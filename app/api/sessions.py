from __future__ import annotations

import logging
from datetime import datetime

import orjson
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db,
    get_kb,
    optional_client_id,
    require_client_id,
    session_not_open,
    session_or_404,
    wait_for_plan_ready,
)
from app.chat.closing import ask_to_close_from_history
from app.content.knowledge_base import CPIndex, KnowledgeBase
from app.db import repository
from app.db.models import SESSION_CLOSED, SESSION_OPEN, SESSION_TEACHER_REQUESTED, ChatMessage, ChatSession
from app.schemas import (
    ChatListResponse,
    ChatSummary,
    CreateSessionRequest,
    CreateSessionResponse,
    MessageOut,
    SessionContext,
    SessionDetail,
    SessionStatusResponse,
    Source,
    TeacherRequestBody,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None

_STARTER_TEMPLATES = [
    "Explain {concept} in simple words",
    "What is {subconcept}?",
    "Give a real-life example of {concept}",
]


def _build_starter_questions(cp_index: CPIndex) -> list[str]:
    """Deterministic, template-based — no LLM. Picks concepts from different chapters."""
    by_chapter: dict[int, list] = {}
    for p in cp_index.positions:
        by_chapter.setdefault(p.chapter_pos, []).append(p)

    picks = [positions[0] for _chapter_pos, positions in sorted(by_chapter.items())][:4]
    if len(picks) < 4:
        for positions in by_chapter.values():
            for p in positions:
                if p not in picks:
                    picks.append(p)
                if len(picks) >= 4:
                    break
            if len(picks) >= 4:
                break

    questions: list[str] = []
    for i, p in enumerate(picks[:4]):
        if "numerical" in p.subconcept_name.lower():
            questions.append(f"Solve a numerical on {p.concept_name}")
        else:
            template = _STARTER_TEMPLATES[i % len(_STARTER_TEMPLATES)]
            questions.append(template.format(concept=p.concept_name, subconcept=p.subconcept_name))
    return questions


def _message_to_out(m: ChatMessage) -> MessageOut:
    sources: list[Source] = []
    if m.sources_json:
        sources = [Source.model_validate(s) for s in orjson.loads(m.sources_json)]
    debug: dict | None = None
    if m.role == "assistant":
        router = orjson.loads(m.router_json) if m.router_json else {}
        debug = {
            "reason": router.get("reason"),
            "matchedTopics": router.get("matched_topic_codes"),
            "usage": orjson.loads(m.usage_json) if m.usage_json else None,
            "latencyMs": orjson.loads(m.latency_json) if m.latency_json else None,
            "costUsd": m.cost_usd,
        }
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        status=m.status,
        verdict=m.verdict,
        suggested_subject=None,
        sources=sources,
        created_at=m.created_at.isoformat(),
        debug=debug,
    )


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    req: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
    kb: KnowledgeBase = Depends(get_kb),
    client_id: str | None = Depends(optional_client_id),
) -> CreateSessionResponse:
    """One open chat per student per subject: if the student (X-Client-Id) already has an open
    chat for this subject it is returned with `resumed: true` instead of creating another. To
    start fresh, the frontend closes the open chat first (POST /sessions/{id}/close)."""
    cp_indices = await wait_for_plan_ready(kb, req.plan_id)
    cp_index = cp_indices.get(req.cp_id)
    if cp_index is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "cp_not_found", "message": "Subject not found in this plan"}},
        )

    session = None
    if client_id is not None:
        session = await repository.find_open_session(db, client_id=client_id, plan_id=req.plan_id, cp_id=req.cp_id)
    resumed = session is not None
    if session is None:
        session = await repository.create_session(
            db,
            plan_id=req.plan_id,
            cp_id=req.cp_id,
            medium=cp_index.meta.medium_name,
            grade=cp_index.meta.grade_name,
            board=cp_index.meta.board_name,
            subject=cp_index.meta.subject_name,
            client_id=client_id,
        )

    return CreateSessionResponse(
        session_id=session.id,
        context=SessionContext(
            medium=cp_index.meta.medium_name,
            grade=cp_index.meta.grade_name,
            board=cp_index.meta.board_name,
            subject=cp_index.meta.subject_name,
            cp_id=req.cp_id,
            plan_id=req.plan_id,
        ),
        starter_questions=_build_starter_questions(cp_index),
        resumed=resumed,
    )


@router.get("/sessions", response_model=ChatListResponse)
async def list_sessions(
    client_id: str = Depends(require_client_id), db: AsyncSession = Depends(get_db)
) -> ChatListResponse:
    """The student's chat history across all subjects (chats with at least one question),
    most recently active first."""
    rows = await repository.list_client_sessions(db, client_id)
    return ChatListResponse(
        sessions=[
            ChatSummary(
                session_id=s.id,
                plan_id=s.plan_id,
                cp_id=s.cp_id,
                medium=s.medium,
                grade=s.grade,
                board=s.board,
                subject=s.subject,
                title=s.title or "",
                status=s.status,
                created_at=s.created_at.isoformat(),
                updated_at=(s.updated_at or s.created_at).isoformat(),
                closed_at=_iso(s.closed_at),
                rating=rating,
            )
            for s, rating in rows
        ]
    )


async def _session_detail(db: AsyncSession, session: ChatSession) -> SessionDetail:
    messages = await repository.list_messages(db, session.id)
    feedback = await repository.get_session_feedback(db, session.id)
    teacher_request = (
        await repository.get_teacher_request(db, session.id) if session.status == SESSION_TEACHER_REQUESTED else None
    )
    return SessionDetail(
        session_id=session.id,
        medium=session.medium,
        grade=session.grade,
        board=session.board,
        subject=session.subject,
        cp_id=session.cp_id,
        plan_id=session.plan_id,
        status=session.status,
        title=session.title,
        ask_to_close=ask_to_close_from_history(session, messages),
        rating=feedback.rating if feedback else None,
        closed_at=_iso(session.closed_at),
        teacher_requested_at=_iso(teacher_request.created_at) if teacher_request else None,
    )


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)) -> SessionDetail:
    session = await session_or_404(db, session_id)
    return await _session_detail(db, session)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def get_messages(session_id: str, db: AsyncSession = Depends(get_db)) -> list[MessageOut]:
    await session_or_404(db, session_id)
    messages = await repository.list_messages(db, session_id)
    return [_message_to_out(m) for m in messages]


@router.post("/sessions/{session_id}/close", response_model=SessionStatusResponse)
async def close_session(session_id: str, db: AsyncSession = Depends(get_db)) -> SessionStatusResponse:
    """The student marks the doubt solved (or starts a new chat). Idempotent; the chat becomes
    read-only history and can then be rated."""
    session = await session_or_404(db, session_id)
    if session.status == SESSION_TEACHER_REQUESTED:
        raise session_not_open("This doubt was already sent to a teacher.")
    if session.status == SESSION_OPEN:
        session = await repository.close_session(db, session)
    return SessionStatusResponse(session_id=session.id, status=session.status, closed_at=_iso(session.closed_at))


@router.post("/sessions/{session_id}/teacher-request", response_model=SessionStatusResponse)
async def request_teacher(
    session_id: str, body: TeacherRequestBody, db: AsyncSession = Depends(get_db)
) -> SessionStatusResponse:
    """The doubt isn't solved and the student wants a teacher to call them. Idempotent; the chat
    becomes read-only and is handed over (see TeacherRequest)."""
    session = await session_or_404(db, session_id)
    if session.status == SESSION_CLOSED:
        raise session_not_open("This doubt is already closed. Start a new chat to ask again.")
    if session.status == SESSION_OPEN:
        note = body.note.strip() if body.note and body.note.strip() else None
        teacher_request = await repository.request_teacher(db, session, note)
        logger.info("teacher call requested: session=%s subject=%s", session.id, session.subject)
    else:
        teacher_request = await repository.get_teacher_request(db, session.id)
    return SessionStatusResponse(
        session_id=session.id,
        status=session.status,
        teacher_requested_at=_iso(teacher_request.created_at) if teacher_request else None,
    )
