"""All SQLite access goes through this module — no raw queries elsewhere."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    SESSION_CLOSED,
    SESSION_OPEN,
    SESSION_TEACHER_REQUESTED,
    ChatMessage,
    ChatSession,
    Feedback,
    SessionFeedback,
    TeacherRequest,
)

COMPLETE_ASSISTANT_STATUSES = ("complete", "interrupted")
TITLE_MAX_CHARS = 120


@dataclass
class Turn:
    user_text: str
    assistant_text: str
    assistant_verdict: str | None = None


async def create_session(
    db: AsyncSession,
    *,
    plan_id: str,
    cp_id: str,
    medium: str,
    grade: str,
    board: str,
    subject: str,
    client_id: str | None = None,
) -> ChatSession:
    session = ChatSession(
        plan_id=plan_id,
        cp_id=cp_id,
        medium=medium,
        grade=grade,
        board=board,
        subject=subject,
        client_id=client_id,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: str) -> ChatSession | None:
    return await db.get(ChatSession, session_id)


async def find_open_session(db: AsyncSession, *, client_id: str, plan_id: str, cp_id: str) -> ChatSession | None:
    result = await db.execute(
        select(ChatSession)
        .where(
            ChatSession.client_id == client_id,
            ChatSession.plan_id == plan_id,
            ChatSession.cp_id == cp_id,
            ChatSession.status == SESSION_OPEN,
        )
        .order_by(ChatSession.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def list_client_sessions(db: AsyncSession, client_id: str) -> list[tuple[ChatSession, int | None]]:
    """The student's chats that have at least one question, most recently active first, each
    with its star rating (or None)."""
    result = await db.execute(
        select(ChatSession, SessionFeedback.rating)
        .outerjoin(SessionFeedback, SessionFeedback.session_id == ChatSession.id)
        .where(ChatSession.client_id == client_id, ChatSession.title.is_not(None))
        .order_by(ChatSession.updated_at.desc())
    )
    return [(row[0], row[1]) for row in result.all()]


async def close_session(db: AsyncSession, session: ChatSession) -> ChatSession:
    session.status = SESSION_CLOSED
    session.closed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(session)
    return session


async def request_teacher(db: AsyncSession, session: ChatSession, note: str | None) -> TeacherRequest:
    request = TeacherRequest(session_id=session.id, note=note)
    session.status = SESSION_TEACHER_REQUESTED
    db.add(request)
    await db.commit()
    await db.refresh(request)
    return request


async def get_teacher_request(db: AsyncSession, session_id: str) -> TeacherRequest | None:
    result = await db.execute(select(TeacherRequest).where(TeacherRequest.session_id == session_id))
    return result.scalars().first()


async def get_session_feedback(db: AsyncSession, session_id: str) -> SessionFeedback | None:
    result = await db.execute(select(SessionFeedback).where(SessionFeedback.session_id == session_id))
    return result.scalars().first()


async def add_session_feedback(
    db: AsyncSession, *, session_id: str, rating: int, comment: str | None
) -> SessionFeedback:
    feedback = SessionFeedback(session_id=session_id, rating=rating, comment=comment)
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return feedback


async def add_message(
    db: AsyncSession,
    *,
    session_id: str,
    role: str,
    content: str = "",
    message_id: str | None = None,
    verdict: str | None = None,
    router_json: str | None = None,
    sources_json: str | None = None,
    usage_json: str | None = None,
    latency_json: str | None = None,
    cost_usd: float | None = None,
    status: str = "complete",
) -> ChatMessage:
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        verdict=verdict,
        router_json=router_json,
        sources_json=sources_json,
        usage_json=usage_json,
        latency_json=latency_json,
        cost_usd=cost_usd,
        status=status,
    )
    if message_id is not None:
        message.id = message_id
    db.add(message)
    session = await db.get(ChatSession, session_id)
    if session is not None:
        session.updated_at = datetime.now(timezone.utc)
        if role == "user" and not session.title:
            session.title = " ".join(content.split())[:TITLE_MAX_CHARS]
    await db.commit()
    await db.refresh(message)
    return message


async def update_message(
    db: AsyncSession,
    message_id: str,
    *,
    content: str | None = None,
    verdict: str | None = None,
    router_json: str | None = None,
    sources_json: str | None = None,
    usage_json: str | None = None,
    latency_json: str | None = None,
    cost_usd: float | None = None,
    status: str | None = None,
) -> ChatMessage | None:
    message = await db.get(ChatMessage, message_id)
    if message is None:
        return None
    if content is not None:
        message.content = content
    if verdict is not None:
        message.verdict = verdict
    if router_json is not None:
        message.router_json = router_json
    if sources_json is not None:
        message.sources_json = sources_json
    if usage_json is not None:
        message.usage_json = usage_json
    if latency_json is not None:
        message.latency_json = latency_json
    if cost_usd is not None:
        message.cost_usd = cost_usd
    if status is not None:
        message.status = status
    await db.commit()
    await db.refresh(message)
    return message


async def list_messages(db: AsyncSession, session_id: str) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    return list(result.scalars().all())


async def recent_complete_turns(db: AsyncSession, session_id: str, limit: int) -> list[Turn]:
    """Pairs each user message with its (eventual) complete/interrupted assistant reply.

    A user message stays "pending" across any error/superseded assistant attempts so a
    later successful retry still pairs with it; only a new user message ends the pairing
    window. Messages with no qualifying assistant reply (e.g. the just-inserted current
    question) are simply never appended.
    """
    messages = await list_messages(db, session_id)
    turns: list[Turn] = []
    pending_user: ChatMessage | None = None
    for m in messages:
        if m.role == "user":
            pending_user = m
            continue
        if m.role == "assistant" and pending_user is not None and m.status in COMPLETE_ASSISTANT_STATUSES:
            turns.append(
                Turn(user_text=pending_user.content, assistant_text=m.content, assistant_verdict=m.verdict)
            )
    return turns[-limit:] if limit else turns


async def get_latest_user_message(db: AsyncSession, session_id: str) -> ChatMessage | None:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id, ChatMessage.role == "user")
        .order_by(ChatMessage.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def supersede_after(db: AsyncSession, session_id: str, user_message_id: str) -> None:
    """Marks any assistant reply after the given user message as superseded (used on retry)."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at)
    )
    user_message = await db.get(ChatMessage, user_message_id)
    if user_message is None:
        return
    for m in result.scalars().all():
        if m.created_at > user_message.created_at and m.status != "superseded":
            m.status = "superseded"
    await db.commit()


async def add_feedback(db: AsyncSession, *, message_id: str, rating: int, comment: str | None) -> Feedback:
    feedback = Feedback(message_id=message_id, rating=rating, comment=comment)
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return feedback
