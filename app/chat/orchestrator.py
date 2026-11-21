"""Ties together the router, retriever and tutor into the SSE event stream in §4.4-4.9."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import openai
import orjson
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import ServerSentEvent
from starlette.requests import Request

from app.chat.closing import should_ask_to_close
from app.chat.cost import estimate_cost
from app.chat.events import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    MatchedTopic,
    MetaEvent,
    RouteEvent,
    SourceEvent,
    StatusEvent,
    sse,
)
from app.chat.history import build_tutor_history
from app.chat.rate_limit import SlidingWindowLimiter
from app.config import Settings
from app.content.knowledge_base import CPIndex, KnowledgeBase
from app.db import repository
from app.db.models import ChatMessage
from app.db.repository import Turn
from app.llm.router import route as router_route
from app.llm.schemas import CallUsage, RouterDecision
from app.llm.tutor import TutorDelta, build_current_turn, stream_answer
from app.retrieval.retriever import RetrievedChunk, retrieve
from app.retrieval.tokenize import tokenize
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)

REFUSAL_FALLBACK_MESSAGE = (
    "Sorry, I wasn't able to put together a good answer to that question. Could you try "
    "rephrasing it, or ask about a different part of the topic?"
)
MAX_TOKENS_SUFFIX = "\n\n_(answer was cut short — ask me to continue)_"
NOT_RELATED_VERDICTS = {"not_related", "greeting", "unclear", "unsafe"}
EMPTY_USAGE = {"router": None, "tutor": None}


def _classify_llm_error(exc: Exception) -> tuple[str, str]:
    """Most-to-least-specific, per §8.3. Returns (sse error code, user-facing message)."""
    if isinstance(exc, openai.RateLimitError):
        # OpenAI reports an empty/over-limit billing account as a 429 too; that's not "busy".
        if exc.code == "insufficient_quota":
            logger.error("OpenAI account is out of quota — add billing at platform.openai.com: %s", exc)
            return "ai_unavailable", "The AI service is temporarily unavailable."
        return "busy", "The tutor is busy right now. Please try again in a moment."
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return "ai_unavailable", "The AI service is temporarily unavailable."
    if isinstance(exc, openai.APIConnectionError):
        return "network", "Could not reach the AI service. Please check your connection and retry."
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
        logger.error(
            "OpenAI API rejected the request — check OPENAI_API_KEY and ROUTER_MODEL/ANSWER_MODEL "
            "in backend/.env: %s",
            exc,
        )
        return "ai_unavailable", "The AI service is temporarily unavailable."
    if isinstance(exc, openai.BadRequestError):
        logger.error("BadRequestError from OpenAI API (likely a config bug): %s", exc)
        return "bad_request", "Something went wrong preparing that request."
    logger.error("Unexpected error calling OpenAI API: %s", exc)
    return "ai_unavailable", "The AI service is temporarily unavailable."


def _code_lookup(cp_index: CPIndex) -> tuple[dict, dict]:
    by_topic_code = {p.topic_code: p for p in cp_index.positions}
    by_concept_code: dict = {}
    for p in cp_index.positions:
        by_concept_code.setdefault(p.concept_code, p)
    return by_topic_code, by_concept_code


def _matched_topics(cp_index: CPIndex, codes: list[str]) -> list[MatchedTopic]:
    by_topic, by_concept = _code_lookup(cp_index)
    topics: list[MatchedTopic] = []
    for code in codes:
        p = by_topic.get(code)
        if p is not None:
            topics.append(
                MatchedTopic(code=code, chapter=p.chapter_name, concept=p.concept_name, subconcept=p.subconcept_name)
            )
            continue
        p = by_concept.get(code)
        if p is not None:
            topics.append(MatchedTopic(code=code, chapter=p.chapter_name, concept=p.concept_name, subconcept=""))
    return topics


def _snippet_body(text: str, max_chars: int = 2000) -> str:
    # Large enough to carry a whole note chunk (NOTE_CHUNK_MAX_CHARS) so the UI can show the
    # full source on demand; the frontend clamps it to a short preview by default.
    body = text.split("\n\n", 1)[-1] if "\n\n" in text else text
    body = body.strip()
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rsplit(" ", 1)[0] + "…"


def _source_events(retrieved: list[RetrievedChunk]) -> list[SourceEvent]:
    return [
        SourceEvent(
            id=rc.source_id,
            type=rc.chunk.content_type,
            title=rc.chunk.title,
            chapter=rc.chunk.chapter_name,
            concept=rc.chunk.concept_name,
            subconcept=rc.chunk.subconcept_name,
            snippet=_snippet_body(rc.chunk.text),
            video_url=rc.chunk.video_url,
            duration_sec=rc.chunk.duration_sec,
            difficulty=rc.chunk.difficulty,
            images=[img.url for img in rc.chunk.images],
        )
        for rc in retrieved
    ]


def _closest_topics(cp_index: CPIndex, query_tokens: list[str], n: int = 3) -> list[str]:
    hits = cp_index.bm25.search(query_tokens, top_n=20)
    seen: set[str] = set()
    names: list[str] = []
    for chunk, _score in hits:
        if chunk.topic_code in seen:
            continue
        seen.add(chunk.topic_code)
        names.append(f"{chunk.chapter_name} › {chunk.concept_name} › {chunk.subconcept_name}")
        if len(names) >= n:
            break
    return names


@dataclass
class _Latency:
    router_ms: float
    retrieval_ms: float = 0.0
    tutor_ms: float = 0.0
    t_start: float = 0.0

    def as_dict(self) -> dict:
        total_ms = (time.perf_counter() - self.t_start) * 1000
        return {"router": self.router_ms, "retrieval": self.retrieval_ms, "tutor": self.tutor_ms, "total": total_ms}


async def _persist_turn(
    db: AsyncSession,
    *,
    assistant_message_id: str,
    session_id: str,
    content: str,
    decision: RouterDecision,
    sources: list[SourceEvent],
    router_usage: CallUsage,
    tutor_usage: CallUsage | None,
    latency: _Latency,
    status: str,
    ask_to_close: bool = False,
) -> ServerSentEvent | None:
    """Persists the assistant message and returns the `done` SSE event — or None if
    `status == "error"`, since DoneEvent.status only admits complete/interrupted; the
    caller sends an `error` event instead in that case."""
    router_cost = estimate_cost(router_usage.model, router_usage)
    tutor_cost = estimate_cost(tutor_usage.model, tutor_usage) if tutor_usage else 0.0
    usage = {"router": vars(router_usage), "tutor": vars(tutor_usage) if tutor_usage else {}}
    latency_dict = latency.as_dict()
    total_cost = router_cost + tutor_cost

    await repository.add_message(
        db,
        message_id=assistant_message_id,
        session_id=session_id,
        role="assistant",
        content=content,
        verdict=decision.verdict,
        router_json=orjson.dumps(decision.model_dump()).decode(),
        sources_json=orjson.dumps([s.model_dump(by_alias=True) for s in sources]).decode(),
        usage_json=orjson.dumps(usage).decode(),
        latency_json=orjson.dumps(latency_dict).decode(),
        cost_usd=total_cost,
        status=status,
    )
    if status == "error":
        return None
    return sse(
        "done",
        DoneEvent(
            message_id=assistant_message_id,
            verdict=decision.verdict,
            status=status,
            usage=usage,
            latency_ms=latency_dict,
            cost_usd=total_cost,
            ask_to_close=ask_to_close,
        ),
    )


class ChatOrchestrator:
    def __init__(
        self,
        settings: Settings,
        kb: KnowledgeBase,
        llm_client: openai.AsyncOpenAI,
        rate_limiter: SlidingWindowLimiter,
    ) -> None:
        self._settings = settings
        self._kb = kb
        self._client = llm_client
        self._rate_limiter = rate_limiter

    async def run(
        self, db: AsyncSession, request: Request, chat_req: ChatRequest
    ) -> AsyncIterator[ServerSentEvent]:
        settings = self._settings
        t_start = time.perf_counter()

        retry_after = self._rate_limiter.check(chat_req.session_id)
        if retry_after is not None:
            yield sse(
                "error",
                ErrorEvent(
                    code="rate_limited",
                    message="Too many questions — please wait a moment.",
                    retry_after=round(retry_after) or 1,
                ),
            )
            return

        session = await repository.get_session(db, chat_req.session_id)
        if session is None:
            yield sse("error", ErrorEvent(code="session_not_found", message="This session no longer exists."))
            return

        if chat_req.retry_of_message_id:
            user_message = await db.get(ChatMessage, chat_req.retry_of_message_id)
            await repository.supersede_after(db, chat_req.session_id, chat_req.retry_of_message_id)
            message = user_message.content
        else:
            user_message = await repository.add_message(
                db, session_id=chat_req.session_id, role="user", content=chat_req.message, status="complete"
            )
            message = chat_req.message

        assistant_message_id = str(uuid.uuid4())
        yield sse("meta", MetaEvent(user_message_id=user_message.id, assistant_message_id=assistant_message_id))
        yield sse("status", StatusEvent(stage="checking"))

        cp_indices = self._kb.get_plan(session.plan_id) or {}
        cp_index = cp_indices.get(session.cp_id)
        if cp_index is None:
            yield sse(
                "error",
                ErrorEvent(code="ai_unavailable", message="This subject's lessons are not available right now."),
            )
            return

        recent_turns: list[Turn] = await repository.recent_complete_turns(
            db, chat_req.session_id, settings.max_history_turns
        )

        t_router = time.perf_counter()
        try:
            decision, router_usage = await router_route(self._client, cp_index, recent_turns, message, settings)
        except Exception as exc:  # noqa: BLE001 - classify and surface as an SSE error
            code, friendly = _classify_llm_error(exc)
            yield sse("error", ErrorEvent(code=code, message=friendly))
            return
        router_latency_ms = (time.perf_counter() - t_router) * 1000

        yield sse(
            "route",
            RouteEvent(
                verdict=decision.verdict,
                reason=decision.reason,
                matched_topics=_matched_topics(cp_index, decision.matched_topic_codes),
                question_type=decision.question_type,
                suggested_subject=decision.suggested_subject,
            ),
        )

        had_related_answer = any(t.assistant_verdict == "related" for t in recent_turns)

        if decision.verdict in NOT_RELATED_VERDICTS:
            content = decision.short_reply or ""
            yield sse("delta", DeltaEvent(text=content))
            yield await _persist_turn(
                db,
                assistant_message_id=assistant_message_id,
                session_id=chat_req.session_id,
                content=content,
                decision=decision,
                sources=[],
                router_usage=router_usage,
                tutor_usage=None,
                latency=_Latency(router_ms=router_latency_ms, t_start=t_start),
                status="complete",
                ask_to_close=should_ask_to_close(
                    status="complete",
                    verdict=decision.verdict,
                    question_type=decision.question_type,
                    had_related_answer=had_related_answer,
                ),
            )
            return

        yield sse("status", StatusEvent(stage="searching"))
        t_retrieval = time.perf_counter()
        retrieved = retrieve(cp_index, decision, settings)
        retrieval_latency_ms = (time.perf_counter() - t_retrieval) * 1000

        if not retrieved and settings.strict_grounding:
            query_tokens = tokenize(decision.standalone_question + " " + " ".join(decision.search_queries))
            topics = _closest_topics(cp_index, query_tokens)
            lines = "\n".join(f"- {t}" for t in topics)
            content = (
                f"That's not covered in your {cp_index.meta.subject_name} lessons yet. "
                f"Here are some related topics you could ask about instead:\n{lines}"
            )
            yield sse("sources", [])
            yield sse("delta", DeltaEvent(text=content))
            yield await _persist_turn(
                db,
                assistant_message_id=assistant_message_id,
                session_id=chat_req.session_id,
                content=content,
                decision=decision,
                sources=[],
                router_usage=router_usage,
                tutor_usage=None,
                latency=_Latency(router_ms=router_latency_ms, retrieval_ms=retrieval_latency_ms, t_start=t_start),
                status="complete",
            )
            return

        source_events = _source_events(retrieved)
        yield sse("sources", source_events)
        yield sse("status", StatusEvent(stage="writing"))

        history_messages = build_tutor_history(recent_turns)
        current_turn_text = build_current_turn(cp_index.meta, decision, retrieved, message)

        accumulated: list[str] = []
        tutor_usage: CallUsage | None = None
        stop_reason: str | None = None
        status = "complete"
        error_event: ErrorEvent | None = None

        t_tutor = time.perf_counter()
        try:
            async for event in stream_answer(self._client, settings, history_messages, current_turn_text):
                if isinstance(event, TutorDelta):
                    if await request.is_disconnected():
                        status = "interrupted"
                        break
                    accumulated.append(event.text)
                    yield sse("delta", DeltaEvent(text=event.text))
                else:
                    tutor_usage = event.usage
                    stop_reason = event.stop_reason
        except Exception as exc:  # noqa: BLE001 - classify and surface as an SSE error
            code, friendly = _classify_llm_error(exc)
            error_event = ErrorEvent(code=code, message=friendly)
            status = "error"
        tutor_latency_ms = (time.perf_counter() - t_tutor) * 1000

        final_text = "".join(accumulated)
        if status == "complete":
            if stop_reason == "content_filter":
                yield sse("delta", DeltaEvent(text=REFUSAL_FALLBACK_MESSAGE))
                if final_text:
                    final_text = f"{final_text}\n\n{REFUSAL_FALLBACK_MESSAGE}".strip()
                else:
                    final_text = REFUSAL_FALLBACK_MESSAGE
            elif stop_reason == "length":
                yield sse("delta", DeltaEvent(text=MAX_TOKENS_SUFFIX))
                final_text += MAX_TOKENS_SUFFIX

        done_event = await _persist_turn(
            db,
            assistant_message_id=assistant_message_id,
            session_id=chat_req.session_id,
            content=final_text,
            decision=decision,
            sources=source_events,
            router_usage=router_usage,
            tutor_usage=tutor_usage or CallUsage(model=settings.answer_model),
            latency=_Latency(
                router_ms=router_latency_ms,
                retrieval_ms=retrieval_latency_ms,
                tutor_ms=tutor_latency_ms,
                t_start=t_start,
            ),
            status=status,
            ask_to_close=stop_reason != "content_filter"
            and should_ask_to_close(
                status=status,
                verdict=decision.verdict,
                question_type=decision.question_type,
                had_related_answer=had_related_answer,
            ),
        )

        if status == "error":
            assert error_event is not None
            yield sse("error", error_event)
            return

        yield done_event
