"""Pydantic models for every SSE event in PROJECT_SPEC.md §10, plus the sse() helper
that serializes one to a sse_starlette ServerSentEvent."""
from __future__ import annotations

from typing import Literal

import orjson
from pydantic import BaseModel
from sse_starlette import ServerSentEvent

from app.schemas import CamelModel


class MetaEvent(CamelModel):
    user_message_id: str
    assistant_message_id: str


class StatusEvent(CamelModel):
    stage: Literal["checking", "searching", "writing"]


class MatchedTopic(CamelModel):
    code: str
    chapter: str
    concept: str
    subconcept: str


class RouteEvent(CamelModel):
    verdict: str
    reason: str
    matched_topics: list[MatchedTopic]
    question_type: str
    suggested_subject: str | None = None


class SourceEvent(CamelModel):
    id: str
    type: str
    title: str
    chapter: str
    concept: str
    subconcept: str
    snippet: str
    video_url: str | None = None
    duration_sec: int | None = None
    difficulty: int | None = None
    images: list[str] = []


class DeltaEvent(CamelModel):
    text: str


class DoneEvent(CamelModel):
    message_id: str
    verdict: str | None
    status: Literal["complete", "interrupted"]
    usage: dict
    latency_ms: dict
    cost_usd: float
    ask_to_close: bool = False  # show "Did this solve your doubt?" under this answer


class ErrorEvent(CamelModel):
    code: Literal["busy", "ai_unavailable", "network", "bad_request", "rate_limited", "session_not_found"]
    message: str
    retry_after: int | None = None


def sse(name: str, data: BaseModel | list[BaseModel]) -> ServerSentEvent:
    if isinstance(data, list):
        payload = [item.model_dump(by_alias=True) for item in data]
    else:
        payload = data.model_dump(by_alias=True)
    return ServerSentEvent(event=name, data=orjson.dumps(payload).decode())
