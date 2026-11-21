"""Pydantic models for REST request/response bodies (not the SSE event payloads —
those live in app/chat/events.py)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

SessionStatus = Literal["open", "closed", "teacher_requested"]


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class CatalogOption(CamelModel):
    cp_id: str
    cp_name: str
    plan_id: str
    medium_id: str
    medium_name: str
    grade_id: str
    grade_name: str
    board_id: str
    board_name: str
    subject_id: str
    subject_name: str
    chapter_count: int
    concept_count: int
    is_active: bool


class PlanInfo(CamelModel):
    plan_id: str
    plan_name: str


class CatalogResponse(CamelModel):
    plans: list[PlanInfo]
    options: list[CatalogOption]
    last_synced_at: str | None = None


class OutlineSubconcept(CamelModel):
    code: str
    name: str


class ContentCounts(CamelModel):
    notes: int
    videos: int
    questions: int


class OutlineConcept(CamelModel):
    code: str
    name: str
    subconcepts: list[OutlineSubconcept]


class OutlineChapter(CamelModel):
    code: str
    name: str
    counts: ContentCounts
    concepts: list[OutlineConcept]


class OutlineResponse(CamelModel):
    cp_id: str
    subject: str
    chapters: list[OutlineChapter]


class CreateSessionRequest(CamelModel):
    plan_id: str
    cp_id: str


class SessionContext(CamelModel):
    medium: str
    grade: str
    board: str
    subject: str
    cp_id: str
    plan_id: str


class CreateSessionResponse(CamelModel):
    session_id: str
    context: SessionContext
    starter_questions: list[str]
    resumed: bool = False  # true when the student's open chat for this subject was returned


class SessionDetail(SessionContext):
    session_id: str
    status: SessionStatus
    title: str | None = None
    ask_to_close: bool = False
    rating: int | None = None
    closed_at: str | None = None
    teacher_requested_at: str | None = None


class ChatSummary(CamelModel):
    session_id: str
    plan_id: str
    cp_id: str
    medium: str
    grade: str
    board: str
    subject: str
    title: str
    status: SessionStatus
    created_at: str
    updated_at: str
    closed_at: str | None = None
    rating: int | None = None


class ChatListResponse(CamelModel):
    sessions: list[ChatSummary]


class TeacherRequestBody(CamelModel):
    note: str | None = Field(default=None, max_length=500)


class SessionStatusResponse(CamelModel):
    session_id: str
    status: SessionStatus
    closed_at: str | None = None
    teacher_requested_at: str | None = None


class SessionFeedbackRequest(CamelModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)


class Source(CamelModel):
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


class MessageOut(CamelModel):
    id: str
    role: str
    content: str
    status: str
    verdict: str | None = None
    suggested_subject: str | None = None
    sources: list[Source] = []
    created_at: str
    debug: dict | None = None


class ChatRequest(CamelModel):
    session_id: str
    message: str
    retry_of_message_id: str | None = None


class FeedbackRequest(CamelModel):
    rating: int
    comment: str | None = None


class FeedbackResponse(CamelModel):
    ok: bool = True


class AdminReloadRequest(CamelModel):
    plan_ids: list[str]


class AdminReloadResponse(CamelModel):
    started: bool = True


class HealthResponse(CamelModel):
    status: str
    plans: dict[str, str]
    models: dict[str, str]
