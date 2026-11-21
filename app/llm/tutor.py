"""Step C of the chat flow: builds the grounded current-turn prompt and streams the
tutor's answer from OpenAI."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import openai

from app.config import Settings
from app.content.knowledge_base import CPMeta
from app.llm.client import call_usage
from app.llm.schemas import CallUsage, RouterDecision
from app.retrieval.retriever import RetrievedChunk

TUTOR_SYSTEM = (Path(__file__).parent / "prompts" / "tutor_system.md").read_text(encoding="utf-8")


@dataclass
class TutorDelta:
    text: str


@dataclass
class TutorFinal:
    text: str
    usage: CallUsage
    stop_reason: str | None


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _render_source(rc: RetrievedChunk) -> str:
    c = rc.chunk
    if c.content_type in ("revision_note", "pdf"):
        attrs = (
            f'id="{rc.source_id}" type="revision_note" chapter="{_escape_attr(c.chapter_name)}" '
            f'concept="{_escape_attr(c.concept_name)}" subconcept="{_escape_attr(c.subconcept_name)}" '
            f'title="{_escape_attr(c.title)}"'
        )
    elif c.content_type == "video_transcript":
        duration_min = round(c.duration_sec / 60) if c.duration_sec else 0
        attrs = (
            f'id="{rc.source_id}" type="video_lesson" title="{_escape_attr(c.title)}" '
            f'duration_min="{duration_min}"'
        )
    else:  # question
        difficulty = c.difficulty if c.difficulty is not None else ""
        attrs = f'id="{rc.source_id}" type="practice_question" difficulty="{difficulty}"'
    return f"<source {attrs}>\n{c.text}\n</source>"


def build_lesson_content_block(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "<lesson_content>None of the lessons matched this question.</lesson_content>"
    sources = "\n".join(_render_source(rc) for rc in retrieved)
    return f"<lesson_content>\n{sources}\n</lesson_content>"


def _learner_block(meta: CPMeta, decision: RouterDecision) -> str:
    return (
        "<learner>\n"
        f"medium: {meta.medium_name} | board: {meta.board_name} | grade: {meta.grade_name} | "
        f"subject: {meta.subject_name}\n"
        f"reply_language: {decision.reply_language}\n"
        f"question_type: {decision.question_type}\n"
        "</learner>"
    )


def _student_question_block(original_message: str, decision: RouterDecision) -> str:
    return (
        "<student_question>\n"
        f"{original_message}\n"
        f"(interpreted as: {decision.standalone_question})\n"
        "</student_question>"
    )


def build_current_turn(
    meta: CPMeta, decision: RouterDecision, retrieved: list[RetrievedChunk], original_message: str
) -> str:
    return "\n\n".join(
        [
            _learner_block(meta, decision),
            build_lesson_content_block(retrieved),
            _student_question_block(original_message, decision),
        ]
    )


async def stream_answer(
    client: openai.AsyncOpenAI,
    settings: Settings,
    history_messages: list[dict],
    current_turn_text: str,
) -> AsyncIterator[TutorDelta | TutorFinal]:
    messages = [
        {"role": "system", "content": TUTOR_SYSTEM},
        *history_messages,
        {"role": "user", "content": current_turn_text},
    ]

    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model=settings.answer_model,
        max_completion_tokens=settings.answer_max_tokens,
        reasoning_effort=settings.answer_effort,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_text_parts: list[str] = []
    finish_reason: str | None = None
    usage = None
    async with stream:
        async for chunk in stream:
            if chunk.usage is not None:  # only on the final chunk, which has no choices
                usage = chunk.usage
            for choice in chunk.choices:
                if choice.delta.content:
                    full_text_parts.append(choice.delta.content)
                    yield TutorDelta(text=choice.delta.content)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

    latency_ms = (time.perf_counter() - t0) * 1000
    yield TutorFinal(
        text="".join(full_text_parts),
        usage=call_usage(settings.answer_model, usage, finish_reason, latency_ms),
        stop_reason=finish_reason,
    )
