"""Step A of the chat flow: a single non-streaming OpenAI call that classifies the
message and (for related questions) picks outline topics + search keywords."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import openai
from pydantic import ValidationError

from app.chat.history import build_router_recent
from app.config import Settings
from app.content.knowledge_base import CPIndex
from app.db.repository import Turn
from app.llm.client import call_usage
from app.llm.schemas import ROUTER_SCHEMA, CallUsage, RouterDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM = (Path(__file__).parent / "prompts" / "router_system.md").read_text(encoding="utf-8")

_MAX_CODES = 6
_MAX_QUERIES = 6
_MAX_ATTEMPTS = 2


def _build_outline_block(cp_index: CPIndex) -> str:
    """Deterministic (no timestamps/session IDs) so system prompt + this block form a stable
    prefix that OpenAI's automatic prompt caching hits on every question about the subject."""
    meta = cp_index.meta
    subject_block = (
        "<subject>\n"
        f"medium: {meta.medium_name} | grade: {meta.grade_name} | board: {meta.board_name} | "
        f"subject: {meta.subject_name}\n"
        "</subject>"
    )
    other_subjects = sorted({s.meta.subject_name for s in cp_index.sibling_cp_indices})
    other_body = "\n".join(other_subjects) if other_subjects else "(none)"
    other_block = f"<other_subjects_in_plan>\n{other_body}\n</other_subjects_in_plan>"
    outline_block = f"<subject_outline>\n{cp_index.outline_text}\n</subject_outline>"
    return f"{subject_block}\n\n{other_block}\n\n{outline_block}"


def _build_turn_block(recent_turns: list[Turn], message: str) -> str:
    recent = build_router_recent(recent_turns)
    return f"{recent}\n\n<new_message>\n{message}\n</new_message>"


def _clean_decision(cp_index: CPIndex, decision: RouterDecision) -> RouterDecision:
    """Drops topic codes that don't exist in this CP's outline and truncates server-side
    (the schema deliberately has no maxItems, per spec)."""
    valid_codes = set(cp_index.code_to_subconcept_id) | set(cp_index.concept_code_to_subconcept_codes)
    codes = [c for c in decision.matched_topic_codes if c in valid_codes][:_MAX_CODES]
    queries = decision.search_queries[:_MAX_QUERIES]
    return decision.model_copy(update={"matched_topic_codes": codes, "search_queries": queries})


def _fallback_decision(message: str) -> RouterDecision:
    return RouterDecision(
        verdict="related",
        reason="router returned invalid output twice; using a BM25-only fallback",
        standalone_question=message,
        reply_language="English",
        question_type="other",
        matched_topic_codes=[],
        search_queries=[message],
        suggested_subject=None,
        short_reply=None,
    )


async def _call_once(
    client: openai.AsyncOpenAI, settings: Settings, outline_block: str, turn_block: str
) -> tuple[RouterDecision, CallUsage]:
    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=settings.router_model,
        max_completion_tokens=settings.router_max_tokens,
        reasoning_effort=settings.router_effort,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "router_decision", "strict": True, "schema": ROUTER_SCHEMA},
        },
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": outline_block},
                    {"type": "text", "text": turn_block},
                ],
            },
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = response.choices[0]
    if choice.message.refusal:
        raise ValueError(f"router refused: {choice.message.refusal}")
    if not choice.message.content:
        raise ValueError(f"router response had no content (finish_reason={choice.finish_reason})")

    payload = json.loads(choice.message.content)
    decision = RouterDecision.model_validate(payload)
    return decision, call_usage(settings.router_model, response.usage, choice.finish_reason, latency_ms)


async def route(
    client: openai.AsyncOpenAI,
    cp_index: CPIndex,
    recent_turns: list[Turn],
    message: str,
    settings: Settings,
) -> tuple[RouterDecision, CallUsage]:
    """OpenAI API errors (rate limit, connection, 5xx, bad request) propagate to the
    caller so the orchestrator can map them to the right SSE error code — only invalid
    JSON/schema output from the model is retried here, per §4.9."""
    outline_block = _build_outline_block(cp_index)
    turn_block = _build_turn_block(recent_turns, message)

    for attempt in range(_MAX_ATTEMPTS):
        try:
            decision, usage = await _call_once(client, settings, outline_block, turn_block)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("router produced invalid output on attempt %d: %s", attempt + 1, exc)
            continue
        return _clean_decision(cp_index, decision), usage

    logger.error("router failed validation twice; falling back for message=%r", message[:200])
    return _fallback_decision(message), CallUsage(model=settings.router_model)
