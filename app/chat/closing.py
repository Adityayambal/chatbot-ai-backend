"""When the UI should ask "Did this solve your doubt?" under the latest answer."""
from __future__ import annotations

import orjson

from app.db.models import SESSION_OPEN, ChatMessage, ChatSession
from app.db.repository import COMPLETE_ASSISTANT_STATUSES


def should_ask_to_close(
    *, status: str, verdict: str | None, question_type: str | None, had_related_answer: bool
) -> bool:
    """Ask only at a natural end point: after a complete answer to a subject doubt, or after a
    "thanks/bye" once a doubt has been answered. Never while a test is in progress (the answer is
    a set of test questions the student still has to answer), after a clarifying question, an
    off-topic or unsafe reply, or an incomplete/failed answer."""
    if status != "complete":
        return False
    if verdict == "related":
        return question_type != "test_request"
    if verdict == "greeting":
        return had_related_answer
    return False


def ask_to_close_from_history(session: ChatSession, messages: list[ChatMessage]) -> bool:
    """Same rule, recomputed from stored messages for a chat being resumed."""
    if session.status != SESSION_OPEN:
        return False
    visible = [m for m in messages if m.status != "superseded"]
    if not visible or visible[-1].role != "assistant":
        return False
    last = visible[-1]
    question_type = orjson.loads(last.router_json).get("question_type") if last.router_json else None
    had_related_answer = any(
        m.role == "assistant" and m.verdict == "related" and m.status in COMPLETE_ASSISTANT_STATUSES
        for m in visible[:-1]
    )
    return should_ask_to_close(
        status=last.status, verdict=last.verdict, question_type=question_type, had_related_answer=had_related_answer
    )
