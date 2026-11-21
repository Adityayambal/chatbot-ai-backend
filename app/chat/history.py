"""Formats stored conversation turns for the router (plain text) and tutor (OpenAI chat
messages)."""
from __future__ import annotations

from app.db.repository import Turn

ROUTER_RECENT_TURNS = 3
ROUTER_ASSISTANT_CUT_CHARS = 300


def build_router_recent(turns: list[Turn]) -> str:
    recent = turns[-ROUTER_RECENT_TURNS:] if turns else []
    if not recent:
        body = "(none — this is the first message in this session)"
    else:
        lines: list[str] = []
        for t in recent:
            lines.append(f"Student: {t.user_text}")
            lines.append(f"Tutor: {t.assistant_text[:ROUTER_ASSISTANT_CUT_CHARS]}")
        body = "\n".join(lines)
    return f"<recent_conversation>\n{body}\n</recent_conversation>"


def build_tutor_history(turns: list[Turn]) -> list[dict]:
    """One user/assistant message pair per turn. `turns` should already be limited to
    MAX_HISTORY_TURNS by the caller."""
    messages: list[dict] = []
    for t in turns:
        messages.append({"role": "user", "content": t.user_text})
        messages.append({"role": "assistant", "content": t.assistant_text})
    return messages
