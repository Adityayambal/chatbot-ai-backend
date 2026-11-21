from app.chat.history import build_router_recent, build_tutor_history
from app.db.repository import Turn


def test_build_router_recent_handles_first_message():
    text = build_router_recent([])
    assert "first message" in text
    assert text.startswith("<recent_conversation>")


def test_build_router_recent_keeps_only_last_three():
    turns = [Turn(user_text=f"q{i}", assistant_text=f"a{i}") for i in range(5)]
    text = build_router_recent(turns)
    assert "q0" not in text
    assert "q1" not in text
    assert "q2" in text
    assert "q3" in text
    assert "q4" in text


def test_build_router_recent_cuts_assistant_text_to_300_chars():
    long_answer = "x" * 500
    turns = [Turn(user_text="q", assistant_text=long_answer)]
    text = build_router_recent(turns)
    tutor_line = next(line for line in text.splitlines() if line.startswith("Tutor:"))
    assert len(tutor_line) == len("Tutor: ") + 300


def test_build_tutor_history_pairs_user_and_assistant_messages():
    turns = [Turn(user_text="q1", assistant_text="a1"), Turn(user_text="q2", assistant_text="a2")]
    messages = build_tutor_history(turns)
    assert messages == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_build_tutor_history_empty_turns():
    assert build_tutor_history([]) == []
