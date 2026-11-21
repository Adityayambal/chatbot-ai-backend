"""Chat history + doubt lifecycle: one open chat per student per subject, close → optional
1-5★ rating, teacher hand-off, read-only closed chats, and when to ask "Did this solve your doubt?".

Runs the real app (routes + error handlers) in-process via httpx's ASGI transport, which
doesn't trigger the lifespan, so app.state is wired by hand: fixture KB, test DB, fake LLM.
"""
import sqlite3

import httpx
import orjson
import pytest

from app.chat.closing import should_ask_to_close
from app.chat.rate_limit import SlidingWindowLimiter
from app.db import repository
from app.db.database import init_db, make_engine
from app.main import create_app
from tests.fakes import FIXTURE_PLAN_ID, FakeAsyncOpenAI

CP_ID = "cp0000000000000000000001"
CLIENT = {"X-Client-Id": "student-browser-0001"}
OTHER_CLIENT = {"X-Client-Id": "student-browser-0002"}


@pytest.fixture
async def api(settings, fixture_kb, db_session_factory):
    app = create_app()
    app.state.settings = settings
    app.state.kb = fixture_kb
    app.state.async_session = db_session_factory
    app.state.llm_client = FakeAsyncOpenAI()
    app.state.rate_limiter = SlidingWindowLimiter(100)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _start_chat(api, headers=CLIENT) -> dict:
    resp = await api.post("/api/sessions", json={"planId": FIXTURE_PLAN_ID, "cpId": CP_ID}, headers=headers)
    assert resp.status_code == 200
    return resp.json()


async def _add_turn(db, session_id: str, question: str, *, verdict="related", question_type="definition") -> None:
    await repository.add_message(db, session_id=session_id, role="user", content=question)
    await repository.add_message(
        db,
        session_id=session_id,
        role="assistant",
        content="answer",
        verdict=verdict,
        router_json=orjson.dumps({"question_type": question_type}).decode(),
    )


@pytest.mark.asyncio
async def test_same_student_and_subject_resumes_the_open_chat(api):
    first = await _start_chat(api)
    again = await _start_chat(api)
    other_student = await _start_chat(api, OTHER_CLIENT)

    assert first["resumed"] is False
    assert again["sessionId"] == first["sessionId"]
    assert again["resumed"] is True
    assert other_student["sessionId"] != first["sessionId"]


@pytest.mark.asyncio
async def test_without_client_id_every_start_is_a_new_chat(api):
    a = await _start_chat(api, headers={})
    b = await _start_chat(api, headers={})
    assert a["sessionId"] != b["sessionId"]


@pytest.mark.asyncio
async def test_closing_lets_the_student_start_a_fresh_chat_and_history_lists_both(api, db):
    old = await _start_chat(api)
    await _add_turn(db, old["sessionId"], "What is   displacement?")
    assert (await api.post(f"/api/sessions/{old['sessionId']}/close")).json()["status"] == "closed"

    new = await _start_chat(api)
    assert new["sessionId"] != old["sessionId"] and new["resumed"] is False
    await _add_turn(db, new["sessionId"], "What is velocity?")

    resp = await api.get("/api/sessions", headers=CLIENT)
    assert resp.status_code == 200
    chats = resp.json()["sessions"]
    assert [c["sessionId"] for c in chats] == [new["sessionId"], old["sessionId"]]  # newest activity first
    assert [c["status"] for c in chats] == ["open", "closed"]
    assert chats[1]["title"] == "What is displacement?"
    assert chats[1]["subject"] == "Science & Technology"
    assert (await api.get("/api/sessions", headers=OTHER_CLIENT)).json()["sessions"] == []


@pytest.mark.asyncio
async def test_chats_without_a_question_are_not_in_history(api):
    await _start_chat(api)
    assert (await api.get("/api/sessions", headers=CLIENT)).json()["sessions"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "status", "code"),
    [({}, 400, "client_id_required"), ({"X-Client-Id": "bad id!"}, 422, "invalid_input")],
)
async def test_history_needs_a_valid_client_id(api, headers, status, code):
    resp = await api.get("/api/sessions", headers=headers)
    assert resp.status_code == status
    assert resp.json()["error"]["code"] == code


@pytest.mark.asyncio
async def test_rating_is_optional_once_and_only_after_closing(api, db):
    chat = await _start_chat(api)
    sid = chat["sessionId"]
    await _add_turn(db, sid, "What is acceleration?")

    early = await api.post(f"/api/sessions/{sid}/feedback", json={"rating": 5})
    assert early.status_code == 409 and early.json()["error"]["code"] == "session_open"

    await api.post(f"/api/sessions/{sid}/close")
    assert (await api.post(f"/api/sessions/{sid}/feedback", json={"rating": 6})).status_code == 422
    ok = await api.post(f"/api/sessions/{sid}/feedback", json={"rating": 4, "comment": "  clear  "})
    assert ok.status_code == 200
    twice = await api.post(f"/api/sessions/{sid}/feedback", json={"rating": 5})
    assert twice.status_code == 409 and twice.json()["error"]["code"] == "feedback_exists"

    detail = (await api.get(f"/api/sessions/{sid}")).json()
    assert detail["status"] == "closed" and detail["rating"] == 4 and detail["closedAt"]
    assert detail["askToClose"] is False
    assert (await api.get("/api/sessions", headers=CLIENT)).json()["sessions"][0]["rating"] == 4


@pytest.mark.asyncio
async def test_closed_chat_is_read_only(api, db):
    chat = await _start_chat(api)
    sid = chat["sessionId"]
    await api.post(f"/api/sessions/{sid}/close")

    resp = await api.post("/api/chat/stream", json={"sessionId": sid, "message": "one more doubt"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "session_not_open"
    assert await repository.list_messages(db, sid) == []
    # Closing is idempotent.
    assert (await api.post(f"/api/sessions/{sid}/close")).status_code == 200


@pytest.mark.asyncio
async def test_teacher_request_hands_the_chat_over(api, db):
    chat = await _start_chat(api)
    sid = chat["sessionId"]
    await _add_turn(db, sid, "Why is the sky blue?")

    resp = await api.post(f"/api/sessions/{sid}/teacher-request", json={"note": "still confused about scattering"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "teacher_requested" and body["teacherRequestedAt"]
    request = await repository.get_teacher_request(db, sid)
    assert request.note == "still confused about scattering"

    again = await api.post(f"/api/sessions/{sid}/teacher-request", json={})
    assert again.status_code == 200 and again.json()["teacherRequestedAt"] == body["teacherRequestedAt"]
    assert (await api.post(f"/api/sessions/{sid}/close")).status_code == 409
    stream = await api.post("/api/chat/stream", json={"sessionId": sid, "message": "hello?"})
    assert stream.status_code == 409
    detail = (await api.get(f"/api/sessions/{sid}")).json()
    assert detail["status"] == "teacher_requested" and detail["teacherRequestedAt"]
    # A new chat in the same subject is a fresh one.
    assert (await _start_chat(api))["sessionId"] != sid


@pytest.mark.asyncio
async def test_teacher_request_on_a_closed_chat_is_rejected(api):
    sid = (await _start_chat(api))["sessionId"]
    await api.post(f"/api/sessions/{sid}/close")
    resp = await api.post(f"/api/sessions/{sid}/teacher-request", json={})
    assert resp.status_code == 409 and resp.json()["error"]["code"] == "session_not_open"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question_type", "expected"), [("definition", True), ("test_request", False)]
)
async def test_resumed_chat_reports_whether_to_ask_to_close(api, db, question_type, expected):
    sid = (await _start_chat(api))["sessionId"]
    assert (await api.get(f"/api/sessions/{sid}")).json()["askToClose"] is False  # nothing answered yet
    await _add_turn(db, sid, "Explain motion", question_type=question_type)
    assert (await api.get(f"/api/sessions/{sid}")).json()["askToClose"] is expected


@pytest.mark.parametrize(
    ("status", "verdict", "question_type", "had_related", "expected"),
    [
        ("complete", "related", "concept_explanation", False, True),
        ("complete", "related", "test_request", False, False),  # test questions still unanswered
        ("interrupted", "related", "definition", False, False),
        ("complete", "greeting", "other", True, True),  # "thanks!" after an answered doubt
        ("complete", "greeting", "other", False, False),  # plain "hi"
        ("complete", "unclear", "other", True, False),
        ("complete", "not_related", "other", True, False),
        ("complete", "unsafe", "other", True, False),
    ],
)
def test_should_ask_to_close(status, verdict, question_type, had_related, expected):
    assert should_ask_to_close(
        status=status, verdict=verdict, question_type=question_type, had_related_answer=had_related
    ) is expected


@pytest.mark.asyncio
async def test_existing_database_gets_the_new_columns(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # the schema before chat history existed
        conn.execute(
            "CREATE TABLE chat_sessions (id VARCHAR(36) PRIMARY KEY, plan_id VARCHAR(64), cp_id VARCHAR(64), "
            "medium VARCHAR(64), grade VARCHAR(64), board VARCHAR(128), subject VARCHAR(128), created_at DATETIME)"
        )
        conn.execute("INSERT INTO chat_sessions (id, plan_id, cp_id) VALUES ('legacy', 'p', 'c')")

    engine = make_engine(f"sqlite+aiosqlite:///{path}")
    await init_db(engine)
    await init_db(engine)  # idempotent
    await engine.dispose()

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)")}
        status = conn.execute("SELECT status FROM chat_sessions WHERE id = 'legacy'").fetchone()[0]
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"client_id", "status", "title", "updated_at", "closed_at"} <= columns
    assert status == "open"
    assert {"session_feedback", "teacher_requests"} <= tables
