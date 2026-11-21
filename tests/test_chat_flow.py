import orjson
import pytest
from httpx2 import Request as Httpx2Request
from httpx2 import Response as Httpx2Response
from openai import APIConnectionError, RateLimitError

from app.chat.orchestrator import MAX_TOKENS_SUFFIX, REFUSAL_FALLBACK_MESSAGE, ChatOrchestrator
from app.chat.rate_limit import SlidingWindowLimiter
from app.db import repository
from app.schemas import ChatRequest
from tests.fakes import FIXTURE_PLAN_ID, FakeAsyncOpenAI, ScriptedRouterResponse, ScriptedTutorResponse


class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


def _event_names(events) -> list[str]:
    return [e.event for e in events]


def _event_data(events, name: str) -> list[dict]:
    return [orjson.loads(e.data) for e in events if e.event == name]


async def _make_session(db, fixture_kb):
    cp_index = next(iter(fixture_kb.get_plan(FIXTURE_PLAN_ID).values()))
    session = await repository.create_session(
        db,
        plan_id=FIXTURE_PLAN_ID,
        cp_id=cp_index.meta.cp_id,
        medium=cp_index.meta.medium_name,
        grade=cp_index.meta.grade_name,
        board=cp_index.meta.board_name,
        subject=cp_index.meta.subject_name,
    )
    return session, cp_index


NOT_RELATED_DECISION = {
    "verdict": "not_related",
    "reason": "asked about cricket, not science",
    "standalone_question": "Who won the IPL?",
    "reply_language": "English",
    "question_type": "other",
    "matched_topic_codes": [],
    "search_queries": ["IPL"],
    "suggested_subject": None,
    "short_reply": "I help with Science & Technology doubts here! Try asking about motion or Newton's laws.",
}

RELATED_DECISION = {
    "verdict": "related",
    "reason": "about newton's second law",
    "standalone_question": "What is Newton's second law of motion?",
    "reply_language": "English",
    "question_type": "definition",
    "matched_topic_codes": ["2.1.1"],
    "search_queries": ["Newton second law", "F = ma"],
    "suggested_subject": None,
    "short_reply": None,
}


@pytest.mark.asyncio
async def test_not_related_flow_never_calls_tutor(db, fixture_kb, settings):
    session, _cp_index = await _make_session(db, fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json=NOT_RELATED_DECISION))

    orchestrator = ChatOrchestrator(settings, fixture_kb, client, SlidingWindowLimiter(100))
    chat_req = ChatRequest(session_id=session.id, message="who won the IPL?")
    events = [e async for e in orchestrator.run(db, _FakeRequest(), chat_req)]

    assert _event_names(events) == ["meta", "status", "route", "delta", "done"]
    assert len(client.tutor_calls) == 0

    done = _event_data(events, "done")[0]
    assert done["verdict"] == "not_related"
    assert done["status"] == "complete"

    messages = await repository.list_messages(db, session.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == NOT_RELATED_DECISION["short_reply"]
    assert messages[1].status == "complete"


@pytest.mark.asyncio
async def test_related_flow_full_event_sequence(db, fixture_kb, settings):
    session, _cp_index = await _make_session(db, fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json=RELATED_DECISION))
    client.tutor_responses.append(
        ScriptedTutorResponse(deltas=["Newton's ", "second law ", "relates force and mass."])
    )

    orchestrator = ChatOrchestrator(settings, fixture_kb, client, SlidingWindowLimiter(100))
    chat_req = ChatRequest(session_id=session.id, message="what is newton's second law?")
    events = [e async for e in orchestrator.run(db, _FakeRequest(), chat_req)]

    names = _event_names(events)
    assert names == [
        "meta",
        "status",
        "route",
        "status",
        "sources",
        "status",
        "delta",
        "delta",
        "delta",
        "done",
    ]

    sources = _event_data(events, "sources")[0]
    assert len(sources) > 0
    assert all("id" in s for s in sources)

    done = _event_data(events, "done")[0]
    assert done["status"] == "complete"
    assert done["verdict"] == "related"
    assert done["usage"]["tutor"]["cache_read_input_tokens"] == 1000
    assert done["costUsd"] > 0

    messages = await repository.list_messages(db, session.id)
    assistant_message = messages[-1]
    assert assistant_message.content == "Newton's second law relates force and mass."
    assert assistant_message.status == "complete"
    assert assistant_message.sources_json is not None
    assert len(orjson.loads(assistant_message.sources_json)) > 0


@pytest.mark.asyncio
async def test_tutor_error_saves_partial_and_emits_error_event(db, fixture_kb, settings):
    session, _cp_index = await _make_session(db, fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json=RELATED_DECISION))
    connection_error = APIConnectionError(
        message="boom", request=Httpx2Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    client.tutor_responses.append(
        ScriptedTutorResponse(deltas=["Here is the "], raise_mid_stream=connection_error)
    )

    orchestrator = ChatOrchestrator(settings, fixture_kb, client, SlidingWindowLimiter(100))
    chat_req = ChatRequest(session_id=session.id, message="what is newton's second law?")
    events = [e async for e in orchestrator.run(db, _FakeRequest(), chat_req)]

    names = _event_names(events)
    assert names[-1] == "error"
    assert "done" not in names

    error = _event_data(events, "error")[0]
    assert error["code"] == "network"

    messages = await repository.list_messages(db, session.id)
    assistant_message = messages[-1]
    assert assistant_message.status == "error"
    assert assistant_message.content == "Here is the "


@pytest.mark.asyncio
async def test_retry_reuses_user_message_without_duplicating_it(db, fixture_kb, settings):
    session, _cp_index = await _make_session(db, fixture_kb)
    client = FakeAsyncOpenAI()

    # First attempt fails mid-stream.
    client.router_responses.append(ScriptedRouterResponse(decision_json=RELATED_DECISION))
    connection_error = APIConnectionError(
        message="boom", request=Httpx2Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    client.tutor_responses.append(ScriptedTutorResponse(deltas=[], raise_mid_stream=connection_error))

    orchestrator = ChatOrchestrator(settings, fixture_kb, client, SlidingWindowLimiter(100))
    chat_req = ChatRequest(session_id=session.id, message="what is newton's second law?")
    first_events = [e async for e in orchestrator.run(db, _FakeRequest(), chat_req)]
    user_message_id = _event_data(first_events, "meta")[0]["userMessageId"]

    messages_after_failure = await repository.list_messages(db, session.id)
    assert [m.role for m in messages_after_failure] == ["user", "assistant"]

    # Retry: same user message id, router+tutor succeed this time.
    client.router_responses.append(ScriptedRouterResponse(decision_json=RELATED_DECISION))
    client.tutor_responses.append(ScriptedTutorResponse(deltas=["F = ma."]))

    retry_req = ChatRequest(
        session_id=session.id, message="what is newton's second law?", retry_of_message_id=user_message_id
    )
    retry_events = [e async for e in orchestrator.run(db, _FakeRequest(), retry_req)]
    assert _event_data(retry_events, "meta")[0]["userMessageId"] == user_message_id
    assert retry_events[-1].event == "done"

    messages_after_retry = await repository.list_messages(db, session.id)
    user_messages = [m for m in messages_after_retry if m.role == "user"]
    assert len(user_messages) == 1  # no duplicate user message was inserted

    assistant_messages = [m for m in messages_after_retry if m.role == "assistant"]
    assert len(assistant_messages) == 2
    assert assistant_messages[0].status == "superseded"
    assert assistant_messages[1].status == "complete"
    assert assistant_messages[1].content == "F = ma."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "expected_tail"),
    [("length", MAX_TOKENS_SUFFIX), ("content_filter", REFUSAL_FALLBACK_MESSAGE)],
)
async def test_tutor_finish_reason_appends_notice(db, fixture_kb, settings, finish_reason, expected_tail):
    session, _cp_index = await _make_session(db, fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json=RELATED_DECISION))
    client.tutor_responses.append(ScriptedTutorResponse(deltas=["Force equals "], finish_reason=finish_reason))

    orchestrator = ChatOrchestrator(settings, fixture_kb, client, SlidingWindowLimiter(100))
    chat_req = ChatRequest(session_id=session.id, message="what is newton's second law?")
    events = [e async for e in orchestrator.run(db, _FakeRequest(), chat_req)]

    assert _event_data(events, "delta")[-1]["text"] == expected_tail
    assert _event_data(events, "done")[0]["status"] == "complete"
    messages = await repository.list_messages(db, session.id)
    assert messages[-1].content.endswith(expected_tail)


def _rate_limit_error(code: str | None) -> RateLimitError:
    request = Httpx2Request("POST", "https://api.openai.com/v1/chat/completions")
    return RateLimitError(
        "429", response=Httpx2Response(429, request=request), body={"code": code, "message": "429"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "expected"), [("insufficient_quota", "ai_unavailable"), (None, "busy")])
async def test_router_rate_limit_quota_vs_busy(db, fixture_kb, settings, code, expected):
    session, _cp_index = await _make_session(db, fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json={}, raise_error=_rate_limit_error(code)))

    orchestrator = ChatOrchestrator(settings, fixture_kb, client, SlidingWindowLimiter(100))
    chat_req = ChatRequest(session_id=session.id, message="what is newton's second law?")
    events = [e async for e in orchestrator.run(db, _FakeRequest(), chat_req)]

    assert _event_names(events) == ["meta", "status", "error"]
    assert _event_data(events, "error")[0]["code"] == expected


GREETING_DECISION = {**NOT_RELATED_DECISION, "verdict": "greeting", "short_reply": "You're welcome!"}


@pytest.mark.asyncio
async def test_done_event_says_when_to_ask_to_close(db, fixture_kb, settings):
    """Ask after an answered doubt and after "thanks"; never right after test questions are set."""
    session, _cp_index = await _make_session(db, fixture_kb)
    client = FakeAsyncOpenAI()
    orchestrator = ChatOrchestrator(settings, fixture_kb, client, SlidingWindowLimiter(100))

    async def ask(message: str, decision: dict, deltas: list[str] | None = None) -> bool:
        client.router_responses.append(ScriptedRouterResponse(decision_json=decision))
        if deltas is not None:
            client.tutor_responses.append(ScriptedTutorResponse(deltas=deltas))
        chat_req = ChatRequest(session_id=session.id, message=message)
        events = [e async for e in orchestrator.run(db, _FakeRequest(), chat_req)]
        return _event_data(events, "done")[0]["askToClose"]

    assert await ask("hi", GREETING_DECISION) is False  # nothing answered yet
    assert await ask("what is newton's second law?", RELATED_DECISION, ["F = ma."]) is True
    assert await ask("thanks!", GREETING_DECISION) is True
    test_request = {**RELATED_DECISION, "question_type": "test_request"}
    assert await ask("quiz me on newton's laws", test_request, ["1. State the second law."]) is False
