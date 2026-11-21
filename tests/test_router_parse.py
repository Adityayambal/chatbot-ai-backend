import pytest

from app.llm.router import route
from tests.fakes import FIXTURE_PLAN_ID, FakeAsyncOpenAI, ScriptedRouterResponse

VALID_DECISION = {
    "verdict": "related",
    "reason": "about newton's second law",
    "standalone_question": "What is Newton's second law of motion?",
    "reply_language": "English",
    "question_type": "definition",
    "matched_topic_codes": ["2.1.1", "9.9.9"],  # 9.9.9 does not exist in the fixture outline
    "search_queries": ["Newton second law", "F = ma"],
    "suggested_subject": None,
    "short_reply": None,
}


def _cp_index(fixture_kb):
    return next(iter(fixture_kb.get_plan(FIXTURE_PLAN_ID).values()))


@pytest.mark.asyncio
async def test_route_parses_valid_json_and_drops_unknown_codes(fixture_kb, settings):
    cp_index = _cp_index(fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json=VALID_DECISION))

    decision, usage = await route(client, cp_index, [], "what is newton's second law?", settings)

    assert decision.verdict == "related"
    assert decision.matched_topic_codes == ["2.1.1"]
    assert usage.input_tokens == 500
    assert len(client.router_calls) == 1


@pytest.mark.asyncio
async def test_route_retries_once_on_invalid_json_then_succeeds(fixture_kb, settings):
    cp_index = _cp_index(fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json={"not": "a valid decision"}))
    client.router_responses.append(ScriptedRouterResponse(decision_json=VALID_DECISION))

    decision, usage = await route(client, cp_index, [], "test", settings)

    assert decision.verdict == "related"
    assert len(client.router_calls) == 2


@pytest.mark.asyncio
async def test_route_falls_back_after_two_invalid_responses(fixture_kb, settings):
    cp_index = _cp_index(fixture_kb)
    client = FakeAsyncOpenAI()
    client.router_responses.append(ScriptedRouterResponse(decision_json={"bad": "data"}))
    client.router_responses.append(ScriptedRouterResponse(decision_json={"still": "bad"}))

    decision, usage = await route(client, cp_index, [], "hello there", settings)

    assert decision.verdict == "related"
    assert decision.matched_topic_codes == []
    assert decision.standalone_question == "hello there"
    assert "fallback" in decision.reason
    assert len(client.router_calls) == 2
