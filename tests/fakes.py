"""Test doubles: a KnowledgeBase pre-loaded from the mini fixture (no network), and a
fake AsyncOpenAI client (scripted router JSON + streamed tutor text, no OpenAI calls)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from app.config import Settings
from app.content.knowledge_base import KnowledgeBase
from app.content.models import BundleResponse

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mini_bundle.json"
FIXTURE_PLAN_ID = "plan000000000000000000001"


def load_fixture_bundle() -> BundleResponse:
    raw = FIXTURE_PATH.read_bytes()
    return BundleResponse.model_validate(orjson.loads(raw))


async def build_fixture_kb(settings: Settings, plan_id: str = FIXTURE_PLAN_ID) -> KnowledgeBase:
    """A KnowledgeBase whose single plan is pre-built from the mini fixture — no HTTP, no disk cache reuse."""
    kb = KnowledgeBase(settings)
    raw = FIXTURE_PATH.read_bytes()
    bundle = BundleResponse.model_validate(orjson.loads(raw))
    cp_indices = await kb._build_cp_indices(plan_id, raw, bundle)  # noqa: SLF001 - test double
    kb._plans[plan_id] = {"status": "ready", "cp_indices": cp_indices, "fetched_at": 1_700_000_000.0}  # noqa: SLF001
    return kb


# prompt_tokens includes cached_tokens, as in the real API.
_DEFAULT_ROUTER_USAGE = {"prompt_tokens": 500, "completion_tokens": 50, "cached_tokens": 0}
_DEFAULT_TUTOR_USAGE = {"prompt_tokens": 2000, "completion_tokens": 300, "cached_tokens": 1000}


def _usage(u: dict) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=u["prompt_tokens"],
        completion_tokens=u["completion_tokens"],
        total_tokens=u["prompt_tokens"] + u["completion_tokens"],
        prompt_tokens_details=PromptTokensDetails(cached_tokens=u["cached_tokens"]),
    )


@dataclass
class ScriptedRouterResponse:
    decision_json: dict
    usage: dict = field(default_factory=lambda: dict(_DEFAULT_ROUTER_USAGE))
    finish_reason: str = "stop"
    raise_error: Exception | None = None


@dataclass
class ScriptedTutorResponse:
    deltas: list[str]
    usage: dict = field(default_factory=lambda: dict(_DEFAULT_TUTOR_USAGE))
    finish_reason: str = "stop"
    raise_before_stream: Exception | None = None
    raise_mid_stream: Exception | None = None


def _chunk(choices: list[ChunkChoice], usage: CompletionUsage | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl_fake_tutor",
        object="chat.completion.chunk",
        created=1_700_000_000,
        model="fake-model",
        choices=choices,
        usage=usage,
    )


class _FakeChatStream:
    """Mimics openai.AsyncStream[ChatCompletionChunk] with stream_options.include_usage."""

    def __init__(self, resp: ScriptedTutorResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> _FakeChatStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False

    async def __aiter__(self):
        for d in self._resp.deltas:
            yield _chunk([ChunkChoice(index=0, delta=ChoiceDelta(content=d), finish_reason=None)])
        if self._resp.raise_mid_stream:
            raise self._resp.raise_mid_stream
        yield _chunk([ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason=self._resp.finish_reason)])
        yield _chunk([], usage=_usage(self._resp.usage))


class _FakeCompletions:
    def __init__(self, client: FakeAsyncOpenAI) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> ChatCompletion | _FakeChatStream:
        if kwargs.get("stream"):
            return self._tutor(kwargs)
        return self._router(kwargs)

    def _router(self, kwargs: dict) -> ChatCompletion:
        self._client.router_calls.append(kwargs)
        if not self._client.router_responses:
            raise RuntimeError("FakeAsyncOpenAI: no more scripted router responses")
        resp = self._client.router_responses.pop(0)
        if resp.raise_error:
            raise resp.raise_error
        return ChatCompletion(
            id="chatcmpl_fake_router",
            object="chat.completion",
            created=1_700_000_000,
            model=kwargs.get("model", "fake-model"),
            choices=[
                Choice(
                    index=0,
                    finish_reason=resp.finish_reason,
                    message=ChatCompletionMessage(role="assistant", content=orjson.dumps(resp.decision_json).decode()),
                )
            ],
            usage=_usage(resp.usage),
        )

    def _tutor(self, kwargs: dict) -> _FakeChatStream:
        self._client.tutor_calls.append(kwargs)
        if not self._client.tutor_responses:
            raise RuntimeError("FakeAsyncOpenAI: no more scripted tutor responses")
        resp = self._client.tutor_responses.pop(0)
        if resp.raise_before_stream:
            raise resp.raise_before_stream
        return _FakeChatStream(resp)


class _FakeChat:
    def __init__(self, client: FakeAsyncOpenAI) -> None:
        self.completions = _FakeCompletions(client)


class FakeAsyncOpenAI:
    """Stands in for openai.AsyncOpenAI: scripted router JSON + streamed tutor text."""

    def __init__(self) -> None:
        self.router_responses: list[ScriptedRouterResponse] = []
        self.tutor_responses: list[ScriptedTutorResponse] = []
        self.router_calls: list[dict] = []
        self.tutor_calls: list[dict] = []
        self.chat = _FakeChat(self)

    async def close(self) -> None:
        pass
