from __future__ import annotations

import logging

import openai
from openai.types import CompletionUsage

from app.config import Settings
from app.llm.schemas import CallUsage

logger = logging.getLogger(__name__)


def make_client(settings: Settings) -> openai.AsyncOpenAI:
    # An empty key is passed through (not None) so a missing key surfaces as a 401 on the first
    # call instead of the SDK raising at construction and taking the whole app down.
    return openai.AsyncOpenAI(api_key=settings.openai_api_key, max_retries=3)


async def verify_models(client: openai.AsyncOpenAI, settings: Settings) -> None:
    """Best-effort startup check. Never raises — a bad model id should log, not crash the app."""
    for model in {settings.router_model, settings.answer_model}:
        try:
            await client.models.retrieve(model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not verify model %s: %s", model, exc)


def call_usage(
    model: str, usage: CompletionUsage | None, finish_reason: str | None, latency_ms: float
) -> CallUsage:
    if usage is None:
        return CallUsage(model=model, stop_reason=finish_reason, latency_ms=latency_ms)
    details = usage.prompt_tokens_details
    cached = (details.cached_tokens or 0) if details else 0
    return CallUsage(
        model=model,
        input_tokens=usage.prompt_tokens - cached,
        output_tokens=usage.completion_tokens,
        cache_read_input_tokens=cached,
        stop_reason=finish_reason,
        latency_ms=latency_ms,
    )
