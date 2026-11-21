from __future__ import annotations

from app.config import PRICES
from app.llm.schemas import CallUsage


def estimate_cost(model: str, usage: CallUsage) -> float:
    prices = PRICES.get(model)
    if prices is None:
        return 0.0
    return (
        usage.input_tokens / 1_000_000 * prices["input"]
        + usage.output_tokens / 1_000_000 * prices["output"]
        + usage.cache_read_input_tokens / 1_000_000 * prices["cache_read"]
    )
