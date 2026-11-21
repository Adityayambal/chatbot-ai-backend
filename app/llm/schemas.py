from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

QuestionType = Literal[
    "definition",
    "concept_explanation",
    "difference",
    "numerical",
    "process_or_diagram",
    "reasoning_why",
    "exam_answer",
    "test_request",
    "other",
]


class RouterDecision(BaseModel):
    verdict: Literal["related", "not_related", "greeting", "unclear", "unsafe"]
    reason: str
    standalone_question: str
    reply_language: str
    question_type: QuestionType
    matched_topic_codes: list[str]
    search_queries: list[str]
    suggested_subject: str | None = None
    short_reply: str | None = None


# Hand-written (not Pydantic-generated) per spec §8.2: every property in `required`,
# nullable fields typed ["string", "null"], additionalProperties false, no maxItems
# (list truncation happens server-side in router.py).
ROUTER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["related", "not_related", "greeting", "unclear", "unsafe"],
        },
        "reason": {"type": "string"},
        "standalone_question": {"type": "string"},
        "reply_language": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": [
                "definition",
                "concept_explanation",
                "difference",
                "numerical",
                "process_or_diagram",
                "reasoning_why",
                "exam_answer",
                "test_request",
                "other",
            ],
        },
        "matched_topic_codes": {"type": "array", "items": {"type": "string"}},
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "suggested_subject": {"type": ["string", "null"]},
        "short_reply": {"type": ["string", "null"]},
    },
    "required": [
        "verdict",
        "reason",
        "standalone_question",
        "reply_language",
        "question_type",
        "matched_topic_codes",
        "search_queries",
        "suggested_subject",
        "short_reply",
    ],
    "additionalProperties": False,
}


@dataclass
class CallUsage:
    model: str
    input_tokens: int = 0  # uncached prompt tokens only; cached ones are in cache_read_input_tokens
    output_tokens: int = 0  # includes reasoning tokens, which OpenAI bills as output
    cache_read_input_tokens: int = 0
    stop_reason: str | None = None  # OpenAI finish_reason: "stop", "length", "content_filter", ...
    latency_ms: float = 0.0
