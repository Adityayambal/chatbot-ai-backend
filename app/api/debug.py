"""Debug-only endpoints, gated by require_debug (only mounted/usable when DEBUG=true)."""
from __future__ import annotations

import orjson
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_kb, require_debug, wait_for_plan_ready
from app.config import Settings, get_settings
from app.content.knowledge_base import KnowledgeBase
from app.db.models import ChatMessage
from app.retrieval.retriever import retrieve

router = APIRouter(tags=["debug"], dependencies=[Depends(require_debug)])


class _DebugDecision:
    """Stands in for a RouterDecision when exercising retrieval with no router call."""

    def __init__(self, question: str) -> None:
        self.standalone_question = question
        self.search_queries: list[str] = [question]
        self.matched_topic_codes: list[str] = []


@router.get("/debug/retrieve")
async def debug_retrieve(
    planId: str = Query(...),
    cpId: str = Query(...),
    q: str = Query(...),
    kb: KnowledgeBase = Depends(get_kb),
    settings: Settings = Depends(get_settings),
) -> dict:
    cp_indices = await wait_for_plan_ready(kb, planId)
    cp_index = cp_indices.get(cpId)
    if cp_index is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "cp_not_found", "message": "Subject not found in this plan"}},
        )

    results = retrieve(cp_index, _DebugDecision(q), settings)
    return {
        "query": q,
        "results": [
            {
                "sourceId": r.source_id,
                "chunkId": r.chunk.chunk_id,
                "score": r.score,
                "contentType": r.chunk.content_type,
                "breadcrumb": f"{r.chunk.chapter_name} › {r.chunk.concept_name} › {r.chunk.subconcept_name}",
                "topicCode": r.chunk.topic_code,
                "snippet": r.chunk.text[:200],
            }
            for r in results
        ],
    }


@router.get("/debug/turn/{message_id}")
async def debug_turn(message_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    message = await db.get(ChatMessage, message_id)
    if message is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "message_not_found", "message": "Unknown message"}}
        )
    return {
        "id": message.id,
        "sessionId": message.session_id,
        "role": message.role,
        "status": message.status,
        "verdict": message.verdict,
        "router": orjson.loads(message.router_json) if message.router_json else None,
        "sources": orjson.loads(message.sources_json) if message.sources_json else None,
        "usage": orjson.loads(message.usage_json) if message.usage_json else None,
        "latencyMs": orjson.loads(message.latency_json) if message.latency_json else None,
        "costUsd": message.cost_usd,
        "content": message.content,
    }
