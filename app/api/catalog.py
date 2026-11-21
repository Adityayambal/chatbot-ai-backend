from __future__ import annotations

import itertools
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_kb, wait_for_plan_ready
from app.content.knowledge_base import KnowledgeBase
from app.schemas import (
    CatalogOption,
    CatalogResponse,
    ContentCounts,
    OutlineChapter,
    OutlineConcept,
    OutlineResponse,
    OutlineSubconcept,
    PlanInfo,
)

router = APIRouter(tags=["catalog"])


@router.get("/catalog", response_model=CatalogResponse)
async def get_catalog(
    planIds: str = Query(..., description="Comma-separated plan IDs"),
    kb: KnowledgeBase = Depends(get_kb),
) -> CatalogResponse:
    plan_ids = [p.strip() for p in planIds.split(",") if p.strip()]
    plans: list[PlanInfo] = []
    options: list[CatalogOption] = []
    last_synced_at: str | None = None

    for plan_id in plan_ids:
        cp_indices = await wait_for_plan_ready(kb, plan_id)
        plan_name = ""
        for idx in cp_indices.values():
            plan_name = idx.meta.plan_name
            options.append(
                CatalogOption(
                    cp_id=idx.meta.cp_id,
                    cp_name=idx.meta.cp_name,
                    plan_id=idx.meta.plan_id,
                    medium_id=idx.meta.medium_id,
                    medium_name=idx.meta.medium_name,
                    grade_id=idx.meta.grade_id,
                    grade_name=idx.meta.grade_name,
                    board_id=idx.meta.board_id,
                    board_name=idx.meta.board_name,
                    subject_id=idx.meta.subject_id,
                    subject_name=idx.meta.subject_name,
                    chapter_count=idx.meta.chapter_count,
                    concept_count=idx.meta.concept_count,
                    is_active=idx.meta.is_active,
                )
            )
        plans.append(PlanInfo(plan_id=plan_id, plan_name=plan_name))
        fetched_at = kb.get_fetched_at(plan_id)
        if fetched_at is not None:
            last_synced_at = datetime.fromtimestamp(fetched_at, tz=timezone.utc).isoformat()
        kb.maybe_schedule_refresh(plan_id)

    return CatalogResponse(plans=plans, options=options, last_synced_at=last_synced_at)


@router.get("/subjects/{cp_id}/outline", response_model=OutlineResponse)
async def get_outline(
    cp_id: str,
    planId: str = Query(...),
    kb: KnowledgeBase = Depends(get_kb),
) -> OutlineResponse:
    cp_indices = await wait_for_plan_ready(kb, planId)
    cp_index = cp_indices.get(cp_id)
    if cp_index is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "cp_not_found", "message": "Subject not found in this plan"}},
        )

    chapter_counts: dict[int, dict[str, int]] = {}
    for chunk in cp_index.chunks:
        bucket = chapter_counts.setdefault(chunk.chapter_pos, {"notes": 0, "videos": 0, "questions": 0})
        if chunk.content_type in ("revision_note", "pdf"):
            bucket["notes"] += 1
        elif chunk.content_type == "video_transcript":
            bucket["videos"] += 1
        elif chunk.content_type == "question":
            bucket["questions"] += 1

    chapters: list[OutlineChapter] = []
    for chapter_pos, chapter_group_iter in itertools.groupby(cp_index.positions, key=lambda p: p.chapter_pos):
        chapter_group = list(chapter_group_iter)
        first = chapter_group[0]
        concepts: list[OutlineConcept] = []
        for _concept_pos, concept_group_iter in itertools.groupby(chapter_group, key=lambda p: p.concept_pos):
            concept_group = list(concept_group_iter)
            concept_first = concept_group[0]
            subconcepts = [
                OutlineSubconcept(code=p.topic_code, name=p.subconcept_name) for p in concept_group
            ]
            concepts.append(
                OutlineConcept(
                    code=concept_first.concept_code,
                    name=concept_first.concept_name,
                    subconcepts=subconcepts,
                )
            )
        counts = chapter_counts.get(chapter_pos, {"notes": 0, "videos": 0, "questions": 0})
        chapters.append(
            OutlineChapter(
                code=str(chapter_pos),
                name=first.chapter_name,
                counts=ContentCounts(**counts),
                concepts=concepts,
            )
        )

    return OutlineResponse(cp_id=cp_id, subject=cp_index.meta.subject_name, chapters=chapters)
