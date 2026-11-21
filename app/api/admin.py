from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.api.deps import get_kb, require_admin
from app.content.knowledge_base import KnowledgeBase
from app.schemas import AdminReloadRequest, AdminReloadResponse

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


@router.post("/admin/reload", response_model=AdminReloadResponse)
async def reload_plans(req: AdminReloadRequest, kb: KnowledgeBase = Depends(get_kb)) -> AdminReloadResponse:
    asyncio.create_task(kb.reload(req.plan_ids))
    return AdminReloadResponse(started=True)
