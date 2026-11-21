from __future__ import annotations

import asyncio
import re
import time

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.content.knowledge_base import CPIndex, KnowledgeBase
from app.db import repository
from app.db.models import ChatSession

PLAN_WAIT_TIMEOUT_SECONDS = 30.0
PLAN_WAIT_POLL_SECONDS = 0.5
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def optional_client_id(x_client_id: str | None = Header(default=None)) -> str | None:
    """The anonymous per-browser student ID the frontend sends as `X-Client-Id`. It stands in
    for a real student ID until the app has accounts."""
    if x_client_id is None:
        return None
    if not _CLIENT_ID_RE.match(x_client_id):
        raise HTTPException(
            status_code=422,
            detail={
                "error": {"code": "invalid_input", "message": "X-Client-Id must be 8-64 letters, digits or dashes"}
            },
        )
    return x_client_id


def require_client_id(client_id: str | None = Depends(optional_client_id)) -> str:
    if client_id is None:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "client_id_required", "message": "X-Client-Id header is required"}},
        )
    return client_id


def get_kb(request: Request) -> KnowledgeBase:
    return request.app.state.kb


async def get_db(request: Request) -> AsyncSession:
    async with request.app.state.async_session() as session:
        yield session


async def session_or_404(db: AsyncSession, session_id: str) -> ChatSession:
    session = await repository.get_session(db, session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "session_not_found", "message": "Unknown session"}}
        )
    return session


def session_not_open(message: str) -> HTTPException:
    return HTTPException(status_code=409, detail={"error": {"code": "session_not_open", "message": message}})


def require_admin(
    x_admin_token: str = Header(default=""), settings: Settings = Depends(get_settings)
) -> None:
    if not settings.admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "unauthorized", "message": "Invalid admin token"}},
        )


def require_debug(settings: Settings = Depends(get_settings)) -> None:
    if not settings.debug:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "not_found", "message": "Not found"}},
        )


async def wait_for_plan_ready(kb: KnowledgeBase, plan_id: str) -> dict[str, CPIndex]:
    """Waits up to PLAN_WAIT_TIMEOUT_SECONDS for a plan to become ready/stale."""
    if plan_id not in kb.known_plan_ids():
        kb.ensure_loading_started(plan_id)

    deadline = time.monotonic() + PLAN_WAIT_TIMEOUT_SECONDS
    while True:
        status = kb.get_plan_status(plan_id)
        if status in ("ready", "stale"):
            cp_indices = kb.get_plan(plan_id)
            if cp_indices is not None:
                return cp_indices
        if status == "error":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": "content_unavailable",
                        "message": "Lessons are unavailable right now",
                    }
                },
            )
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": "content_loading",
                        "message": "Preparing lessons…",
                        "retryAfter": 5,
                    }
                },
            )
        await asyncio.sleep(PLAN_WAIT_POLL_SECONDS)
