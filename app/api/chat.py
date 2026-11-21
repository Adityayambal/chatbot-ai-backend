from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sse_starlette import EventSourceResponse

from app.api.deps import session_not_open
from app.chat.orchestrator import ChatOrchestrator
from app.db import repository
from app.db.models import SESSION_OPEN
from app.schemas import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat/stream")
async def chat_stream(chat_req: ChatRequest, request: Request) -> EventSourceResponse:
    settings = request.app.state.settings
    kb = request.app.state.kb
    client = request.app.state.llm_client
    rate_limiter = request.app.state.rate_limiter
    async_session = request.app.state.async_session

    message = chat_req.message.strip()
    if not message or len(message) > settings.max_message_chars:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "invalid_input", "message": "Your message is empty or too long."}},
        )

    async with async_session() as validation_db:
        session = await repository.get_session(validation_db, chat_req.session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "session_not_found", "message": "Unknown session"}},
            )
        if session.status != SESSION_OPEN:
            raise session_not_open("This doubt is closed. Start a new chat to ask more.")

        if chat_req.retry_of_message_id:
            latest_user = await repository.get_latest_user_message(validation_db, chat_req.session_id)
            if latest_user is None or latest_user.id != chat_req.retry_of_message_id:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "code": "invalid_input",
                            "message": "retryOfMessageId must be the latest user message in this session.",
                        }
                    },
                )

    orchestrator = ChatOrchestrator(settings, kb, client, rate_limiter)

    async def event_gen():
        async with async_session() as db:
            async for event in orchestrator.run(db, request, chat_req):
                yield event

    return EventSourceResponse(
        event_gen(),
        ping=15,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
