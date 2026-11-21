from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, session_or_404
from app.db import repository
from app.db.models import SESSION_OPEN, ChatMessage
from app.schemas import FeedbackRequest, FeedbackResponse, SessionFeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/messages/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    message_id: str, req: FeedbackRequest, db: AsyncSession = Depends(get_db)
) -> FeedbackResponse:
    if req.rating not in (1, -1):
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "invalid_input", "message": "rating must be 1 or -1"}},
        )
    message = await db.get(ChatMessage, message_id)
    if message is None:
        raise HTTPException(
            status_code=404, detail={"error": {"code": "message_not_found", "message": "Unknown message"}}
        )
    await repository.add_feedback(db, message_id=message_id, rating=req.rating, comment=req.comment)
    return FeedbackResponse(ok=True)


@router.post("/sessions/{session_id}/feedback", response_model=FeedbackResponse)
async def submit_session_feedback(
    session_id: str, req: SessionFeedbackRequest, db: AsyncSession = Depends(get_db)
) -> FeedbackResponse:
    """Optional 1-5 star rating for a whole chat, once, after it is closed or handed to a teacher."""
    session = await session_or_404(db, session_id)
    if session.status == SESSION_OPEN:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "session_open", "message": "Close this doubt before rating it."}},
        )
    if await repository.get_session_feedback(db, session_id) is not None:
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "feedback_exists", "message": "You have already rated this chat."}},
        )
    comment = req.comment.strip() if req.comment and req.comment.strip() else None
    await repository.add_session_feedback(db, session_id=session_id, rating=req.rating, comment=comment)
    return FeedbackResponse(ok=True)
