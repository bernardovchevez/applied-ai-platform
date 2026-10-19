"""POST /v1/feedback."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from deps import verify_bff_api_key
from feedback import insert_feedback
from models import FeedbackRequest, FeedbackResponse
from rag_helpers import prompt_version

router = APIRouter(dependencies=[Depends(verify_bff_api_key)])


@router.post("/v1/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    body: FeedbackRequest,
    x_client_platform: str | None = Header(default=None, alias="X-Client-Platform"),
) -> FeedbackResponse:
    try:
        row = insert_feedback(
            session_id=body.session_id,
            prompt=body.prompt,
            response=body.response,
            rating=body.rating,
            sources=[s.model_dump() for s in (body.sources or [])],
            comment=body.comment,
            client_platform=x_client_platform,
            prompt_version=prompt_version(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "feedback_store_failed", "message": str(exc)},
        ) from exc

    return FeedbackResponse(id=row["id"], created_at=row["created_at"])
