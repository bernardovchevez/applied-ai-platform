"""POST /v1/retrieve."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from deps import verify_bff_api_key
from models import ChunkHitOut, RetrieveRequest, RetrieveResponse
from rag_helpers import chunk_id, run_retrieve, source_from_metadata, sources_from_hits

router = APIRouter(dependencies=[Depends(verify_bff_api_key)])


@router.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    x_client_platform: str | None = Header(default=None, alias="X-Client-Platform"),
) -> RetrieveResponse:
    del x_client_platform
    try:
        hits = run_retrieve(body.question, body.top_k, body.filters)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": str(exc)},
        ) from exc

    chunks: list[ChunkHitOut] = []
    for hit in hits:
        meta = hit.metadata
        citation = source_from_metadata(meta, distance=hit.distance)
        chunks.append(
            ChunkHitOut(
                chunk_id=chunk_id(meta),
                text=hit.text,
                source=citation,
            )
        )

    return RetrieveResponse(
        question=body.question,
        chunks=chunks,
        sources=sources_from_hits(hits),
    )
