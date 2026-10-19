"""POST /v1/chat/stream (SSE RAG)."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, Header
from ops.langfuse_tracing import ChatStreamTrace, active_chat_model
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from deps import verify_bff_api_key
from health_checks import check_pgvector
from models import ChatRequest
from rag_helpers import (
    build_rag_user_message,
    get_store,
    load_system_prompt,
    pgvector_unavailable_message,
    run_retrieve,
    sources_from_hits,
    sse_error,
    sse_headers,
    prompt_version,
)
from shared.llm import llm_provider, stream_chat_completion
from shared.rag_text import clean_rag_answer, unique_source_filenames

router = APIRouter(dependencies=[Depends(verify_bff_api_key)])


@router.post("/v1/chat/stream")
async def chat_stream(
    body: ChatRequest,
    x_client_platform: str | None = Header(default=None, alias="X-Client-Platform"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> EventSourceResponse:
    start = time.perf_counter()
    session_id = x_session_id or str(uuid.uuid4())
    pv = prompt_version()
    trace = ChatStreamTrace(
        session_id=session_id,
        question=body.question,
        prompt_version=pv,
        client_platform=x_client_platform,
        top_k=body.top_k,
        format=body.format,
    )

    pg = check_pgvector(get_store())
    if pg["status"] != "ok":
        msg = pgvector_unavailable_message(pg.get("error"))
        trace.end_error(error_type="pgvector_unavailable", message=msg, stage="preflight")
        return sse_error(msg)

    try:
        hits = run_retrieve(body.question, body.top_k, body.filters)
    except Exception as exc:
        msg = pgvector_unavailable_message(str(exc))
        trace.end_error(error_type="retrieve_error", message=msg, stage="retrieve")
        return sse_error(msg)

    sources = sources_from_hits(hits)
    trace.end_retrieve([s.model_dump() for s in sources], len(hits))
    system = load_system_prompt(compact=body.format == "compact")
    user_msg = build_rag_user_message(body.question, hits)
    max_tokens = 200 if body.format == "compact" else 600

    async def event_generator():
        yield {
            "event": "sources",
            "data": json.dumps([s.model_dump() for s in sources]),
        }
        parts: list[str] = []
        trace.start_generation(
            model=active_chat_model(),
            system=system,
            user_msg=user_msg,
            max_tokens=max_tokens,
        )
        try:
            async for token in iterate_in_threadpool(
                stream_chat_completion(
                    user_msg,
                    system=system,
                    max_tokens=max_tokens,
                )
            ):
                parts.append(token)
                yield {"event": "token", "data": token}
        except Exception as exc:
            trace.end_error(error_type="llm_error", message=str(exc), stage="llm")
            yield {
                "event": "error",
                "data": json.dumps({"error": "llm_error", "message": str(exc)}),
            }
            return

        final_answer = clean_rag_answer(
            "".join(parts),
            source_files=unique_source_filenames([s.model_dump() for s in sources]),
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        provider = llm_provider()
        trace.end_success(
            final_answer=final_answer,
            latency_ms=elapsed_ms,
            provider=provider,
            prompt_version=pv,
        )
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "session_id": session_id,
                    "provider": provider,
                    "latency_ms": elapsed_ms,
                    "prompt_version": pv,
                    "client_platform": x_client_platform,
                    "final_answer": final_answer,
                }
            ),
        }

    return EventSourceResponse(event_generator(), headers=sse_headers())
