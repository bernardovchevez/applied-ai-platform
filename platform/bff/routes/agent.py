"""POST /v1/agent/run."""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from ops.langfuse_tracing import AgentRunTrace

from agent_runner import execute_agent, serialize_agent_result
from agent_sessions import resolve_session, save_pending_session
from deps import verify_bff_api_key
from models import AgentPausedResponse, AgentRunRequest, AgentRunResponse
from rag_helpers import agent_run_response, agent_sources

router = APIRouter(dependencies=[Depends(verify_bff_api_key)])


@router.post(
    "/v1/agent/run",
    response_model=None,
    responses={
        200: {"model": AgentRunResponse},
        202: {"model": AgentPausedResponse},
    },
)
async def agent_run(
    body: AgentRunRequest,
    x_client_platform: str | None = Header(default=None, alias="X-Client-Platform"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> AgentRunResponse | JSONResponse:
    if body.thread_id and body.resume:
        session_id = x_session_id or body.thread_id
        trace = AgentRunTrace(
            session_id=session_id,
            prompt=None,
            client_platform=x_client_platform,
            resume=body.resume,
            thread_id=body.thread_id,
        )
        session = resolve_session(body.thread_id, body.resume)
        if not session:
            trace.end_error(
                error_type="session_not_found",
                message=f"No pending session for thread_id={body.thread_id}",
                stage="resume",
            )
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "session_not_found",
                    "message": f"No pending session for thread_id={body.thread_id}",
                },
            )
        if body.resume == "reject":
            tool_trace = [{"tool": "human_approval", "status": "rejected"}]
            trace.end_resume(decision="rejected", answer="", tool_trace=tool_trace)
            return JSONResponse(
                status_code=200,
                content=AgentRunResponse(
                    answer="",
                    sources=[],
                    tool_trace=tool_trace,
                    status="rejected",
                ).model_dump(),
            )
        payload = dict(session["result"])
        trace_list = list(payload.get("tool_trace") or [])
        trace_list.append(
            {
                "tool": "human_approval",
                "status": "approved",
                "thread_id": body.thread_id,
            }
        )
        payload["tool_trace"] = trace_list
        trace.end_resume(
            decision="approved",
            answer=payload.get("answer") or "",
            tool_trace=trace_list,
        )
        return agent_run_response(payload, status="approved")

    if not body.prompt:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "prompt is required unless resuming with thread_id + resume",
            },
        )

    pending_thread_id = str(uuid.uuid4()) if not body.auto_approve else None
    session_id = x_session_id or pending_thread_id or str(uuid.uuid4())
    trace = AgentRunTrace(
        session_id=session_id,
        prompt=body.prompt,
        client_platform=x_client_platform,
        auto_approve=body.auto_approve,
        thread_id=pending_thread_id,
    )
    start = time.perf_counter()

    try:
        result = await asyncio.to_thread(execute_agent, body.prompt)
        payload = serialize_agent_result(result)
    except Exception as exc:
        trace.end_error(error_type="agent_failed", message=str(exc), stage="execute")
        raise HTTPException(
            status_code=503,
            detail={"error": "agent_failed", "message": str(exc)},
        ) from exc

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if body.auto_approve:
        trace.end_completed(
            answer=payload["answer"],
            sources=payload.get("sources") or [],
            tool_trace=payload.get("tool_trace") or [],
            latency_ms=elapsed_ms,
            status="completed",
        )
        return agent_run_response(payload)

    thread_id = save_pending_session(
        prompt=body.prompt,
        proposed_output=payload["answer"],
        result_payload=payload,
        client_platform=x_client_platform,
        thread_id=pending_thread_id,
    )
    trace.end_paused(
        thread_id=thread_id,
        proposed_output=payload["answer"],
        tool_trace=payload.get("tool_trace") or [],
        latency_ms=elapsed_ms,
    )
    paused = AgentPausedResponse(
        thread_id=thread_id,
        message="Review the draft below. Approve to publish it to the workflow record; reject to discard.",
        proposed_output=payload["answer"],
        prompt=body.prompt,
        sources=agent_sources(payload.get("sources") or []),
        tool_trace=payload.get("tool_trace") or [],
    )
    return JSONResponse(status_code=202, content=paused.model_dump())
