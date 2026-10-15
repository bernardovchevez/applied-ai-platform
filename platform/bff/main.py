"""Media Ops Applied AI Platform BFF.

Run from this directory:
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

BFF_DIR = Path(__file__).resolve().parent
LAB_ROOT = BFF_DIR.parent.parent
OPENAPI_PATH = BFF_DIR / "openapi.yaml"
PROMPTS_DIR = LAB_ROOT / "guardrails" / "prompts"
API_VERSION = "0.1.0"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "rag"))
sys.path.insert(0, str(BFF_DIR.parent))

from ops.langfuse_tracing import ChatStreamTrace, AgentRunTrace, active_chat_model  # noqa: E402

from query_pgvector import retrieve as pgvector_retrieve  # noqa: E402
from feedback import insert_feedback  # noqa: E402
from evals_summary import build_eval_summary, eval_summary_has_data  # noqa: E402
from agent_runner import execute_agent, serialize_agent_result  # noqa: E402
from agent_sessions import resolve_session, save_pending_session  # noqa: E402
from health_checks import build_health_report, check_pgvector  # noqa: E402
from shared.config import app_env, get_env, is_production, state_store_backend  # noqa: E402
from shared.embeddings import active_embedding_model_id  # noqa: E402
from shared.rag_text import clean_rag_answer, normalize_chunk_text, rag_question_hints, unique_source_filenames  # noqa: E402
from shared.llm import llm_provider, stream_chat_completion  # noqa: E402
from stores.pgvector import PgVectorStore  # noqa: E402
from stores.base import ChunkHit  # noqa: E402


def _docs_enabled() -> bool:
    explicit = get_env("BFF_ENABLE_DOCS")
    if explicit is not None:
        return explicit.strip().lower() in ("1", "true", "yes")
    return not is_production()


app = FastAPI(
    title="Media Ops Applied AI Platform BFF",
    version=API_VERSION,
    description=(
        "Presentation layer for web, mobile, and CTV clients. "
        "Platform logic lives in applied-ai-platform (pgvector, agents, MCP)."
    ),
    docs_url="/docs" if _docs_enabled() else None,
    redoc_url="/redoc" if _docs_enabled() else None,
)


def _cors_origins() -> list[str]:
    raw = get_env("BFF_CORS_ORIGINS")
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def verify_bff_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Verify BFF_API_KEY when configured (stage/production). No-op when unset."""
    expected = get_env("BFF_API_KEY")
    if not expected:
        return

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Valid BFF API key required."},
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-Client-Platform",
        "X-Session-Id",
        "Authorization",
        "X-API-Key",
    ],
    expose_headers=["X-Prompt-Version", "X-Embed-Model", "X-Environment"],
)


def _custom_openapi() -> dict:
    if OPENAPI_PATH.exists():
        return yaml.safe_load(OPENAPI_PATH.read_text(encoding="utf-8"))
    return app.openapi_schema or {}


app.openapi = _custom_openapi  # type: ignore[method-assign]


class MetadataFilters(BaseModel):
    doc_type: str | None = None
    confidentiality: str | None = None
    rights_tier: Literal["cleared", "pending", "denied", "excluded", "unknown"] | None = None
    department: str | None = None


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    filters: MetadataFilters | None = None


class SourceCitation(BaseModel):
    file_name: str
    doc_type: str | None = None
    rights_tier: str | None = None
    sharepoint_path: str | None = None
    chunk_index: int | None = None
    distance: float | None = None


class ChunkHitOut(BaseModel):
    chunk_id: str
    text: str
    source: SourceCitation


class RetrieveResponse(BaseModel):
    question: str
    chunks: list[ChunkHitOut]
    sources: list[SourceCitation]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    filters: MetadataFilters | None = None
    format: Literal["full", "compact"] = "full"


FeedbackRating = Literal["up", "down", "approve", "reject"]


class FeedbackRequest(BaseModel):
    session_id: str
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    rating: FeedbackRating
    sources: list[SourceCitation] | None = None
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: str
    created_at: str


class EvalBackendScore(BaseModel):
    backend: str
    passed: int
    total: int
    note: str | None = None


class PromptfooSummary(BaseModel):
    passed: int
    total: int
    failures: int | None = None
    errors: int | None = None
    eval_id: str | None = None
    timestamp: str | None = None


class EvalSummaryResponse(BaseModel):
    gold_questions: int
    agent_scenarios: int = 0
    keyword_eval: list[EvalBackendScore]
    keyword_eval_timestamp: str | None = None
    ragas_eval: list[EvalBackendScore] = Field(default_factory=list)
    ragas_eval_timestamp: str | None = None
    ragas_judge_model: str | None = None
    agent_eval: EvalBackendScore | None = None
    agent_eval_timestamp: str | None = None
    promptfoo: PromptfooSummary | None = None
    sources: dict[str, str] | None = None


class PgvectorHealthOk(BaseModel):
    status: Literal["ok"]
    chunk_count: int
    document_count: int


class PgvectorHealthFailed(BaseModel):
    status: Literal["unreachable"]
    error: str


PgvectorHealth = Annotated[
    PgvectorHealthOk | PgvectorHealthFailed,
    Field(discriminator="status"),
]


class LlmHealthOk(BaseModel):
    provider: str
    status: Literal["ok"]


class LlmHealthFailed(BaseModel):
    provider: str
    status: Literal["unreachable"]
    error: str


LlmHealth = Annotated[
    LlmHealthOk | LlmHealthFailed,
    Field(discriminator="status"),
]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: Literal["localhost", "stage", "production"] = "localhost"
    state_store: Literal["sqlite", "postgres"] = "sqlite"
    pgvector: PgvectorHealth
    llm: LlmHealth


AgentResumeDecision = Literal["approve", "reject"]


class AgentRunRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=1)
    auto_approve: bool = False
    thread_id: str | None = None
    resume: AgentResumeDecision | None = None


class AgentRunResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    tool_trace: list[dict[str, Any]]
    status: Literal["approved", "completed", "rejected"] = "completed"


class AgentPausedResponse(BaseModel):
    thread_id: str
    message: str
    proposed_output: str | None = None
    prompt: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)


def _sse_headers() -> dict[str, str]:
    return {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache, no-transform",
        "X-Environment": app_env(),
    }


def _sse_error(message: str, *, error: str = "service_unavailable") -> EventSourceResponse:
    async def error_events():
        yield {
            "event": "error",
            "data": json.dumps({"error": error, "message": message}),
        }

    return EventSourceResponse(error_events(), headers=_sse_headers())


def _pgvector_error_detail(exc: str | None) -> str | None:
    """Short user-facing detail from psycopg2 / connection errors."""
    if not exc:
        return None
    lower = exc.lower()
    if "connection refused" in lower:
        return "Postgres is not running on localhost:5432."
    if "timeout expired" in lower or "timed out" in lower:
        return "Postgres connection timed out."
    if "password authentication failed" in lower:
        return "Postgres authentication failed."
    # Drop duplicate psycopg2 paragraphs (IPv4 + IPv6 repeats).
    first = exc.strip().split("\n")[0].strip()
    if len(first) > 120:
        return first[:117] + "..."
    return first


def _pgvector_unavailable_message(detail: str | None = None) -> str:
    base = (
        "Document search is unavailable (pgvector/Postgres). "
        "Start the database: cd applied-ai-platform && docker compose up -d"
    )
    short = _pgvector_error_detail(detail)
    if short:
        return f"{base}. {short}"
    return base


def _chunk_id(meta: dict) -> str:
    stem = Path(meta.get("path") or meta.get("source") or "doc").stem
    index = int(meta.get("chunk_index") or 0)
    return hashlib.md5(f"{stem}-{index}".encode()).hexdigest()


def _source_from_metadata(meta: dict, distance: float | None = None) -> SourceCitation:
    return SourceCitation(
        file_name=meta.get("source") or "unknown",
        doc_type=meta.get("doc_type"),
        rights_tier=meta.get("rights_tier"),
        sharepoint_path=meta.get("sharepoint_path"),
        chunk_index=meta.get("chunk_index"),
        distance=distance,
    )


def _filters_kwargs(filters: MetadataFilters | None) -> dict:
    if not filters:
        return {}
    return {k: v for k, v in filters.model_dump().items() if v is not None}


def _prompt_version() -> str:
    prompt_path = PROMPTS_DIR / "system_rag.txt"
    if not prompt_path.exists():
        return "unknown"
    return hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:12]


def _load_system_prompt(*, compact: bool) -> str:
    path = PROMPTS_DIR / "system_rag.txt"
    system = path.read_text(encoding="utf-8") if path.exists() else "Answer from context only."
    if compact:
        system += "\n\nRespond in 2-4 concise bullet points for mobile/CTV. Keep under 120 words."
    return system


def _build_rag_user_message(question: str, hits: list[ChunkHit]) -> str:
    blocks: list[str] = []
    for h in hits:
        source = h.metadata.get("source") or "unknown"
        body = normalize_chunk_text(h.text.strip())
        blocks.append(f'"{source}"\n---\n{body}')
    context = "\n\n".join(blocks)
    hints = rag_question_hints(question)
    return f"Context:\n{context}\n\nQuestion: {question}{hints}"


def _sources_from_hits(hits: list[ChunkHit]) -> list[SourceCitation]:
    seen_files: set[str] = set()
    sources: list[SourceCitation] = []
    for hit in hits:
        name = hit.metadata.get("source")
        if name and name not in seen_files:
            seen_files.add(name)
            sources.append(_source_from_metadata(hit.metadata, distance=hit.distance))
    return sources


def _run_retrieve(question: str, top_k: int, filters: MetadataFilters | None) -> list[ChunkHit]:
    store = _get_store()
    return pgvector_retrieve(
        store,
        question,
        top_k=top_k,
        **_filters_kwargs(filters),
    )


def _agent_sources(sources: list[dict[str, Any]]) -> list[SourceCitation]:
    return [
        SourceCitation(
            file_name=row.get("file_name") or "unknown",
            doc_type=row.get("doc_type"),
            rights_tier=row.get("rights_tier"),
            sharepoint_path=row.get("sharepoint_path"),
        )
        for row in sources
    ]


def _agent_run_response(
    payload: dict[str, Any],
    *,
    status: Literal["approved", "completed", "rejected"] = "completed",
) -> AgentRunResponse:
    return AgentRunResponse(
        answer=payload["answer"],
        sources=_agent_sources(payload.get("sources") or []),
        tool_trace=payload.get("tool_trace") or [],
        status=status,
    )


_store: PgVectorStore | None = None


def _get_store() -> PgVectorStore:
    global _store
    if _store is None:
        _store = PgVectorStore()
    return _store


def _stub(platform: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "message": "Not implemented in platform/bff/main.py",
            "spec": "/openapi.yaml",
            "client_platform": platform,
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    responses={503: {"model": HealthResponse}},
)
def health() -> HealthResponse | JSONResponse:
    report = build_health_report(version=API_VERSION)
    response = HealthResponse.model_validate(report)
    if response.status == "degraded":
        return JSONResponse(status_code=503, content=response.model_dump(exclude_none=True))
    return response


@app.get("/v1/version")
def version() -> JSONResponse:
    prompt_version = _prompt_version()
    embed_model = active_embedding_model_id()
    env_name = app_env()

    body = {
        "api_version": API_VERSION,
        "environment": env_name,
        "state_store": state_store_backend(),
        "prompt_version": prompt_version,
        "embed_model": embed_model,
        "index_document_count": None,
    }
    return JSONResponse(
        content=body,
        headers={
            "X-Prompt-Version": prompt_version,
            "X-Embed-Model": embed_model,
            "X-Environment": env_name,
        },
    )


@app.get("/openapi.yaml", include_in_schema=False)
def serve_openapi_file() -> FileResponse:
    return FileResponse(OPENAPI_PATH, media_type="application/yaml")


@app.post("/v1/retrieve", response_model=RetrieveResponse, dependencies=[Depends(verify_bff_api_key)])
async def retrieve(
    body: RetrieveRequest,
    x_client_platform: str | None = Header(default=None, alias="X-Client-Platform"),
) -> RetrieveResponse:
    try:
        hits = _run_retrieve(body.question, body.top_k, body.filters)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": str(exc)},
        ) from exc

    chunks: list[ChunkHitOut] = []
    for hit in hits:
        meta = hit.metadata
        citation = _source_from_metadata(meta, distance=hit.distance)
        chunks.append(
            ChunkHitOut(
                chunk_id=_chunk_id(meta),
                text=hit.text,
                source=citation,
            )
        )

    return RetrieveResponse(
        question=body.question,
        chunks=chunks,
        sources=_sources_from_hits(hits),
    )


@app.post("/v1/chat/stream", dependencies=[Depends(verify_bff_api_key)])
async def chat_stream(
    body: ChatRequest,
    x_client_platform: str | None = Header(default=None, alias="X-Client-Platform"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> EventSourceResponse:
    start = time.perf_counter()
    session_id = x_session_id or str(uuid.uuid4())
    prompt_version = _prompt_version()
    trace = ChatStreamTrace(
        session_id=session_id,
        question=body.question,
        prompt_version=prompt_version,
        client_platform=x_client_platform,
        top_k=body.top_k,
        format=body.format,
    )

    pg = check_pgvector(_get_store())
    if pg["status"] != "ok":
        msg = _pgvector_unavailable_message(pg.get("error"))
        trace.end_error(error_type="pgvector_unavailable", message=msg, stage="preflight")
        return _sse_error(msg)

    try:
        hits = _run_retrieve(body.question, body.top_k, body.filters)
    except Exception as exc:
        msg = _pgvector_unavailable_message(str(exc))
        trace.end_error(error_type="retrieve_error", message=msg, stage="retrieve")
        return _sse_error(msg)

    sources = _sources_from_hits(hits)
    trace.end_retrieve([s.model_dump() for s in sources], len(hits))
    system = _load_system_prompt(compact=body.format == "compact")
    user_msg = _build_rag_user_message(body.question, hits)
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
            prompt_version=prompt_version,
        )
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "session_id": session_id,
                    "provider": provider,
                    "latency_ms": elapsed_ms,
                    "prompt_version": prompt_version,
                    "client_platform": x_client_platform,
                    "final_answer": final_answer,
                }
            ),
        }

    return EventSourceResponse(event_generator(), headers=_sse_headers())


@app.post("/v1/feedback", response_model=FeedbackResponse, status_code=201, dependencies=[Depends(verify_bff_api_key)])
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
            prompt_version=_prompt_version(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "feedback_store_failed", "message": str(exc)},
        ) from exc

    return FeedbackResponse(id=row["id"], created_at=row["created_at"])


@app.get("/v1/evals/summary", response_model=EvalSummaryResponse, dependencies=[Depends(verify_bff_api_key)])
def evals_summary() -> EvalSummaryResponse:
    data = build_eval_summary()
    if not eval_summary_has_data(data):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "evals_not_found",
                "message": "No eval artifacts found under rag/evals/ or agents/evals/",
            },
        )
    agent_eval = data.get("agent_eval")
    return EvalSummaryResponse(
        gold_questions=data["gold_questions"],
        agent_scenarios=data.get("agent_scenarios") or 0,
        keyword_eval=data["keyword_eval"],
        keyword_eval_timestamp=data.get("keyword_eval_timestamp"),
        ragas_eval=data.get("ragas_eval") or [],
        ragas_eval_timestamp=data.get("ragas_eval_timestamp"),
        ragas_judge_model=data.get("ragas_judge_model"),
        agent_eval=EvalBackendScore(
            backend=agent_eval["backend"],
            passed=agent_eval["passed"],
            total=agent_eval["total"],
            note=agent_eval.get("note"),
        )
        if agent_eval
        else None,
        agent_eval_timestamp=agent_eval.get("timestamp") if agent_eval else None,
        promptfoo=data.get("promptfoo"),
        sources=data.get("sources"),
    )


@app.post(
    "/v1/agent/run",
    response_model=None,
    dependencies=[Depends(verify_bff_api_key)],
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
        return _agent_run_response(payload, status="approved")

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
        return _agent_run_response(payload)

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
        sources=_agent_sources(payload.get("sources") or []),
        tool_trace=payload.get("tool_trace") or [],
    )
    return JSONResponse(status_code=202, content=paused.model_dump())


@app.get("/", include_in_schema=False)
def root() -> PlainTextResponse:
    return PlainTextResponse(
        f"Media Ops AI Platform BFF v{API_VERSION} ({app_env()})\n"
        "Docs: /docs\n"
        "OpenAPI: /openapi.yaml\n"
        "Health: /health\n"
    )
