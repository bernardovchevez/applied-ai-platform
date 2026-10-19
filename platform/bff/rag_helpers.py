"""RAG retrieve, citations, prompts, and SSE helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from sse_starlette.sse import EventSourceResponse

from bootstrap import PROMPTS_DIR
from models import AgentRunResponse, MetadataFilters, SourceCitation
from query_pgvector import retrieve as pgvector_retrieve
from shared.config import app_env
from shared.rag_text import normalize_chunk_text, rag_question_hints
from stores.base import ChunkHit
from stores.pgvector import PgVectorStore


def sse_headers() -> dict[str, str]:
    return {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache, no-transform",
        "X-Environment": app_env(),
    }


def sse_error(message: str, *, error: str = "service_unavailable") -> EventSourceResponse:
    async def error_events():
        yield {
            "event": "error",
            "data": json.dumps({"error": error, "message": message}),
        }

    return EventSourceResponse(error_events(), headers=sse_headers())


def pgvector_error_detail(exc: str | None) -> str | None:
    if not exc:
        return None
    lower = exc.lower()
    if "connection refused" in lower:
        return "Postgres is not running on localhost:5432."
    if "timeout expired" in lower or "timed out" in lower:
        return "Postgres connection timed out."
    if "password authentication failed" in lower:
        return "Postgres authentication failed."
    first = exc.strip().split("\n")[0].strip()
    if len(first) > 120:
        return first[:117] + "..."
    return first


def pgvector_unavailable_message(detail: str | None = None) -> str:
    base = (
        "Document search is unavailable (pgvector/Postgres). "
        "Start the database: cd applied-ai-platform && docker compose up -d"
    )
    short = pgvector_error_detail(detail)
    if short:
        return f"{base}. {short}"
    return base


def chunk_id(meta: dict) -> str:
    stem = Path(meta.get("path") or meta.get("source") or "doc").stem
    index = int(meta.get("chunk_index") or 0)
    return hashlib.md5(f"{stem}-{index}".encode()).hexdigest()


def source_from_metadata(meta: dict, distance: float | None = None) -> SourceCitation:
    return SourceCitation(
        file_name=meta.get("source") or "unknown",
        doc_type=meta.get("doc_type"),
        rights_tier=meta.get("rights_tier"),
        sharepoint_path=meta.get("sharepoint_path"),
        chunk_index=meta.get("chunk_index"),
        distance=distance,
    )


def filters_kwargs(filters: MetadataFilters | None) -> dict:
    if not filters:
        return {}
    return {k: v for k, v in filters.model_dump().items() if v is not None}


def prompt_version() -> str:
    prompt_path = PROMPTS_DIR / "system_rag.txt"
    if not prompt_path.exists():
        return "unknown"
    return hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:12]


def load_system_prompt(*, compact: bool) -> str:
    path = PROMPTS_DIR / "system_rag.txt"
    system = path.read_text(encoding="utf-8") if path.exists() else "Answer from context only."
    if compact:
        system += "\n\nRespond in 2-4 concise bullet points for mobile/CTV/Slack. Keep under 120 words."
    return system


def build_rag_user_message(question: str, hits: list[ChunkHit]) -> str:
    blocks: list[str] = []
    for h in hits:
        source = h.metadata.get("source") or "unknown"
        body = normalize_chunk_text(h.text.strip())
        blocks.append(f'"{source}"\n---\n{body}')
    context = "\n\n".join(blocks)
    hints = rag_question_hints(question)
    return f"Context:\n{context}\n\nQuestion: {question}{hints}"


def sources_from_hits(hits: list[ChunkHit]) -> list[SourceCitation]:
    seen_files: set[str] = set()
    sources: list[SourceCitation] = []
    for hit in hits:
        name = hit.metadata.get("source")
        if name and name not in seen_files:
            seen_files.add(name)
            sources.append(source_from_metadata(hit.metadata, distance=hit.distance))
    return sources


_store: PgVectorStore | None = None


def get_store() -> PgVectorStore:
    global _store
    if _store is None:
        _store = PgVectorStore()
    return _store


def run_retrieve(question: str, top_k: int, filters: MetadataFilters | None) -> list[ChunkHit]:
    store = get_store()
    return pgvector_retrieve(
        store,
        question,
        top_k=top_k,
        **filters_kwargs(filters),
    )


def agent_sources(sources: list[dict[str, Any]]) -> list[SourceCitation]:
    return [
        SourceCitation(
            file_name=row.get("file_name") or "unknown",
            doc_type=row.get("doc_type"),
            rights_tier=row.get("rights_tier"),
            sharepoint_path=row.get("sharepoint_path"),
        )
        for row in sources
    ]


def agent_run_response(
    payload: dict[str, Any],
    *,
    status: Literal["approved", "completed", "rejected"] = "completed",
) -> AgentRunResponse:
    return AgentRunResponse(
        answer=payload["answer"],
        sources=agent_sources(payload.get("sources") or []),
        tool_trace=payload.get("tool_trace") or [],
        status=status,
    )
