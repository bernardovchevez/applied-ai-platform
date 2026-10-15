"""Dependency probes for BFF /health."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from shared.config import get_env  # noqa: E402
from shared.llm import llm_provider  # noqa: E402
from shared.ollama_client import is_available as ollama_available  # noqa: E402
from stores.pgvector import PgVectorStore  # noqa: E402


def check_pgvector(store: PgVectorStore | None = None) -> dict[str, Any]:
    """Ping Postgres/pgvector and return chunk + document counts."""
    try:
        pg = store or PgVectorStore()
        with pg._connect_raw() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks")
                chunk_count = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM documents")
                document_count = int(cur.fetchone()[0])
        return {
            "status": "ok",
            "chunk_count": chunk_count,
            "document_count": document_count,
        }
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


def check_llm() -> dict[str, Any]:
    """Report active LLM provider and whether it is reachable."""
    provider = llm_provider()
    if provider == "ollama":
        if ollama_available():
            return {"provider": provider, "status": "ok"}
        return {
            "provider": provider,
            "status": "unreachable",
            "error": "Ollama not reachable (ollama serve)",
        }
    if provider == "openai":
        if get_env("OPENAI_API_KEY"):
            return {"provider": provider, "status": "ok"}
        return {
            "provider": provider,
            "status": "unreachable",
            "error": "OPENAI_API_KEY not set",
        }
    return {"provider": provider, "status": "unreachable", "error": "unknown provider"}


def build_health_report(*, version: str) -> dict[str, Any]:
    pg = check_pgvector()
    llm = check_llm()
    checks_ok = pg["status"] == "ok" and llm["status"] == "ok"
    overall = "ok" if checks_ok else "degraded"
    return {
        "status": overall,
        "version": version,
        "pgvector": pg,
        "llm": llm,
    }
