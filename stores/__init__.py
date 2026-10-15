"""Vector store adapters."""

from __future__ import annotations

from typing import Any

from stores.base import ChunkHit, VectorStore
from stores.pgvector import PgVectorStore

__all__ = ["ChunkHit", "PgVectorStore", "PineconeStore", "VectorStore"]


def __getattr__(name: str) -> Any:
    if name == "PineconeStore":
        from stores.pinecone import PineconeStore

        return PineconeStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
