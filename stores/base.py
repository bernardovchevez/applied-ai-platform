"""Vector store adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkHit:
    """One retrieved chunk with metadata and similarity score."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


@dataclass
class IngestStats:
    documents: int
    chunks: int
    lora_tagged_chunks: int = 0


class VectorStore(ABC):
    """Common interface for Chroma, pgvector, Pinecone backends."""

    @abstractmethod
    def reset(self) -> None:
        """Drop and recreate store tables/index."""

    @abstractmethod
    def ingest(self, data_dir=None, reset: bool = True) -> IngestStats:
        """Load corpus from data_dir and embed chunks."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 4,
        *,
        doc_type: str | None = None,
        confidentiality: str | None = None,
        rights_tier: str | None = None,
        department: str | None = None,
    ) -> list[ChunkHit]:
        """Similarity search with optional relational metadata filters."""
