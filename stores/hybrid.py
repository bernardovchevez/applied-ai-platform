"""Hybrid BM25 + dense vector retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from stores.corpus import iter_corpus_chunks

RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class CorpusChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class HybridHit:
    text: str
    metadata: dict[str, Any]
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None


class BM25ChunkIndex:
    """In-memory BM25 index over corpus chunks."""

    def __init__(self, chunks: list[CorpusChunk]) -> None:
        self._chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((self._chunks[i].chunk_id, float(scores[i])) for i in range(len(self._chunks))),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def get(self, chunk_id: str) -> CorpusChunk | None:
        return self._by_id.get(chunk_id)


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25ChunkIndex:
    chunks = [
        CorpusChunk(chunk_id=cid, text=text, metadata=meta)
        for cid, text, meta in iter_corpus_chunks()
    ]
    if not chunks:
        raise RuntimeError("No corpus chunks found for BM25 index")
    return BM25ChunkIndex(chunks)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Merge ranked chunk IDs with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_hybrid_hits(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[tuple[str, float]],
    top_k: int,
) -> list[HybridHit]:
    """Combine vector and BM25 results into fused HybridHit list."""
    bm25_index = get_bm25_index()
    vector_ranked = [h["chunk_id"] for h in vector_hits if h.get("chunk_id")]
    bm25_ranked = [chunk_id for chunk_id, _ in bm25_hits]

    fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked])[:top_k]
    vector_rank_map = {cid: i + 1 for i, cid in enumerate(vector_ranked)}
    bm25_rank_map = {cid: i + 1 for i, cid in enumerate(bm25_ranked)}
    vector_by_id = {h["chunk_id"]: h for h in vector_hits if h.get("chunk_id")}

    hits: list[HybridHit] = []
    for chunk_id, score in fused:
        if chunk_id in vector_by_id:
            src = vector_by_id[chunk_id]
            hits.append(
                HybridHit(
                    text=src["text"],
                    metadata=src["metadata"],
                    score=score,
                    vector_rank=vector_rank_map.get(chunk_id),
                    bm25_rank=bm25_rank_map.get(chunk_id),
                )
            )
            continue
        chunk = bm25_index.get(chunk_id)
        if chunk:
            hits.append(
                HybridHit(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=score,
                    vector_rank=vector_rank_map.get(chunk_id),
                    bm25_rank=bm25_rank_map.get(chunk_id),
                )
            )
    return hits
