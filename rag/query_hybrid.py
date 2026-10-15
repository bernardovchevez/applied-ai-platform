#!/usr/bin/env python3
"""Hybrid BM25 + vector RAG over Chroma."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from shared.embeddings import get_embedding_function
from shared.llm import chat_completion, llm_provider
from stores.hybrid import fuse_hybrid_hits, get_bm25_index

CHROMA_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "media_docs"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "guardrails" / "prompts"
DEFAULT_FETCH_K = 20


def unique_sources(sources: list[str | None]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for source in sources:
        if source and source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


def get_collection():
    embedding_fn = get_embedding_function()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve_vector(question: str, fetch_k: int = DEFAULT_FETCH_K) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=fetch_k)
    hits: list[dict] = []
    for doc, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return hits


def retrieve(
    question: str,
    top_k: int = 4,
    fetch_k: int = DEFAULT_FETCH_K,
) -> list[dict]:
    vector_hits = retrieve_vector(question, fetch_k=fetch_k)
    bm25_hits = get_bm25_index().search(question, top_k=fetch_k)
    fused = fuse_hybrid_hits(vector_hits, bm25_hits, top_k=top_k)
    return [
        {
            "text": hit.text,
            "metadata": hit.metadata,
            "distance": 1.0 / (1.0 + hit.score),
            "hybrid_score": hit.score,
            "vector_rank": hit.vector_rank,
            "bm25_rank": hit.bm25_rank,
        }
        for hit in fused
    ]


def answer(question: str, top_k: int = 4, fetch_k: int = DEFAULT_FETCH_K) -> dict:
    chunks = retrieve(question, top_k=top_k, fetch_k=fetch_k)
    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['text']}" for c in chunks
    )

    system_path = PROMPTS_DIR / "system_rag.txt"
    system = system_path.read_text() if system_path.exists() else "Answer from context only."
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    answer_text = chat_completion(user_msg, system=system)

    return {
        "question": question,
        "answer": answer_text,
        "provider": llm_provider(),
        "sources": unique_sources([c["metadata"].get("source") for c in chunks]),
        "chunks": chunks,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid BM25 + vector RAG query")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks with vector/BM25 ranks",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    if args.show_context:
        chunks = retrieve(question, top_k=args.top_k, fetch_k=args.fetch_k)
        print("=== Hybrid retrieved context (RRF) ===")
        for i, c in enumerate(chunks, 1):
            src = c["metadata"].get("source", "unknown")
            print(
                f"\n--- Chunk {i} ({src}, score={c['hybrid_score']:.4f}, "
                f"vec_rank={c.get('vector_rank')}, bm25_rank={c.get('bm25_rank')}) ---"
            )
            print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
        print("\n=== Answer ===")

    result = answer(question, top_k=args.top_k, fetch_k=args.fetch_k)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
