#!/usr/bin/env python3
"""Query ChromaDB RAG pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from shared.config import get_env
from shared.embeddings import get_embedding_function
from shared.llm import chat_completion, llm_provider

CHROMA_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "media_docs"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "guardrails" / "prompts"


def unique_sources(sources: list[str | None]) -> list[str]:
    """Preserve order, drop duplicates and empty values."""
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


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def answer(question: str, top_k: int = 4) -> dict:
    chunks = retrieve(question, top_k=top_k)
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

    parser = argparse.ArgumentParser(description="Query media ops RAG corpus")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks sent to the LLM (debug)",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    if args.show_context:
        chunks = retrieve(question)
        print("=== Retrieved context ===")
        for i, c in enumerate(chunks, 1):
            src = c["metadata"].get("source", "unknown")
            print(f"\n--- Chunk {i} ({src}, distance={c['distance']:.4f}) ---")
            print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
        print("\n=== Answer ===")

    result = answer(question)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
