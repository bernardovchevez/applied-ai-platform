#!/usr/bin/env python3
"""Query pgvector RAG pipeline with metadata filters."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import get_env
from shared.llm import chat_completion, llm_provider
from shared.rag_text import clean_rag_answer, normalize_chunk_text, rag_question_hints
from stores.pgvector import PgVectorStore

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "guardrails" / "prompts"


def unique_sources(sources: list[str | None]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for source in sources:
        if source and source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


def retrieve(
    store: PgVectorStore,
    question: str,
    top_k: int = 4,
    **filters,
) -> list:
    return store.search(question, top_k=top_k, **filters)


def answer(question: str, top_k: int = 4, **filters) -> dict:
    store = PgVectorStore()
    chunks = retrieve(store, question, top_k=top_k, **filters)
    context = "\n\n".join(
        f'"{c.metadata.get("source", "unknown")}"\n---\n{normalize_chunk_text(c.text.strip())}'
        for c in chunks
    )

    system_path = PROMPTS_DIR / "system_rag.txt"
    system = system_path.read_text() if system_path.exists() else "Answer from context only."
    user_msg = f"Context:\n{context}\n\nQuestion: {question}{rag_question_hints(question)}"
    answer_text = clean_rag_answer(
        chat_completion(user_msg, system=system),
        source_files=unique_sources([c.metadata.get("source") for c in chunks]),
    )

    return {
        "question": question,
        "answer": answer_text,
        "provider": llm_provider(),
        "filters": {k: v for k, v in filters.items() if v is not None},
        "sources": unique_sources([c.metadata.get("source") for c in chunks]),
        "chunks": chunks,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Query media ops corpus via pgvector")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--doc-type", dest="doc_type", help="Filter: doc_type")
    parser.add_argument("--confidentiality", help="Filter: confidentiality")
    parser.add_argument("--rights-tier", dest="rights_tier", help="Filter: rights_tier")
    parser.add_argument("--department", help="Filter: department")
    parser.add_argument(
        "--sql-demo",
        action="store_true",
        help="Run warehouse join demo (rights_clearance + cleared)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks sent to the LLM",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    store = PgVectorStore()

    if args.sql_demo:
        print("=== SQL join demo: cleared rights_clearance memos ===")
        print(f"DB: {get_env('DATABASE_URL', store.database_url).split('@')[-1]}")
        for row in store.sql_join_demo(question, top_k=args.top_k):
            print(
                f"\n{row['file']} ({row['doc_type']}, {row['rights_tier']}) "
                f"dist={row['distance']}"
            )
            print(f"  SharePoint: {row['sharepoint_path']}")
            print(f"  {row['snippet']}")
        return

    filters = {
        "doc_type": args.doc_type,
        "confidentiality": args.confidentiality,
        "rights_tier": args.rights_tier,
        "department": args.department,
    }

    if args.show_context:
        chunks = retrieve(store, question, top_k=args.top_k, **filters)
        active = {k: v for k, v in filters.items() if v}
        print("=== Retrieved context ===")
        if active:
            print("Filters:", active)
        for i, c in enumerate(chunks, 1):
            src = c.metadata.get("source", "unknown")
            print(
                f"\n--- Chunk {i} ({src}, distance={c.distance:.4f}, "
                f"doc_type={c.metadata.get('doc_type')}) ---"
            )
            print(c.text[:400] + ("..." if len(c.text) > 400 else ""))
        print("\n=== Answer ===")

    result = answer(question, top_k=args.top_k, **filters)
    if result["filters"]:
        print("Filters:", result["filters"])
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
