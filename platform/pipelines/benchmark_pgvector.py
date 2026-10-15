#!/usr/bin/env python3
"""Benchmark pgvector retrieval latency (compare with Databricks AI Search notebook)."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from stores.pgvector import PgVectorStore  # noqa: E402


def benchmark(
    question: str,
    *,
    top_k: int,
    doc_type: str | None,
    runs: int,
) -> dict:
    store = PgVectorStore()
    filters = {"doc_type": doc_type} if doc_type else {}

    # Warm-up (embed query + connection)
    store.search(question, top_k=top_k, **filters)

    timings: list[float] = []
    last_hits = []
    for _ in range(runs):
        start = time.perf_counter()
        last_hits = store.search(question, top_k=top_k, **filters)
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "question": question,
        "filters": filters,
        "top_k": top_k,
        "runs": runs,
        "ms_p50": statistics.median(timings),
        "ms_mean": statistics.mean(timings),
        "ms_min": min(timings),
        "ms_max": max(timings),
        "hit_count": len(last_hits),
        "sources": [h.metadata.get("source") for h in last_hits],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pgvector search latency")
    parser.add_argument(
        "question",
        nargs="?",
        default="Is the theme song cleared for EMEA streaming?",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--doc-type", default="rights_clearance")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    result = benchmark(
        args.question,
        top_k=args.top_k,
        doc_type=args.doc_type,
        runs=args.runs,
    )

    print(f"Question: {result['question']}")
    if result["filters"]:
        print(f"Filters: {result['filters']}")
    print(f"Runs: {result['runs']} | hits: {result['hit_count']}")
    print(
        f"Latency (ms): p50={result['ms_p50']:.1f} "
        f"mean={result['ms_mean']:.1f} min={result['ms_min']:.1f} max={result['ms_max']:.1f}"
    )
    print("Sources:", ", ".join(s for s in result["sources"] if s))


if __name__ == "__main__":
    main()
