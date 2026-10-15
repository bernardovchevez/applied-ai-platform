#!/usr/bin/env python3
"""Keyword + source checks against gold_qa.json."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

GOLD_PATH = Path(__file__).parent / "gold_qa.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
BASELINE_PATH = Path(__file__).parent / "baseline.json"
RAG_DIR = Path(__file__).resolve().parent.parent
PLATFORM_DIR = Path(__file__).resolve().parent.parent.parent / "platform"


def _maybe_log_mlflow(summary: list[dict[str, Any]], *, gold_path: str, artifact_path: Path) -> None:
    sys.path.insert(0, str(PLATFORM_DIR))
    from ops.mlflow_eval import log_rag_keyword_backend, mlflow_available, tracking_uri

    if not mlflow_available():
        print("MLflow not installed — pip install mlflow")
        return
    for row in summary:
        run_id = log_rag_keyword_backend(row, artifact_path=artifact_path, gold_path=gold_path)
        if run_id:
            print(f"MLflow run {row['backend']}: {run_id}")
    print(f"MLflow tracking URI: {tracking_uri()}")


def load_module(name: str, filename: str):
    path = RAG_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_chroma_runner():
    mod = load_module("query_chroma", "query.py")

    def run(question: str) -> dict[str, Any]:
        return mod.answer(question)

    return run


def make_langchain_runner():
    mod = load_module("query_langchain", "query_langchain.py")
    vectorstore = mod.get_vectorstore()
    chain = mod.build_chain(vectorstore)

    def run(question: str) -> dict[str, Any]:
        return mod.answer(question, chain)

    return run


def make_llamaindex_runner():
    mod = load_module("query_llamaindex", "query_llamaindex.py")
    index = mod.load_index()
    query_engine = mod.build_query_engine(index)

    def run(question: str) -> dict[str, Any]:
        return mod.answer(question, query_engine)

    return run


def make_pgvector_runner():
    mod = load_module("query_pgvector", "query_pgvector.py")

    def run(question: str) -> dict[str, Any]:
        return mod.answer(question)

    return run


def make_pinecone_runner():
    mod = load_module("query_pinecone", "query_pinecone.py")

    def run(question: str) -> dict[str, Any]:
        return mod.answer(question)

    return run


def make_hybrid_runner():
    mod = load_module("query_hybrid", "query_hybrid.py")

    def run(question: str) -> dict[str, Any]:
        return mod.answer(question)

    return run


BACKENDS: dict[str, Callable[[], Callable[[str], dict[str, Any]]]] = {
    "chroma": make_chroma_runner,
    "langchain": make_langchain_runner,
    "llamaindex": make_llamaindex_runner,
    "pgvector": make_pgvector_runner,
    "pinecone": make_pinecone_runner,
    "hybrid": make_hybrid_runner,
}


def evaluate_item(item: dict, result: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    answer_text = (result.get("answer") or "").lower()
    sources = result.get("sources") or []

    missing_required = [
        k for k in item.get("expected_contains", []) if k.lower() not in answer_text
    ]
    if missing_required:
        reasons.append(f"missing required keywords: {missing_required}")

    any_keywords = item.get("expected_contains_any") or []
    if any_keywords and not any(k.lower() in answer_text for k in any_keywords):
        reasons.append(f"need at least one of: {any_keywords}")

    forbidden = item.get("expected_contains_not_any") or []
    found_forbidden = [k for k in forbidden if k.lower() in answer_text]
    if found_forbidden:
        reasons.append(f"must not contain: {found_forbidden}")

    expected_sources = item.get("expected_sources") or []
    if expected_sources and not any(exp in sources for exp in expected_sources):
        reasons.append(f"expected source in {expected_sources}, got {sources}")

    return (not reasons, reasons)


def run_backend(name: str, gold: list[dict], verbose: bool = False) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"Backend: {name}")
    print("=" * 60)

    runner = BACKENDS[name]()
    passed = 0
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for item in gold:
        t_q = time.perf_counter()
        result = runner(item["question"])
        elapsed = time.perf_counter() - t_q
        ok, reasons = evaluate_item(item, result)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] ({elapsed:.1f}s) {item['question'][:70]}")

        entry: dict[str, Any] = {
            "question": item["question"],
            "pass": ok,
            "elapsed_s": round(elapsed, 2),
            "sources": result.get("sources"),
            "answer": result.get("answer") or "",
        }
        if not ok:
            entry["reasons"] = reasons
            entry["expected"] = {
                "sources": item.get("expected_sources"),
                "contains": item.get("expected_contains"),
                "contains_any": item.get("expected_contains_any"),
            }
        results.append(entry)

        if verbose or not ok:
            if not ok:
                print(f"       reasons: {'; '.join(reasons)}")
            print(f"       sources: {result.get('sources')}")
            preview = (result.get("answer") or "").replace("\n", " ")[:160]
            print(f"       answer:  {preview}...")

    total_time = time.perf_counter() - t0
    score = f"{passed}/{len(gold)}"
    print(f"\nScore: {score}  |  total time: {total_time:.1f}s")
    return {
        "backend": name,
        "passed": passed,
        "total": len(gold),
        "total_time_s": round(total_time, 2),
        "results": results,
    }


def write_results(summary: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(GOLD_PATH),
        "backends": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults written to {output_path}")
    return payload


def write_keyword_baseline(summary: list[dict[str, Any]], *, timestamp: str) -> None:
    """Merge keyword pass rates into baseline.json (preserves RAGAS section if present)."""
    baseline: dict[str, Any] = {}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline["keyword_timestamp"] = timestamp
    baseline["keyword"] = {
        row["backend"]: {
            "passed": row["passed"],
            "total": row["total"],
            "pass_rate": round(row["passed"] / row["total"], 4) if row.get("total") else None,
            "total_time_s": row.get("total_time_s"),
        }
        for row in summary
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n")
    print(f"Keyword baseline written to {BASELINE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gold Q&A eval against RAG backends")
    parser.add_argument(
        "--backend",
        choices=[*BACKENDS.keys(), "all"],
        default="all",
        help="Which pipeline to test (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print details for every question")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help=f"Write JSON results (default: {RESULTS_PATH.name})",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing results JSON",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log results to MLflow experiment media-rag-evals",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=f"Merge pass rates into {BASELINE_PATH.name}",
    )
    args = parser.parse_args()

    gold = json.loads(GOLD_PATH.read_text())
    print(f"Gold set: {len(gold)} questions from {GOLD_PATH.name}")

    backends = list(BACKENDS.keys()) if args.backend == "all" else [args.backend]
    summary: list[dict[str, Any]] = []

    for name in backends:
        summary.append(run_backend(name, gold, verbose=args.verbose))

    if len(summary) > 1:
        print(f"\n{'=' * 60}")
        print("Summary")
        print("=" * 60)
        for row in summary:
            print(f"  {row['backend']:12} {row['passed']}/{row['total']}")

    payload: dict[str, Any] | None = None
    if not args.no_write:
        payload = write_results(summary, args.output)
        if args.write_baseline:
            write_keyword_baseline(summary, timestamp=payload["timestamp"])
    elif args.write_baseline:
        write_keyword_baseline(
            summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    if args.mlflow:
        _maybe_log_mlflow(
            summary,
            gold_path=str(GOLD_PATH),
            artifact_path=args.output if not args.no_write else RESULTS_PATH,
        )


if __name__ == "__main__":
    main()
