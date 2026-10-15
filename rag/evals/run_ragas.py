#!/usr/bin/env python3
"""RAGAS evaluation over gold Q&A (faithfulness, relevancy, context precision)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.ragas_compat import get_ragas_embeddings, get_ragas_llm, patch_ragas_imports

patch_ragas_imports()

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
)
from ragas.run_config import RunConfig  # noqa: E402

EVALS_DIR = Path(__file__).parent
PLATFORM_DIR = EVALS_DIR.parent.parent / "platform"
GOLD_PATH = EVALS_DIR / "gold_qa.json"
RAGAS_RESULTS_PATH = EVALS_DIR / "ragas_results.json"
BASELINE_PATH = EVALS_DIR / "baseline.json"

METRIC_BUILDERS = {
    "faithfulness": lambda llm, _emb: Faithfulness(),
    "answer_relevancy": lambda llm, emb: AnswerRelevancy(),
    "context_precision": lambda llm, _emb: LLMContextPrecisionWithoutReference(),
}


def load_run_eval_module():
    path = EVALS_DIR / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_mod", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_contexts(result: dict[str, Any]) -> list[str]:
    chunks = result.get("chunks")
    if chunks:
        first = chunks[0]
        if hasattr(first, "page_content"):
            return [doc.page_content for doc in chunks if getattr(doc, "page_content", None)]
        if isinstance(first, dict) and "text" in first:
            return [c["text"] for c in chunks if c.get("text")]

    nodes = result.get("source_nodes")
    if nodes:
        return [node.get_content() for node in nodes if node.get_content()]

    return []


def build_rag_samples(gold: list[dict], runner) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for item in gold:
        t0 = time.perf_counter()
        result = runner(item["question"])
        elapsed = time.perf_counter() - t0
        contexts = extract_contexts(result)
        if not contexts and result.get("answer"):
            # Fallback: still score faithfulness with empty context if retrieval failed
            contexts = []

        samples.append(
            {
                "question": item["question"],
                "user_input": item["question"],
                "response": result.get("answer") or "",
                "retrieved_contexts": contexts,
                "reference": item.get("reference"),
                "sources": result.get("sources") or [],
                "elapsed_s": round(elapsed, 2),
                "notes": item.get("notes"),
            }
        )
    return samples


def build_metrics(names: list[str], llm, embeddings):
    metrics = []
    for name in names:
        if name not in METRIC_BUILDERS:
            raise ValueError(f"Unknown metric: {name}. Choose from: {list(METRIC_BUILDERS)}")
        metrics.append(METRIC_BUILDERS[name](llm, embeddings))
    return metrics


def safe_mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def result_to_rows(eval_result, samples: list[dict]) -> list[dict[str, Any]]:
    """Merge per-row RAGAS scores with sample metadata."""
    df = eval_result.to_pandas()
    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        row = {"question": sample["question"], "elapsed_s": sample["elapsed_s"]}
        for col in df.columns:
            if col in {"user_input", "response", "retrieved_contexts", "reference"}:
                continue
            val = df.iloc[i][col]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                row[col] = round(float(val), 4)
            else:
                row[col] = None
        rows.append(row)
    return rows


def aggregate_metric_means(rows: list[dict], metric_names: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in metric_names:
        key = name if name != "context_precision" else "llm_context_precision_without_reference"
        values = [r.get(key) for r in rows]
        mean = safe_mean(values)
        out[name] = round(mean, 4) if mean is not None else None
    return out


def run_backend(
    name: str,
    gold: list[dict],
    llm,
    embeddings,
    metric_names: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"RAGAS backend: {name}")
    print("=" * 60)

    run_eval = load_run_eval_module()
    runner = run_eval.BACKENDS[name]()
    samples = build_rag_samples(gold, runner)

    dataset = Dataset.from_dict(
        {
            "user_input": [s["user_input"] for s in samples],
            "response": [s["response"] for s in samples],
            "retrieved_contexts": [s["retrieved_contexts"] for s in samples],
        }
    )

    metrics = build_metrics(metric_names, llm, embeddings)
    print(f"Running RAGAS on {len(samples)} questions, metrics: {', '.join(metric_names)}")
    print("(Local llama3.2 judge may return NaN on some rows; use --limit to smoke-test.)")

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        run_config = RunConfig(timeout=300, max_workers=1, max_retries=3)
        eval_result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
    total_time = time.perf_counter() - t0

    rows = result_to_rows(eval_result, samples)
    means = aggregate_metric_means(rows, metric_names)

    print(f"\nMean scores ({total_time:.1f}s total):")
    for metric, score in means.items():
        label = f"{score:.4f}" if score is not None else "n/a"
        print(f"  {metric:20} {label}")

    if verbose:
        print("\nPer-question:")
        for row in rows:
            scores = {k: v for k, v in row.items() if k not in {"question", "elapsed_s"}}
            print(f"  • {row['question'][:65]}")
            print(f"    {scores}")

    return {
        "backend": name,
        "metrics": means,
        "total_time_s": round(total_time, 2),
        "questions": len(samples),
        "results": rows,
    }


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {path}")


def write_ragas_baseline(
    summary: list[dict[str, Any]],
    *,
    timestamp: str,
    judge_model: str,
    embed_model: str,
) -> None:
    """Merge RAGAS means into baseline.json (preserves keyword section if present)."""
    baseline: dict[str, Any] = {}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline["ragas_timestamp"] = timestamp
    baseline["judge_model"] = judge_model
    baseline["embedding_model"] = embed_model
    baseline["ragas"] = {row["backend"]: row["metrics"] for row in summary}
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n")
    print(f"RAGAS baseline written to {BASELINE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS eval on gold Q&A")
    parser.add_argument(
        "--backend",
        choices=["chroma", "langchain", "llamaindex", "all"],
        default="chroma",
        help="RAG pipeline to evaluate (default: chroma, fastest smoke test)",
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness",
        help="Comma-separated: faithfulness, answer_relevancy, context_precision "
        "(default: faithfulness, most reliable with local Ollama judge)",
    )
    parser.add_argument(
        "--judge",
        choices=["ollama", "openai", "auto"],
        default="auto",
        help="RAGAS judge LLM (default: auto, openai if key set, else ollama)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N gold questions only")
    parser.add_argument("--verbose", action="store_true", help="Print per-question scores")
    parser.add_argument(
        "--output",
        type=Path,
        default=RAGAS_RESULTS_PATH,
        help=f"JSON output path (default: {RAGAS_RESULTS_PATH.name})",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=f"Also write aggregate scores to {BASELINE_PATH.name}",
    )
    parser.add_argument("--no-write", action="store_true", help="Skip writing JSON output")
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log results to MLflow experiment media-rag-evals",
    )
    args = parser.parse_args()

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    gold = json.loads(GOLD_PATH.read_text())
    if args.limit > 0:
        gold = gold[: args.limit]

    from shared.config import get_env
    from shared.ragas_compat import resolve_judge_provider

    judge_arg = None if args.judge == "auto" else args.judge
    judge_provider = resolve_judge_provider(judge_arg)

    llm, judge_model = get_ragas_llm(judge_arg)
    embeddings, embed_model = get_ragas_embeddings(judge_arg)

    print(f"Gold set: {len(gold)} questions from {GOLD_PATH.name}")
    print(f"Judge: {judge_model} | Embeddings: {embed_model}")
    if judge_provider == "ollama" and len(metric_names) > 1:
        print(
            "Note: local Ollama judge often fails JSON parsing for multi-metric runs. "
            "Use --metrics faithfulness or --judge openai for full suite."
        )

    backends = ["chroma", "langchain", "llamaindex"] if args.backend == "all" else [args.backend]
    summary: list[dict[str, Any]] = []

    for name in backends:
        summary.append(run_backend(name, gold, llm, embeddings, metric_names, verbose=args.verbose))

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(GOLD_PATH),
        "judge_model": judge_model,
        "embedding_model": embed_model,
        "judge_provider": judge_provider,
        "metric_names": metric_names,
        "backends": summary,
    }

    if not args.no_write:
        write_json(payload, args.output)
        if args.write_baseline:
            write_ragas_baseline(
                summary,
                timestamp=payload["timestamp"],
                judge_model=judge_model,
                embed_model=embed_model,
            )
    elif args.write_baseline:
        write_ragas_baseline(
            summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
            judge_model=judge_model,
            embed_model=embed_model,
        )

    if args.mlflow:
        sys.path.insert(0, str(PLATFORM_DIR))
        from ops.mlflow_eval import log_ragas_backend, mlflow_available, tracking_uri

        if not mlflow_available():
            print("MLflow not installed (pip install mlflow)")
        else:
            artifact = args.output if not args.no_write else RAGAS_RESULTS_PATH
            for row in summary:
                run_id = log_ragas_backend(row, payload=payload, artifact_path=artifact)
                if run_id:
                    print(f"MLflow run {row['backend']}: {run_id}")
            print(f"MLflow tracking URI: {tracking_uri()}")

    if len(summary) > 1:
        print(f"\n{'=' * 60}")
        print("Summary (mean RAGAS scores)")
        print("=" * 60)
        for row in summary:
            parts = "  ".join(f"{k}={v}" for k, v in row["metrics"].items())
            print(f"  {row['backend']:12} {parts}")


if __name__ == "__main__":
    main()
