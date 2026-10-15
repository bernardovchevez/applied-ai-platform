#!/usr/bin/env python3
"""Print MLflow eval runs from the platform sqlite store (terminal fallback for MLflow 3 UI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "platform"))

from ops.mlflow_eval import EXPERIMENT_AGENT, EXPERIMENT_RAG, tracking_uri  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="List MLflow eval runs in the platform store")
    parser.add_argument(
        "--experiment",
        choices=["agent", "rag", "all"],
        default="all",
        help="Which experiment to show (default: all)",
    )
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError:
        raise SystemExit("mlflow not installed (pip install mlflow)") from None

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    print(f"Tracking URI: {uri}\n")

    targets: list[str] = []
    if args.experiment in ("agent", "all"):
        targets.append(EXPERIMENT_AGENT)
    if args.experiment in ("rag", "all"):
        targets.append(EXPERIMENT_RAG)

    found = False
    for name in targets:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            print(f"[{name}] experiment not found")
            continue
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
        print(f"=== {name} (id {exp.experiment_id}): {len(runs)} run(s) ===")
        if runs.empty:
            print("  (no runs)\n")
            continue
        found = True
        for _, row in runs.iterrows():
            run_name = row.get("tags.mlflow.runName") or row.get("run_id")
            metrics = []
            for col in row.index:
                if col.startswith("metrics."):
                    metrics.append(f"{col.removeprefix('metrics.')}={row[col]}")
            metric_str = ", ".join(metrics) if metrics else "no metrics"
            backend = row.get("params.backend", row.get("params.eval_type", ""))
            eval_type = row.get("params.eval_type", "")
            print(f"  • {run_name}")
            print(f"    {metric_str}")
            if eval_type:
                print(f"    eval_type={eval_type}", end="")
                if backend and backend != eval_type:
                    print(f", backend={backend}")
                else:
                    print()
            elif backend:
                print(f"    backend={backend}")
            print()
    if not found:
        print("No runs logged yet. Example:")
        print("  python agents/evals/run_agent_eval.py --mlflow")


if __name__ == "__main__":
    main()
