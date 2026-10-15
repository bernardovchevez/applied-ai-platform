"""MLflow experiment logging for platform eval runs.

Optional — pass --mlflow to eval scripts or set MLFLOW_TRACKING_URI in .env.
Default tracking store: platform/ops/mlflow_runs/
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from shared.config import get_env

logger = logging.getLogger(__name__)

OPS_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"sqlite:///{OPS_DIR / 'mlflow.db'}"
EXPERIMENT_RAG = "media-rag-evals"
EXPERIMENT_AGENT = "media-agent-evals"


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def tracking_uri() -> str:
    return get_env("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def _configure_mlflow() -> Any:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    return mlflow


@contextmanager
def eval_run(
    *,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Context manager wrapping mlflow.start_run. No-op if mlflow is not installed."""
    if not mlflow_available():
        logger.warning("mlflow not installed — skip logging (pip install mlflow)")
        yield None
        return

    mlflow = _configure_mlflow()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        if params:
            safe_params = {k: (v if v is not None else "null") for k, v in params.items()}
            mlflow.log_params(safe_params)
        yield run


def _log_metrics(metrics: dict[str, float | int | None]) -> None:
    import mlflow

    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, float(value))


def _log_artifact(path: Path | None) -> None:
    if not path or not path.is_file():
        return
    import mlflow

    mlflow.log_artifact(str(path))


def log_rag_keyword_backend(
    row: dict[str, Any],
    *,
    artifact_path: Path | None = None,
    gold_path: str | None = None,
) -> str | None:
    """Log one backend row from run_eval.py summary."""
    backend = row["backend"]
    run_name = f"rag-keyword-{backend}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    params = {
        "eval_type": "rag_keyword",
        "backend": backend,
        "gold_path": gold_path,
        "total_questions": row.get("total"),
    }
    metrics = {
        "passed": row.get("passed"),
        "total": row.get("total"),
        "pass_rate": (row["passed"] / row["total"]) if row.get("total") else None,
        "total_time_s": row.get("total_time_s"),
    }
    with eval_run(
        experiment_name=EXPERIMENT_RAG,
        run_name=run_name,
        params=params,
        tags={"eval_type": "rag_keyword", "backend": backend},
    ) as run:
        if run is None:
            return None
        _log_metrics(metrics)
        _log_artifact(artifact_path)
        return run.info.run_id


def log_ragas_backend(
    row: dict[str, Any],
    *,
    payload: dict[str, Any],
    artifact_path: Path | None = None,
) -> str | None:
    """Log one backend row from run_ragas.py summary."""
    backend = row["backend"]
    run_name = f"rag-ragas-{backend}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    params = {
        "eval_type": "rag_ragas",
        "backend": backend,
        "judge_model": payload.get("judge_model"),
        "embedding_model": payload.get("embedding_model"),
        "judge_provider": payload.get("judge_provider"),
        "metric_names": ",".join(payload.get("metric_names") or []),
        "gold_path": payload.get("gold_path"),
        "questions": row.get("questions"),
    }
    metrics: dict[str, float | int | None] = {
        "questions": row.get("questions"),
        "total_time_s": row.get("total_time_s"),
    }
    for name, score in (row.get("metrics") or {}).items():
        metrics[name] = score

    with eval_run(
        experiment_name=EXPERIMENT_RAG,
        run_name=run_name,
        params=params,
        tags={"eval_type": "rag_ragas", "backend": backend},
    ) as run:
        if run is None:
            return None
        _log_metrics(metrics)
        _log_artifact(artifact_path)
        return run.info.run_id


def log_agent_eval(
    payload: dict[str, Any],
    *,
    artifact_path: Path | None = None,
) -> str | None:
    """Log agent scenario harness results."""
    backend = payload.get("backend") or "agent_v1"
    run_name = f"agent-{backend}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    passed = payload.get("passed", 0)
    total = payload.get("total", 0)
    params = {
        "eval_type": "agent_scenarios",
        "backend": backend,
        "scenarios_path": payload.get("scenarios_path"),
        "total_scenarios": total,
    }
    metrics = {
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) if total else None,
    }
    with eval_run(
        experiment_name=EXPERIMENT_AGENT,
        run_name=run_name,
        params=params,
        tags={"eval_type": "agent_scenarios", "backend": backend},
    ) as run:
        if run is None:
            return None
        _log_metrics(metrics)
        _log_artifact(artifact_path)
        return run.info.run_id
