"""Load evaluation artifacts for BFF admin summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
RAG_EVALS_DIR = LAB_ROOT / "rag" / "evals"
AGENT_EVALS_DIR = LAB_ROOT / "agents" / "evals"

RESULTS_PATH = RAG_EVALS_DIR / "results.json"
RAGAS_PATH = RAG_EVALS_DIR / "ragas_results.json"
PROMPTFOO_PATH = RAG_EVALS_DIR / "promptfoo-results.json"
GOLD_QA_PATH = RAG_EVALS_DIR / "gold_qa.json"
AGENT_RESULTS_PATH = AGENT_EVALS_DIR / "agent_results.json"
AGENT_SCENARIOS_PATH = AGENT_EVALS_DIR / "agent_scenarios.json"


def _gold_question_count() -> int:
    if not GOLD_QA_PATH.exists():
        return 0
    data = json.loads(GOLD_QA_PATH.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    return int(data.get("questions") or 0)


def _agent_scenario_count() -> int:
    if not AGENT_SCENARIOS_PATH.exists():
        return 0
    data = json.loads(AGENT_SCENARIOS_PATH.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    return 0


def _load_keyword_eval() -> tuple[list[dict[str, Any]], str | None]:
    if not RESULTS_PATH.exists():
        return [], None
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in payload.get("backends") or []:
        passed = int(row.get("passed") or 0)
        total = int(row.get("total") or 0)
        note = None
        if total and passed < total:
            note = f"{total - passed} failing"
        rows.append(
            {
                "backend": row.get("backend") or "unknown",
                "passed": passed,
                "total": total,
                "note": note,
            }
        )
    return rows, payload.get("timestamp")


def _load_ragas_eval() -> tuple[list[dict[str, Any]], str | None, str | None]:
    if not RAGAS_PATH.exists():
        return [], None, None
    payload = json.loads(RAGAS_PATH.read_text(encoding="utf-8"))
    metric_names = payload.get("metric_names") or []
    rows: list[dict[str, Any]] = []
    for row in payload.get("backends") or []:
        metrics = row.get("metrics") or {}
        note_parts: list[str] = []
        for name in metric_names:
            val = metrics.get(name)
            if val is None:
                note_parts.append(f"{name}=n/a")
            else:
                note_parts.append(f"{name}={val}")
        rows.append(
            {
                "backend": row.get("backend") or "unknown",
                "passed": int(row.get("questions") or 0),
                "total": int(row.get("questions") or 0),
                "note": ", ".join(note_parts) if note_parts else None,
            }
        )
    judge = payload.get("judge_model")
    return rows, payload.get("timestamp"), judge


def _load_agent_eval() -> dict[str, Any] | None:
    if not AGENT_RESULTS_PATH.exists():
        return None
    payload = json.loads(AGENT_RESULTS_PATH.read_text(encoding="utf-8"))
    passed = int(payload.get("passed") or 0)
    total = int(payload.get("total") or 0)
    if total == 0:
        return None
    note = None
    if passed < total:
        note = f"{total - passed} failing"
    return {
        "backend": payload.get("backend") or "agent_v1",
        "passed": passed,
        "total": total,
        "note": note,
        "timestamp": payload.get("timestamp"),
    }


def _load_promptfoo_summary() -> dict[str, Any] | None:
    if not PROMPTFOO_PATH.exists():
        return None
    payload = json.loads(PROMPTFOO_PATH.read_text(encoding="utf-8"))
    stats = (payload.get("results") or {}).get("stats") or {}
    successes = int(stats.get("successes") or 0)
    failures = int(stats.get("failures") or 0)
    errors = int(stats.get("errors") or 0)
    total = successes + failures + errors
    if total == 0:
        return None
    return {
        "passed": successes,
        "total": total,
        "failures": failures,
        "errors": errors,
        "eval_id": payload.get("evalId"),
        "timestamp": (payload.get("results") or {}).get("timestamp"),
    }


def build_eval_summary() -> dict[str, Any]:
    keyword_eval, keyword_ts = _load_keyword_eval()
    ragas_eval, ragas_ts, ragas_judge = _load_ragas_eval()
    promptfoo = _load_promptfoo_summary()
    agent_eval = _load_agent_eval()
    return {
        "gold_questions": _gold_question_count(),
        "agent_scenarios": _agent_scenario_count(),
        "keyword_eval": keyword_eval,
        "keyword_eval_timestamp": keyword_ts,
        "ragas_eval": ragas_eval,
        "ragas_eval_timestamp": ragas_ts,
        "ragas_judge_model": ragas_judge,
        "agent_eval": agent_eval,
        "promptfoo": promptfoo,
        "sources": {
            "results_json": str(RESULTS_PATH.relative_to(LAB_ROOT)),
            "ragas_json": str(RAGAS_PATH.relative_to(LAB_ROOT)),
            "promptfoo_json": str(PROMPTFOO_PATH.relative_to(LAB_ROOT)),
            "gold_qa_json": str(GOLD_QA_PATH.relative_to(LAB_ROOT)),
            "agent_results_json": str(AGENT_RESULTS_PATH.relative_to(LAB_ROOT)),
        },
    }


def eval_summary_has_data(data: dict[str, Any]) -> bool:
    return bool(
        data.get("gold_questions")
        or data.get("keyword_eval")
        or data.get("ragas_eval")
        or data.get("agent_eval")
        or data.get("promptfoo")
    )
