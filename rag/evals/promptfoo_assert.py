#!/usr/bin/env python3
"""promptfoo Python assert — mirrors evals/run_eval.py keyword + source checks."""

from __future__ import annotations

import json


def _parse_jsonish(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return [str(v) for v in json.loads(text)]
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(value)]


def check(output: str, context) -> bool | float | dict:
    """
    Validate RAG JSON output against test vars from gold_qa.json.

    Returns True/False, or a dict with pass/score/reason for promptfoo UI.
    """
    vars_ = (context or {}).get("vars") or {}
    try:
        data = json.loads(output) if output else {}
    except json.JSONDecodeError:
        data = {"answer": output or ""}

    answer_text = (data.get("answer") or "").lower()
    sources = data.get("sources") or []
    reasons: list[str] = []

    for keyword in _parse_jsonish(vars_.get("expected_contains")):
        if keyword.lower() not in answer_text:
            reasons.append(f"missing required keyword: {keyword}")

    any_keywords = _parse_jsonish(vars_.get("expected_contains_any"))
    if any_keywords and not any(k.lower() in answer_text for k in any_keywords):
        reasons.append(f"need at least one of: {any_keywords}")

    forbidden = _parse_jsonish(vars_.get("expected_contains_not_any"))
    found = [k for k in forbidden if k.lower() in answer_text]
    if found:
        reasons.append(f"must not contain: {found}")

    expected_sources = _parse_jsonish(vars_.get("expected_sources"))
    if expected_sources and not any(src in sources for src in expected_sources):
        reasons.append(f"expected source in {expected_sources}, got {sources}")

    passed = not reasons
    if passed:
        return True

    return {
        "pass": False,
        "score": 0,
        "reason": "; ".join(reasons),
    }
