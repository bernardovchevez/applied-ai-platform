#!/usr/bin/env python3
"""Generate promptfoo test cases from evals/gold_qa.json."""

from __future__ import annotations

import json
from pathlib import Path

GOLD_PATH = Path(__file__).parent / "gold_qa.json"


def generate_tests() -> list[dict]:
    gold = json.loads(GOLD_PATH.read_text())
    tests: list[dict] = []

    for item in gold:
        vars_: dict = {
            "question": item["question"],
            "notes": item.get("notes", ""),
        }
        if item.get("expected_contains"):
            vars_["expected_contains"] = json.dumps(item["expected_contains"])
        if item.get("expected_contains_any"):
            vars_["expected_contains_any"] = json.dumps(item["expected_contains_any"])
        if item.get("expected_contains_not_any"):
            vars_["expected_contains_not_any"] = json.dumps(item["expected_contains_not_any"])
        if item.get("expected_sources"):
            vars_["expected_sources"] = json.dumps(item["expected_sources"])

        tests.append(
            {
                "description": item["question"][:80],
                "vars": vars_,
            }
        )

    return tests


if __name__ == "__main__":
    print(json.dumps(generate_tests(), indent=2))
