"""Agent workflow adapter for BFF /v1/agent/run."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS_DIR = LAB_ROOT / "agents"

sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(AGENTS_DIR))

from agent_v1 import run_agent  # noqa: E402
from workflow import AgentResult  # noqa: E402


def execute_agent(prompt: str, *, max_turns: int = 8) -> AgentResult:
    return run_agent(prompt, max_turns=max_turns)


def _sources_from_result(result: AgentResult) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []

    def add_file(name: str | None, **extra: Any) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        sources.append({"file_name": name, **extra})

    if result.search_docs:
        for name in result.search_docs.get("sources") or []:
            add_file(name)

    if result.lookup_rights:
        for row in result.lookup_rights.get("clearance_records") or []:
            add_file(row.get("source") or row.get("file_name"))

    return sources


def _tool_trace_from_result(result: AgentResult) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    if result.lookup_rights:
        trace.append({"tool": "lookup_rights", **result.lookup_rights})
    if result.search_docs:
        trace.append({"tool": "search_docs", **result.search_docs})
    if result.escalation:
        trace.append({"tool": "escalate_legal", **result.escalation})
    if result.draft_brief:
        trace.append({"tool": "draft_brief", "brief": result.draft_brief})
    flags = {
        "grounding_override": result.grounding_override,
        "clearance_grounding_override": result.clearance_grounding_override,
        "auto_escalated": result.auto_escalated,
        "auto_drafted": result.auto_drafted,
    }
    if any(flags.values()):
        trace.append({"tool": "workflow_flags", **flags})
    return trace


def serialize_agent_result(result: AgentResult) -> dict[str, Any]:
    return {
        "answer": result.full_output,
        "sources": _sources_from_result(result),
        "tool_trace": _tool_trace_from_result(result),
    }
