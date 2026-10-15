"""Optional LangSmith tracing for LangGraph agent evals.

Enable with --langsmith on run_agent_eval.py or set LANGCHAIN_TRACING_V2=true in .env.
Requires LANGCHAIN_API_KEY (LangSmith API key).
"""

from __future__ import annotations

import os
from typing import Any

from shared.config import get_env

DEFAULT_PROJECT = "applied-ai-platform-agents"


def langsmith_available() -> bool:
    return bool(get_env("LANGCHAIN_API_KEY") or get_env("LANGSMITH_API_KEY"))


def langsmith_project() -> str:
    return (
        get_env("LANGCHAIN_PROJECT")
        or get_env("LANGSMITH_PROJECT")
        or DEFAULT_PROJECT
    )


def enable_langsmith_tracing(*, project: str | None = None) -> bool:
    """Turn on LangChain/LangGraph export for this process. Returns False if no API key."""
    api_key = get_env("LANGCHAIN_API_KEY") or get_env("LANGSMITH_API_KEY")
    if not api_key:
        return False
    project_name = project or langsmith_project()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGSMITH_PROJECT"] = project_name
    return True


def invoke_config_for_scenario(
    scenario_id: str,
    *,
    backend: str,
    max_turns: int = 8,
) -> dict[str, Any]:
    """RunnableConfig fields LangSmith picks up on agent.invoke()."""
    return {
        "recursion_limit": max_turns,
        "run_name": f"agent-eval-{scenario_id}",
        "metadata": {
            "scenario_id": scenario_id,
            "backend": backend,
            "eval_harness": "agents/evals/run_agent_eval.py",
        },
        "tags": ["agent-eval", backend, scenario_id],
    }


def langsmith_ui_hint() -> str:
    return (
        "Open https://smith.langchain.com → project "
        f"'{langsmith_project()}' → filter tags:agent-eval"
    )
