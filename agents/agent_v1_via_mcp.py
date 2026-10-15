#!/usr/bin/env python3
"""ReAct agent that loads tools via MCP (langchain-mcp-adapters).

Same guard pipeline as ``agent_v1.py``, but tools come from ``mcp_server/server.py``
over stdio (or SSE) instead of importing ``tools.py`` directly.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
LAB_ROOT = AGENTS_DIR.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(AGENTS_DIR))

from mcp_adapter_compat import apply_mcp2_compat

apply_mcp2_compat()

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from shared.llm import llm_provider
from workflow import (
    AgentResult,
    apply_auto_draft_brief,
    apply_auto_escalation,
    apply_clearance_grounding_guard,
    apply_grounding_guard,
    ensure_lookup_rights,
    extract_draft_brief,
    extract_final_text,
    extract_tool_summaries,
    format_assessment_footer,
    get_llm,
    human_approval_gate,
)

PROMPTS_DIR = LAB_ROOT / "guardrails" / "prompts"
MCP_SERVER = AGENTS_DIR / "mcp_server" / "server.py"
SERVER_NAME = "media-workflow"
CORE_TOOL_NAMES = frozenset({"search_docs", "lookup_rights", "draft_brief", "escalate_legal"})


def load_system_prompt() -> str:
    rag_path = PROMPTS_DIR / "system_rag.txt"
    rag_rules = rag_path.read_text(encoding="utf-8").strip() if rag_path.exists() else ""
    agent_rules = (
        "You are a production workflow assistant for an enterprise media company. "
        "Always call search_docs before answering factual questions. "
        "Use lookup_rights first for territory clearance questions, then search_docs for detail. "
        "If clearance is pending, unknown, or confidence is below 0.7, call escalate_legal. "
        "For summary or brief requests, call lookup_rights and search_docs first, then draft_brief "
        "with gathered context before your final answer. "
        "Your final answer MUST agree with tool results; never contradict search_docs or lookup_rights. "
        "Never answer from general knowledge or training data. "
        "If search_docs says information is not in the documents, repeat that refusal; do not invent an answer. "
        "Cite source document filenames when answering from the corpus."
    )
    if rag_rules:
        return f"{agent_rules}\n\n{rag_rules}"
    return agent_rules


def mcp_connection_config(*, sse_url: str | None = None) -> dict:
    """Stdio spawn config (default) or SSE URL for a running MCP server."""
    if sse_url:
        return {"url": sse_url, "transport": "sse"}
    python = LAB_ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)
    return {
        "command": str(python),
        "args": [str(MCP_SERVER.relative_to(LAB_ROOT))],
        "transport": "stdio",
        "cwd": str(LAB_ROOT),
    }


def build_mcp_client(*, sse_url: str | None = None) -> MultiServerMCPClient:
    return MultiServerMCPClient({SERVER_NAME: mcp_connection_config(sse_url=sse_url)})


async def _run_agent_async(
    user_message: str,
    *,
    max_turns: int = 8,
    sse_url: str | None = None,
) -> AgentResult:
    client = build_mcp_client(sse_url=sse_url)
    async with client.session(SERVER_NAME) as session:
        tools = await load_mcp_tools(session, server_name=SERVER_NAME)
        tools = [tool for tool in tools if tool.name in CORE_TOOL_NAMES]

        agent = create_react_agent(
            get_llm(),
            tools,
            prompt=load_system_prompt(),
        )
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"recursion_limit": max_turns},
        )

    messages = result.get("messages", [])
    lookup, search, escalation = extract_tool_summaries(messages)
    draft = extract_draft_brief(messages)
    return apply_auto_draft_brief(
        user_message,
        apply_auto_escalation(
            apply_clearance_grounding_guard(
                apply_grounding_guard(
                    user_message,
                    ensure_lookup_rights(
                        user_message,
                        AgentResult(
                            answer=extract_final_text(messages),
                            lookup_rights=lookup,
                            search_docs=search,
                            escalation=escalation,
                            draft_brief=draft,
                        ),
                    ),
                    messages,
                )
            )
        ),
    )


def run_agent(
    user_message: str,
    max_turns: int = 8,
    *,
    sse_url: str | None = None,
) -> AgentResult:
    return asyncio.run(
        _run_agent_async(user_message, max_turns=max_turns, sse_url=sse_url)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReAct agent with tools via MCP (langchain-mcp-adapters)"
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="User request (default: Harbor Line S01E04 EMEA clearance summary)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip human approval gate (for scripts/CI)",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--sse",
        help="Connect to a running MCP SSE server (e.g. http://127.0.0.1:8765/sse)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tool-derived clearance assessment before approval",
    )
    args = parser.parse_args()

    user_message = " ".join(args.prompt) if args.prompt else (
        "Prepare a clearance summary for Harbor Line S01E04 EMEA streaming launch."
    )

    print(f"Provider: {llm_provider()}")
    print(f"Tools: MCP ({'SSE ' + args.sse if args.sse else 'stdio → mcp_server/server.py'})")
    print(f"Task: {user_message}\n")

    result = run_agent(user_message, max_turns=args.max_turns, sse_url=args.sse)
    if not result.answer:
        print("Agent finished without a text response.")
        if result.lookup_rights:
            print("\n" + format_assessment_footer(result))
        return

    if args.verbose and result.lookup_rights:
        print("=== Tool-derived assessment ===")
        print(format_assessment_footer(result))
        print()

    output = result.full_output
    approved = human_approval_gate(output, auto_approve=args.auto_approve)
    if approved:
        print("\n=== APPROVED OUTPUT ===\n")
        print(approved)


if __name__ == "__main__":
    main()
