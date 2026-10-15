#!/usr/bin/env python3
"""Single agent with LangChain + Ollama tool calling (LangGraph ReAct)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from shared.llm import llm_provider
from tools import get_tools
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

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "guardrails" / "prompts"

# Re-export for evals/run_agent_eval.py (load_agent_module)
__all__ = [
    "AgentResult",
    "answer_contradicts_lookup",
    "apply_auto_draft_brief",
    "apply_auto_escalation",
    "apply_clearance_grounding_guard",
    "apply_grounding_guard",
    "claims_cleared",
    "claims_pending",
    "ensure_lookup_rights",
    "format_assessment_footer",
    "human_approval_gate",
    "is_refusal",
    "run_agent",
]

from workflow import (  # noqa: E402 - re-exports for eval harness
    answer_contradicts_lookup,
    claims_cleared,
    claims_pending,
    is_refusal,
)


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


def run_agent(user_message: str, max_turns: int = 8, *, run_config: dict | None = None) -> AgentResult:
    llm = get_llm()
    agent = create_react_agent(
        llm,
        get_tools(),
        prompt=load_system_prompt(),
    )
    config: dict = {"recursion_limit": max_turns}
    if run_config:
        for key in ("run_name", "metadata", "tags", "callbacks", "recursion_limit"):
            if key in run_config:
                config[key] = run_config[key]
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    messages = result.get("messages", [])
    lookup, search, escalation = extract_tool_summaries(messages)
    draft = extract_draft_brief(messages)
    agent_result = apply_auto_draft_brief(
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
    return agent_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Production research agent")
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
        "--verbose",
        action="store_true",
        help="Print tool-derived clearance assessment before approval",
    )
    args = parser.parse_args()

    user_message = " ".join(args.prompt) if args.prompt else (
        "Prepare a clearance summary for Harbor Line S01E04 EMEA streaming launch."
    )

    print(f"Provider: {llm_provider()}")
    print(f"Task: {user_message}\n")

    result = run_agent(user_message, max_turns=args.max_turns)
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
