#!/usr/bin/env python3
"""LangGraph multi-agent production workflow.

Researcher → Compliance → Supervisor → Producer (optional) → Critic → Finalize → Human approval
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt

from role_agents import run_compliance, run_critic, run_producer, run_researcher
from shared.llm import llm_provider
from workflow import (
    AgentResult,
    apply_auto_draft_brief,
    apply_auto_escalation,
    apply_clearance_grounding_guard,
    apply_grounding_guard,
    format_assessment_footer,
    format_clearance_answer,
    format_escalation_block,
    human_approval_gate,
    is_clearance_question,
    is_summary_flow,
)

CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "workflow.db"

CheckpointMode = Literal["memory", "sqlite", "none"]
Route = Literal["researcher", "compliance", "producer", "critic"]
CriticRoute = Literal["producer", "finalize"]


class WorkflowState(TypedDict, total=False):
    user_message: str
    auto_approve: bool
    use_react: bool
    use_llm: bool
    search_docs: dict | None
    web_search: dict | None
    lookup_rights: dict | None
    ip_flags: dict | None
    compliance_notes: str | None
    escalation: dict | None
    draft_brief: str | None
    answer: str
    full_output: str
    compliance_loops: int
    next_route: Route
    grounding_override: bool
    clearance_grounding_override: bool
    auto_escalated: bool
    auto_drafted: bool
    approved: bool | None
    _has_checkpointer: bool
    critic_review: dict | None
    critic_loops: int


def format_ip_flags_footer(ip_flags: dict | None) -> str:
    if not ip_flags or not ip_flags.get("flags"):
        return ""
    lines = ["---", "**IP / sponsorship flags (Compliance)**"]
    if ip_flags.get("needs_review"):
        lines.append("- **Action:** review flagged sponsorship / placement docs before launch")
    for row in ip_flags["flags"][:3]:
        lines.append(
            f"- **{row.get('source')}** [{row.get('severity')}]: "
            f"{(row.get('snippet') or '')[:120]}"
        )
    return "\n".join(lines)


def format_compliance_notes_footer(notes: str | None) -> str:
    if not notes or not notes.strip():
        return ""
    return f"---\n**Compliance review (LLM)**\n{notes.strip()}"


def format_critic_footer(review: dict | None) -> str:
    if not review:
        return ""
    lines = ["---", "**Critic review**"]
    if review.get("approved"):
        lines.append("- **Verdict:** approved — draft aligns with tool results")
    else:
        lines.append("- **Verdict:** issues noted (Finalize guards may still apply)")
    for issue in review.get("issues") or []:
        lines.append(f"- {issue}")
    for concern in review.get("llm_concerns") or []:
        lines.append(f"- **LLM concern:** {concern}")
    if review.get("llm_notes"):
        lines.append(f"\n{review['llm_notes'].strip()}")
    if review.get("needs_revision"):
        lines.append("- **Action:** sent back to Producer for one revision pass")
    return "\n".join(lines)


def researcher_node(state: WorkflowState) -> dict:
    """Researcher — ReAct sub-agent over search_docs."""
    use_react = state.get("use_react", True)
    return run_researcher(state["user_message"], use_react=use_react)


def compliance_node(state: WorkflowState) -> dict:
    """Compliance — ReAct sub-agent over lookup_rights + escalate_legal."""
    use_react = state.get("use_react", True)
    loops = state.get("compliance_loops", 0) + 1
    updates = run_compliance(
        state["user_message"],
        search_docs_result=state.get("search_docs"),
        use_react=use_react,
    )
    updates["compliance_loops"] = loops
    return updates


def supervisor_route(state: WorkflowState) -> Route:
    """Supervisor — route to compliance loop, producer, or finalize."""
    message = state["user_message"]
    lookup = state.get("lookup_rights")
    loops = state.get("compliance_loops", 0)

    if is_clearance_question(message) and not lookup and loops < 2:
        return "compliance"

    if is_clearance_question(message) and lookup and lookup.get("needs_escalation"):
        if not state.get("escalation") and loops < 3:
            return "compliance"

    if is_summary_flow(message) and not state.get("draft_brief"):
        return "producer"

    return "critic"


def _build_producer_context(state: WorkflowState) -> str:
    message = state["user_message"]
    context_parts = [f"User request: {message}"]

    lookup = state.get("lookup_rights")
    if lookup:
        context_parts.append(
            f"Clearance: {lookup.get('status')} "
            f"(confidence {lookup.get('confidence')}, "
            f"territory {lookup.get('territory_normalized') or lookup.get('territory')})"
        )
        for row in lookup.get("clearance_records", [])[:4]:
            context_parts.append(
                f"- {row.get('source')} [{row.get('rights_tier')}]: "
                f"{(row.get('snippet') or '')[:220]}"
            )

    search = state.get("search_docs")
    if search:
        context_parts.append(f"RAG answer: {search.get('answer', '')[:1200]}")
        sources = search.get("sources") or []
        if sources:
            context_parts.append(f"RAG sources: {', '.join(sources[:6])}")

    if state.get("escalation"):
        context_parts.append(f"Legal escalation: {state['escalation'].get('reason')}")
    if state.get("compliance_notes"):
        context_parts.append(f"Compliance review: {state['compliance_notes']}")
    if state.get("ip_flags", {}).get("flags"):
        context_parts.append(f"IP flags: {state['ip_flags']}")

    review = state.get("critic_review") or {}
    issues = review.get("issues") or []
    if issues:
        context_parts.append(
            "Critic feedback (address in revision):\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )

    return "\n\n".join(context_parts)


def producer_node(state: WorkflowState) -> dict:
    """Producer — ReAct sub-agent over draft_brief."""
    use_react = state.get("use_react", True)
    review = state.get("critic_review") or {}
    if review.get("needs_revision"):
        use_react = False
    context = _build_producer_context(state)
    return run_producer(state["user_message"], context, use_react=use_react)


def critic_node(state: WorkflowState) -> dict:
    """Critic — review draft against lookup_rights and search_docs before finalize."""
    draft = state.get("draft_brief") or _draft_answer(state)
    return run_critic(
        state["user_message"],
        draft,
        lookup=state.get("lookup_rights"),
        search=state.get("search_docs"),
        critic_loops=state.get("critic_loops", 0),
        use_llm=state.get("use_llm", True),
    )


def critic_route(state: WorkflowState) -> CriticRoute:
    """Route revision loop to Producer or proceed to Finalize."""
    review = state.get("critic_review") or {}
    if review.get("needs_revision"):
        return "producer"
    return "finalize"


def _draft_answer(state: WorkflowState) -> str:
    if state.get("draft_brief"):
        return state["draft_brief"]
    lookup = state.get("lookup_rights")
    if lookup and is_clearance_question(state["user_message"]):
        return format_clearance_answer(lookup)
    search = state.get("search_docs")
    if search and search.get("answer"):
        return str(search["answer"]).strip()
    return ""


def finalize_node(state: WorkflowState) -> dict:
    """Apply grounding guards and build assessment footer."""
    message = state["user_message"]
    result = AgentResult(
        answer=_draft_answer(state),
        lookup_rights=state.get("lookup_rights"),
        search_docs=state.get("search_docs"),
        escalation=state.get("escalation"),
        draft_brief=state.get("draft_brief"),
        auto_escalated=bool(state.get("auto_escalated")),
        auto_drafted=bool(state.get("auto_drafted")),
    )

    result = apply_grounding_guard(message, result, messages=[])
    result = apply_clearance_grounding_guard(result)
    result = apply_auto_escalation(result)
    result = apply_auto_draft_brief(message, result)

    if result.escalation and result.auto_escalated:
        block = format_escalation_block(result.escalation, auto=True)
        if block not in result.answer:
            result.answer = f"{result.answer.rstrip()}\n\n{block}"

    full_output = result.full_output
    for extra in (
        format_ip_flags_footer(state.get("ip_flags")),
        format_compliance_notes_footer(state.get("compliance_notes")),
        format_critic_footer(state.get("critic_review")),
    ):
        if extra and extra not in full_output:
            full_output = f"{full_output.rstrip()}\n\n{extra}"

    return {
        "answer": result.answer,
        "full_output": full_output,
        "lookup_rights": result.lookup_rights,
        "search_docs": result.search_docs,
        "escalation": result.escalation,
        "draft_brief": result.draft_brief,
        "grounding_override": result.grounding_override,
        "clearance_grounding_override": result.clearance_grounding_override,
        "auto_escalated": result.auto_escalated,
        "auto_drafted": result.auto_drafted,
    }


def human_approval_node(state: WorkflowState) -> dict:
    """Human-in-the-loop gate — interrupt when checkpointed, else pass-through."""
    output = state.get("full_output") or state.get("answer") or ""
    if state.get("auto_approve"):
        return {"approved": True, "full_output": output}

    # Without a checkpointer, LangGraph cannot interrupt — CLI gate runs in main().
    if not state.get("_has_checkpointer"):
        return {"approved": None, "full_output": output}

    decision = interrupt(
        {
            "type": "human_approval",
            "proposed_output": output,
            "prompt": "Approve and send? Respond with 'y' to approve.",
        }
    )
    approved = str(decision).strip().lower() in {"y", "yes", "approve", "approved"}
    return {"approved": approved, "full_output": output if approved else output}


def build_graph(*, checkpointer=None, pause_before_approval: bool = False):
    graph = StateGraph(WorkflowState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("producer", producer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("human_approval", human_approval_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "compliance")
    graph.add_conditional_edges(
        "compliance",
        supervisor_route,
        {
            "compliance": "compliance",
            "producer": "producer",
            "critic": "critic",
        },
    )
    graph.add_edge("producer", "critic")
    graph.add_conditional_edges(
        "critic",
        critic_route,
        {
            "producer": "producer",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", "human_approval")
    graph.add_edge("human_approval", END)

    interrupt_before = ["human_approval"] if pause_before_approval else None
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def get_sqlite_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    return SqliteSaver(conn)


def resolve_checkpoint_mode(
    *,
    explicit: CheckpointMode | None,
    auto_approve: bool,
    thread_id: str | None,
) -> CheckpointMode:
    if explicit:
        return explicit
    if thread_id:
        return "sqlite"
    return "memory" if auto_approve else "sqlite"


def get_checkpointer(mode: CheckpointMode):
    if mode == "none":
        return None
    if mode == "sqlite":
        return get_sqlite_checkpointer()
    return InMemorySaver()


def state_to_agent_result(state: WorkflowState) -> AgentResult:
    return AgentResult(
        answer=state.get("answer") or "",
        lookup_rights=state.get("lookup_rights"),
        escalation=state.get("escalation"),
        search_docs=state.get("search_docs"),
        draft_brief=state.get("draft_brief"),
        grounding_override=bool(state.get("grounding_override")),
        clearance_grounding_override=bool(state.get("clearance_grounding_override")),
        auto_escalated=bool(state.get("auto_escalated")),
        auto_drafted=bool(state.get("auto_drafted")),
    )


def _graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def build_app(*, checkpoint: CheckpointMode = "sqlite", pause_before_approval: bool = False):
    """Compiled graph + checkpointer for a given backend."""
    checkpointer = get_checkpointer(checkpoint)
    app = build_graph(
        checkpointer=checkpointer,
        pause_before_approval=pause_before_approval,
    )
    return app, checkpointer


def get_thread_snapshot(thread_id: str, *, checkpoint: CheckpointMode = "sqlite"):
    """Load persisted graph state for a thread (None if unknown)."""
    app, _ = build_app(checkpoint=checkpoint)
    return app.get_state(_graph_config(thread_id))


def start_graph_until_interrupt(
    user_message: str,
    *,
    thread_id: str | None = None,
    checkpoint: CheckpointMode = "sqlite",
    use_react: bool = True,
) -> tuple[str, dict]:
    """Run workflow until human-approval interrupt; returns (thread_id, interrupt_payload)."""
    if checkpoint == "none":
        raise ValueError("Checkpoint resume requires memory or sqlite backend")

    app, checkpointer = build_app(
        checkpoint=checkpoint,
        pause_before_approval=True,
    )
    tid = thread_id or str(uuid.uuid4())
    config = _graph_config(tid)

    initial: WorkflowState = {
        "user_message": user_message,
        "auto_approve": False,
        "use_react": use_react,
        "use_llm": True,
        "compliance_loops": 0,
        "_has_checkpointer": checkpointer is not None,
    }

    app.invoke(initial, config=config)
    snapshot = app.get_state(config)
    if not snapshot or not snapshot.next:
        values = snapshot.values if snapshot else {}
        raise RuntimeError(
            "Graph finished without interrupt — use --auto-approve or check workflow state"
        )

    values = snapshot.values or {}
    output = values.get("full_output") or values.get("answer") or ""
    payload = {
        "type": "human_approval",
        "proposed_output": output,
        "prompt": "Approve and send? Respond with 'y' to approve.",
        "next_nodes": list(snapshot.next),
    }
    return tid, payload


def resume_graph_thread(
    thread_id: str,
    decision: str = "y",
    *,
    checkpoint: CheckpointMode = "sqlite",
) -> AgentResult:
    """Resume a paused thread at the human-approval interrupt."""
    app, _ = build_app(checkpoint=checkpoint, pause_before_approval=True)
    config = _graph_config(thread_id)
    snapshot = app.get_state(config)
    if not snapshot or not snapshot.next:
        if snapshot and snapshot.values:
            return state_to_agent_result(snapshot.values)
        raise RuntimeError(f"No checkpoint state for thread {thread_id!r}")

    app.invoke(Command(resume=decision), config=config)
    snapshot = app.get_state(config)
    if not snapshot or not snapshot.values:
        raise RuntimeError(f"Resume failed for thread {thread_id!r}")
    return state_to_agent_result(snapshot.values)


def run_graph(
    user_message: str,
    *,
    auto_approve: bool = True,
    thread_id: str | None = None,
    checkpoint: CheckpointMode | None = None,
    use_react: bool = True,
    use_llm: bool = True,
    resume_decision: str | None = None,
) -> AgentResult:
    """Execute the multi-agent workflow; returns AgentResult (eval-compatible)."""
    if thread_id and resume_decision is not None:
        mode = resolve_checkpoint_mode(
            explicit=checkpoint,
            auto_approve=False,
            thread_id=thread_id,
        )
        return resume_graph_thread(thread_id, resume_decision, checkpoint=mode)

    mode = resolve_checkpoint_mode(
        explicit=checkpoint,
        auto_approve=auto_approve,
        thread_id=thread_id,
    )
    checkpointer = get_checkpointer(mode)
    pause = not auto_approve and checkpointer is not None
    app = build_graph(checkpointer=checkpointer, pause_before_approval=pause)
    tid = thread_id or str(uuid.uuid4())
    config = _graph_config(tid) if checkpointer else {}

    if thread_id and checkpointer:
        snapshot = app.get_state(config)
        if snapshot and snapshot.next:
            raise RuntimeError(
                f"Thread {thread_id!r} is paused at human approval. "
                f"Resume with: graph.py --thread-id {thread_id} --resume y"
            )

    initial: WorkflowState = {
        "user_message": user_message,
        "auto_approve": auto_approve,
        "use_react": use_react,
        "use_llm": use_llm,
        "compliance_loops": 0,
        "_has_checkpointer": checkpointer is not None,
    }

    final_state = app.invoke(initial, config=config)

    if not auto_approve and checkpointer:
        snapshot = app.get_state(config)
        if snapshot and snapshot.next:
            payload = {
                "proposed_output": (snapshot.values or {}).get("full_output")
                or (snapshot.values or {}).get("answer")
                or "",
            }
            approved_text = human_approval_gate(payload.get("proposed_output", ""), auto_approve=False)
            if approved_text:
                app.invoke(Command(resume="y"), config=config)
                snapshot = app.get_state(config)
                final_state = snapshot.values if snapshot else final_state
            else:
                final_state = {**(snapshot.values or {}), "approved": False}
        elif final_state.get("__interrupt__"):
            interrupts = final_state["__interrupt__"]
            payload = interrupts[0].value if interrupts else {}
            proposed = payload.get("proposed_output", "")
            approved_text = human_approval_gate(proposed, auto_approve=False)
            if approved_text:
                app.invoke(Command(resume="y"), config=config)
                snapshot = app.get_state(config)
                final_state = snapshot.values if snapshot else final_state
            else:
                final_state = {**final_state, "approved": False}

    return state_to_agent_result(final_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent production workflow")
    parser.add_argument(
        "prompt",
        nargs="*",
        help="User request (default: Harbor Line S01E04 EMEA clearance summary)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip human approval gate",
    )
    parser.add_argument(
        "--thread-id",
        help="Resume / inspect a persisted graph thread (implies --checkpoint sqlite)",
    )
    parser.add_argument(
        "--resume",
        metavar="DECISION",
        help="Resume a paused thread (--thread-id required), e.g. y or n",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["memory", "sqlite", "none"],
        help="Checkpoint backend (default: memory when --auto-approve, else sqlite)",
    )
    parser.add_argument(
        "--no-react",
        action="store_true",
        help="Use deterministic tool calls instead of ReAct sub-agents",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM critic/compliance notes (deterministic checks only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print clearance assessment before approval",
    )
    args = parser.parse_args()

    mode = resolve_checkpoint_mode(
        explicit=args.checkpoint,
        auto_approve=args.auto_approve,
        thread_id=args.thread_id,
    )

    if args.resume:
        if not args.thread_id:
            raise SystemExit("--resume requires --thread-id")
        print(f"Provider: {llm_provider()}")
        print(f"Resuming thread: {args.thread_id}")
        print(f"Checkpoint: {mode}\n")
        result = run_graph(
            "",
            auto_approve=False,
            thread_id=args.thread_id,
            checkpoint=args.checkpoint or "sqlite",
            resume_decision=args.resume,
        )
        print("\n=== APPROVED OUTPUT ===\n" if args.resume.lower() in {"y", "yes"} else "\n=== OUTPUT (withheld) ===\n")
        print(result.full_output)
        return

    user_message = " ".join(args.prompt) if args.prompt else (
        "Prepare a clearance summary for Harbor Line S01E04 EMEA streaming launch."
    )

    print(f"Provider: {llm_provider()}")
    print(f"Workflow: Researcher → Compliance → Supervisor → Producer → Critic → Approval")
    print(f"Checkpoint: {mode}" + (f" ({CHECKPOINT_PATH})" if mode == "sqlite" else ""))
    print(f"Sub-agents: {'deterministic' if args.no_react else 'ReAct'}")
    print(f"Critic LLM: {'off' if args.no_llm else 'on'}")
    print(f"Task: {user_message}\n")

    result = run_graph(
        user_message,
        auto_approve=args.auto_approve,
        thread_id=args.thread_id,
        checkpoint=args.checkpoint,
        use_react=not args.no_react,
        use_llm=not args.no_llm,
    )

    if args.verbose and result.lookup_rights:
        print("=== Tool-derived assessment ===")
        print(format_assessment_footer(result))
        print()

    output = result.full_output
    if args.auto_approve:
        print("\n=== APPROVED OUTPUT ===\n")
        print(output)
        return

    approved = human_approval_gate(output, auto_approve=False)
    if approved:
        print("\n=== APPROVED OUTPUT ===\n")
        print(approved)


if __name__ == "__main__":
    main()
