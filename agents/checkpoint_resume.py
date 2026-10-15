#!/usr/bin/env python3
"""Checkpoint resume CLI: pause at human approval, then resume later.

Simulates a producer closing the terminal at the approval gate and resuming
from SQLite the next day with the same thread-id.

Usage:
  # Step 1: run until interrupt (fast: --no-react inside)
  python agents/checkpoint_resume.py start

  # Step 2: resume (separate shell / later session)
  python agents/checkpoint_resume.py resume

  # Or one-shot walkthrough (start + resume)
  python agents/checkpoint_resume.py full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import (
    CHECKPOINT_PATH,
    get_thread_snapshot,
    resume_graph_thread,
    start_graph_until_interrupt,
)
from shared.llm import llm_provider

DEMO_PROMPT = (
    "Prepare a clearance summary for Harbor Line S01E04 EMEA streaming launch."
)
THREAD_FILE = Path(__file__).resolve().parent / ".checkpoints" / "demo_thread.json"


def _save_thread(thread_id: str, *, prompt: str) -> None:
    THREAD_FILE.parent.mkdir(parents=True, exist_ok=True)
    THREAD_FILE.write_text(
        json.dumps({"thread_id": thread_id, "prompt": prompt}, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_thread() -> dict:
    if not THREAD_FILE.exists():
        raise SystemExit(
            f"No saved thread at {THREAD_FILE}. Run: checkpoint_resume.py start"
        )
    return json.loads(THREAD_FILE.read_text(encoding="utf-8"))


def cmd_start(prompt: str, *, use_react: bool) -> str:
    print("=== Step 1: Start workflow (SQLite checkpoint) ===")
    print(f"Provider: {llm_provider()}")
    print(f"Checkpoint DB: {CHECKPOINT_PATH}")
    print(f"Prompt: {prompt}\n")
    print("Running Researcher → Compliance → Producer → Finalize → Human approval…\n")

    thread_id, payload = start_graph_until_interrupt(
        prompt,
        checkpoint="sqlite",
        use_react=use_react,
    )
    _save_thread(thread_id, prompt=prompt)

    proposed = payload.get("proposed_output", "")
    preview = proposed[:600] + ("…" if len(proposed) > 600 else "")

    print("⏸  PAUSED at human-approval interrupt")
    print(f"   Thread ID: {thread_id}")
    print(f"   Saved to:  {THREAD_FILE}\n")
    print("--- Proposed output (preview) ---")
    print(preview)
    print("--- end preview ---\n")

    snapshot = get_thread_snapshot(thread_id)
    if snapshot and snapshot.next:
        print(f"   Graph next node(s): {snapshot.next}")

    print("\nResume in a new session:")
    print("  python agents/checkpoint_resume.py resume")
    print("\nOr via graph.py directly:")
    print(f"  python agents/graph.py --thread-id {thread_id} --resume y")
    return thread_id


def cmd_resume(*, decision: str) -> None:
    saved = _load_thread()
    thread_id = saved["thread_id"]
    prompt = saved.get("prompt", DEMO_PROMPT)

    print("=== Step 2: Resume paused thread ===")
    print(f"Provider: {llm_provider()}")
    print(f"Checkpoint DB: {CHECKPOINT_PATH}")
    print(f"Thread ID: {thread_id}")
    print(f"Original prompt: {prompt}\n")

    snapshot = get_thread_snapshot(thread_id)
    if snapshot and snapshot.next:
        print(f"Loaded checkpoint, next node(s): {snapshot.next}\n")
    elif snapshot and snapshot.values:
        print("Thread already completed, showing final output:\n")
        from graph import state_to_agent_result

        print(state_to_agent_result(snapshot.values).full_output)
        return

    print(f"Resuming with decision={decision!r}…\n")
    result = resume_graph_thread(thread_id, decision, checkpoint="sqlite")

    label = "APPROVED OUTPUT" if decision.lower() in {"y", "yes", "approve"} else "OUTPUT (withheld)"
    print(f"=== {label} ===\n")
    print(result.full_output)


def cmd_full(prompt: str, *, use_react: bool, decision: str) -> None:
    thread_id = cmd_start(prompt, use_react=use_react)
    print("\n" + "=" * 60 + "\n")
    cmd_resume(decision=decision)
    print(f"\n✓ Demo complete: thread {thread_id} resumed from {CHECKPOINT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph SQLite checkpoint pause → resume demo"
    )
    parser.add_argument(
        "command",
        choices=["start", "resume", "full"],
        help="start=pause at interrupt; resume=continue saved thread; full=both steps",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Override demo prompt (start/full only)",
    )
    parser.add_argument(
        "--react",
        action="store_true",
        help="Use ReAct sub-agents (slower; default skips for demo speed)",
    )
    parser.add_argument(
        "--decision",
        default="y",
        help="Approval decision for resume/full (default: y)",
    )
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else DEMO_PROMPT
    use_react = args.react

    if args.command == "start":
        cmd_start(prompt, use_react=use_react)
    elif args.command == "resume":
        cmd_resume(decision=args.decision)
    else:
        cmd_full(prompt, use_react=use_react, decision=args.decision)


if __name__ == "__main__":
    main()
