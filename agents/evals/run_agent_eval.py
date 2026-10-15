#!/usr/bin/env python3
"""Run agent evaluation scenarios from agent_scenarios.json."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS_DIR = Path(__file__).resolve().parent.parent
PLATFORM_DIR = LAB_ROOT / "platform"
SCENARIOS_PATH = Path(__file__).parent / "agent_scenarios.json"
RESULTS_PATH = Path(__file__).parent / "agent_results.json"

sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(AGENTS_DIR))


def load_agent_module():
    path = AGENTS_DIR / "agent_v1.py"
    spec = importlib.util.spec_from_file_location("agents_agent_v1", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agents_agent_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_graph_module():
    path = AGENTS_DIR / "graph.py"
    spec = importlib.util.spec_from_file_location("agents_graph", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agents_graph"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_mcp_agent_module():
    path = AGENTS_DIR / "agent_v1_via_mcp.py"
    spec = importlib.util.spec_from_file_location("agents_agent_v1_mcp", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agents_agent_v1_mcp"] = mod
    spec.loader.exec_module(mod)
    return mod


def evaluate_scenario(scenario: dict, result: Any) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    answer = (result.answer or "").lower()
    lookup = result.lookup_rights or {}
    sources_from_lookup = [
        row.get("source")
        for row in lookup.get("clearance_records", [])
        if row.get("source")
    ]
    search_sources = result.search_docs.get("sources") if result.search_docs else []
    all_sources = list(dict.fromkeys([*sources_from_lookup, *(search_sources or [])]))

    if expected_status := scenario.get("expected_status"):
        actual = lookup.get("status")
        if actual != expected_status:
            reasons.append(f"expected status {expected_status!r}, got {actual!r}")

    if "expected_confidence" in scenario:
        actual = lookup.get("confidence")
        if actual != scenario["expected_confidence"]:
            reasons.append(
                f"expected confidence {scenario['expected_confidence']}, got {actual}"
            )

    if "expected_confidence_max" in scenario:
        actual = lookup.get("confidence")
        if actual is not None and actual > scenario["expected_confidence_max"]:
            reasons.append(
                f"expected confidence <= {scenario['expected_confidence_max']}, got {actual}"
            )

    if "needs_escalation" in scenario:
        actual = lookup.get("needs_escalation")
        if lookup and actual != scenario["needs_escalation"]:
            reasons.append(f"expected needs_escalation={scenario['needs_escalation']}, got {actual}")
        elif not lookup and scenario["needs_escalation"] is False:
            pass
        elif not lookup and scenario.get("expected_refusal"):
            pass
        elif not lookup and scenario["needs_escalation"]:
            reasons.append("expected needs_escalation=True but lookup_rights was not called")

    for kw in scenario.get("expected_contains", []):
        if kw.lower() not in answer:
            reasons.append(f"missing required keyword: {kw!r}")

    any_kws = scenario.get("expected_contains_any") or []
    if any_kws and not any(k.lower() in answer for k in any_kws):
        reasons.append(f"need at least one of: {any_kws}")

    for kw in scenario.get("expected_contains_not_any", []):
        if kw.lower() in answer:
            reasons.append(f"must not contain: {kw!r}")

    expected_sources = scenario.get("expected_sources_any") or []
    if expected_sources and not any(s in all_sources for s in expected_sources):
        reasons.append(f"expected source in {expected_sources}, got {all_sources}")

    if scenario.get("expected_refusal"):
        if not load_agent_module().is_refusal(result.answer):
            reasons.append("expected refusal phrasing in answer")

    if scenario.get("expect_escalation_tool") and not result.escalation:
        reasons.append("expected escalate_legal tool call")

    if scenario.get("expect_draft_brief") and not result.draft_brief:
        reasons.append("expected draft_brief in summary flow")

    agent_mod = load_agent_module()
    if scenario.get("answer_must_align_with_lookup") and result.lookup_rights:
        if agent_mod.answer_contradicts_lookup(result.answer, result.lookup_rights):
            reasons.append(
                f"answer contradicts lookup status {result.lookup_rights.get('status')!r}"
            )

    if scenario.get("forbid_cleared_claim") and agent_mod.claims_cleared(result.answer):
        reasons.append("answer must not claim cleared")

    if scenario.get("forbid_pending_claim") and agent_mod.claims_pending(result.answer):
        reasons.append("answer must not claim pending")

    return (not reasons, reasons)


def run_eval(
    scenarios: list[dict],
    max_turns: int = 8,
    verbose: bool = False,
    use_graph: bool = False,
    use_mcp: bool = False,
    langsmith: bool = False,
) -> list[dict[str, Any]]:
    agent_mod = load_agent_module()
    graph_mod = load_graph_module() if use_graph else None
    mcp_mod = load_mcp_agent_module() if use_mcp else None
    summary: list[dict[str, Any]] = []
    passed = 0

    if use_graph:
        backend = "graph.py (LangGraph multi-agent)"
        backend_key = "graph"
    elif use_mcp:
        backend = "agent_v1_via_mcp.py (MCP tools)"
        backend_key = "mcp"
    else:
        backend = "agent_v1.py (ReAct single-agent)"
        backend_key = "agent_v1"
    print(f"Agent scenarios: {len(scenarios)} from {SCENARIOS_PATH.name}")
    print(f"Backend: {backend}")
    if langsmith:
        sys.path.insert(0, str(PLATFORM_DIR))
        from ops.langsmith_tracing import langsmith_ui_hint

        print(f"LangSmith: enabled → {langsmith_ui_hint()}\n")
    else:
        print()

    for scenario in scenarios:
        sid = scenario.get("id", scenario["prompt"][:40])
        print("=" * 60)
        print(f"Scenario: {sid}")
        print(scenario["prompt"])
        t0 = time.perf_counter()
        invoke_config = None
        if langsmith and backend_key == "agent_v1":
            from ops.langsmith_tracing import invoke_config_for_scenario

            invoke_config = invoke_config_for_scenario(sid, backend=backend_key, max_turns=max_turns)
        if use_graph:
            result = graph_mod.run_graph(scenario["prompt"], auto_approve=True)
        elif use_mcp:
            result = mcp_mod.run_agent(scenario["prompt"], max_turns=max_turns)
        else:
            result = agent_mod.run_agent(
                scenario["prompt"],
                max_turns=max_turns,
                run_config=invoke_config,
            )
        elapsed = time.perf_counter() - t0
        ok, reasons = evaluate_scenario(scenario, result)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] ({elapsed:.1f}s)")

        entry: dict[str, Any] = {
            "id": sid,
            "prompt": scenario["prompt"],
            "pass": ok,
            "elapsed_s": round(elapsed, 2),
            "answer": result.answer,
            "lookup_status": (result.lookup_rights or {}).get("status"),
            "lookup_confidence": (result.lookup_rights or {}).get("confidence"),
            "needs_escalation": (result.lookup_rights or {}).get("needs_escalation"),
            "grounding_override": result.grounding_override,
            "clearance_grounding_override": result.clearance_grounding_override,
            "escalation_called": bool(result.escalation),
            "auto_escalated": result.auto_escalated,
            "draft_brief_called": bool(result.draft_brief),
            "auto_drafted": result.auto_drafted,
        }
        if not ok:
            entry["reasons"] = reasons
        summary.append(entry)

        if verbose or not ok:
            if not ok:
                print(f"  reasons: {'; '.join(reasons)}")
            lookup = result.lookup_rights or {}
            if lookup:
                print(
                    f"  lookup: status={lookup.get('status')} "
                    f"confidence={lookup.get('confidence')} "
                    f"escalate={lookup.get('needs_escalation')}"
                )
            print(f"  answer: {(result.answer or '')[:200]}...")

    print("\n" + "=" * 60)
    print(f"Score: {passed}/{len(scenarios)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent scenarios")
    parser.add_argument("--limit", type=int, help="Run first N scenarios only")
    parser.add_argument("--id", help="Run a single scenario by id")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Run scenarios through graph.py (multi-agent) instead of agent_v1",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run scenarios through agent_v1_via_mcp.py (MCP tools)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help=f"Write JSON results (default: {RESULTS_PATH.name})",
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log results to MLflow experiment media-agent-evals",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Export LangGraph ReAct traces to LangSmith (requires LANGCHAIN_API_KEY)",
    )
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="LangSmith project name (default: LANGCHAIN_PROJECT or applied-ai-platform-agents)",
    )
    args = parser.parse_args()

    if args.graph and args.mcp:
        raise SystemExit("Use only one of --graph or --mcp")

    langsmith_on = False
    if args.langsmith:
        sys.path.insert(0, str(PLATFORM_DIR))
        from ops.langsmith_tracing import enable_langsmith_tracing

        if not enable_langsmith_tracing(project=args.langsmith_project):
            raise SystemExit(
                "LangSmith requires LANGCHAIN_API_KEY (or LANGSMITH_API_KEY) in .env; "
                "get one at https://smith.langchain.com/settings"
            )
        langsmith_on = True
        if args.graph or args.mcp:
            print(
                "Note: --langsmith with --graph/--mcp enables project tracing; "
                "per-scenario run_name tags apply to agent_v1 only.\n"
            )

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if args.id:
        scenarios = [s for s in scenarios if s.get("id") == args.id]
        if not scenarios:
            raise SystemExit(f"No scenario with id={args.id!r}")
    elif args.limit:
        scenarios = scenarios[: args.limit]

    summary = run_eval(
        scenarios,
        max_turns=args.max_turns,
        verbose=args.verbose,
        use_graph=args.graph,
        use_mcp=args.mcp,
        langsmith=langsmith_on,
    )

    if not args.no_write:
        if args.graph:
            backend_key = "graph"
        elif args.mcp:
            backend_key = "mcp"
        else:
            backend_key = "agent_v1"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenarios_path": str(SCENARIOS_PATH),
            "backend": backend_key,
            "passed": sum(1 for r in summary if r["pass"]),
            "total": len(summary),
            "results": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\nResults written to {args.output}")
    else:
        backend_key = "graph" if args.graph else ("mcp" if args.mcp else "agent_v1")
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenarios_path": str(SCENARIOS_PATH),
            "backend": backend_key,
            "passed": sum(1 for r in summary if r["pass"]),
            "total": len(summary),
            "results": summary,
        }

    if args.mlflow:
        sys.path.insert(0, str(PLATFORM_DIR))
        from ops.mlflow_eval import log_agent_eval, mlflow_available, tracking_uri

        if not mlflow_available():
            print("MLflow not installed (pip install mlflow)")
        else:
            artifact = args.output if not args.no_write else RESULTS_PATH
            run_id = log_agent_eval(payload, artifact_path=artifact)
            if run_id:
                print(f"MLflow run: {run_id}")
            print(f"MLflow tracking URI: {tracking_uri()}")

    if langsmith_on:
        from ops.langsmith_tracing import langsmith_ui_hint

        print(f"\nLangSmith traces → {langsmith_ui_hint()}")


if __name__ == "__main__":
    main()
