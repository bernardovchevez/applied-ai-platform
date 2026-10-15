#!/usr/bin/env python3
"""MCP server exposing RAG + workflow agent tools.

Run (stdio for IDE / desktop MCP clients):
  python agents/mcp_server/server.py

Run (SSE for local HTTP):
  python agents/mcp_server/server.py --transport sse --port 8765
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(AGENTS_DIR))

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

from tools import (
    check_ip_flags_impl,
    draft_brief_impl_for_guard,
    escalate_legal_impl_for_guard,
    lookup_rights_impl_for_guard,
    search_docs_impl_for_guard,
    search_web_stub_impl,
)

CORPUS_META = LAB_ROOT / "data" / "synthetic" / "sharepoint-meta.json"
EVAL_RESULTS = AGENTS_DIR / "evals" / "agent_results.json"

_TOOL_USE_RULES = (
    "Always call search_docs before answering factual questions. "
    "For territory clearance, call lookup_rights first, then search_docs for detail. "
    "If clearance is pending, unknown, or confidence is below 0.7, call escalate_legal. "
    "Your answer MUST agree with tool results; never contradict search_docs or lookup_rights. "
    "Cite source document filenames when answering from the corpus."
)

server = MCPServer(
    "media-workflow",
    title="Media Operations Workflow",
    instructions=(
        "Production workflow tools for media operations: corpus RAG (search_docs), "
        "territory clearance lookup (lookup_rights), production briefs (draft_brief), "
        "and legal escalation (escalate_legal). Requires pgvector ingest for full functionality."
    ),
)


@server.custom_route("/", methods=["GET"])
@server.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    """Browser-friendly status; MCP protocol is on /sse, not /."""
    tools = server._tool_manager.list_tools()
    resources = server._resource_manager.list_resources()
    prompts = server._prompt_manager.list_prompts()
    return JSONResponse(
        {
            "service": "media-workflow",
            "status": "ok",
            "mcp_sse": "/sse",
            "mcp_messages": "/messages/",
            "tools": [t.name for t in tools],
            "resources": [str(r.uri) for r in resources],
            "prompts": [p.name for p in prompts],
            "note": "Open / in a browser for this page; MCP clients connect to /sse.",
        }
    )


@server.tool(
    name="search_docs",
    description="Search media ops documents (rights, production, ad sales, legal).",
)
def mcp_search_docs(question: str) -> str:
    """RAG over the synthetic media ops corpus (pgvector, Chroma fallback)."""
    return json.dumps(search_docs_impl_for_guard(question), default=str)


@server.tool(
    name="lookup_rights",
    description=(
        "Look up structured music/content clearance for a series in a territory "
        "(SQL clearance_territories + vector fallback)."
    ),
)
def mcp_lookup_rights(series: str, territory: str) -> str:
    """Territory clearance status with confidence and source memos."""
    return json.dumps(lookup_rights_impl_for_guard(series, territory), default=str)


@server.tool(
    name="draft_brief",
    description="Draft a production brief from gathered research and clearance context.",
)
def mcp_draft_brief(topic: str, context: str) -> str:
    """LLM-synthesized production brief with citations from context."""
    return draft_brief_impl_for_guard(topic, context)


@server.tool(
    name="escalate_legal",
    description="Escalate to legal when clearance is pending, unclear, or confidence is below 0.7.",
)
def mcp_escalate_legal(reason: str, confidence: float) -> str:
    """Format a legal escalation payload for the compliance workflow."""
    return json.dumps(escalate_legal_impl_for_guard(reason, confidence), default=str)


@server.tool(
    name="check_ip_flags",
    description="Scan sponsorship and product-placement docs for IP review flags.",
)
def mcp_check_ip_flags(series: str, user_message: str) -> str:
    """Rule-based IP / sponsorship flag scan for launch summaries."""
    return json.dumps(check_ip_flags_impl(series, user_message), default=str)


@server.tool(
    name="search_web_stub",
    description="Disabled web search stub in corpus-only mode (returns not configured).",
)
def mcp_search_web_stub(query: str) -> str:
    """Placeholder for external web research (intentionally disabled)."""
    return json.dumps(search_web_stub_impl(query), default=str)


@server.prompt(
    name="clearance-summary",
    title="Clearance summary",
    description="Prepare a territory clearance summary for a series streaming launch.",
)
def clearance_summary_prompt(series: str = "Harbor Line", territory: str = "EMEA") -> str:
    """User-facing template for compliance / distribution clearance workflows."""
    return (
        f"Prepare a clearance summary for {series} in {territory} for streaming launch.\n\n"
        f"{_TOOL_USE_RULES}"
    )


@server.prompt(
    name="corpus-research",
    title="Corpus research",
    description="Answer a factual question using the media ops document corpus.",
)
def corpus_research_prompt(
    question: str = "Is Harbor Line cleared for EMEA streaming?",
) -> str:
    """User-facing template for RAG Q&A over production, legal, and ad sales docs."""
    return (
        f"Research question: {question}\n\n"
        f"Call search_docs with this question first. {_TOOL_USE_RULES}"
    )


@server.prompt(
    name="draft-production-brief",
    title="Draft production brief",
    description="Gather clearance context and draft a production brief for the team.",
)
def draft_production_brief_prompt(
    topic: str = "S01E04 streaming launch readiness",
    series: str = "Harbor Line",
    territory: str = "EMEA",
) -> str:
    """User-facing template for Producer-style brief requests."""
    return (
        f"Draft a production brief on: {topic}\n"
        f"Series: {series} | Territory: {territory}\n\n"
        "Call lookup_rights and search_docs first, then draft_brief with gathered context. "
        f"{_TOOL_USE_RULES}"
    )


@server.prompt(
    name="legal-escalation-review",
    title="Legal escalation review",
    description="Format a legal escalation when clearance is unclear or below confidence threshold.",
)
def legal_escalation_review_prompt(
    reason: str = "Clearance status unclear or confidence below threshold.",
    confidence: float = 0.5,
) -> str:
    """User-facing template for Compliance → Legal handoff."""
    return (
        f"Review this clearance issue for legal escalation:\n{reason}\n\n"
        f"Reported confidence: {confidence}\n\n"
        "Call lookup_rights if series/territory are mentioned, then escalate_legal with the "
        f"actual confidence from lookup_rights when available. {_TOOL_USE_RULES}"
    )


@server.resource(
    "media://corpus/clearance-index",
    name="clearance-index",
    description="Rights clearance memo filenames and territory metadata from the synthetic corpus.",
    mime_type="application/json",
)
def clearance_index_resource() -> str:
    """JSON index of clearance-related documents."""
    if not CORPUS_META.exists():
        return json.dumps({"error": "sharepoint-meta.json not found; run ingest first"})
    meta = json.loads(CORPUS_META.read_text(encoding="utf-8"))
    rows = meta.get("documents", meta if isinstance(meta, list) else [])
    clearance_docs = [
        {
            "filename": row.get("filename") or row.get("file_name"),
            "doc_type": row.get("doc_type"),
            "rights_tier": row.get("rights_tier"),
            "territory_status": row.get("territory_status"),
            "clearance_territories": row.get("clearance_territories"),
        }
        for row in rows
        if row.get("doc_type") == "rights_clearance" or row.get("rights_tier")
    ]
    return json.dumps({"clearance_documents": clearance_docs, "count": len(clearance_docs)}, indent=2)


@server.resource(
    "media://evals/agent-results",
    name="agent-eval-results",
    description="Latest agent scenario eval scores (graph + agent_v1).",
    mime_type="application/json",
)
def agent_eval_results_resource() -> str:
    """Most recent agent_results.json from evals/run_agent_eval.py."""
    if not EVAL_RESULTS.exists():
        return json.dumps(
            {
                "message": "No eval results yet",
                "run": "python agents/evals/run_agent_eval.py --graph",
            }
        )
    return EVAL_RESULTS.read_text(encoding="utf-8")


def smoke_test() -> None:
    """Print tools/resources/prompts and run one lookup for terminal verification."""
    tools = server._tool_manager.list_tools()
    resources = server._resource_manager.list_resources()
    prompts = server._prompt_manager.list_prompts()
    print("MCP server smoke test (media-workflow)")
    print(f"  Tools ({len(tools)}): {', '.join(t.name for t in tools)}")
    print(f"  Resources ({len(resources)}): {', '.join(str(r.uri) for r in resources)}")
    print(f"  Prompts ({len(prompts)}): {', '.join(p.name for p in prompts)}")
    sample = json.loads(mcp_lookup_rights("Harbor Line", "EMEA"))
    print(
        f"  lookup_rights('Harbor Line', 'EMEA'): "
        f"status={sample.get('status')} confidence={sample.get('confidence')}"
    )
    print("OK: use stdio via an MCP client, or --transport sse for HTTP.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Media workflow MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport (stdio for local MCP clients; sse for HTTP clients)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="List tools and run one lookup, then exit (for terminal testing)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.transport == "sse":
        base = f"http://{args.host}:{args.port}"
        print(f"MCP SSE server running", flush=True)
        print(f"  Health:  {base}/health", flush=True)
        print(f"  MCP SSE: {base}/sse  (not {base}/; use an MCP client)", flush=True)
        server.run(transport="sse", host=args.host, port=args.port)
    else:
        print(
            "MCP stdio server running: waiting for client on stdin.\n"
            "This terminal will look idle until a client connects. Ctrl+C to stop.\n"
            "To test without a client: python .../server.py --smoke-test",
            file=sys.stderr,
            flush=True,
        )
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
