#!/usr/bin/env python3
"""Thin MCP client demo: list tools/resources/prompts and call lookup_rights.

Spawns the local server over stdio (default) or connects to SSE.

Usage:
  python agents/mcp_server/client_demo.py
  python agents/mcp_server/client_demo.py --sse http://127.0.0.1:8765/sse
  python agents/mcp_server/client_demo.py --tool search_docs --question "EMEA clearance?"
  python agents/mcp_server/client_demo.py --prompt clearance-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anyio

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"
sys.path.insert(0, str(LAB_ROOT))

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client


def _content_text(result) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else ""


def _prompt_message_text(result) -> str:
    parts: list[str] = []
    for msg in getattr(result, "messages", []) or []:
        content = getattr(msg, "content", None)
        if content is None:
            continue
        text = getattr(content, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


async def run_demo(
    *,
    sse_url: str | None,
    tool: str | None,
    prompt: str | None,
    series: str,
    territory: str,
    question: str,
) -> None:
    if sse_url:
        async with sse_client(sse_url) as streams:
            await _session_demo(
                *streams,
                tool=tool,
                prompt=prompt,
                series=series,
                territory=territory,
                question=question,
            )
    else:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVER_SCRIPT)],
            cwd=str(LAB_ROOT),
        )
        async with stdio_client(params) as streams:
            await _session_demo(
                *streams,
                tool=tool,
                prompt=prompt,
                series=series,
                territory=territory,
                question=question,
            )


async def _session_demo(
    read_stream,
    write_stream,
    *,
    tool: str | None,
    prompt: str | None,
    series: str,
    territory: str,
    question: str,
) -> None:
    async with ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        info = session.server_info
        print(f"Connected: {info.name if info else 'media-workflow'}")

        tools = await session.list_tools()
        print(f"\nTools ({len(tools.tools)}):")
        for t in tools.tools:
            print(f"  - {t.name}: {(t.description or '')[:70]}")

        resources = await session.list_resources()
        print(f"\nResources ({len(resources.resources)}):")
        for r in resources.resources:
            print(f"  - {r.uri}")

        prompts = await session.list_prompts()
        print(f"\nPrompts ({len(prompts.prompts)}):")
        for p in prompts.prompts:
            title = p.title or p.name
            print(f"  - {p.name}: {title}: {(p.description or '')[:55]}")

        if prompt:
            args = {
                "clearance-summary": {"series": series, "territory": territory},
                "corpus-research": {"question": question},
                "draft-production-brief": {
                    "topic": question,
                    "series": series,
                    "territory": territory,
                },
                "legal-escalation-review": {
                    "reason": question,
                    "confidence": "0.5",
                },
            }.get(prompt, {"series": series, "territory": territory})
            print(f"\n=== get_prompt {prompt!r} ===")
            rendered = await session.get_prompt(prompt, args)
            print(_prompt_message_text(rendered)[:800])
            return

        if tool == "search_docs":
            print(f"\n=== call_tool search_docs ===")
            result = await session.call_tool("search_docs", {"question": question})
            print(_content_text(result)[:800])
            return

        print(f"\n=== call_tool lookup_rights ({series!r}, {territory!r}) ===")
        result = await session.call_tool(
            "lookup_rights",
            {"series": series, "territory": territory},
        )
        text = _content_text(result)
        try:
            data = json.loads(text)
            print(json.dumps(data, indent=2)[:1200])
            print(f"\n→ status={data.get('status')} confidence={data.get('confidence')}")
        except json.JSONDecodeError:
            print(text[:800])

        print("\n=== read_resource media://corpus/clearance-index (preview) ===")
        try:
            resource = await session.read_resource("media://corpus/clearance-index")
            blocks = getattr(resource, "contents", []) or []
            preview = ""
            for block in blocks:
                preview = getattr(block, "text", None) or preview
            print((preview or str(resource))[:400] + ("…" if len(preview) > 400 else ""))
        except Exception as exc:
            print(f"(resource read skipped: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP client demo for media-workflow")
    parser.add_argument(
        "--sse",
        help="SSE server URL (default: spawn server.py over stdio)",
    )
    parser.add_argument(
        "--tool",
        choices=["lookup_rights", "search_docs"],
        help="Tool to call (default: lookup_rights)",
    )
    parser.add_argument(
        "--prompt",
        choices=[
            "clearance-summary",
            "corpus-research",
            "draft-production-brief",
            "legal-escalation-review",
        ],
        help="Render an MCP prompt template (default flow: lookup_rights tool call)",
    )
    parser.add_argument("--series", default="Harbor Line")
    parser.add_argument("--territory", default="EMEA")
    parser.add_argument(
        "--question",
        default="Is Harbor Line cleared for EMEA streaming?",
    )
    args = parser.parse_args()

    from functools import partial

    anyio.run(
        partial(
            run_demo,
            sse_url=args.sse,
            tool=args.tool,
            prompt=args.prompt,
            series=args.series,
            territory=args.territory,
            question=args.question,
        )
    )


if __name__ == "__main__":
    main()
