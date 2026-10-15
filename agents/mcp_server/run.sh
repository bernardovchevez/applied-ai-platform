#!/usr/bin/env bash
# MCP stdio launcher: uses project venv (system `python` may lack `mcp`).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/agents/mcp_server/server.py" "$@"
