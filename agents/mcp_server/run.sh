#!/usr/bin/env bash
# Cursor MCP launcher — uses project venv (system `python` lacks `mcp`).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/agents/mcp_server/server.py" "$@"
