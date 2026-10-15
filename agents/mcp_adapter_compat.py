"""Bridge langchain-mcp-adapters 0.3.x with MCP Python SDK 2.x.

langchain-mcp-adapters targets mcp 1.x (FastMCP imports, camelCase result
fields, ``list_tools(cursor=...)``). Our MCP server uses mcp 2.x (MCPServer).
Call ``apply_mcp2_compat()`` before importing langchain_mcp_adapters.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from pydantic import BaseModel

_applied = False


def apply_mcp2_compat() -> None:
    """Patch MCP SDK gaps so langchain-mcp-adapters works with mcp 2.x."""
    global _applied
    if _applied:
        return
    _applied = True

    import mcp.shared.context as mcp_context

    if not hasattr(mcp_context, "RequestContext"):
        mcp_context.RequestContext = mcp_context.BaseContext

    if "mcp.shared.session" not in sys.modules:
        from mcp.shared.dispatcher import ProgressFnT

        shim = types.ModuleType("mcp.shared.session")
        shim.ProgressFnT = ProgressFnT
        sys.modules["mcp.shared.session"] = shim

    if "mcp.server.fastmcp.tools" not in sys.modules:
        fastmcp = types.ModuleType("mcp.server.fastmcp")
        util_pkg = types.ModuleType("mcp.server.fastmcp.utilities")
        tools_mod = types.ModuleType("mcp.server.fastmcp.tools")
        util_mod = types.ModuleType("mcp.server.fastmcp.utilities.func_metadata")

        class _Tool:
            pass

        class _ArgModelBase(BaseModel):
            pass

        class _FuncMetadata:
            def __init__(self, **kwargs: Any) -> None:
                pass

        tools_mod.Tool = _Tool
        util_mod.ArgModelBase = _ArgModelBase
        util_mod.FuncMetadata = _FuncMetadata
        sys.modules["mcp.server.fastmcp"] = fastmcp
        sys.modules["mcp.server.fastmcp.tools"] = tools_mod
        sys.modules["mcp.server.fastmcp.utilities"] = util_pkg
        sys.modules["mcp.server.fastmcp.utilities.func_metadata"] = util_mod

    from mcp.client.session import ClientSession
    from mcp.types import PaginatedRequestParams

    if not getattr(ClientSession, "_lc_mcp2_compat", False):

        async def _list_tools_compat(
            self: ClientSession,
            *,
            cursor: str | None = None,
            params: PaginatedRequestParams | None = None,
            **kwargs: Any,
        ):
            if cursor is not None:
                params = PaginatedRequestParams(cursor=cursor)
            result = await ClientSession._lc_orig_list_tools(self, params=params)  # type: ignore[attr-defined]
            if not hasattr(result, "nextCursor"):
                object.__setattr__(result, "nextCursor", getattr(result, "next_cursor", None))
            return result

        ClientSession._lc_orig_list_tools = ClientSession.list_tools  # type: ignore[attr-defined]
        ClientSession.list_tools = _list_tools_compat  # type: ignore[method-assign]
        ClientSession._lc_mcp2_compat = True  # type: ignore[attr-defined]

    if not getattr(ClientSession, "_lc_mcp2_call_compat", False):

        async def _call_tool_compat(
            self: ClientSession,
            name: str,
            arguments: dict[str, Any] | None = None,
            *args: Any,
            **kwargs: Any,
        ):
            result = await ClientSession._lc_orig_call_tool(  # type: ignore[attr-defined]
                self, name, arguments, *args, **kwargs
            )
            if not hasattr(result, "isError"):
                object.__setattr__(result, "isError", getattr(result, "is_error", False))
            if not hasattr(result, "structuredContent"):
                object.__setattr__(
                    result,
                    "structuredContent",
                    getattr(result, "structured_content", None),
                )
            return result

        ClientSession._lc_orig_call_tool = ClientSession.call_tool  # type: ignore[attr-defined]
        ClientSession.call_tool = _call_tool_compat  # type: ignore[method-assign]
        ClientSession._lc_mcp2_call_compat = True  # type: ignore[attr-defined]

    _patch_adapter_tool_listing()


def _patch_adapter_tool_listing() -> None:
    """Add camelCase aliases on MCP Tool models for adapter conversion."""
    import langchain_mcp_adapters.tools as adapter_tools

    if getattr(adapter_tools, "_lc_mcp2_compat", False):
        return

    _orig_list = adapter_tools._list_all_tools

    async def _list_all_tools_compat(session):  # type: ignore[no-untyped-def]
        tools = await _orig_list(session)
        for tool in tools:
            if not hasattr(tool, "inputSchema"):
                object.__setattr__(tool, "inputSchema", getattr(tool, "input_schema", {}))
        return tools

    adapter_tools._list_all_tools = _list_all_tools_compat  # type: ignore[assignment]
    adapter_tools._lc_mcp2_compat = True
