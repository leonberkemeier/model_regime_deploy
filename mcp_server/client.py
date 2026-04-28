"""
MCP Client — Synchronous Wrapper
=================================
Provides a simple synchronous interface for calling tools on the remote
MCP server from the AI PC's synchronous task pipeline.

Usage:
    from mcp_server.client import call_tool, is_server_available

    result = call_tool("get_quantitative_risk_tool", {"ticker": "AAPL"})
    data   = json.loads(result)

Set MCP_SERVER_URL in .env to the Tailscale address of the webserver:
    MCP_SERVER_URL=http://your-server.tailnet.ts.net:9876
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9876")


async def _call_tool_async(tool_name: str, args: dict) -> str:
    """Async implementation — call a named tool on the MCP server via SSE."""
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession

    sse_url = f"{MCP_SERVER_URL.rstrip('/')}/sse"

    async with sse_client(sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)

            # Extract text content from the MCP result
            if result.content:
                return result.content[0].text
            return "{}"


def call_tool(tool_name: str, args: dict | None = None) -> str:
    """
    Synchronously call a tool on the remote MCP server.

    Returns the tool's JSON string response, or an error JSON string
    if the server is unreachable.

    Args:
        tool_name: The registered tool name (e.g. "get_quantitative_risk_tool")
        args:      Tool arguments as a dict (default: empty)
    """
    if args is None:
        args = {}

    try:
        return asyncio.run(_call_tool_async(tool_name, args))
    except Exception as exc:
        logger.error(f"MCP tool call failed [{tool_name}]: {exc}")
        return json.dumps({"error": str(exc), "tool": tool_name})


def is_server_available() -> bool:
    """
    Quick connectivity check — returns True if the MCP server is reachable.
    Used by the agentic loop to decide whether to enable tool-calling mode.
    """
    import requests
    try:
        resp = requests.get(f"{MCP_SERVER_URL.rstrip('/')}/sse", timeout=3, stream=True)
        # SSE endpoint returns 200 and keeps connection open; any 2xx means it's up
        resp.close()
        return resp.status_code < 400
    except Exception:
        return False
