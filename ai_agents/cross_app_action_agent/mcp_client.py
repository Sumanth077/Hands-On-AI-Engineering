"""Small adapter around Liner Actions MCP.

FastMCP owns the OAuth 2.1 browser flow. Liner owns the credentials for the
connected services, so this application never receives Slack or GitHub tokens.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from fastmcp.client.auth import OAuth


@dataclass(frozen=True)
class ActionTool:
    """A tool advertised by the remote Liner MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class LinerActionsClient:
    """Connect, discover, and call Liner Actions through one MCP endpoint."""

    def __init__(self, url: str | None = None, callback_port: int | None = None) -> None:
        self.url = url or os.environ.get(
            "LINER_ACTIONS_MCP_URL", "https://actions.liner.com/api/v1/mcp"
        )
        configured_port = callback_port or int(
            os.environ.get("LINER_OAUTH_CALLBACK_PORT", "8765")
        )
        self.oauth = OAuth(client_name="Cross-App Action Agent", callback_port=configured_port)
        self.tools: list[ActionTool] = []

    async def connect(self) -> list[ActionTool]:
        """Open Liner's OAuth flow if needed, then cache the server's tools."""
        async with Client(self.url, auth=self.oauth, init_timeout=20, timeout=60) as client:
            remote_tools = await client.list_tools()
        self.tools = []
        for tool in remote_tools:
            # MCP's wire field is inputSchema. Some adapters expose a Pythonic
            # input_schema alias, while the installed FastMCP version returns the
            # original MCP field. Support both without changing the remote schema.
            input_schema = getattr(tool, "input_schema", None)
            if input_schema is None:
                input_schema = getattr(tool, "inputSchema", {})
            self.tools.append(
                ActionTool(
                    name=tool.name,
                    description=tool.description or "No description supplied by Liner.",
                    input_schema=dict(input_schema),
                )
            )
        return self.tools

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one Liner tool and return structured output when available."""
        async with Client(self.url, auth=self.oauth, init_timeout=20, timeout=60) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
        if result.is_error:
            detail = "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )
            raise RuntimeError(detail or f"Liner action {name} failed.")
        if result.data is not None:
            return result.data
        return "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )

    def connect_sync(self) -> list[ActionTool]:
        return asyncio.run(self.connect())

    def call_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        return asyncio.run(self.call(name, arguments))

    @staticmethod
    def format_result(value: Any) -> str:
        """Keep tool output readable for both the model and the activity log."""
        return json.dumps(value, indent=2, default=str)[:12_000]

