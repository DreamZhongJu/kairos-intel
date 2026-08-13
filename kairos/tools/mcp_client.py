"""Dynamic MCP client for the Feishu research assistant.

Loads external MCP servers from a JSON config and exposes their tools as
LangChain tools at runtime, so new capabilities (search engines, file
systems, GitHub, databases, ...) can be added without writing per-tool code.

Supported transports:
  - "streamable_http": remote MCP servers over HTTPS (e.g. Exa).
  - "stdio": local MCP servers launched as subprocesses (e.g. npx servers).

Config file (default: <project>/mcp_servers.json):
  {
    "servers": [
      {"name": "exa", "transport": "streamable_http", "url": "https://mcp.exa.ai/mcp", "enabled": true},
      {"name": "filesystem", "transport": "stdio", "command": "npx",
       "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/data"], "enabled": true}
    ]
  }

Disabled or unreachable servers are skipped with a warning; the assistant
keeps working with its built-in tools when no MCP server is configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field, create_model

from kairos.infrastructure.settings import PROJECT_ROOT

LOG = logging.getLogger("kairos.mcp")

MCP_CONFIG_PATH = Path(os.getenv("MCP_CONFIG_PATH", str(PROJECT_ROOT / "mcp_servers.json")))

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_server(server: dict[str, Any]) -> list[str]:
    """Return a list of config problems for one server entry."""
    errors: list[str] = []
    name = str(server.get("name", "")).strip()
    if not name:
        errors.append("缺少 name")
    transport = server.get("transport", "stdio")
    if transport not in {"stdio", "streamable_http"}:
        errors.append(f"不支持的 transport: {transport}（支持 stdio / streamable_http）")
    if transport == "stdio" and not str(server.get("command", "")).strip():
        errors.append("stdio server 缺少 command")
    if transport == "streamable_http" and not str(server.get("url", "")).strip():
        errors.append("streamable_http server 缺少 url")
    return errors


def load_config() -> list[dict[str, Any]]:
    """Read enabled servers from the config file; empty when absent."""
    if not MCP_CONFIG_PATH.exists():
        return []
    try:
        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        LOG.warning("MCP config unreadable: %s", exc)
        return []
    servers = data.get("servers", []) if isinstance(data, dict) else []
    enabled: list[dict[str, Any]] = []
    for server in servers:
        if not isinstance(server, dict) or not server.get("enabled", True):
            continue
        errors = validate_server(server)
        if errors:
            LOG.warning(
                "MCP server %r 配置无效，已跳过：%s",
                server.get("name", "?"),
                "；".join(errors),
            )
            continue
        enabled.append(server)
    return enabled


@asynccontextmanager
async def _open_session(server: dict[str, Any]) -> AsyncIterator[ClientSession]:
    """Async context manager yielding a connected ClientSession."""
    transport = server.get("transport", "stdio")
    if transport == "streamable_http":
        url = server.get("url", "")
        if not url:
            raise ValueError("streamable_http server requires 'url'")
        streams = streamable_http_client(url)
    elif transport == "stdio":
        command = server.get("command", "")
        if not command:
            raise ValueError("stdio server requires 'command'")
        params = StdioServerParameters(
            command=command,
            args=server.get("args", []) or [],
            env=server.get("env") or None,
        )
        streams = stdio_client(params)
    else:
        raise ValueError(f"unsupported MCP transport: {transport}")

    async with streams as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _discover_async(server: dict[str, Any]) -> list[dict[str, Any]]:
    async with _open_session(server) as session:
        result = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema or {},
            }
            for tool in result.tools
        ]


async def _call_async(server: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> str:
    async with _open_session(server) as session:
        result = await session.call_tool(tool_name, arguments)
        parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            parts.append(text if isinstance(text, str) else str(item))
        payload = "\n".join(part for part in parts if part)
        if getattr(result, "isError", False):
            raise RuntimeError(payload or f"MCP tool {tool_name} returned an error")
        return payload or "(空结果)"


def discover_tools(server: dict[str, Any]) -> list[dict[str, Any]]:
    """Sync wrapper: list tools exposed by one server."""
    return asyncio.run(_discover_async(server))


def _call_sync(server: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> str:
    return asyncio.run(_call_async(server, tool_name, arguments))


def _json_schema_to_pydantic(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON Schema input schema into a Pydantic args model."""
    fields: dict[str, tuple[Any, Any]] = {}
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    for prop_name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        field_type = _TYPE_MAP.get(prop.get("type", "string"), str)
        description = str(prop.get("description", "")) or prop_name
        if prop_name in required:
            fields[prop_name] = (field_type, Field(description=description))
        else:
            fields[prop_name] = (field_type, Field(default=None, description=description))
    if not fields:
        fields["_empty"] = (str, Field(default="", description="unused placeholder"))
    safe_name = re.sub(r"\W+", "_", name) or "mcp_tool"
    return create_model(f"{safe_name}_args", **fields)


def make_tool(server: dict[str, Any], info: dict[str, Any]) -> BaseTool:
    """Build one LangChain tool that invokes a remote MCP tool on demand."""
    tool_name = str(info.get("name", ""))
    args_model = _json_schema_to_pydantic(tool_name, info.get("input_schema", {}) or {})

    def invoke_func(_server: dict[str, Any] = server, _tool: str = tool_name, **kwargs: Any) -> str:
        arguments = {key: value for key, value in kwargs.items() if value is not None}
        return _call_sync(_server, _tool, arguments)

    return StructuredTool.from_function(
        name=tool_name,
        description=str(info.get("description", "") or f"MCP tool: {tool_name}"),
        func=invoke_func,
        args_schema=args_model,
    )


def load_mcp_tools() -> list[BaseTool]:
    """Discover tools across all enabled servers; failures are isolated."""
    tools: list[BaseTool] = []
    for server in load_config():
        server_name = str(server.get("name", "?"))
        try:
            discovered = discover_tools(server)
        except Exception as exc:
            LOG.warning("MCP server %s unavailable, skipped: %s", server_name, exc)
            continue
        for info in discovered:
            try:
                tools.append(make_tool(server, info))
                LOG.info("MCP tool registered: %s/%s", server_name, info.get("name"))
            except Exception:
                LOG.exception("failed to build MCP tool %s/%s", server_name, info.get("name"))
    return tools
