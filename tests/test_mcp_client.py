"""Tests for the dynamic MCP client (no network / no real servers)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="feishu_mcp_test_")
os.environ["MCP_CONFIG_PATH"] = str(Path(_TMP) / "mcp_servers.json")
os.environ["ASSISTANT_DATA_DIR"] = _TMP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.tools import mcp_client  # noqa: E402


class McpClientTest(unittest.TestCase):
    def setUp(self) -> None:
        mcp_client.MCP_CONFIG_PATH.unlink(missing_ok=True)

    def test_no_config_yields_no_tools(self) -> None:
        self.assertEqual(mcp_client.load_config(), [])
        self.assertEqual(mcp_client.load_mcp_tools(), [])

    def test_config_filters_enabled_servers(self) -> None:
        cfg = {
            "servers": [
                {"name": "a", "transport": "stdio", "command": "x", "enabled": True},
                {"name": "b", "transport": "stdio", "command": "y", "enabled": False},
                {"name": "c", "transport": "stdio", "command": "z"},
            ]
        }
        mcp_client.MCP_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        names = [s["name"] for s in mcp_client.load_config()]
        self.assertEqual(names, ["a", "c"])

    def test_invalid_server_config_is_skipped_with_warning(self) -> None:
        cfg = {
            "servers": [
                {"name": "bad-stdio", "transport": "stdio", "enabled": True},
                {"name": "bad-http", "transport": "streamable_http", "enabled": True},
                {"name": "bad-transport", "transport": "carrier_pigeon", "command": "x", "enabled": True},
                {"name": "good", "transport": "stdio", "command": "python", "enabled": True},
            ]
        }
        mcp_client.MCP_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        with self.assertLogs("kairos.mcp", level="WARNING") as logs:
            names = [s["name"] for s in mcp_client.load_config()]
        self.assertEqual(names, ["good"])
        self.assertEqual(len(logs.output), 3)

    def test_schema_conversion(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "limit": {"type": "integer", "description": "max results"},
                "optional_flag": {"type": "boolean", "description": "flag"},
            },
            "required": ["query"],
        }
        model = mcp_client._json_schema_to_pydantic("web_search_exa", schema)
        instance = model(query="hello", limit=5)
        self.assertEqual(instance.query, "hello")
        self.assertEqual(instance.limit, 5)
        self.assertIsNone(instance.optional_flag)

    def test_make_tool_passes_arguments(self) -> None:
        server = {"name": "fake", "transport": "stdio", "command": "nope"}
        info = {
            "name": "fake_search",
            "description": "a fake MCP tool",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "q"}},
                "required": ["query"],
            },
        }
        tool = mcp_client.make_tool(server, info)
        self.assertEqual(tool.name, "fake_search")
        with patch.object(mcp_client, "_call_sync", return_value="结果") as mocked:
            out = tool.invoke({"query": "hello"})
        self.assertEqual(out, "结果")
        mocked.assert_called_once_with(server, "fake_search", {"query": "hello"})

    def test_load_mcp_tools_isolates_failures(self) -> None:
        cfg = {
            "servers": [
                {"name": "broken", "transport": "streamable_http", "url": "http://127.0.0.1:1/mcp", "enabled": True}
            ]
        }
        mcp_client.MCP_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        with patch.object(mcp_client, "discover_tools", side_effect=RuntimeError("boom")):
            self.assertEqual(mcp_client.load_mcp_tools(), [])

    def test_load_mcp_tools_registers_discovered_tools(self) -> None:
        cfg = {
            "servers": [
                {"name": "ok", "transport": "stdio", "command": "fake", "enabled": True}
            ]
        }
        mcp_client.MCP_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        fake_tool = {
            "name": "github_search",
            "description": "search github",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "q"}},
                "required": ["query"],
            },
        }
        with patch.object(mcp_client, "discover_tools", return_value=[fake_tool]):
            tools = mcp_client.load_mcp_tools()
        self.assertEqual([t.name for t in tools], ["github_search"])

    @classmethod
    def tearDownClass(cls) -> None:
        mcp_client.MCP_CONFIG_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
