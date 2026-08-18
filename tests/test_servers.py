"""Tests for the MCP server and OpenAPI REST service surfaces."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.server import api  # noqa: E402
from kairos.server import mcp_server  # noqa: E402


class RestApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = api.create_app().test_client()

    def test_health(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_openapi_document(self) -> None:
        schema = api.build_openapi()
        self.assertIn("openapi", schema)
        self.assertIn("/api/chat", schema["paths"])

    def test_tools_endpoint(self) -> None:
        tools = self.client.get("/api/tools").get_json()
        self.assertIsInstance(tools, list)
        names = {t["name"] for t in tools}
        self.assertIn("web_search", names)

    def test_knowledge_stats_endpoint(self) -> None:
        self.assertEqual(self.client.get("/api/knowledge/stats").status_code, 200)

    def test_chat_requires_question(self) -> None:
        resp = self.client.post("/api/chat", json={})
        self.assertEqual(resp.status_code, 400)


class McpServerTest(unittest.TestCase):
    def test_tools_registered(self) -> None:
        tools = asyncio.run(mcp_server.mcp.list_tools())
        names = {t.name for t in tools}
        self.assertIn("ask_agent", names)
        self.assertIn("web_search", names)
        self.assertIn("knowledge_graph_query", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)