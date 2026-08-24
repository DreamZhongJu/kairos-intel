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

    def test_knowledge_ingest_validation(self) -> None:
        self.assertEqual(self.client.post("/api/knowledge/ingest", json={}).status_code, 400)
        resp = self.client.post(
            "/api/knowledge/ingest",
            json={"channel_id": "123", "messages": [{"user_id": "1", "text": "only one"}]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_knowledge_ingest_requires_token_when_configured(self) -> None:
        from unittest.mock import patch

        from kairos.infrastructure import settings

        with patch.object(settings, "KAIROS_API_TOKEN", "secret"):
            self.assertEqual(
                self.client.post("/api/knowledge/ingest", json={"channel_id": "1", "messages": []}).status_code,
                401,
            )
            ok = self.client.post(
                "/api/knowledge/ingest",
                headers={"X-Token": "secret"},
                json={"channel_id": "1", "messages": []},
            )
            self.assertEqual(ok.status_code, 400)  # authed, now fails on payload

    def test_knowledge_query_endpoint(self) -> None:
        resp = self.client.get("/api/knowledge/query?q=anything")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for key in ("query", "entity", "chunks", "graph", "answer"):
            self.assertIn(key, data)
        self.assertIn("answer", data)


class McpServerTest(unittest.TestCase):
    def test_tools_registered(self) -> None:
        tools = asyncio.run(mcp_server.mcp.list_tools())
        names = {t.name for t in tools}
        self.assertIn("ask_agent", names)
        self.assertIn("web_search", names)
        self.assertIn("knowledge_graph_query", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)