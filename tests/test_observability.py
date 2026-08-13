"""Smoke tests for the observability layer and the web panel (no network)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="feishu_obs_test_")
os.environ["ASSISTANT_DATA_DIR"] = _TMP
os.environ["PANEL_TOKEN"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from kairos.agent import runtime as agent_runtime  # noqa: E402
from kairos.memory import store as memory_store  # noqa: E402
from kairos.observability import metrics  # noqa: E402
from web_panel import create_app  # noqa: E402


class _FakeGraph:
    def __init__(self, messages: list) -> None:
        self.messages = messages

    def invoke(self, state: dict, config: dict) -> dict:
        return {"messages": self.messages}


class ObservabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        memory_store.init_db()
        metrics.init_metrics_table()
        metrics.log_request(
            request_id="test-ok-1",
            owner_id="user-1",
            chat_id="chat-1",
            question="最近的机器翻译进展如何？",
            context_len=300,
            tool_sequence=["web_search", "read_webpage"],
            answer="机器翻译近期有若干进展。",
            prompt_tokens=120,
            completion_tokens=40,
            latency_ms=1500,
            status="ok",
        )
        metrics.log_request(
            request_id="test-err-1",
            owner_id="user-2",
            chat_id="chat-2",
            question="帮我归档到知识库",
            context_len=100,
            tool_sequence=["archive_to_knowledge_base"],
            answer="权限不足",
            prompt_tokens=80,
            completion_tokens=5,
            latency_ms=800,
            status="error",
            error_type="PermissionError",
        )

    def test_metrics_summary(self) -> None:
        s = metrics.summary()
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["ok"], 1)
        self.assertEqual(s["errors"], 1)
        self.assertGreater(s["total_tokens"], 0)
        self.assertGreater(s["estimated_cost_usd"], 0)
        self.assertEqual(len(s["last_7_days"]), 7)

    def test_tool_stats(self) -> None:
        stats = {t["tool"]: t for t in metrics.tool_stats()}
        self.assertGreaterEqual(stats["web_search"]["calls"], 1)
        self.assertGreaterEqual(stats["archive_to_knowledge_base"]["errors"], 1)

    def test_runtime_answer_records_log(self) -> None:
        graph = _FakeGraph(
            [
                HumanMessage(content="input"),
                AIMessage(content="模拟回答", tool_calls=[{"name": "web_search", "args": {}, "id": "1", "type": "tool_call"}]),
                AIMessage(content="最终回答内容"),
            ]
        )
        out = agent_runtime.answer(graph, "测试问题", "上下文", "owner-x", "chat-x")
        self.assertEqual(out, "最终回答内容")
        rows, _ = metrics.list_logs(page=1, status="ok")
        latest = rows[0]
        self.assertEqual(latest["question"], "测试问题")
        self.assertEqual(latest["tool_sequence"], ["web_search"])
        self.assertEqual(latest["chat_id"], "chat-x")

    def test_web_panel_pages(self) -> None:
        client = create_app().test_client()
        for path in ("/", "/logs", "/logs/1", "/health", "/memories", "/api/summary", "/api/logs", "/api/tools"):
            resp = client.get(path)
            self.assertEqual(resp.status_code, 200, path)
        # The dashboard must render real HTML, not Jinja-escaped source text.
        dash = client.get("/").get_data(as_text=True)
        self.assertIn('<div class="cards">', dash)
        self.assertNotIn("&lt;div", dash)
        logs = client.get("/logs").get_data(as_text=True)
        self.assertIn("<table>", logs)
        self.assertNotIn("&lt;table", logs)
        memories = client.get("/memories").get_data(as_text=True)
        self.assertIn("记忆", memories)
        summary = client.get("/api/summary").get_json()
        self.assertGreaterEqual(summary["total"], 2)
        detail = client.get("/api/logs/1").get_json()
        self.assertIn("tool_sequence", detail)

    def test_redact(self) -> None:
        self.assertNotIn("sk-", metrics.redact("key sk-abcdefghijklmnop and https://x.com/?token=abc"))
        self.assertNotIn("abc", metrics.redact("https://x.com/?token=abc")[0:0] + metrics.redact("https://x.com/?token=abc"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
