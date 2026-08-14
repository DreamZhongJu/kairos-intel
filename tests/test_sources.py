"""Tests for verifiable source citation extracted from the tool-call chain."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from kairos.agent.runtime import _extract_sources, _source_footer  # noqa: E402


def _tool_call(name: str, call_id: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class SourceExtractionTest(unittest.TestCase):
    def test_search_json_results_become_sources(self) -> None:
        messages = [
            AIMessage(content="", tool_calls=[_tool_call("web_search", "1", {"query": "最新进展"})]),
            ToolMessage(
                content='[{"title": "A 站文章", "url": "https://example.com/a", "snippet": "..."}]',
                tool_call_id="1",
                name="web_search",
            ),
            AIMessage(content="根据检索结果，进展如下。"),
        ]
        sources, path = _extract_sources(messages)
        self.assertEqual(sources, [("A 站文章", "https://example.com/a")])
        self.assertEqual(path, ["web_search"])
        footer = _source_footer(sources, path)
        self.assertIn("https://example.com/a", footer)
        self.assertIn("联网搜索", footer)
        self.assertIn("参考来源", footer)

    def test_read_webpage_original_ranks_first(self) -> None:
        messages = [
            AIMessage(content="", tool_calls=[_tool_call("web_search", "1", {"query": "topic"})]),
            ToolMessage(
                content='[{"title": "列表页", "url": "https://example.com/list", "snippet": "..."}]',
                tool_call_id="1",
                name="web_search",
            ),
            AIMessage(content="", tool_calls=[_tool_call("read_webpage", "2", {"url": "https://example.com/article"})]),
            ToolMessage(
                content='{"title": "Article", "url": "https://example.com/article", "text": "..."}',
                tool_call_id="2",
                name="read_webpage",
            ),
            AIMessage(content="原文要点如下。"),
        ]
        sources, path = _extract_sources(messages)
        self.assertEqual(sources[0], ("原文", "https://example.com/article"))
        self.assertIn(("列表页", "https://example.com/list"), sources)
        self.assertEqual(path, ["web_search", "read_webpage"])

    def test_plain_text_url_fallback(self) -> None:
        messages = [
            ToolMessage(content="已创建云文档：https://my.feishu.cn/docx/abc123", tool_call_id="1", name="save_cloud_document"),
        ]
        sources, _ = _extract_sources(messages)
        self.assertEqual(sources, [("", "https://my.feishu.cn/docx/abc123")])

    def test_no_tools_yields_no_footer(self) -> None:
        messages = [AIMessage(content="直接回答，无工具。")]
        sources, path = _extract_sources(messages)
        self.assertEqual(sources, [])
        self.assertEqual(_source_footer(sources, path), "")

    def test_dedupe_repeated_urls(self) -> None:
        messages = [
            ToolMessage(content='[{"url": "https://example.com/a"}]', tool_call_id="1", name="web_search"),
            ToolMessage(content='{"url": "https://example.com/a", "title": "dup"}', tool_call_id="2", name="paper_lookup"),
        ]
        sources, _ = _extract_sources(messages)
        self.assertEqual(len(sources), 1)

    def test_cap_and_path_label_fallback(self) -> None:
        urls = [f"https://example.com/{index}" for index in range(20)]
        payload = [{"url": url} for url in urls]
        messages = [ToolMessage(content=str(payload).replace("'", '"'), tool_call_id="1", name="mcp_custom_tool")]
        sources, _ = _extract_sources(messages)
        self.assertEqual(len(sources), 6)
        footer = _source_footer(sources, ["mcp_custom_tool"])
        self.assertIn("mcp_custom_tool", footer)  # unknown tools keep their raw name


if __name__ == "__main__":
    unittest.main(verbosity=2)
