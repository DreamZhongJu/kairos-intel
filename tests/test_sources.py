"""Tests for verifiable source citation extracted from the tool-call chain."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from kairos.agent.runtime import (  # noqa: E402
    _extract_sources,
    _model_citation_urls,
    _source_footer,
    _validated_sources,
)


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


class ModelCitationValidationTest(unittest.TestCase):
    def test_citation_block_urls_extracted_and_stripped(self) -> None:
        content = "调研结论如下。\n\n参考来源：\nhttps://cs.hubu.edu.cn/szdw.htm\nhttps://www.hubu.edu.cn/\n"
        urls, index = _model_citation_urls(content)
        self.assertEqual(urls, ["https://cs.hubu.edu.cn/szdw.htm", "https://www.hubu.edu.cn/"])
        self.assertEqual(index, 2)
        stripped = "\n".join(content.splitlines()[:index]).strip()
        self.assertEqual(stripped, "调研结论如下。")
        self.assertNotIn("参考来源", stripped)

    def test_citation_marker_with_colon_and_prefix(self) -> None:
        content = "正文。\n——\n📎 参考来源：\n1. https://example.com/a\n"
        urls, index = _model_citation_urls(content)
        self.assertEqual(urls, ["https://example.com/a"])
        self.assertGreater(index, 0)

    def test_only_cited_sources_kept_in_order(self) -> None:
        sources = [
            ("原文", "https://example.com/article"),
            ("列表页", "https://example.com/list"),
            ("无关仓库", "https://github.com/someone/repo"),
        ]
        cited = ["https://github.com/someone/repo", "https://example.com/article"]
        validated = _validated_sources(sources, cited)
        # Order follows the real source list, not the citation order.
        self.assertEqual(
            validated,
            [("原文", "https://example.com/article"), ("无关仓库", "https://github.com/someone/repo")],
        )

    def test_hallucinated_citation_filtered_out(self) -> None:
        sources = [("原文", "https://example.com/article")]
        cited = ["https://example.com/article", "https://not-in-tool-output.com/fake"]
        validated = _validated_sources(sources, cited)
        self.assertEqual(validated, [("原文", "https://example.com/article")])

    def test_empty_citation_means_no_validation(self) -> None:
        sources = [("A", "https://example.com/a"), ("B", "https://example.com/b")]
        self.assertEqual(_validated_sources(sources, []), sources)

    def test_no_citation_block_in_answer(self) -> None:
        urls, index = _model_citation_urls("没有检索依据的普通回答。")
        self.assertEqual(urls, [])
        self.assertEqual(index, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
