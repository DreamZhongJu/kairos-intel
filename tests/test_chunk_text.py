"""Tests for Feishu reply chunking (streaming-style delivery)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.channels.feishu import chunk_text  # noqa: E402


class ChunkTextTest(unittest.TestCase):
    def test_short_text_single_chunk(self) -> None:
        self.assertEqual(chunk_text("短回答。"), ["短回答。"])

    def test_empty_text(self) -> None:
        self.assertEqual(chunk_text(""), [""])

    def test_long_text_split_on_paragraphs(self) -> None:
        text = "\n".join(f"这是第 {index} 段的完整内容内容内容内容。" for index in range(30))
        chunks = chunk_text(text, limit=120)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        self.assertEqual("".join(chunks).replace("\n", ""), text.replace("\n", ""))

    def test_overlong_single_paragraph_hard_split(self) -> None:
        text = "字" * 500
        chunks = chunk_text(text, limit=100)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(len(c) == 100 for c in chunks))

    def test_footer_stays_attached_to_last_content(self) -> None:
        body = "\n".join(f"内容内容内容内容内容内容 {index}" for index in range(20))
        footer = "\n——\n📎 参考来源（可点击核对）：\n1. https://example.com/a"
        chunks = chunk_text(f"{body}\n{footer}", limit=100)
        self.assertGreater(len(chunks), 1)
        self.assertIn("https://example.com/a", chunks[-1])

    def test_whitespace_only_lines(self) -> None:
        # Short text is returned untouched; blank lines collapse only when
        # the text is actually split into chunks.
        text = "第一段。\n\n\n\n   \n第二段。"
        self.assertEqual(chunk_text(text), [text])
        long_text = ("长内容" * 60) + "\n\n\n\n   \n" + ("后续内容" * 60)
        chunks = chunk_text(long_text, limit=120)
        self.assertGreater(len(chunks), 1)
        self.assertNotIn("\n\n\n", "\n".join(chunks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
