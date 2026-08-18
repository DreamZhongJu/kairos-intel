"""Tests for the embedding-free knowledge graph engine (SQLite + FTS5)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kairos.knowledge import engine  # noqa: E402


class KnowledgeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="kg_test_"))
        self._kv = patch.object(engine, "DB_PATH", self._tmp / "knowledge.db")
        self._kv.start()
        engine.init()

    def tearDown(self) -> None:
        self._kv.stop()

    def test_chunk_text_splits_long_text(self) -> None:
        text = "\n".join(f"第 {i} 段内容内容内容" for i in range(20))
        chunks = engine.chunk_text(text, limit=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c for c in chunks))

    def test_add_document_and_keyword_search(self) -> None:
        doc_id = engine.add_document("测试文档", "凯伊是一款个人情报助手，支持飞书与知识图谱。")
        self.assertGreaterEqual(doc_id, 1)
        hits = engine.keyword_search("知识图谱")
        self.assertTrue(any(d["doc_id"] == doc_id for d in hits))

    def test_entity_dedupe(self) -> None:
        a = engine.upsert_entity("湖北大学", "机构")
        b = engine.upsert_entity("湖北大学", "机构")
        self.assertEqual(a, b)  # normalized-name dedupe
        c = engine.upsert_entity("湖北 大学", "机构")  # spaces removed
        self.assertEqual(a, c)

    def test_relation_and_graph_query(self) -> None:
        s = engine.upsert_entity("湖北大学", "机构")
        o = engine.upsert_entity("计算机学院", "机构")
        engine.add_relation("湖北大学", "下设", "计算机学院")
        result = engine.graph_query("湖北大学")
        self.assertTrue(result["found"])
        self.assertIn(s, [n["id"] for n in result["nodes"]])
        self.assertTrue(any(e["predicate"] == "下设" for e in result["edges"]))

    def test_graph_query_missing_entity(self) -> None:
        result = engine.graph_query("不存在的实体")
        self.assertFalse(result["found"])

    def test_stats(self) -> None:
        engine.add_document("x", "湖北省武汉市，湖北大学计算机学院。")
        stats = engine.stats()
        self.assertGreaterEqual(stats["documents"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)