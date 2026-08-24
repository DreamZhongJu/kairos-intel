"""Tests for the embedding-free knowledge graph engine (SQLite + FTS5)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kairos.knowledge import engine  # noqa: E402
from kairos.knowledge import extract as kg_extract  # noqa: E402
from kairos.knowledge import ingest as kg_ingest  # noqa: E402
from kairos.knowledge import tools as kg_tools  # noqa: E402


class ExtractionTest(unittest.TestCase):
    """Pure, keyless tests for the chunked/gleanings extractor (GraphRAG/LightRAG style)."""

    def test_parse_json_strips_fences(self) -> None:
        data = kg_extract._parse_json('```json\n{"entities": [], "relations": []}\n```')
        self.assertEqual(data, {"entities": [], "relations": []})

    def test_parse_json_tolerates_surrounding_text(self) -> None:
        data = kg_extract._parse_json('前置说明 {"entities": [{"name": "A"}], "relations": []} 后置说明')
        self.assertEqual(data["entities"][0]["name"], "A")

    def test_parse_json_rejects_garbage(self) -> None:
        self.assertEqual(kg_extract._parse_json("完全没有 JSON"), {})

    def test_junk_name_filters_code_identifiers(self) -> None:
        for bad in ["dropType", "occPercent", "poolId", "getIntValue()", "gacha.normal", "put()", "bubble.private"]:
            self.assertTrue(kg_extract._is_junk_name(bad), bad)
        for good in ["项目A", "FastAPI", "Java", "NumberFormatException", "肉鸽玩法", "Spring Boot", "tshark"]:
            self.assertFalse(kg_extract._is_junk_name(good), good)

    def test_merge_dedupes_and_cross_checks_endpoints(self) -> None:
        entities = [
            {"name": "项目A", "type": "项目", "description": "Java 服务端"},
            {"name": "项目 A", "type": "项目", "description": "重复"},
            {"name": "Java", "type": "技术", "description": "语言"},
        ]
        relations = [
            {"subject": "项目A", "predicate": "使用", "object": "Java", "confidence": 9},
            {"subject": "项目 A", "predicate": "使用", "object": "Java", "confidence": 8},  # dup
            {"subject": "项目A", "predicate": "使用", "object": "不存在的实体", "confidence": 5},  # dropped
        ]
        merged = kg_extract._merge(entities, relations)
        self.assertEqual({e["name"] for e in merged["entities"]}, {"项目A", "Java"})
        self.assertEqual(len(merged["relations"]), 1)
        self.assertEqual(merged["relations"][0]["subject"], "项目A")
        self.assertEqual(merged["relations"][0]["confidence"], 9)

    def test_extract_returns_empty_without_llm(self) -> None:
        with patch.object(kg_extract, "llm", None):
            self.assertEqual(kg_extract.extract("随便写点什么"), {"entities": [], "relations": []})

    def test_extract_orchestrates_parallel_chunks(self) -> None:
        text = "\n".join(["甲" * 90, "乙" * 90])  # two long paragraphs -> multiple chunks

        def fake_call(chunk: str, continue_pass: bool = False) -> dict:
            return {"entities": [{"name": "项目A", "type": "项目", "description": "d"}], "relations": []}

        with patch.object(kg_extract, "llm", object()), patch.object(
            kg_extract, "_call_extract", side_effect=fake_call
        ) as mock_call:
            result = kg_extract.extract(text, chunk_limit=60)
        self.assertEqual(result["entities"], [{"name": "项目A", "type": "项目", "description": "d"}])
        self.assertGreaterEqual(mock_call.call_count, 2)  # every chunk got extracted


class AliasBaseTest(unittest.TestCase):
    """Pure tests for deterministic alias-base resolution."""

    def test_strips_known_suffix(self) -> None:
        self.assertEqual(engine._alias_base("肉鸽玩法"), "肉鸽")
        self.assertEqual(engine._alias_base("肉鸽系统"), "肉鸽")
        self.assertEqual(engine._alias_base("社交系统"), "社交")

    def test_synonym_map(self) -> None:
        self.assertEqual(engine._alias_base("Roguelike"), "肉鸽")
        self.assertEqual(engine._alias_base("roguelike"), "肉鸽")

    def test_leaves_ordinary_names_alone(self) -> None:
        self.assertEqual(engine._alias_base("肉鸽"), "肉鸽")
        self.assertEqual(engine._alias_base("Java"), "java")
        self.assertEqual(engine._alias_base("项目A"), "项目a")

    def test_no_substring_overmerge(self) -> None:
        # Conservative by design: never substring-match, so these stay distinct.
        self.assertNotEqual(engine._alias_base("Java"), engine._alias_base("JavaScript"))
        self.assertNotEqual(engine._alias_base("MySQL"), engine._alias_base("SQL"))


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

    def test_export_graph(self) -> None:
        engine.add_relation("湖北大学", "下设", "计算机学院")
        exported = engine.export_graph()
        self.assertEqual(exported["stats"]["entities"], 2)
        self.assertEqual(exported["stats"]["relations"], 1)
        names = {n["name"] for n in exported["nodes"]}
        self.assertEqual(names, {"湖北大学", "计算机学院"})
        self.assertEqual(exported["edges"][0]["predicate"], "下设")
        # degrees should reflect the single edge
        degrees = {n["name"]: n["degree"] for n in exported["nodes"]}
        self.assertEqual(degrees["湖北大学"], 1)
        self.assertEqual(degrees["计算机学院"], 1)

    def test_export_graph_empty(self) -> None:
        exported = engine.export_graph()
        self.assertEqual(exported["nodes"], [])
        self.assertEqual(exported["edges"], [])
        self.assertEqual(exported["stats"]["entities"], 0)

    def test_upsert_entity_merges_aliases(self) -> None:
        a = engine.upsert_entity("肉鸽", "领域")
        b = engine.upsert_entity("肉鸽玩法", "领域")
        c = engine.upsert_entity("Roguelike", "领域")
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_search_entities_fuzzy(self) -> None:
        engine.upsert_entity("本地服务端", "项目")
        engine.upsert_entity("FastAPI", "技术")
        hits = engine.search_entities("服务端")
        self.assertEqual([h["name"] for h in hits], ["本地服务端"])
        self.assertEqual(engine.search_entities(""), [])

    def test_graph_tool_fuzzy_fallback(self) -> None:
        engine.add_relation("本地服务端", "基于", "Python")
        # Single fuzzy candidate -> auto-resolved and queried.
        out = kg_tools.native_knowledge_graph_query.invoke({"entity": "本地服务"})
        self.assertIn("本地服务端", out)
        self.assertIn("基于", out)
        # Multiple candidates -> the tool lists them instead of guessing.
        engine.add_relation("开源服务端项目", "使用", "Java")
        out2 = kg_tools.native_knowledge_graph_query.invoke({"entity": "服务端"})
        self.assertIn("候选", out2)
        self.assertIn("本地服务端", out2)
        self.assertIn("开源服务端项目", out2)

    def test_upsert_explicit_canonical(self) -> None:
        a = engine.upsert_entity("小明", "人名", canonical="qq:10001")
        b = engine.upsert_entity("明明", "人名", canonical="qq:10001")  # rename, same person
        self.assertEqual(a, b)
        row = engine.find_entity("qq:10001")
        self.assertIsNotNone(row)
        self.assertEqual(row["name"], "明明")  # newest nickname wins
        # Display-name exact match still resolves to the namespaced node.
        self.assertEqual(engine.find_entity("明明")["id"], a)
        g1 = engine.upsert_entity("群830070676", "群组", canonical="group:830070676")
        self.assertEqual(engine.find_entity("group:830070676")["id"], g1)

    def test_add_relation_by_ids(self) -> None:
        s = engine.upsert_entity("小明", "人名", canonical="qq:1")
        o = engine.upsert_entity("群123", "群组", canonical="group:123")
        self.assertTrue(engine.add_relation_by_ids(s, "活跃于", o))
        self.assertFalse(engine.add_relation_by_ids(s, "活跃于", o))  # dup ignored
        self.assertFalse(engine.add_relation_by_ids(s, "自环", s))  # self-loop rejected
        exported = engine.export_graph()
        self.assertEqual(exported["stats"]["relations"], 1)
        self.assertEqual(exported["edges"][0]["predicate"], "活跃于")

    def test_ingest_dedupe_aliases_keeps_namespaced_ids(self) -> None:
        pid = engine.upsert_entity("小明", "人名", canonical="qq:42")
        engine.dedupe_aliases()
        row = engine.find_entity("qq:42")
        self.assertIsNotNone(row)
        self.assertEqual(row["id"], pid)  # namespace survives re-canonicalization

    def test_dedupe_aliases_merges_legacy_rows(self) -> None:
        import sqlite3

        con = sqlite3.connect(str(engine.DB_PATH))
        for name, canon in [("肉鸽", "肉鸽"), ("肉鸽玩法", "肉鸽玩法"), ("Roguelike", "roguelike"), ("战斗系统", "战斗系统")]:
            con.execute(
                "INSERT INTO entities (name, type, canonical, created_at) VALUES (?,?,?,?)",
                (name, "领域", canon, "t"),
            )
        con.commit()
        eid = {r[1]: r[0] for r in con.execute("SELECT id, name FROM entities")}
        for src in ("肉鸽", "肉鸽玩法", "Roguelike"):
            con.execute(
                "INSERT INTO relations (subject_id, predicate, object_id, confidence) VALUES (?,?,?,?)",
                (eid[src], "涉及", eid["战斗系统"], 5),
            )
        con.commit()
        con.close()

        summary = engine.dedupe_aliases()
        self.assertEqual(summary["merged"], 2)
        exported = engine.export_graph()
        self.assertEqual(exported["stats"]["entities"], 2)
        self.assertEqual(exported["stats"]["relations"], 1)
        self.assertEqual({n["name"] for n in exported["nodes"]}, {"肉鸽", "战斗系统"})


class ChatIngestTest(unittest.TestCase):
    def _patch_db(self):
        tmp = Path(tempfile.mkdtemp(prefix="kg_chat_"))
        return patch.object(engine, "DB_PATH", tmp / "knowledge.db")

    def test_render_window(self) -> None:
        text = kg_ingest.render_window(
            [
                {"user_id": "1", "nickname": "小明", "time": "08-23 21:02", "text": "第一行\n第二行"},
                {"user_id": "2", "nickname": "", "time": "", "content": "用 content 字段也行"},
                {"user_id": "3", "text": ""},
            ]
        )
        self.assertIn("[08-23 21:02] 小明(qq:1): 第一行 第二行", text)
        self.assertIn("(qq:2): 用 content 字段也行", text)
        self.assertNotIn("qq:3", text)  # empty message skipped entirely

    def test_ingest_chat_window_end_to_end_without_llm(self) -> None:
        with self._patch_db(), patch.object(kg_extract, "llm", None):
            result = kg_ingest.ingest_chat_window(
                "830070676",
                [
                    {"user_id": "100", "nickname": "小明", "text": "大家好"},
                    {"user_id": "200", "nickname": "小红", "text": "你好呀"},
                    {"user_id": "300", "nickname": "小刚", "text": "签到"},
                    {"user_id": "100", "nickname": "明明", "text": "改名了"},
                ],
            )
            self.assertGreaterEqual(result["persons"], 3)
            self.assertEqual(result["members_linked"], 3)  # distinct persons linked once
            # Deterministic nodes exist with namespaced ids.
            person = engine.find_entity("qq:100")
            group = engine.find_entity("group:830070676")
            self.assertIsNotNone(person)
            self.assertEqual(person["name"], "明明")
            self.assertEqual(group["type"], "群组")
            exported = engine.export_graph()
        predicates = {e["predicate"] for e in exported["edges"]}
        self.assertIn("活跃于", predicates)

    def test_query_knowledge_assembles_answer(self) -> None:
        with self._patch_db():
            engine.init()
            engine.add_document("测试文档", "凯伊是一款自托管情报助手，支持知识图谱检索。")
            engine.add_relation("项目A", "使用", "Java")
            out = kg_ingest.query_knowledge(q="知识图谱", entity="项目A")
            self.assertTrue(out["chunks"])
            self.assertTrue(out["graph"]["found"])
            self.assertIn("【知识片段】", out["answer"])
            self.assertIn("【图谱关联】", out["answer"])
            miss = kg_ingest.query_knowledge(entity="不存在的玩意")
            self.assertFalse(miss["graph"]["found"])


if __name__ == "__main__":
    unittest.main(verbosity=2)