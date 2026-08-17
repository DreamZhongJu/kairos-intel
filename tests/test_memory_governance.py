"""Tests for layered memory governance (core vs archive, ops, budgets)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="feishu_mem_test_")
os.environ["ASSISTANT_DATA_DIR"] = _TMP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.memory import store  # noqa: E402

OWNER = "owner-governance-test"


class MemoryGovernanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        store.init_db()
        store.forget_memories(OWNER)

    def tearDown(self) -> None:
        store.forget_memories(OWNER)

    def test_apply_ops_add_update_delete(self) -> None:
        changed = store.apply_memory_ops(
            OWNER,
            "msg-1",
            [
                {"op": "add", "category": "偏好", "is_core": True, "content": "研究方向是知识图谱增强的RAG"},
                {"op": "add", "category": "项目", "is_core": False, "content": "正在开发 Kairós"},
                {"op": "noop", "content": ""},
            ],
        )
        self.assertEqual(changed, 2)
        rows = store.list_memories(OWNER)
        self.assertEqual(len(rows), 2)
        core = [r for r in rows if r["is_core"]]
        self.assertEqual(len(core), 1)
        target = core[0]["memory_id"]

        # update merges into the same row instead of duplicating
        changed = store.apply_memory_ops(
            OWNER,
            "msg-2",
            [{"op": "update", "target_id": target, "category": "研究", "is_core": True, "content": "研究方向是KG增强的RAG与AI Agent"}],
        )
        self.assertEqual(changed, 1)
        rows = store.list_memories(OWNER)
        self.assertEqual(len(rows), 2)
        updated = next(r for r in rows if r["memory_id"] == target)
        self.assertIn("AI Agent", updated["content"])
        self.assertEqual(updated["category"], "研究")

        changed = store.apply_memory_ops(OWNER, "msg-3", [{"op": "delete", "target_id": target}])
        self.assertEqual(changed, 1)
        self.assertEqual(len(store.list_memories(OWNER)), 1)

    def test_sensitive_content_is_rejected(self) -> None:
        store.apply_memory_ops(OWNER, "m", [{"op": "add", "category": "偏好", "content": "密码是 abc123456"}])
        self.assertEqual(store.list_memories(OWNER), [])

    def test_core_budget_prunes_oldest(self) -> None:
        for batch in range(2):
            ops = [
                {"op": "add", "category": "偏好", "is_core": True, "content": f"核心记忆条目 {batch * 5 + i}"}
                for i in range(5)
            ]
            store.apply_memory_ops(OWNER, "m", ops)
        core = [r for r in store.list_memories(OWNER, limit=100) if r["is_core"]]
        self.assertLessEqual(len(core), store.CORE_LIMIT)
        contents = {r["content"] for r in core}
        self.assertNotIn("核心记忆条目 0", contents)
        self.assertIn("核心记忆条目 9", contents)

    def test_archive_budget_prunes_by_last_access(self) -> None:
        ops = [
            {"op": "add", "category": "项目", "content": f"存档条目 {i}"}
            for i in range(store.ARCHIVE_LIMIT + 10)
        ]
        store.apply_memory_ops(OWNER, "m", ops)
        rows = store.list_memories(OWNER, limit=200)
        archives = [r for r in rows if not r["is_core"]]
        self.assertLessEqual(len(archives), store.ARCHIVE_LIMIT)

    def test_recall_updates_access_stats_and_layers_core_first(self) -> None:
        store.apply_memory_ops(
            OWNER,
            "m",
            [
                {"op": "add", "category": "身份", "is_core": True, "content": "用户是苏州大学研究生"},
                {"op": "add", "category": "项目", "content": "研究机器翻译的最新进展"},
                {"op": "add", "category": "项目", "content": "学习做菜"},
            ],
        )
        context = store.combined_memory_context(OWNER, "机器翻译有什么进展？")
        self.assertIn("苏州大学研究生", context)  # core always present
        self.assertIn("机器翻译", context)  # relevant archive present
        self.assertNotIn("做菜", context)  # irrelevant archive not recalled
        rows = store.list_memories(OWNER)
        mt = next(r for r in rows if "机器翻译" in r["content"])
        self.assertGreaterEqual(mt["access_count"], 1)

    def test_extract_node_applies_llm_ops(self) -> None:
        fake_completion = unittest.mock.MagicMock()
        fake_completion.choices[0].message.content = (
            '{"ops":[{"op":"add","category":"研究","is_core":true,"content":"研究方向是KG-RAG"}]}'
        )
        if store.llm is None:
            # CI runs without credentials; give the patched call a surface.
            store.llm = unittest.mock.MagicMock()
        with patch.object(store.llm.chat.completions, "create", return_value=fake_completion):
            store.memory_extract_node({"owner_id": OWNER, "message_id": "m1", "question": "我的研究方向是知识图谱增强的RAG"})
        rows = store.list_memories(OWNER)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_core"])
        self.assertIn("KG-RAG", rows[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
