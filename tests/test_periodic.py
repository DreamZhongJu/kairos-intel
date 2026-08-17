"""Tests for the periodic (weekly/monthly) report helpers."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.reports import periodic  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")


class PeriodicHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="periodic_test_"))

    def test_recent_daily_reports_newest_first_and_limited(self) -> None:
        for name in ("structured-2026-08-10.md", "structured-2026-08-11.md", "structured-2026-08-12.md"):
            (self._tmp / name).write_text(f"# 日报 {name}\n内容。\n", encoding="utf-8")
        with patch.object(periodic.settings, "REPORT_DIR", self._tmp):
            bodies = periodic.recent_daily_reports("weekly")
        self.assertEqual(len(bodies), 3)
        self.assertIn("structured-2026-08-12", bodies[0])
        self.assertIn("structured-2026-08-10", bodies[-1])

    def test_recent_daily_reports_caps_at_period_days(self) -> None:
        for index in range(12):
            (self._tmp / f"structured-2026-08-{index + 1:02d}.md").write_text("x\n", encoding="utf-8")
        with patch.object(periodic.settings, "REPORT_DIR", self._tmp):
            weekly = periodic.recent_daily_reports("weekly")
            monthly = periodic.recent_daily_reports("monthly")
        self.assertEqual(len(weekly), 7)
        self.assertEqual(len(monthly), 12)  # only 12 files exist

    def test_request_stats_aggregates_rows_in_window(self) -> None:
        db = self._tmp / "assistant.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE request_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, owner_hash TEXT,"
            "chat_id TEXT, question TEXT, tool_sequence TEXT, answer TEXT, prompt_tokens INTEGER DEFAULT 0,"
            "completion_tokens INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0, latency_ms INTEGER DEFAULT 0,"
            "status TEXT DEFAULT 'ok', error_type TEXT DEFAULT '', created_at TEXT)"
        )
        now = datetime.now(TZ)
        for index in range(5):
            con.execute(
                "INSERT INTO request_logs (question, tool_sequence, total_tokens, prompt_tokens, completion_tokens, status, created_at) VALUES (?,?,?,?,?,?,?)",
                (f"问题 {index % 2}", '["web_search","read_webpage"]', 100, 80, 20, "ok", now.isoformat()),
            )
        con.execute(
            "INSERT INTO request_logs (question, tool_sequence, total_tokens, status, created_at) VALUES (?,?,?,?,?)",
            ("旧问题", "[]", 999, "ok", (now - timedelta(days=60)).isoformat()),
        )
        con.commit()
        con.close()
        with patch.object(periodic.settings, "DB_PATH", db):
            stats = periodic.request_stats("weekly")
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["total_tokens"], 500)
        self.assertGreater(stats["estimated_cost_usd"], 0)
        self.assertIn("问题 1", stats["top_questions"])
        self.assertIn(("web_search", 5), stats["top_tools"])

    def test_period_mapping(self) -> None:
        self.assertEqual(periodic.PERIOD_DAYS["weekly"], 7)
        self.assertEqual(periodic.PERIOD_DAYS["monthly"], 30)
        self.assertIn("每周情报周报", periodic.PERIOD_TITLES["weekly"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
