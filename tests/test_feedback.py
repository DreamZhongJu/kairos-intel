"""Tests for the feedback store (👍/👎 capture)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kairos.observability import feedback  # noqa: E402


class FeedbackStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="feedback_test_"))
        self._patch = patch.object(feedback, "DB_PATH", self._tmp / "assistant.db")
        self._patch.start()
        feedback._inited = False
        feedback.init()

    def tearDown(self) -> None:
        self._patch.stop()

    def test_record_answer_and_summary(self) -> None:
        rid = feedback.record_answer("u1", "chat1", "问题", "回答文本", ["om_a1", "om_a2"])
        self.assertGreater(rid, 0)
        self.assertEqual(feedback.feedback_summary()["total"], 1)
        self.assertEqual(feedback.feedback_summary()["likes"], 0)

    def test_mark_thumbs_up_on_reply_message(self) -> None:
        feedback.record_answer("u1", "c", "q", "a", ["om_a1", "om_a2"])
        ok = feedback.mark("om_a2", True)
        self.assertTrue(ok)
        self.assertEqual(feedback.feedback_summary()["likes"], 1)

    def test_mark_unknown_message_is_noop(self) -> None:
        self.assertFalse(feedback.mark("om_unknown", True))

    def test_disliked_samples(self) -> None:
        feedback.record_answer("u1", "c", "q1", "a1", ["om_1"])
        feedback.mark("om_1", False)
        samples = feedback.disliked_samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["question"], "q1")


if __name__ == "__main__":
    unittest.main(verbosity=2)