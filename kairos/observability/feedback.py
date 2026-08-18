"""Feedback store (👍/👎) for a closed evaluation loop.

Records each answer Kai Yi sends (keyed by the reply message ids so a Feishu
reaction can be attributed), and stores thumbs-up / thumbs-down so prompt and
retrieval can be improved against real usage.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone

from kairos.infrastructure.settings import DB_PATH

LOG = logging.getLogger("kairos.observability.feedback")

_FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT '',
    chat_id TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    message_ids TEXT NOT NULL DEFAULT '[]',
    likes INTEGER NOT NULL DEFAULT 0,
    dislikes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_inited = False
_lock = threading.Lock()


def init() -> None:
    global _inited
    if _inited:
        return
    with _lock:
        if _inited:
            return
        con = sqlite3.connect(DB_PATH)
        try:
            con.execute(_FEEDBACK_SCHEMA)
            con.execute("CREATE INDEX IF NOT EXISTS idx_feedback_owner ON feedback(owner_id, created_at)")
            con.commit()
            _inited = True
        finally:
            con.close()


def _now() -> str:
    return datetime.now().isoformat()


def record_answer(owner_id: str, chat_id: str, question: str, answer: str, message_ids: list[str]) -> int:
    """Store one delivered answer with its reply message ids for reaction mapping."""
    init()
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute(
                "INSERT INTO feedback (owner_id, chat_id, question, answer, message_ids, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (owner_id, chat_id, question, answer, json.dumps(message_ids or [], ensure_ascii=False), _now(), _now()),
            )
            con.commit()
            return int(cur.lastrowid)
        finally:
            con.close()
    except Exception as exc:
        LOG.warning("record feedback failed: %s", exc)
        return -1


def mark(message_id: str, positive: bool) -> bool:
    """Apply a reaction to the answer row owning this reply message id."""
    init()
    column = "likes" if positive else "dislikes"
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute(
                "UPDATE feedback SET {} = {} + 1, updated_at = ? "
                "WHERE EXISTS (SELECT 1 FROM json_each(feedback.message_ids) WHERE json_each.value = ?)".format(column, column),
                (_now(), message_id),
            )
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()
    except Exception as exc:
        LOG.warning("mark feedback failed: %s", exc)
        return False


def feedback_summary() -> dict:
    init()
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            total = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            likes = con.execute("SELECT COALESCE(SUM(likes),0) FROM feedback").fetchone()[0]
            dislikes = con.execute("SELECT COALESCE(SUM(dislikes),0) FROM feedback").fetchone()[0]
            return {"total": int(total), "likes": int(likes), "dislikes": int(dislikes)}
        finally:
            con.close()
    except Exception as exc:
        LOG.warning("feedback summary failed: %s", exc)
        return {"total": 0, "likes": 0, "dislikes": 0}


def disliked_samples(limit: int = 20) -> list[dict]:
    """Return recent negatively-rated answers for regression review."""
    init()
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT owner_id, question, answer, created_at FROM feedback "
                "WHERE dislikes > 0 ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()
    except Exception as exc:
        LOG.warning("feedback samples failed: %s", exc)
        return []