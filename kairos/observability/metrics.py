"""Durable request observability for the Feishu research assistant.

Records one row per user request (question, tool sequence, token usage,
latency, status) in SQLite and provides aggregation queries for the web
panel. Secrets are redacted before storage; owner identifiers are hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any

from kairos.infrastructure.settings import DB_PATH

# Cost estimation (USD per million tokens). Overridable via environment;
# defaults approximate DeepSeek chat pricing and are clearly labelled as
# estimates on the dashboard.
_INPUT_PRICE_PER_M = float(os.getenv("DEEPSEEK_INPUT_PRICE_PER_M", "0.27"))
_OUTPUT_PRICE_PER_M = float(os.getenv("DEEPSEEK_OUTPUT_PRICE_PER_M", "1.10"))

_LOCK = threading.Lock()
_TABLE = "request_logs"

_SECRET_PATTERNS = [
    re.compile(
        r"https?://[^\s]*(?:[?&](?:token|code|access_token|refresh_token|app_secret|api_key|key)=)[^\s]*",
        re.IGNORECASE,
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(?:api[_ -]?key|app[_ -]?secret|authorization|bearer)\s*[:：=]\s*\S+"),
]


def redact(text: str | None, limit: int = 2000) -> str:
    """Strip credentials and collapse whitespace before persistence."""
    value = text or ""
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[redacted]", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def owner_hash(owner_id: str) -> str:
    return hashlib.sha256((owner_id or "").encode("utf-8")).hexdigest()[:12]


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_metrics_table() -> None:
    with _LOCK:
        with _connect() as con:
            con.execute(
                f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    owner_hash TEXT NOT NULL DEFAULT '',
                    chat_id TEXT NOT NULL DEFAULT '',
                    question TEXT NOT NULL DEFAULT '',
                    context_len INTEGER NOT NULL DEFAULT 0,
                    tool_sequence TEXT NOT NULL DEFAULT '[]',
                    answer TEXT NOT NULL DEFAULT '',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ok',
                    error_type TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )"""
            )
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{_TABLE}_created ON {_TABLE}(created_at DESC)")
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{_TABLE}_status ON {_TABLE}(status)")


def log_request(
    *,
    request_id: str,
    owner_id: str,
    chat_id: str,
    question: str,
    context_len: int,
    tool_sequence: list[str],
    answer: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    status: str = "ok",
    error_type: str = "",
) -> None:
    """Persist one request row. Never raises: observability must not break the bot."""
    try:
        init_metrics_table()
        with _LOCK:
            with _connect() as con:
                con.execute(
                    f"""INSERT INTO {_TABLE} (
                        request_id, owner_hash, chat_id, question, context_len, tool_sequence,
                        answer, prompt_tokens, completion_tokens, total_tokens, latency_ms,
                        status, error_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        request_id,
                        owner_hash(owner_id),
                        redact(chat_id, 120),
                        redact(question, 2000),
                        int(context_len or 0),
                        json.dumps(list(tool_sequence or []), ensure_ascii=False),
                        redact(answer, 4000),
                        int(prompt_tokens or 0),
                        int(completion_tokens or 0),
                        int(prompt_tokens or 0) + int(completion_tokens or 0),
                        int(latency_ms or 0),
                        status,
                        redact(error_type, 80),
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
    except Exception:  # pragma: no cover - fail open
        import logging

        logging.getLogger("kairos.observability").exception("log_request failed")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["tool_sequence"] = json.loads(data.get("tool_sequence") or "[]")
    return data


def summary() -> dict[str, Any]:
    """Aggregate dashboard numbers."""
    init_metrics_table()
    with _connect() as con:
        total = con.execute(f"SELECT COUNT(*) AS c FROM {_TABLE}").fetchone()["c"]
        ok = con.execute(f"SELECT COUNT(*) AS c FROM {_TABLE} WHERE status='ok'").fetchone()["c"]
        errors = total - ok
        row = con.execute(
            f"SELECT AVG(latency_ms) AS lat, SUM(total_tokens) AS tok, "
            f"SUM(prompt_tokens) AS pt, SUM(completion_tokens) AS ct FROM {_TABLE}"
        ).fetchone()
        avg_latency = round(row["lat"] or 0, 1)
        total_tokens = int(row["tok"] or 0)
        prompt_tokens = int(row["pt"] or 0)
        completion_tokens = int(row["ct"] or 0)
        estimated_cost_usd = round(
            prompt_tokens / 1_000_000 * _INPUT_PRICE_PER_M
            + completion_tokens / 1_000_000 * _OUTPUT_PRICE_PER_M,
            4,
        )

        today = datetime.now().strftime("%Y-%m-%d")
        today_count = con.execute(
            f"SELECT COUNT(*) AS c FROM {_TABLE} WHERE substr(created_at,1,10)=?", (today,)
        ).fetchone()["c"]
        today_errors = con.execute(
            f"SELECT COUNT(*) AS c FROM {_TABLE} WHERE substr(created_at,1,10)=? AND status!='ok'",
            (today,),
        ).fetchone()["c"]

        # last 7 days, oldest first
        days: dict[str, int] = {}
        for offset in range(6, -1, -1):
            days[(datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")] = 0
        for row in con.execute(
            f"SELECT substr(created_at,1,10) AS d, COUNT(*) AS c FROM {_TABLE} GROUP BY d"
        ):
            if row["d"] in days:
                days[row["d"]] = row["c"]

    return {
        "total": total,
        "ok": ok,
        "errors": errors,
        "success_rate": round(ok / total, 4) if total else 1.0,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "today_count": today_count,
        "today_errors": today_errors,
        "last_7_days": [{"date": d, "count": c} for d, c in days.items()],
    }


def tool_stats(limit: int = 20000) -> list[dict[str, Any]]:
    """Count tool invocations and per-tool error rates from recent rows."""
    counts: dict[str, dict[str, int]] = {}
    with _connect() as con:
        rows = con.execute(f"SELECT tool_sequence, status FROM {_TABLE} ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            for name in json.loads(row["tool_sequence"] or "[]"):
                item = counts.setdefault(name, {"calls": 0, "errors": 0})
                item["calls"] += 1
                if row["status"] != "ok":
                    item["errors"] += 1
    ranked = sorted(
        ({"tool": k, **v} for k, v in counts.items()),
        key=lambda x: x["calls"],
        reverse=True,
    )
    return ranked[:20]


def list_logs(
    page: int = 1,
    page_size: int = 20,
    status: str = "",
    tool: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """Return (rows, total_count) with optional status/tool filters."""
    init_metrics_table()
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    where: list[str] = []
    params: list[Any] = []
    if status in {"ok", "error"}:
        where.append("status = ?")
        params.append(status)
    if tool:
        where.append("tool_sequence LIKE ?")
        params.append(f"%{tool}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _connect() as con:
        total = con.execute(f"SELECT COUNT(*) AS c FROM {_TABLE} {clause}", params).fetchone()["c"]
        rows = con.execute(
            f"SELECT * FROM {_TABLE} {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, page_size, (page - 1) * page_size],
        ).fetchall()
    return [_row_to_dict(r) for r in rows], total


def get_log(log_id: int) -> dict[str, Any] | None:
    init_metrics_table()
    with _connect() as con:
        row = con.execute(f"SELECT * FROM {_TABLE} WHERE id=?", (int(log_id),)).fetchone()
    return _row_to_dict(row) if row else None
