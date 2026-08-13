"""Memory storage, recall, and durable-memory job orchestration.

Memory is governed in two layers, following the mem0 / Letta pattern:

- Core memory: a small set of long-lived user facts (identity, research
  direction, stable preferences). Always loaded into every request context
  and budgeted to a few entries.
- Archive memory: searchable durable facts (projects, habits, decisions)
  recalled by relevance and pruned by access recency.

Extraction produces explicit operations (add / update / delete / noop) so
the same fact is merged instead of duplicated, and outdated facts can be
replaced or forgotten.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any, TypedDict

import requests
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from kairos.infrastructure.settings import (
    CLAUDE_MEM_PLATFORM,
    CLAUDE_MEM_URL,
    DB_PATH,
    MEMORY_LIMIT,
)
from kairos.infrastructure.llm import build_client_optional, model_name

LOG = logging.getLogger("kairos.memory")

direct_http = requests.Session()
direct_http.trust_env = False
direct_http.headers["User-Agent"] = "Kairós/1.0"
# Module must import without credentials so CI and tests can run keyless.
llm = build_client_optional()

MEMORY_LOCK = threading.Lock()
_memory_checkpointer: SqliteSaver | None = None
MEMORY_GRAPH: Any | None = None

CORE_LIMIT = 8
ARCHIVE_LIMIT = 120
SAFE_CATEGORIES = {"偏好", "研究", "项目", "习惯", "决定", "身份"}
_FORGET_PHRASES = ("不要记住", "别记住", "忘记这条", "忘掉")


def claude_mem_scope(owner_id: str) -> str:
    """Stable per-user scope; never send a Feishu user ID to the memory API."""
    digest = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:24]
    return f"kaiyi-feishu-{digest}"


def memory_safe_text(text: str, limit: int = 2800) -> str:
    """Keep memory useful while excluding credentials, OAuth links and raw files."""
    value = re.sub(r"https?://[^\s]*(?:[?&](?:token|code|access_token|refresh_token|app_secret|api_key)=)[^\s]*", "[redacted]", text, flags=re.IGNORECASE)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[redacted]", value)
    value = re.sub(r"(?i)\b(?:api[_ -]?key|app[_ -]?secret|password|password|token|授权码|口令)\s*[:：]\s*\S+", "[redacted]", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def claude_mem_search(owner_id: str, query: str, limit: int = 5) -> str:
    """Search only the current user's local Claude-Mem scope."""
    if not CLAUDE_MEM_URL:
        return ""
    safe_query = memory_safe_text(query, 500)
    if not safe_query:
        return ""
    try:
        response = direct_http.get(
            f"{CLAUDE_MEM_URL}/api/search/observations",
            params={
                "query": safe_query,
                "project": claude_mem_scope(owner_id),
                "platformSource": CLAUDE_MEM_PLATFORM,
                "limit": max(1, min(limit, 8)),
            },
            timeout=4,
        )
        response.raise_for_status()
        payload = response.json()
        blocks = payload.get("content", []) if isinstance(payload, dict) else []
        text = "\n".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict))
        return memory_safe_text(text, 3600)
    except Exception as exc:
        LOG.info("Claude-Mem search unavailable: %s", type(exc).__name__)
        return ""


def claude_mem_record(owner_id: str, message_id: str, question: str, answer_text: str) -> None:
    """Asynchronously record a redacted conversational observation locally."""
    if not CLAUDE_MEM_URL:
        return
    safe_question = memory_safe_text(question)
    safe_answer = memory_safe_text(answer_text, 3600)
    if not safe_question or not safe_answer:
        return
    scope = claude_mem_scope(owner_id)
    session_id = f"{scope}-conversation"
    try:
        direct_http.post(
            f"{CLAUDE_MEM_URL}/api/sessions/init",
            json={
                "contentSessionId": session_id,
                "project": scope,
                "prompt": safe_question,
                "platformSource": CLAUDE_MEM_PLATFORM,
                "customTitle": "Feishu conversation",
            },
            timeout=8,
        ).raise_for_status()
        direct_http.post(
            f"{CLAUDE_MEM_URL}/api/sessions/observations",
            json={
                "contentSessionId": session_id,
                "tool_name": "FeishuConversation",
                "tool_input": {
                    "user_message": safe_question,
                    "message_ref": hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:12],
                },
                "tool_response": {"assistant_reply": safe_answer},
                "cwd": "/kaiyi",
                "platformSource": CLAUDE_MEM_PLATFORM,
            },
            timeout=8,
        ).raise_for_status()
    except Exception as exc:
        LOG.info("Claude-Mem record unavailable: %s", type(exc).__name__)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS handled_messages "
            "(message_id TEXT PRIMARY KEY, handled_at TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS pending_notes (code TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS long_term_memories ("
            "owner_id TEXT NOT NULL, memory_id TEXT PRIMARY KEY, category TEXT NOT NULL, "
            "content TEXT NOT NULL, source_message_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memories_owner ON long_term_memories(owner_id, updated_at DESC)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS archive_batches ("
            "code TEXT PRIMARY KEY, plan_json TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        # Governance columns added incrementally so existing databases upgrade in place.
        _add_column(con, "long_term_memories", "is_core", "INTEGER NOT NULL DEFAULT 0")
        _add_column(con, "long_term_memories", "last_accessed_at", "TEXT")
        _add_column(con, "long_term_memories", "access_count", "INTEGER NOT NULL DEFAULT 0")
        con.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memories_core ON long_term_memories(owner_id, is_core, updated_at DESC)")


def _add_column(con: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def claim_message(message_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as con:
        try:
            con.execute(
                "INSERT INTO handled_messages(message_id, handled_at) VALUES (?, ?)",
                (message_id, datetime.now(UTC).isoformat()),
            )
            con.execute("DELETE FROM handled_messages WHERE handled_at < datetime('now', '-3 days')")
            return True
        except sqlite3.IntegrityError:
            return False


def memory_owner_id(sender: dict[str, Any], chat_id: str) -> str:
    """Prefer a person scope; fall back to the chat when Feishu omits an ID."""
    sender_id = sender.get("sender_id", {}) if isinstance(sender, dict) else {}
    return str(sender_id.get("open_id") or sender_id.get("user_id") or chat_id)


def _memory_terms(text: str) -> set[str]:
    lowered = text.lower()
    ascii_terms = set(re.findall(r"[a-z0-9_+-]{2,}", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    chinese_bigrams = {chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))}
    return ascii_terms | chinese_bigrams


def core_memories(owner_id: str) -> str:
    """Core layer: always loaded, small and stable user facts."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT category, content FROM long_term_memories "
            "WHERE owner_id = ? AND is_core = 1 ORDER BY updated_at DESC LIMIT ?",
            (owner_id, CORE_LIMIT),
        ).fetchall()
    if not rows:
        return "（暂无核心记忆）"
    return "\n".join(f"- {category}: {content}" for category, content in rows)


def relevant_memories(owner_id: str, question: str) -> str:
    """Archive layer: recall a small, scoped set of durable facts."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT memory_id, category, content, updated_at FROM long_term_memories "
            "WHERE owner_id = ? AND is_core = 0 ORDER BY updated_at DESC LIMIT 120",
            (owner_id,),
        ).fetchall()
    if not rows:
        return "（暂无已保存的长期记忆）"
    query_terms = _memory_terms(question)
    ranked = []
    for memory_id, category, content, updated_at in rows:
        overlap = len(query_terms & _memory_terms(content))
        ranked.append((overlap, updated_at, memory_id, category, content))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item for item in ranked if item[0] > 0][:MEMORY_LIMIT]
    if not selected:
        selected = ranked[: min(3, MEMORY_LIMIT)]
    if selected:
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(DB_PATH) as con:
            con.executemany(
                "UPDATE long_term_memories SET last_accessed_at = ?, access_count = access_count + 1 "
                "WHERE owner_id = ? AND memory_id = ?",
                [(now, owner_id, item[2]) for item in selected],
            )
    return "\n".join(f"- {category}: {content}" for _, _, _, category, content in selected)


def combined_memory_context(owner_id: str, question: str) -> str:
    """Layer core memory, relevant archive facts, and Claude-Mem recall."""
    core = core_memories(owner_id)
    archive = relevant_memories(owner_id, question)
    recalled = claude_mem_search(owner_id, question)
    parts = [f"核心记忆（长期稳定）：\n{core}", f"相关存档记忆：\n{archive}"]
    if recalled:
        parts.append(f"相关历史对话记忆（仅供参考，可能不完整）：\n{recalled}")
    return "\n\n".join(parts)


class LongTermMemoryState(TypedDict, total=False):
    owner_id: str
    message_id: str
    question: str


def _existing_memory_brief(owner_id: str, limit: int = 40) -> str:
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT memory_id, category, content FROM long_term_memories "
            "WHERE owner_id = ? ORDER BY updated_at DESC LIMIT ?",
            (owner_id, limit),
        ).fetchall()
    if not rows:
        return "（无）"
    return "\n".join(f"- {memory_id}: [{category}] {content[:60]}" for memory_id, category, content in rows)


def _parse_ops(raw: str) -> list[dict[str, Any]]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    payload = json.loads(match.group(0) if match else "{}")
    return payload.get("ops", []) if isinstance(payload, dict) else []


def apply_memory_ops(owner_id: str, message_id: str, ops: list[dict[str, Any]]) -> int:
    """Execute add/update/delete/noop operations and enforce memory budgets."""
    now = datetime.now(UTC).isoformat()
    changed = 0
    with sqlite3.connect(DB_PATH) as con:
        for op in ops[:6]:
            if not isinstance(op, dict):
                continue
            action = str(op.get("op", "")).strip().lower()
            content = re.sub(r"\s+", " ", str(op.get("content", ""))).strip()[:160]
            if content and re.search(r"(?:sk-|api[_ -]?key|密码|口令|token|授权码)", content, re.IGNORECASE):
                continue
            category = str(op.get("category", "偏好"))[:16]
            if category not in SAFE_CATEGORIES or (len(content) < 4 and action != "delete"):
                continue
            is_core = 1 if op.get("is_core") in (True, "true", 1, "1") else 0
            target_id = str(op.get("target_id", "")).strip()
            if action == "add" and content:
                memory_id = uuid.uuid4().hex
                con.execute(
                    "INSERT INTO long_term_memories(owner_id,memory_id,category,content,source_message_id,created_at,updated_at,is_core) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (owner_id, memory_id, category, content, message_id, now, now, is_core),
                )
                changed += 1
            elif action == "update" and target_id and content:
                cursor = con.execute(
                    "UPDATE long_term_memories SET category=?, content=?, is_core=?, updated_at=?, source_message_id=? "
                    "WHERE owner_id=? AND memory_id=?",
                    (category, content, is_core, now, message_id, owner_id, target_id),
                )
                changed += cursor.rowcount
            elif action == "delete" and target_id:
                cursor = con.execute(
                    "DELETE FROM long_term_memories WHERE owner_id=? AND memory_id=?",
                    (owner_id, target_id),
                )
                changed += cursor.rowcount
            # update/delete with a missing target fall back to a fresh add if content exists.
            elif action == "update" and content and not target_id:
                memory_id = uuid.uuid4().hex
                con.execute(
                    "INSERT INTO long_term_memories(owner_id,memory_id,category,content,source_message_id,created_at,updated_at,is_core) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (owner_id, memory_id, category, content, message_id, now, now, is_core),
                )
                changed += 1
    enforce_memory_budget(owner_id)
    return changed


def enforce_memory_budget(owner_id: str) -> int:
    """Prune core memories to CORE_LIMIT and archives to ARCHIVE_LIMIT."""
    pruned = 0
    with sqlite3.connect(DB_PATH) as con:
        core_ids = [
            row[0]
            for row in con.execute(
                "SELECT memory_id FROM long_term_memories WHERE owner_id=? AND is_core=1 "
                "ORDER BY updated_at DESC, rowid DESC LIMIT -1 OFFSET ?",
                (owner_id, CORE_LIMIT),
            ).fetchall()
        ]
        if core_ids:
            pruned += con.executemany(
                "DELETE FROM long_term_memories WHERE owner_id=? AND memory_id=?",
                [(owner_id, mid) for mid in core_ids],
            ).rowcount
        keep = [
            row[0]
            for row in con.execute(
                "SELECT memory_id FROM long_term_memories WHERE owner_id=? AND is_core=0 "
                "ORDER BY COALESCE(last_accessed_at, created_at) DESC, rowid DESC LIMIT ?",
                (owner_id, ARCHIVE_LIMIT),
            ).fetchall()
        ]
        if keep:
            placeholders = ",".join("?" for _ in keep)
            pruned += con.execute(
                f"DELETE FROM long_term_memories WHERE owner_id=? AND is_core=0 AND memory_id NOT IN ({placeholders})",
                [owner_id, *keep],
            ).rowcount
    return pruned


def memory_extract_node(state: LongTermMemoryState) -> dict[str, Any]:
    """Distill durable facts into governed add/update/delete/noop operations."""
    question = state.get("question", "").strip()
    if not question or any(phrase in question for phrase in _FORGET_PHRASES):
        return {}
    if llm is None:
        return {}
    existing = _existing_memory_brief(state.get("owner_id", ""))
    prompt = (
        "你是长期记忆管理员。根据用户当前消息与已有记忆，决定记忆操作。\n\n"
        f"用户已有记忆：\n{existing}\n\n"
        f"当前消息：{question[:4000]}\n\n"
        "规则：\n"
        "- add：新的长期事实/偏好/决定，且不与已有记忆重复。\n"
        "- update：当前消息补充或修正某条已有记忆，target_id 指向它，content 写合并后的完整内容。\n"
        "- delete：用户明确要求删除或推翻某条已有记忆，target_id 指向它。\n"
        "- noop：没有值得长期记住的内容（临时聊天、新闻、敏感信息）。\n"
        "category 取值：偏好|研究|项目|习惯|决定|身份。\n"
        'is_core=true 只用于用户长期身份与核心偏好（姓名、方向、重要偏好），数量要少。\n'
        '只输出 JSON：{"ops":[{"op":"add|update|delete|noop","target_id":"","category":"偏好","is_core":false,"content":"不超过120字的事实"}]}'
    )
    try:
        raw = llm.chat.completions.create(
            model=model_name(),
            messages=[
                {"role": "system", "content": "你是隐私优先的长期记忆管理员。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        ).choices[0].message.content or "{}"
        ops = _parse_ops(raw)
    except Exception as exc:
        LOG.warning("long-term memory extraction unavailable: %s", exc)
        return {}
    try:
        changed = apply_memory_ops(state.get("owner_id", ""), state.get("message_id", ""), ops)
        if changed:
            LOG.info("memory ops applied: %d", changed)
    except Exception:
        LOG.exception("apply memory ops failed")
    return {}


def build_memory_graph() -> Any:
    graph = StateGraph(LongTermMemoryState)
    graph.add_node("extract", memory_extract_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", END)
    return graph


def init_memory_runtime() -> None:
    """Use LangGraph's SQLite checkpointer for durable memory jobs."""
    global _memory_checkpointer, MEMORY_GRAPH
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    _memory_checkpointer = SqliteSaver(connection)
    _memory_checkpointer.setup()
    MEMORY_GRAPH = build_memory_graph().compile(checkpointer=_memory_checkpointer)


def remember_async(owner_id: str, message_id: str, question: str) -> None:
    """Checkpoint the extraction job; answering the user never waits for memory writes."""
    if MEMORY_GRAPH is None:
        return
    try:
        with MEMORY_LOCK:
            MEMORY_GRAPH.invoke(
                {"owner_id": owner_id, "message_id": message_id, "question": question},
                {"configurable": {"thread_id": f"memory:{owner_id}:{message_id}"}},
            )
    except Exception:
        LOG.exception("long-term memory job failed")


def persist_memory_async(owner_id: str, message_id: str, question: str, answer_text: str) -> None:
    """Keep the old durable facts and add Claude-Mem observation storage."""
    remember_async(owner_id, message_id, question)
    claude_mem_record(owner_id, message_id, question, answer_text)


def list_memories(owner_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Expose memory rows (used by the web panel and diagnostics)."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT memory_id, category, content, is_core, access_count, created_at, updated_at, last_accessed_at "
            "FROM long_term_memories WHERE owner_id=? ORDER BY is_core DESC, updated_at DESC LIMIT ?",
            (owner_id, limit),
        ).fetchall()
    return [
        {
            "memory_id": r[0],
            "category": r[1],
            "content": r[2],
            "is_core": bool(r[3]),
            "access_count": r[4],
            "created_at": r[5],
            "updated_at": r[6],
            "last_accessed_at": r[7],
        }
        for r in rows
    ]


def all_owners() -> list[str]:
    """Distinct memory owners (used by the web panel)."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT DISTINCT owner_id FROM long_term_memories ORDER BY owner_id").fetchall()
    return [row[0] for row in rows]


def list_all_memories(limit: int = 300) -> list[dict[str, Any]]:
    """All memory rows across owners, newest first."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT owner_id, memory_id, category, content, is_core, access_count, created_at, updated_at "
            "FROM long_term_memories ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "owner_id": r[0],
            "memory_id": r[1],
            "category": r[2],
            "content": r[3],
            "is_core": bool(r[4]),
            "access_count": r[5],
            "created_at": r[6],
            "updated_at": r[7],
        }
        for r in rows
    ]


def count_memories() -> dict[str, int]:
    """Totals for the memory dashboard."""
    with sqlite3.connect(DB_PATH) as con:
        total = con.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0]
        core = con.execute("SELECT COUNT(*) FROM long_term_memories WHERE is_core=1").fetchone()[0]
        owners = con.execute("SELECT COUNT(DISTINCT owner_id) FROM long_term_memories").fetchone()[0]
    return {"total": total, "core": core, "archive": total - core, "owners": owners}


def delete_memory(memory_id: str) -> bool:
    """Delete one memory row; returns whether a row was removed."""
    with sqlite3.connect(DB_PATH) as con:
        cursor = con.execute("DELETE FROM long_term_memories WHERE memory_id = ?", (memory_id,))
        return cursor.rowcount > 0


def forget_memories(owner_id: str) -> int:
    with sqlite3.connect(DB_PATH) as con:
        cursor = con.execute("DELETE FROM long_term_memories WHERE owner_id = ?", (owner_id,))
        return cursor.rowcount
