"""Memory storage, recall, and durable-memory job orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, TypedDict

import requests
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from assistant.infrastructure.settings import (
    CLAUDE_MEM_PLATFORM,
    CLAUDE_MEM_URL,
    DB_PATH,
    DEEPSEEK_KEY,
    MEMORY_LIMIT,
    MODEL,
)

LOG = logging.getLogger("feishu-assistant.memory")

direct_http = requests.Session()
direct_http.trust_env = False
direct_http.headers["User-Agent"] = "FeishuResearchAssistant/1.0"
llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

MEMORY_LOCK = threading.Lock()
_memory_checkpointer: SqliteSaver | None = None
MEMORY_GRAPH: Any | None = None


def claude_mem_scope(owner_id: str) -> str:
    """Stable per-user scope; never send a Feishu user ID to the memory API."""
    digest = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:24]
    return f"kaiyi-feishu-{digest}"


def memory_safe_text(text: str, limit: int = 2800) -> str:
    """Keep memory useful while excluding credentials, OAuth links and raw files."""
    value = re.sub(r"https?://[^\s]*(?:[?&](?:token|code|access_token|refresh_token|app_secret|api_key)=)[^\s]*", "[redacted]", text, flags=re.I)
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


def claim_message(message_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as con:
        try:
            con.execute(
                "INSERT INTO handled_messages(message_id, handled_at) VALUES (?, ?)",
                (message_id, datetime.now(timezone.utc).isoformat()),
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


def relevant_memories(owner_id: str, question: str) -> str:
    """Retrieve a small, scoped set of durable facts without exposing other chats."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT category, content, updated_at FROM long_term_memories "
            "WHERE owner_id = ? ORDER BY updated_at DESC LIMIT 80",
            (owner_id,),
        ).fetchall()
    if not rows:
        return "（暂无已保存的长期记忆）"
    query_terms = _memory_terms(question)
    ranked = []
    for category, content, updated_at in rows:
        overlap = len(query_terms & _memory_terms(content))
        ranked.append((overlap, updated_at, category, content))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item for item in ranked if item[0] > 0][:MEMORY_LIMIT]
    if not selected:
        selected = ranked[:min(3, MEMORY_LIMIT)]
    return "\n".join(f"- {category}: {content}" for _, _, category, content in selected)


def combined_memory_context(owner_id: str, question: str) -> str:
    """Layer concise durable facts with Claude-Mem's richer conversation recall."""
    durable = relevant_memories(owner_id, question)
    recalled = claude_mem_search(owner_id, question)
    if recalled:
        return f"稳定偏好与决定：\n{durable}\n\n相关历史对话记忆（仅供参考，可能不完整）：\n{recalled}"
    return durable


class LongTermMemoryState(TypedDict, total=False):
    owner_id: str
    message_id: str
    question: str


def memory_extract_node(state: LongTermMemoryState) -> dict[str, Any]:
    """Use the model only to distill durable, user-approved-by-context facts."""
    question = state.get("question", "").strip()
    if not question or any(phrase in question for phrase in ("不要记住", "别记住", "忘记这条")):
        return {}
    prompt = (
        "从用户这条消息中提取值得长期记住的信息。只保留：明确的偏好、身份、研究方向、长期项目、稳定习惯、已经做出的决定或明确要求。\n"
        "不要保存临时聊天、一次性新闻、账号密码/API 密钥、精确地址、健康或其他敏感信息；没有合适内容就返回空数组。\n"
        '只输出 JSON：{"memories":[{"category":"偏好|研究|项目|习惯|决定","content":"不超过100字的事实"}]}\n\n'
        f"用户消息：{question[:4000]}"
    )
    try:
        raw = llm.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是隐私优先的长期记忆提取器。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        ).choices[0].message.content or "{}"
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0) if match else "{}")
    except Exception as exc:
        LOG.warning("long-term memory extraction unavailable: %s", exc)
        return {}
    memories = payload.get("memories", []) if isinstance(payload, dict) else []
    now = datetime.now(timezone.utc).isoformat()
    safe_categories = {"偏好", "研究", "项目", "习惯", "决定"}
    with sqlite3.connect(DB_PATH) as con:
        for memory in memories[:5]:
            if not isinstance(memory, dict):
                continue
            category = str(memory.get("category", "偏好"))[:16]
            content = re.sub(r"\s+", " ", str(memory.get("content", ""))).strip()[:100]
            if category not in safe_categories or len(content) < 4:
                continue
            if re.search(r"(?:sk-|api[_ -]?key|密码|口令|token|授权码)", content, re.IGNORECASE):
                continue
            memory_id = hashlib.sha256(f"{state['owner_id']}:{category}:{content}".encode("utf-8")).hexdigest()
            con.execute(
                "INSERT INTO long_term_memories(owner_id,memory_id,category,content,source_message_id,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(memory_id) DO UPDATE SET updated_at=excluded.updated_at, source_message_id=excluded.source_message_id",
                (state["owner_id"], memory_id, category, content, state.get("message_id", ""), now, now),
            )
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


def forget_memories(owner_id: str) -> int:
    with sqlite3.connect(DB_PATH) as con:
        cursor = con.execute("DELETE FROM long_term_memories WHERE owner_id = ?", (owner_id,))
        return cursor.rowcount

