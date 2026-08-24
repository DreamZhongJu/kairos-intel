"""Group-chat ingestion: turn chat windows into knowledge-graph updates.

The Koishi collector batches whitelisted group messages into windows and POSTs
them to :func:`ingest_chat_window` (via the REST layer). Deterministic
person/group nodes are registered first — keyed by QQ ids so nickname changes
never split a node — then the rendered conversation text goes through the
standard chunked/gleanings extraction pipeline for topics and relations.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.knowledge import engine
from kairos.knowledge import extract as kg_extract

LOG = logging.getLogger("kairos.knowledge.ingest")


def _msg_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("content") or "").strip().replace("\n", " ")


def render_window(messages: list[dict[str, Any]]) -> str:
    """Render a message window as speaker-labelled lines for extraction."""
    lines: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        text = _msg_text(item)
        if not text:
            continue
        uid = str(item.get("user_id") or "").strip()
        nick = str(item.get("nickname") or "").strip() or uid
        stamp = str(item.get("time") or "").strip()
        who = f"{nick}(qq:{uid})" if uid else nick
        prefix = f"[{stamp}] " if stamp else ""
        lines.append(f"{prefix}{who}: {text[:500]}")
    return "\n".join(lines)


def ingest_chat_window(
    channel_id: str,
    messages: list[dict[str, Any]],
    source: str = "qq",
    title: str = "",
) -> dict[str, Any]:
    """Ingest one window of group-chat messages; returns counts and stats."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    channel_id = str(channel_id or "").strip()
    if not channel_id:
        raise ValueError("channel_id required")
    if len(msgs) < 2:
        raise ValueError("需要至少 2 条消息组成一个窗口")

    engine.init()

    # 1) Deterministic group node.
    group_id = engine.upsert_entity(f"群{channel_id}", "群组", canonical=f"group:{channel_id}")

    # 2) Deterministic person nodes (QQ id is the identity; nickname is a label).
    persons: dict[str, int] = {}
    for item in msgs:
        uid = str(item.get("user_id") or "").strip()
        if not uid:
            continue
        nick = str(item.get("nickname") or "").strip() or uid
        persons[uid] = engine.upsert_entity(nick, "人名", canonical=f"qq:{uid}")

    # 3) Render + store the document, then run LLM extraction on it.
    window_text = render_window(msgs)
    doc_title = title or f"{source}群{channel_id} 聊天记录"
    doc_id = engine.add_document(doc_title, window_text, kind=source or "chat", source=f"{source}:{channel_id}")
    extracted = kg_extract.extract(window_text)
    entities = extracted.get("entities", [])
    relations = extracted.get("relations", [])
    for ent in entities:
        try:
            engine.upsert_entity(ent["name"], ent["type"])
        except Exception:  # noqa: BLE001
            pass
    for rel in relations:
        try:
            engine.add_relation(
                rel["subject"], rel["predicate"], rel["object"], confidence=int(rel.get("confidence", 1))
            )
        except Exception:  # noqa: BLE001
            pass

    # 4) Deterministic membership edges (zero LLM cost, never missed).
    members_linked = 0
    for pid in sorted(set(persons.values())):
        try:
            if engine.add_relation_by_ids(pid, "活跃于", group_id):
                members_linked += 1
        except Exception:  # noqa: BLE001
            pass

    return {
        "doc_id": doc_id,
        "group": {"id": group_id, "canonical": f"group:{channel_id}"},
        "persons": len(persons),
        "entities": len(entities),
        "relations": len(relations),
        "members_linked": members_linked,
        "stats": engine.stats(),
    }


def query_knowledge(q: str = "", entity: str = "", limit: int = 6) -> dict[str, Any]:
    """Lightweight retrieval for bot Q&A: keyword chunks plus a graph subgraph.

    Shared by the REST ``/api/knowledge/query`` endpoint and the MCP surface so
    both stay consistent. Pure SQLite — millisecond latency, no LLM call.
    """
    q = (q or "").strip()
    entity = (entity or "").strip()
    limit = max(1, min(int(limit or 6), 12))
    engine.init()

    chunks: list[dict[str, Any]] = []
    if q:
        for hit in engine.keyword_search(q, limit=limit):
            chunks.append({"title": hit.get("title") or "未命名文档", "text": (hit.get("text") or "")[:400]})

    graph: dict[str, Any] = {"entity": entity, "found": False, "lines": []}
    if entity:
        result = engine.graph_query(entity)
        if not result.get("found"):
            candidates = engine.search_entities(entity)
            if len(candidates) == 1:
                result = engine.graph_query(candidates[0]["name"])
            elif candidates:
                graph["candidates"] = [c["name"] for c in candidates[:8]]
        nodes = {n["id"]: n["name"] for n in result.get("nodes", [])}
        lines = []
        for edge in result.get("edges", []):
            s = nodes.get(edge.get("subject"), str(edge.get("subject")))
            o = nodes.get(edge.get("object"), str(edge.get("object")))
            lines.append(f"- {s} --[{edge.get('predicate', '相关')}]--> {o}")
        graph = {"entity": entity, "found": bool(result.get("found")), "lines": lines[:40],
                 **({"candidates": graph["candidates"]} if graph.get("candidates") else {})}

    # Assemble a plain-text answer block for bots that just want to reply.
    parts: list[str] = []
    if chunks:
        parts.append("【知识片段】")
        for c in chunks[:6]:
            parts.append(f"[{c['title']}]\n{c['text']}")
    if graph.get("found") and graph["lines"]:
        parts.append(f"【图谱关联】「{entity}」（1-2 跳）")
        parts.extend(graph["lines"])
    if graph.get("candidates"):
        parts.append(f"未找到「{entity}」，最接近的实体：" + "、".join(graph["candidates"]))
    if not parts:
        parts.append("知识库中没有匹配的内容。")

    return {"query": q, "entity": entity, "chunks": chunks, "graph": graph, "answer": "\n".join(parts)}
