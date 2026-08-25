"""Group-chat ingestion: turn chat windows into knowledge-graph updates.

The Koishi collector batches whitelisted group messages into windows and POSTs
them to :func:`ingest_chat_window` (via the REST layer). Deterministic
person/group nodes are registered first — keyed by QQ ids so nickname changes
never split a node — then the rendered conversation text goes through the
standard chunked/gleanings extraction pipeline for topics and relations.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from kairos.infrastructure import settings
from kairos.knowledge import engine
from kairos.knowledge import extract as kg_extract
from kairos.knowledge.privacy import pseudonymize_messages

LOG = logging.getLogger("kairos.knowledge.ingest")


def _today() -> str:
    return date.today().isoformat()


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
    provider: str = "",
) -> dict[str, Any]:
    """Ingest one window of group-chat messages; returns counts and stats."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    channel_id = str(channel_id or "").strip()
    if not channel_id:
        raise ValueError("channel_id required")
    if len(msgs) < 2:
        raise ValueError("需要至少 2 条消息组成一个窗口")

    # Privacy gate: non-exempt channels get nicknames pseudonymized, mentions
    # rewritten and identity literals masked BEFORE storage or any LLM call.
    msgs = pseudonymize_messages(channel_id, msgs, settings.PRIVACY_EXEMPT_GROUPS)

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
    extracted = kg_extract.extract(window_text, provider=provider)
    entities = extracted.get("entities", [])
    relations = extracted.get("relations", [])
    # Map per-chunk provenance tags to real chunk ids for traceability.
    _con = engine._connect()
    try:
        chunk_ids = [r[0] for r in _con.execute(
            "SELECT id FROM chunks WHERE doc_id=? ORDER BY seq", (doc_id,)
        ).fetchall()]
    finally:
        _con.close()
    for ent in entities:
        try:
            engine.upsert_entity(ent["name"], ent["type"])
        except Exception:  # noqa: BLE001
            pass
    for rel in relations:
        try:
            ci = int(rel.pop("_ci", 0) or 0)
            chunk_id = chunk_ids[ci] if 0 <= ci < len(chunk_ids) else None
            engine.add_relation(
                rel["subject"],
                rel["predicate"],
                rel["object"],
                chunk_id=chunk_id,
                confidence=int(rel.get("confidence", 1)),
                valid_from=rel.get("time_start"),
                valid_to=rel.get("time_end"),
                playful=bool(rel.get("playful")),
            )
        except Exception:  # noqa: BLE001
            pass

    # 4) Deterministic membership edges (zero LLM cost, never missed).
    members_linked = 0
    for pid in sorted(set(persons.values())):
        try:
            # Membership is observed "now" and stays open-ended.
            if engine.add_relation_by_ids(pid, "活跃于", group_id, valid_from=_today()):
                members_linked += 1
        except Exception:  # noqa: BLE001
            pass

    # Mirror the new data into Neo4j (debounced; no-op when unavailable).
    try:
        from kairos.knowledge import graph_sync

        graph_sync.schedule_resync()
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


def expand_entity(name: str, limit: int = 30) -> dict[str, Any]:
    """All direct (non-noise) relations of one entity, best-confidence first.

    The primitive behind iterative deep-dive (ToG-style): after a first
    search round, the model picks an entity from the evidence and calls this
    to walk one more hop, following whatever looks promising.
    """
    name = (name or "").strip()
    limit = max(1, min(int(limit or 30), 60))
    if not name:
        return {"entity": name, "found": False, "relations": []}
    engine.init()
    con = engine._connect()
    try:
        row = con.execute(
            "SELECT id, name FROM entities WHERE name=? OR canonical=? OR canonical=? LIMIT 1",
            (name, name.lower(), engine._alias_base(name)),
        ).fetchone()
        if not row:
            cands = con.execute(
                "SELECT name FROM entities WHERE name LIKE ? LIMIT 8", (f"%{name}%",)
            ).fetchall()
            return {"entity": name, "found": False,
                    "candidates": [c["name"] for c in cands]}
        eid, resolved = int(row["id"]), row["name"]
        rows = con.execute(
            """
            SELECT r.predicate p, r.confidence c, r.valid_from vf, r.valid_to vt,
                   r.is_playful jp, e.id oid, e.name oname, e.type otype,
                   CASE WHEN r.subject_id=:eid THEN 1 ELSE 0 END AS outward
            FROM relations r JOIN entities e ON e.id = CASE WHEN r.subject_id=:eid THEN r.object_id ELSE r.subject_id END
            WHERE (r.subject_id=:eid OR r.object_id=:eid) AND r.predicate NOT IN
                  ('提及','提到','回复','询问','艾特','呼叫','调侃','评论','引用','@')
            ORDER BY r.confidence DESC LIMIT :lim
            """,
            {"eid": eid, "lim": limit},
        ).fetchall()
    finally:
        con.close()
    relations = []
    for r in rows:
        arrow = f"{r['p']}{r['oname']}" if r["outward"] else f"\u53cd\u5411:{r['p']}{r['oname']}"
        meta = []
        if r["vf"] or r["vt"]:
            meta.append(f"{r['vf'] or '?'}~{r['vt'] or '\u4eca'}")
        if r["jp"]:
            meta.append("\u73a9\u7b11")
        suffix = f" \uff08{'/'.join(meta)}\uff09" if meta else ""
        relations.append(f"- {arrow}{suffix} \uff08\u7f6e\u4fe1{int(r['c'])}\uff09")
    return {"entity": resolved, "found": True, "relations": relations}


def query_knowledge(
    q: str = "", entity: str = "", expand: str = "", limit: int = 6
) -> dict[str, Any]:
    """Retrieve knowledge for a question / entity / deep-dive expansion.

    ``expand`` (or ``q``/``entity`` prefixed with ``expand:``) switches to the
    iterative deep-dive primitive: return every direct relation of one entity
    so the caller can walk the graph hop by hop.
    """
    """Lightweight retrieval for bot Q&A: keyword chunks plus a graph subgraph.

    Shared by the REST ``/api/knowledge/query`` endpoint and the MCP surface so
    both stay consistent. Pure SQLite — millisecond latency, no LLM call.
    """
    q = (q or "").strip()
    entity = (entity or "").strip()
    limit = max(1, min(int(limit or 6), 12))
    engine.init()

    # Iterative deep-dive primitive: expand one entity's direct relations.
    if expand:
        return expand_entity(expand, limit=30)
    if entity.startswith("expand:"):
        return expand_entity(entity[len("expand:"):], limit=30)
    if q.startswith("expand:"):
        return expand_entity(q[len("expand:"):], limit=30)

    # Neo4j mirror first (richer multi-hop context); SQLite fallback below.
    neo4j_used = False
    try:
        from kairos.knowledge import graph_store

        if graph_store.available():
            neo4j_used = True
    except Exception:  # noqa: BLE001
        neo4j_used = False

    chunks: list[dict[str, Any]] = []
    if q:
        for hit in engine.keyword_search(q, limit=limit):
            chunks.append({"title": hit.get("title") or "未命名文档", "text": (hit.get("text") or "")[:400]})

    # Hyper-RAG: link entities from the raw question and pull reasoning-chain
    # evidence — runs even when the caller did not supply an entity.
    hyper: dict[str, Any] = {"linked": [], "evidence": []}
    if q:
        try:
            from kairos.knowledge import hyper_rag

            hyper = hyper_rag.hyper_query(q)
        except Exception:  # noqa: BLE001
            hyper = {"linked": [], "hits": [], "evidence": []}

    graph: dict[str, Any] = {"entity": entity, "found": False, "lines": []}
    if entity and neo4j_used:
        try:
            from kairos.knowledge import graph_store

            nb = graph_store.neighborhood(entity, hops=2, limit=40)
            if nb.get("found"):
                graph["found"] = True
                graph["lines"] = [f"- {e['src']} --[{e['pred']}]--> {e['dst']}" for e in nb.get("edges", [])]
        except Exception:  # noqa: BLE001
            pass
    if entity and not graph.get("found"):
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
    if hyper.get("evidence"):
        linked_names = "、".join(e["name"] for e in hyper.get("linked", []))
        parts.append(f"【推理链证据】（关联：{linked_names}）")
        parts.extend(hyper["evidence"])
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

    return {"query": q, "entity": entity, "chunks": chunks, "graph": graph,
            "hyper": {"linked": hyper.get("linked", []), "hits": hyper.get("hits", []),
                      "evidence": hyper.get("evidence", [])},
            "answer": "\n".join(parts)}
