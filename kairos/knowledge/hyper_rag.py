"""Hyper-RAG query path: natural-language question -> hyperedge evidence.

Closes the loop for AI augmentation: bots send the raw user utterance to
``hyper_query`` and get back linked entities plus formatted reasoning-chain
evidence ready for prompt injection.

Entity linking (no LLM call, millisecond scale):
1. scan every known entity display name as a substring of the question
   (names shorter than 2 chars are skipped — single-char Chinese names like
   「文」/「紫」 false-positive inside ordinary sentences);
2. map each hit through the conservative mention registry so aliases such as
   「甲申嘉平」 resolve onto the same person node;
3. keep at most ``max_entities`` distinct nodes, preferring longer names.
"""

from __future__ import annotations

import time
from typing import Any

from kairos.knowledge import coref, engine, hypergraph

_MIN_NAME_LEN = 2
_MAX_ENTITIES = 3

# Generic nouns the LLM extractor sometimes materialized as entities; they
# must never become link targets ("哪所学校" is not about node 学校).
_STOPWORDS = {
    "学校", "群", "游戏", "朋友", "老师", "同学", "大家", "他们", "她们", "我们",
    "你们", "自己", "人名", "人物", "东西", "地方", "时间", "问题", "内容", "消息",
    "记录", "聊天", "群友", "成员", "用户", "名字", "昵称", "大学", "城市", "国家",
}

_DICT_TTL = 300.0
_dict_cache: list[tuple[str, int]] | None = None
_dict_at = 0.0


def _node_id_by_canonical(canon: str) -> int | None:
    con = engine._connect()
    try:
        row = con.execute("SELECT id FROM entities WHERE canonical=? LIMIT 1", (canon,)).fetchone()
        return int(row["id"]) if row else None
    finally:
        con.close()


def _build_link_dictionary() -> list[tuple[str, int]]:
    """Surface-form -> entity_id pairs for question linking.

    Sources: person-type entity display names, conservative mention aliases,
    and nickname evidence (safe for linking because it maps to whoever typed
    with that nickname). Protected names and stopwords are excluded.
    """
    protected = {
        alias for alias, v in coref._read_json(coref.DATA_DIR / "coref_overrides.json").items()
        if not str(v).strip()
    }
    pairs: dict[str, int] = {}
    con = engine._connect()
    try:
        rows = con.execute(
            "SELECT id, name FROM entities WHERE type='人名' AND length(name)>=?",
            (_MIN_NAME_LEN,),
        ).fetchall()
    finally:
        con.close()
    for r in rows:
        n = r["name"].strip()
        if n and n not in protected and n not in _STOPWORDS:
            pairs.setdefault(n, int(r["id"]))
    for reg in (coref._get_mention_registry(),):
        for surface, canon in reg.items():
            if surface in protected or surface in _STOPWORDS or len(surface) < _MIN_NAME_LEN:
                continue
            nid = _node_id_by_canonical(canon)
            if nid:
                pairs.setdefault(surface, nid)
    invalidate = False
    for surface, canon in coref._get_registry().items():
        if surface in protected or surface in _STOPWORDS or len(surface) < _MIN_NAME_LEN:
            continue
        if surface in pairs:
            continue
        nid = _node_id_by_canonical(canon)
        if nid:
            pairs[surface] = nid
            invalidate = True
    del invalidate
    out = sorted(pairs.items(), key=lambda kv: -len(kv[0]))
    return out


def _get_link_dictionary() -> list[tuple[str, int]]:
    global _dict_cache, _dict_at
    now = time.time()
    if _dict_cache is None or now - _dict_at > _DICT_TTL:
        _dict_cache = _build_link_dictionary()
        _dict_at = now
    return _dict_cache


def link_entities(q: str, max_entities: int = _MAX_ENTITIES) -> list[dict[str, Any]]:
    q = (q or "").strip()
    if not q:
        return []
    by_id: dict[int, dict[str, Any]] = {}
    for surface, eid in _get_link_dictionary():
        if surface in q:
            prev = by_id.get(eid)
            if prev is None or len(surface) > len(prev["name"]):
                by_id[eid] = {"name": surface, "entity_id": eid}
            if len(by_id) >= max_entities * 3:
                break
    linked = sorted(by_id.values(), key=lambda e: -len(e["name"]))[:max_entities]
    return linked


def format_evidence(hits: list[dict[str, Any]]) -> list[str]:
    """Hyperedge chains -> concise Chinese evidence lines for prompt injection.

    Time-aware: validity ranges render inline (「2023~2025」) and expired
    chains get an explicit 「已失效」 marker so the bot won't present stale
    facts as current.
    """
    lines: list[str] = []
    for h in hits:
        names = h.get("names") or []
        preds = h.get("predicates") or []
        if not names or not preds:
            continue
        segs = [names[0]]
        for i, p in enumerate(preds):
            nxt = names[i + 1] if i + 1 < len(names) else "?"
            segs.append(f"{p}{nxt}")
        conf = float(h.get("confidence") or 0)
        line = "".join(segs)
        since, until = h.get("since"), h.get("until")
        if since and until:
            span = f"（{since}~{until}）"
        elif since:
            span = f"（自{since}）"
        elif until:
            span = f"（至{until}）"
        else:
            span = ""
        expired = "【已失效】" if h.get("expired") else ""
        lines.append(f"- {expired}{line}{span}（置信{conf:.2f}）")
    return lines


def _node_name(eid: int) -> str:
    con = engine._connect()
    try:
        row = con.execute("SELECT name FROM entities WHERE id=?", (int(eid),)).fetchone()
        return row["name"] if row else ""
    finally:
        con.close()


def hyper_query(q: str, top_m: int = 5) -> dict[str, Any]:
    linked = link_entities(q)
    if not linked:
        return {"linked": [], "hits": [], "evidence": []}
    # resolve surfaces back to canonical display names for retrieval
    real_names = []
    for e in linked:
        nm = _node_name(e["entity_id"])
        e["resolved"] = nm
        if nm:
            real_names.append(nm)
    hits = hypergraph.retrieve(real_names, top_m=top_m).get("hits", []) if real_names else []
    return {"linked": linked, "hits": hits, "evidence": format_evidence(hits)}
