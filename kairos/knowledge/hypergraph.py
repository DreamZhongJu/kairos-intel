"""Reasoning-chain hypergraph layer over the existing KG.

Implements the offline stage of the reasoning-chain-hyperedge framework:
mine multi-hop chains from the SQLite relation store, score them with a
confidence heuristic, and persist them as reusable hyperedges in Neo4j.
The online stage retrieves hyperedges by structural overlap with query
entities (binary-projection PPR-style scoring).

Neo4j model
-----------
(:Entity)-[:IN_HEDGE {order}]->(:HyperEdge {
    id, predicates:[..], names:[..], types:[..],
    start_id, end_id, confidence, n_pos, source, created_at
})

The hyperedge id is a deterministic hash of its full chain signature so that
re-mining the same chain is an idempotent MERGE (n_pos increments instead).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from kairos.knowledge import graph_store

# --- predicate taxonomy ------------------------------------------------------

STRONG_KNOWLEDGE = {
    "属于", "位于", "就读于", "拥有", "毕业于", "工作于", "作者是", "出生于",
    "学习", "购买", "收集", "创作", "导演", "主演", "包含", "成员", "简称",
    "全称", "别名", "发行", "出版", "建立于", "首播",
}
EVENT = {
    "游玩", "参加", "参与", "前往", "观看", "推荐", "讨论", "分享", "发布",
    "评价", "称赞", "批评", "喜欢", "反感", "支持", "质疑", "采用", "采用",
    "游玩于", "组织",
}
NOISE = {
    "提及", "提到", "回复", "艾特", "呼叫", "调侃", "询问", "提问", "引用",
    "评论", "私发消息给", "活跃于", "聊天", "邀请", "求助", "回应", "@",
}


def _pscore(pred: str) -> float:
    if pred in STRONG_KNOWLEDGE:
        return 1.0
    if pred in EVENT:
        return 0.6
    if pred in NOISE:
        return 0.0
    return 0.25  # unknown predicate — mild credit


# Only whitelisted predicates may appear in a mined chain, and only strong
# entity types may act as the middle anchor. This kills the "X 辱骂 Y 学习
# C语言" class of pseudo-inference where two unrelated facts are glued by a
# person node and an interaction verb.
_ANCHOR_TYPES = {"机构", "地点", "事件", "产品", "项目", "技术", "会议", "领域", "作品"}
_MINABLE_PREDS = STRONG_KNOWLEDGE | EVENT


def _type_diversity(types: list[str]) -> float:
    return len(set(types)) / max(1, len(types))


def chain_confidence(p1: str, p2: str, mid_degree: int, types: list[str]) -> float:
    """Heuristic v0 confidence — replaces the LLM scorer of later stages."""
    kscore = (_pscore(p1) + _pscore(p2)) / 2.0
    tdiv = _type_diversity(types)
    hub = 1.0 / (1.0 + pow(max(0, mid_degree), 0.35))  # soft hub penalty
    return round(max(0.0, min(1.0, 0.55 * kscore + 0.25 * tdiv + 0.20 * hub)), 4)


# --- offline: mine chains from SQLite ----------------------------------------

_NOISE_SQL = ",".join(f"'{p}'" for p in NOISE)
_MINABLE_SQL = ",".join(f"'{p}'" for p in _MINABLE_PREDS)
_ANCHOR_SQL = ",".join(f"'{t}'" for t in _ANCHOR_TYPES)

_MINING_SQL = f"""
SELECT r1.subject_id AS s, r1.predicate AS p1, r1.object_id AS m,
       r2.predicate AS p2, r2.object_id AS o,
       r1.valid_from AS v1s, r1.valid_to AS v1e,
       r2.valid_from AS v2s, r2.valid_to AS v2e,
       r1.is_playful AS j1, r2.is_playful AS j2,
       e1.name AS sn, e1.type AS st, e1.canonical AS sc,
       e2.name AS mn, e2.type AS mt, e2.canonical AS mc,
       e3.name AS oname, e3.type AS ot, e3.canonical AS oc
FROM relations r1
JOIN relations r2 ON r1.object_id = r2.subject_id
JOIN entities e1 ON e1.id = r1.subject_id
JOIN entities e2 ON e2.id = r1.object_id
JOIN entities e3 ON e3.id = r2.object_id
WHERE r1.predicate IN ({_MINABLE_SQL})
  AND r2.predicate IN ({_MINABLE_SQL})
  AND e2.type IN ({_ANCHOR_SQL})
  AND e2.canonical NOT LIKE 'group:%'
  AND r1.subject_id != r2.object_id
"""


def mine_chains(min_confidence: float = 0.45, limit: int | None = None) -> list[dict[str, Any]]:
    """Two-hop chains with at least one knowledge/event anchor, scored."""
    from kairos.knowledge import engine

    engine.init()
    con = engine._connect()
    try:
        deg_rows = con.execute(
            "SELECT subject_id AS id FROM relations UNION ALL SELECT object_id FROM relations"
        ).fetchall()
        degrees: dict[int, int] = {}
        for r in deg_rows:
            degrees[int(r["id"])] = degrees.get(int(r["id"]), 0) + 1

        rows = con.execute(_MINING_SQL).fetchall()
    finally:
        con.close()

    chains: dict[str, dict[str, Any]] = {}
    for row in rows:
        p1, p2 = row["p1"], row["p2"]
        if _pscore(p1) == 0.25 and _pscore(p2) == 0.25:
            continue  # no anchor at all
        mid_deg = degrees.get(int(row["m"]), 0)
        conf = chain_confidence(p1, p2, mid_deg, [row["st"], row["mt"], row["ot"]])
        if conf < min_confidence:
            continue
        sig = f"{row['sc']}|{p1}|{row['mc']}|{p2}|{row['oc']}"
        hid = hashlib.sha1(sig.encode()).hexdigest()[:16]
        if hid in chains:
            chains[hid]["n_evidence"] += 1
            continue
        # Chain validity interval = intersection of the two hops' intervals
        # (Zep-style t_valid). Empty bounds mean open-ended.
        since_parts = [v for v in (row["v1s"], row["v2s"]) if v]
        until_parts = [v for v in (row["v1e"], row["v2e"]) if v]
        chain_since = max(since_parts) if since_parts else None
        chain_until = min(until_parts) if until_parts else None
        if chain_since and chain_until and chain_until < chain_since:
            continue  # hops contradict each other in time — not a stable chain
        # Tone: any playful hop marks the whole chain as banter (Zep-style
        # fact flagging) — halve confidence so jokes never outrank facts.
        playful = bool(row["j1"] or row["j2"])
        chains[hid] = {
            "id": hid,
            "entity_ids": [int(row["s"]), int(row["m"]), int(row["o"])],
            "names": [row["sn"], row["mn"], row["oname"]],
            "types": [row["st"], row["mt"], row["ot"]],
            "canonicals": [row["sc"], row["mc"], row["oc"]],
            "predicates": [p1, p2],
            "start_id": int(row["s"]),
            "end_id": int(row["o"]),
            "confidence": round(conf * (0.5 if playful else 1.0), 4),
            "signature": sig,
            "since": chain_since,
            "until": chain_until,
            "playful": playful,
            # how many distinct relation edges back this exact chain — real
            # evidence weight, unlike the old n_pos that merely counted re-mines
            "n_evidence": 1,
        }
    out = sorted(chains.values(), key=lambda c: -c["confidence"])
    return out[:limit] if limit else out


# --- offline: mine event-star (true n-ary) hyperedges -------------------------

# Participant predicates for event stars: same whitelist as chains keeps the
# discipline — interaction verbs (提及/艾特/辱骂…) never make a participation.
_EVENT_STAR_SQL = f"""
SELECT c.id AS cid, c.name AS cname, c.canonical AS cc,
       r.predicate AS p, r.is_playful AS j,
       r.valid_from AS vf, r.valid_to AS vt,
       e.id AS pid, e.name AS pname, e.type AS ptype, e.canonical AS pc
FROM entities c
JOIN relations r ON r.subject_id = c.id OR r.object_id = c.id
JOIN entities e ON e.id = CASE WHEN r.subject_id = c.id
                               THEN r.object_id ELSE r.subject_id END
WHERE c.type = '\u4e8b\u4ef6'
  AND c.canonical NOT LIKE 'group:%'
  AND e.type = '\u4eba\u540d'
  AND e.canonical NOT LIKE 'group:%'
  AND r.predicate IN ({_MINABLE_SQL})
"""


def mine_event_stars(min_participants: int = 3, max_participants: int = 12) -> list[dict[str, Any]]:
    """True n-ary hyperedges: one event center, many person participants.

    A center qualifies when >= min_participants *distinct* persons connect to
    it through whitelisted predicates. This is what two-hop chains cannot
    express — group activities like a KFC dinner or a New-Year party link
    several people at once, and materializing them is the whole point of a
    hypergraph over a plain graph.
    """
    from kairos.knowledge import engine

    engine.init()
    con = engine._connect()
    try:
        rows = con.execute(_EVENT_STAR_SQL).fetchall()
    finally:
        con.close()

    centers: dict[int, dict[int, dict[str, Any]]] = {}
    meta: dict[int, tuple[str, str]] = {}
    for row in rows:
        cid = int(row["cid"])
        meta[cid] = (row["cname"], row["cc"])
        centers.setdefault(cid, {})
        part = centers[cid]
        pid = int(row["pid"])
        cur = part.get(pid)
        if cur is None or _pscore(row["p"]) > _pscore(cur["p"]):
            part[pid] = {
                "p": row["p"], "j": bool(row["j"]),
                "vf": row["vf"], "vt": row["vt"],
                "pname": row["pname"], "ptype": row["ptype"],
            }
        else:
            # keep widest time window across duplicate participant edges
            if row["vf"] and (not cur["vf"] or row["vf"] < cur["vf"]):
                cur["vf"] = row["vf"]
            if row["vt"] and (not cur["vt"] or row["vt"] > cur["vt"]):
                cur["vt"] = row["vt"]

    stars: dict[str, dict[str, Any]] = {}
    for cid, parts in centers.items():
        persons = {pid: info for pid, info in parts.items()}
        if len(persons) < min_participants:
            continue
        ordered = sorted(
            persons.items(),
            key=lambda kv: (-_pscore(kv[1]["p"]), kv[0]),
        )[:max_participants]
        cname, cc = meta[cid]
        preds = [info["p"] for _, info in ordered]
        names = [cname] + [info["pname"] for _, info in ordered]
        types = ["\u4e8b\u4ef6"] + [info["ptype"] for _, info in ordered]
        playful = any(info["j"] for info in persons.values())
        since_parts = [i["vf"] for i in persons.values() if i["vf"]]
        until_parts = [i["vt"] for i in persons.values() if i["vt"]]
        avg_score = sum(_pscore(p) for p in preds) / len(preds)
        breadth = min(len(persons), 10) / 10.0
        tanchor = 1.0 if (since_parts or until_parts) else 0.6
        conf = round(max(0.0, min(1.0, 0.45 * avg_score + 0.35 * breadth + 0.20 * tanchor))
                     * (0.5 if playful else 1.0), 4)
        sig = "star|" + cc + "|" + ",".join(str(pid) for pid, _ in sorted(ordered))
        hid = hashlib.sha1(sig.encode()).hexdigest()[:16]
        stars[hid] = {
            "id": hid,
            "kind": "event_star",
            "entity_ids": [cid] + [pid for pid, _ in ordered],
            "names": names,
            "types": types,
            "canonicals": [cc] + [info.get("pc", "") for _, info in ordered],  # noqa: FURB
            "predicates": preds,
            "start_id": cid,
            "end_id": cid,
            "confidence": conf,
            "signature": sig,
            "since": max(since_parts) if since_parts else None,
            "until": min(until_parts) if until_parts else None,
            "playful": playful,
            "n_evidence": len(parts),
        }
    out = sorted(stars.values(), key=lambda s: -s["confidence"])
    return out


# --- persistence into Neo4j ---------------------------------------------------

def save_hyper_edges(chains: list[dict[str, Any]], source: str = "chain_mining") -> int:
    """Idempotent upsert; repeated chains increment n_pos.

    Each chain is its own short transaction with deadlock retry — the Neo4j
    mirror may be writing concurrently (post-ingest resync).
    """
    import time as _time

    saved = 0
    driver = graph_store._driver()
    for c in chains:
        for attempt in range(4):
            try:
                with driver.session() as s:
                    s.execute_write(lambda tx: _upsert_one(tx, c, source))
                saved += 1
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if ("DeadlockDetected" in msg or "TransientError" in msg) and attempt < 3:
                    _time.sleep(0.5 * (attempt + 1))
                    continue
                raise
    return saved


def _upsert_one(tx, c: dict[str, Any], source: str) -> None:
    nev = int(c.get("n_evidence", 1))
    kind = c.get("kind", "chain")
    tx.run(
        """
        MERGE (h:HyperEdge {id: $id})
        ON CREATE SET h.predicates=$preds, h.names=$names, h.types=$types,
            h.entity_ids=$eids, h.start_id=$sid, h.end_id=$oid,
            h.confidence=$conf, h.n_pos=$nev, h.n_evidence=$nev,
            h.source=$src, h.created_at=$now, h.kind=$kind,
            h.signature=$sig, h.since=$since, h.until=$until, h.playful=$playful
        ON MATCH SET h.n_evidence=$nev,
            h.confidence=$conf, h.kind=$kind,
            h.since=coalesce(h.since,$since),
            h.until=coalesce(h.until,$until),
            h.playful = CASE WHEN $playful THEN true ELSE coalesce(h.playful, false) END
        """,
        id=c["id"], preds=c["predicates"], names=c["names"], types=c["types"],
        eids=c["entity_ids"], sid=c["start_id"], oid=c["end_id"],
        conf=c["confidence"], src=source, sig=c["signature"],
        since=c.get("since"), until=c.get("until"),
        playful=bool(c.get("playful")),
        now=time.strftime("%Y-%m-%d %H:%M:%S"),
        nev=nev,
        kind=kind,
    )
    for i, eid in enumerate(c["entity_ids"]):
        tx.run(
            """
            MATCH (e:Entity {id:$eid}), (h:HyperEdge {id:$hid})
            MERGE (e)-[r:IN_HEDGE]->(h) SET r.order=$i
            """,
            eid=eid, hid=c["id"], i=i,
        )


# --- online retrieval ---------------------------------------------------------

def retrieve(query_entities: list[str], top_m: int = 8) -> dict[str, Any]:
    """Score hyperedges by structural overlap with the query entity set.

    Two-phase indexed lookup: (1) resolve query entities and their one-hop
    neighbors via the unique-id index; (2) pull hyperedges through the
    IN_HEDGE membership edges for those ids and score in Python.
    Millisecond scale at current graph sizes.
    """
    qset = [q.strip() for q in query_entities if q.strip()]
    if not qset:
        return {"query_entities": [], "hits": []}

    with graph_store._driver().session() as s:
        # phase 1: seed ids (exact) + one-hop neighbor ids (bounded)
        seeds = s.run(
            "MATCH (e:Entity) WHERE e.name IN $names "
            "RETURN e.id AS id, e.name AS name LIMIT 20",
            names=qset,
        ).data()
        if not seeds:
            return {"query_entities": qset, "hits": []}
        seed_ids = [r["id"] for r in seeds]
        seed_names = {r["name"] for r in seeds}
        nbrs = s.run(
            "MATCH (q:Entity)-[r]-(nb:Entity) WHERE q.id IN $ids "
            "RETURN nb.id AS id LIMIT 300",
            ids=seed_ids,
        ).data()
        nbr_ids = [r["id"] for r in nbrs]

        # phase 2: hyperedges reachable from either set (indexed id lookups)
        rows = s.run(
            """
            MATCH (e:Entity)-[:IN_HEDGE]->(h:HyperEdge)
            WHERE e.id IN $seed_ids OR e.id IN $nbr_ids
            RETURN h.id AS hid, h.names AS names, h.types AS types,
                   h.predicates AS preds, h.confidence AS conf,
                   h.since AS since, h.until AS until, h.kind AS kind,
                   coalesce(h.playful, false) AS playful,
                   collect(e.id) AS member_ids
            """,
            seed_ids=seed_ids, nbr_ids=nbr_ids,
        ).data()

    today = time.strftime("%Y-%m-%d")
    scored = []
    for r in rows:
        members = set(r["member_ids"])
        direct = len(members.intersection(seed_ids))
        neighbor = len(members.intersection(nbr_ids))
        conf = float(r["conf"] or 0)
        until = r.get("until")
        expired = bool(until and until < today)
        eff_conf = conf * (0.3 if expired else 1.0)
        score = (direct * 0.7 + min(neighbor, 2) * 0.3) * eff_conf
        if direct == 0:
            score *= 0.5  # neighbor-only chains must not outrank direct hits
        kind = r.get("kind") or "chain"
        if kind == "event_star" and r.get("names"):
            center = r["names"][0]
            spokes = [
                f"{r['names'][i + 1]} --[{r['preds'][i]}]--> {center}"
                for i in range(len(r["preds"] or []))
                if i + 1 < len(r["names"])
            ]
            chain = f"({len(spokes)}-ary) " + "; ".join(spokes[:4]) + ("; …" if len(spokes) > 4 else "")
        else:
            chain = " -> ".join(
                f"{r['names'][i]} --[{r['preds'][i]}]--> {r['names'][i + 1]}"
                for i in range(len(r["preds"]))
            ) if r.get("preds") else ""
        scored.append({
            "kind": kind,
            "names": r["names"], "types": r["types"], "predicates": r["preds"],
            "confidence": conf, "direct": direct, "neighbor": neighbor,
            "since": r.get("since"), "until": until, "expired": expired,
            "playful": bool(r.get("playful")),
            "score": round(score, 4),
            "chain": chain,
        })
    scored.sort(key=lambda x: -x["score"])
    hits = [{k: v for k, v in h.items() if not k.startswith("_")} for h in scored[:top_m]]
    return {"query_entities": sorted(seed_names), "hits": hits}
