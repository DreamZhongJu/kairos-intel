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

_MINING_SQL = f"""
SELECT r1.subject_id AS s, r1.predicate AS p1, r1.object_id AS m,
       r2.predicate AS p2, r2.object_id AS o,
       e1.name AS sn, e1.type AS st, e1.canonical AS sc,
       e2.name AS mn, e2.type AS mt, e2.canonical AS mc,
       e3.name AS oname, e3.type AS ot, e3.canonical AS oc
FROM relations r1
JOIN relations r2 ON r1.object_id = r2.subject_id
JOIN entities e1 ON e1.id = r1.subject_id
JOIN entities e2 ON e2.id = r1.object_id
JOIN entities e3 ON e3.id = r2.object_id
WHERE r1.predicate NOT IN ({_NOISE_SQL})
  AND r2.predicate NOT IN ({_NOISE_SQL})
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
            continue
        chains[hid] = {
            "id": hid,
            "entity_ids": [int(row["s"]), int(row["m"]), int(row["o"])],
            "names": [row["sn"], row["mn"], row["oname"]],
            "types": [row["st"], row["mt"], row["ot"]],
            "canonicals": [row["sc"], row["mc"], row["oc"]],
            "predicates": [p1, p2],
            "start_id": int(row["s"]),
            "end_id": int(row["o"]),
            "confidence": conf,
            "signature": sig,
        }
    out = sorted(chains.values(), key=lambda c: -c["confidence"])
    return out[:limit] if limit else out


# --- persistence into Neo4j ---------------------------------------------------

def save_hyper_edges(chains: list[dict[str, Any]], source: str = "chain_mining") -> int:
    """Idempotent upsert; repeated chains increment n_pos."""
    saved = 0
    with graph_store._driver().session() as s:
        for c in chains:
            s.run(
                """
                MERGE (h:HyperEdge {id: $id})
                ON CREATE SET h.predicates=$preds, h.names=$names, h.types=$types,
                    h.entity_ids=$eids, h.start_id=$sid, h.end_id=$oid,
                    h.confidence=$conf, h.n_pos=1, h.source=$src, h.created_at=$now,
                    h.signature=$sig
                ON MATCH SET h.n_pos = coalesce(h.n_pos,1)+1
                """,
                id=c["id"], preds=c["predicates"], names=c["names"], types=c["types"],
                eids=c["entity_ids"], sid=c["start_id"], oid=c["end_id"],
                conf=c["confidence"], src=source, sig=c["signature"],
                now=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            for i, eid in enumerate(c["entity_ids"]):
                s.run(
                    """
                    MATCH (e:Entity {id:$eid}), (h:HyperEdge {id:$hid})
                    MERGE (e)-[r:IN_HEDGE]->(h) SET r.order=$i
                    """,
                    eid=eid, hid=c["id"], i=i,
                )
            saved += 1
    return saved


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
                   collect(e.id) AS member_ids
            """,
            seed_ids=seed_ids, nbr_ids=nbr_ids,
        ).data()

    scored = []
    for r in rows:
        members = set(r["member_ids"])
        direct = len(members.intersection(seed_ids))
        neighbor = len(members.intersection(nbr_ids))
        score = (direct * 0.6 + min(neighbor, 4) * 0.4) * float(r["conf"] or 0)
        scored.append({
            "names": r["names"], "types": r["types"], "predicates": r["preds"],
            "confidence": r["conf"], "direct": direct, "neighbor": neighbor,
            "score": round(score, 4),
            "chain": " -> ".join(
                f"{r['names'][i]} --[{r['preds'][i]}]--> {r['names'][i + 1]}"
                for i in range(len(r["preds"]))
            ) if r.get("preds") else "",
        })
    scored.sort(key=lambda x: -x["score"])
    hits = [{k: v for k, v in h.items() if not k.startswith("_")} for h in scored[:top_m]]
    return {"query_entities": sorted(seed_names), "hits": hits}
