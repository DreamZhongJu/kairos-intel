"""Neo4j mirror of the SQLite knowledge graph for AI retrieval.

SQLite stays the source of truth (ingest path unchanged). This module pushes
entities/relations/chunks into Neo4j and answers retrieval queries there with
Cypher: fulltext keyword match, N-hop neighborhood expansion, and optional
vector similarity when embeddings are configured.

Env:
    NEO4J_URI      default bolt://neo4j:7687 (docker network) / bolt://127.0.0.1:7687
    NEO4J_USER     default neo4j
    NEO4J_PASSWORD default kairos-graph
"""

from __future__ import annotations

import os
from typing import Any

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "kairos-graph")

_DRIVER: Any | None = None


def _driver():
    global _DRIVER
    if _DRIVER is None:
        from neo4j import GraphDatabase

        _DRIVER = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    return _DRIVER


def available(timeout: float = 3.0) -> bool:
    """True when Neo4j is reachable; cheap gate for callers."""
    try:
        _driver().verify_connectivity()
        return True
    except Exception:  # noqa: BLE001
        return False


def init_schema() -> None:
    """Idempotent constraints + fulltext indexes."""
    with _driver().session() as s:
        s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        s.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        s.run(
            "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name]"
        )
        s.run(
            "CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.text]"
        )


def sync_all(batch_size: int = 500) -> dict[str, int]:
    """Full mirror from SQLite. Safe to re-run (MERGE is idempotent)."""
    from kairos.knowledge import engine

    engine.init()
    counts = {"entities": 0, "relations": 0, "chunks": 0}
    con = engine._connect()
    try:
        entities = con.execute("SELECT id, name, type, canonical FROM entities").fetchall()
        relations = con.execute(
            "SELECT subject_id, predicate, object_id, confidence FROM relations"
        ).fetchall()
        chunks = con.execute(
            "SELECT c.id, c.doc_id, c.seq, c.text, d.title FROM chunks c JOIN documents d ON c.doc_id=d.id"
        ).fetchall()
    finally:
        con.close()

    def _tx_entities(tx, rows):
        for row in rows:
            tx.run(
                "MERGE (e:Entity {id:$id}) SET e.name=$name, e.type=$type, e.canonical=$canonical",
                id=int(row["id"]), name=row["name"], type=row["type"], canonical=row["canonical"],
            )

    def _tx_chunks(tx, rows):
        for row in rows:
            tx.run(
                "MERGE (d:Document {id:$did}) SET d.title=$title "
                "MERGE (c:Chunk {id:$cid}) SET c.text=$text, c.seq=$seq "
                "MERGE (d)-[:HAS_CHUNK]->(c)",
                did=int(row["doc_id"]), title=row["title"], cid=int(row["id"]),
                text=(row["text"] or "")[:2000], seq=int(row["seq"]),
            )

    with _driver().session() as s:
        for start in range(0, len(entities), batch_size):
            s.execute_write(_tx_entities, [dict(r) for r in entities[start:start + batch_size]])
            counts["entities"] += min(batch_size, len(entities) - start)
        chunk_ids = set()
        for start in range(0, len(chunks), batch_size):
            rows = [dict(r) for r in chunks[start:start + batch_size]]
            s.execute_write(_tx_chunks, rows)
            counts["chunks"] += len(rows)
            chunk_ids.update(int(r["id"]) for r in rows)
        for start in range(0, len(relations), batch_size):
            rows = [dict(r) for r in relations[start:start + batch_size]]
            s.execute_write(_tx_relations, rows)
            counts["relations"] += len(rows)

    # mention edges from relation.source_chunk_id
    con = engine._connect()
    try:
        mentions = con.execute(
            "SELECT DISTINCT source_chunk_id, subject_id FROM relations WHERE source_chunk_id IS NOT NULL "
            "UNION SELECT DISTINCT source_chunk_id, object_id FROM relations WHERE source_chunk_id IS NOT NULL"
        ).fetchall()
    finally:
        con.close()

    def _tx_mentions(tx, rows):
        for row in rows:
            tx.run(
                "MATCH (c:Chunk {id:$cid}), (e:Entity {id:$eid}) MERGE (c)-[:MENTIONS]->(e)",
                cid=int(row["source_chunk_id"]), eid=int(row["subject_id"]),
            )

    with _driver().session() as s:
        for start in range(0, len(mentions), batch_size):
            s.execute_write(_tx_mentions, [dict(r) for r in mentions[start:start + batch_size]])
    return counts


def _tx_relations(tx, rows):
    for row in rows:
        tx.run(
            "MATCH (a:Entity {id:$s}), (b:Entity {id:$o}) "
            "MERGE (a)-[r:REL {predicate:$p}]->(b) SET r.confidence=$conf",
            s=int(row["subject_id"]), o=int(row["object_id"]),
            p=row["predicate"], conf=int(row.get("confidence") or 1),
        )


def upsert_entity(eid: int, name: str, etype: str, canonical: str) -> None:
    with _driver().session() as s:
        s.run(
            "MERGE (e:Entity {id:$id}) SET e.name=$name, e.type=$type, e.canonical=$canonical",
            id=eid, name=name, type=etype, canonical=canonical,
        )


def upsert_relation(sid: int, predicate: str, oid: int, confidence: int = 1) -> None:
    with _driver().session() as s:
        s.run(
            "MATCH (a:Entity {id:$s}), (b:Entity {id:$o}) "
            "MERGE (a)-[r:REL {predicate:$p}]->(b) SET r.confidence=$conf",
            s=sid, o=oid, p=predicate, conf=max(1, min(10, int(confidence))),
        )


def add_chunk(cid: int, doc_id: int, doc_title: str, seq: int, text: str) -> None:
    with _driver().session() as s:
        s.run(
            "MERGE (d:Document {id:$did}) SET d.title=$title "
            "MERGE (c:Chunk {id:$cid}) SET c.text=$text, c.seq=$seq "
            "MERGE (d)-[:HAS_CHUNK]->(c)",
            did=doc_id, title=doc_title, cid=cid, seq=seq, text=text[:2000],
        )


def link_mention(cid: int, eid: int) -> None:
    with _driver().session() as s:
        s.run("MATCH (c:Chunk {id:$cid}), (e:Entity {id:$eid}) MERGE (c)-[:MENTIONS]->(e)", cid=cid, eid=eid)


def search(keyword: str, limit: int = 8) -> list[dict[str, Any]]:
    """Entity lookup by substring (CJK-safe) with one-hop context."""
    cypher = (
        "MATCH (e:Entity) WHERE e.name CONTAINS $kw "
        "WITH e, count { (e)--() } AS deg "
        "OPTIONAL MATCH (e)-[r]-(other) "
        "WITH e, deg, collect({pred: type(r), other: other.name})[0..6] AS ctx "
        "RETURN e.id AS id, e.name AS name, e.type AS type, deg, ctx "
        "ORDER BY deg DESC LIMIT $limit"
    )
    with _driver().session() as s:
        rows = s.run(cypher, kw=keyword.strip(), limit=limit).data()
    for row in rows:
        row["ctx"] = [c for c in row["ctx"] if c.get("other")]
    return rows


def neighborhood(name: str, hops: int = 2, limit: int = 40) -> dict[str, Any]:
    """N-hop subgraph around the closest name match — single Cypher query."""
    q = (
        "MATCH (c:Entity) WHERE c.name CONTAINS $name "
        "WITH c ORDER BY size(c.name) ASC LIMIT 1 "
        "MATCH (c)-[rels*1.." + str(max(1, min(hops, 3))) + "]-(n) "
        "UNWIND rels AS r "
        "WITH DISTINCT startNode(r) AS a, r AS edge, endNode(r) AS b "
        "RETURN a.name AS src, coalesce(edge.predicate, type(edge)) AS pred, b.name AS dst, "
        "count { (a)--() } AS w ORDER BY w DESC LIMIT $limit"
    )
    with _driver().session() as s:
        edges = s.run(q, name=name.strip(), limit=limit).data()
    if not edges:
        return {"found": False}
    center = min((e["src"] for e in edges), key=len)
    return {"found": True, "center": center, "edges": edges}
