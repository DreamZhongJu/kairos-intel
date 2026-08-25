"""Knowledge graph engine: SQLite + FTS5 keyword + graph traversal.

No embedding model is required — retrieval relies on SQLite FTS5 (trigram
tokenizer for CJK substring matching) plus entity/relation graph traversal,
so it runs fully offline on a self-hosted box.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from kairos.infrastructure.settings import DATA_DIR

LOG = logging.getLogger("kairos.knowledge.graph")

DB_PATH = DATA_DIR / "knowledge.db"
_LOCK = threading.Lock()

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL DEFAULT '',
        kind TEXT NOT NULL DEFAULT 'note',
        source TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        seq INTEGER NOT NULL DEFAULT 0,
        text TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'entity',
        canonical TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical)",
    """
    CREATE TABLE IF NOT EXISTS relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER NOT NULL,
        predicate TEXT NOT NULL,
        object_id INTEGER NOT NULL,
        source_chunk_id INTEGER,
        confidence INTEGER DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id)",
]


def _fts_supported() -> bool:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        con.close()
        return True
    except sqlite3.OperationalError:
        return False


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init() -> None:
    with _LOCK:
        con = _connect()
        try:
            for statement in SCHEMA:
                con.execute(statement)
            if _fts_supported():
                con.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='trigram')"
                )
            con.commit()
        finally:
            con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunk_text(text: str, limit: int = 900) -> list[str]:
    """Split text into chunks on paragraph boundaries (plain, embedding-free)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        piece = paragraph.strip()
        if not piece:
            continue
        if current and len(current) + len(piece) + 1 > limit:
            chunks.append(current)
            current = ""
        if current:
            current = f"{current}\n{piece}"
        else:
            current = piece
            while len(current) > limit:
                chunks.append(current[:limit])
                current = current[limit:]
    if current:
        chunks.append(current)
    return chunks or [text]


def _normalize(name: str) -> str:
    name = (name or "").strip()
    name = name.replace("\u3000", " ").replace(" ", "")
    width = {"（": "(", "）": ")", "，": ",", "。": "."}
    return "".join(width.get(ch, ch) for ch in name).lower()


# Functional suffixes stripped when resolving an entity's alias base, so
# "肉鸽玩法" / "肉鸽系统" collapse onto "肉鸽". Deliberately conservative:
# we only strip a small whitelist of modifiers, never do arbitrary substring
# matching (which would wrongly merge e.g. Java/JavaScript or SQL/MySQL).
_KNOWN_SUFFIXES = ("系统", "玩法", "模式", "协议", "模块", "机制", "功能")

# Explicit cross-language / curated synonyms (normalized key -> canonical base).
_SYNONYM_ALIASES = {"roguelike": "肉鸽"}


def _alias_base(name: str) -> str:
    """Map a name to its canonical alias base for entity dedup."""
    base = _normalize(name)
    base = _SYNONYM_ALIASES.get(base, base)
    for suffix in _KNOWN_SUFFIXES:
        if base.endswith(suffix) and len(base) - len(suffix) >= 2:
            base = base[: -len(suffix)]
            break
    return base


def add_document(title: str, text: str, kind: str = "note", source: str = "") -> int:
    """Store a document's chunks and return the new document id."""
    chunks = chunk_text(text)
    con = _connect()
    try:
        cur = con.execute(
            "INSERT INTO documents (title, kind, source, created_at) VALUES (?,?,?,?)",
            (title, kind, source, _now()),
        )
        doc_id = cur.lastrowid
        for seq, chunk in enumerate(chunks):
            cur = con.execute(
                "INSERT INTO chunks (doc_id, seq, text) VALUES (?,?,?)",
                (doc_id, seq, chunk),
            )
            chunk_id = cur.lastrowid
            try:
                con.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?,?)", (chunk_id, chunk))
            except sqlite3.OperationalError:
                pass  # FTS disabled or not supported
        con.commit()
        return int(doc_id)
    finally:
        con.close()


def upsert_entity(name: str, etype: str = "entity", canonical: str | None = None) -> int:
    """Insert an entity deduplicated by alias-aware or explicit canonical id.

    ``canonical`` pins a deterministic identity (e.g. "qq:123456",
    "group:830070676") so display-name changes never split the node; on an
    explicit-canonical collision the newest name wins (a rename).

    Person mentions without an explicit canonical are routed through the
    coreference registry: a known alias attaches to its existing qq:* node
    instead of forking a bare-nickname duplicate.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("entity name required")
    etype = etype or "entity"
    explicit = (canonical or "").strip()
    key = _normalize(explicit) if explicit else _alias_base(name)
    con = _connect()
    try:
        row = con.execute("SELECT id FROM entities WHERE canonical=? LIMIT 1", (key,)).fetchone()
        if row:
            rid = int(row["id"])
            if explicit:
                con.execute("UPDATE entities SET name=?, type=? WHERE id=?", (name, etype, rid))
                con.commit()
            return rid
        resolved_canon = None
        if not explicit and etype == "人名":
            try:
                from kairos.knowledge import coref

                resolved_canon = coref.resolve_person(name)
            except Exception:  # noqa: BLE001
                resolved_canon = None
        if resolved_canon:
            row = con.execute(
                "SELECT id FROM entities WHERE canonical=? LIMIT 1", (resolved_canon,)
            ).fetchone()
            if row:
                return int(row["id"])  # keep existing display name; just attach
        if not explicit:
            row = con.execute("SELECT id FROM entities WHERE name=? LIMIT 1", (name,)).fetchone()
            if row:
                return int(row["id"])
        final_key = resolved_canon or key
        cur = con.execute(
            "INSERT INTO entities (name, type, canonical, created_at) VALUES (?,?,?,?)",
            (name, etype, final_key, _now()),
        )
        con.commit()
        return int(cur.lastrowid)
    finally:
        con.close()


def add_relation_by_ids(subject_id: int, predicate: str, object_id: int, confidence: int = 1) -> bool:
    """Insert a relation between two known entity ids (deterministic edges)."""
    predicate = (predicate or "").strip()[:40]
    sid, oid = int(subject_id), int(object_id)
    if not predicate or sid == oid:
        return False
    con = _connect()
    try:
        exists = con.execute(
            "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
            (sid, predicate, oid),
        ).fetchone()
        if exists:
            return False
        con.execute(
            "INSERT INTO relations (subject_id, predicate, object_id, confidence) VALUES (?,?,?,?)",
            (sid, predicate, oid, max(1, min(10, int(confidence)))),
        )
        con.commit()
        return True
    finally:
        con.close()


def add_relation(subject: str, predicate: str, obj: str, chunk_id: int | None = None, confidence: int = 1) -> None:
    subject_id = upsert_entity(subject)
    object_id = upsert_entity(obj)
    con = _connect()
    try:
        exists = con.execute(
            "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
            (subject_id, predicate, object_id),
        ).fetchone()
        if exists:
            return
        con.execute(
            "INSERT INTO relations (subject_id, predicate, object_id, source_chunk_id, confidence) VALUES (?,?,?,?,?)",
            (subject_id, predicate, object_id, chunk_id, confidence),
        )
        con.commit()
    finally:
        con.close()


def find_entity(name: str) -> dict | None:
    key = _alias_base(name)
    plain = (name or "").strip()
    con = _connect()
    try:
        row = con.execute("SELECT * FROM entities WHERE canonical=? LIMIT 1", (key,)).fetchone()
        if not row and plain:
            # Exact display-name fallback: covers deterministic nodes whose
            # canonical is a namespaced id (qq:/group:) rather than the name.
            row = con.execute("SELECT * FROM entities WHERE name=? LIMIT 1", (plain,)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def search_entities(keyword: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fuzzy-match entity names for when an exact graph lookup misses."""
    kw = _normalize(keyword)
    if not kw:
        return []
    con = _connect()
    try:
        rows = con.execute(
            "SELECT id, name, type FROM entities WHERE canonical LIKE ? OR name LIKE ? ORDER BY id LIMIT ?",
            (f"%{kw}%", f"%{kw}%", int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def graph_query(entity: str, depth: int = 2, limit: int = 40) -> dict[str, Any]:
    """Return the entity's 1-2 hop neighborhood as nodes + edges."""
    entity_row = find_entity(entity)
    if not entity_row:
        return {"entity": entity, "found": False, "nodes": [], "edges": []}
    eid = int(entity_row["id"])
    con = _connect()
    try:
        edges_raw = con.execute(
            "SELECT subject_id, predicate, object_id FROM relations WHERE subject_id=? OR object_id=?",
            (eid, eid),
        ).fetchall()
        ids = {eid}
        edges: list[dict[str, Any]] = []
        for row in edges_raw:
            edges.append({"subject": row["subject_id"], "predicate": row["predicate"], "object": row["object_id"]})
            ids.add(row["subject_id"])
            ids.add(row["object_id"])
        # 2nd hop
        if depth >= 2 and ids:
            placeholders = ",".join("?" * len(ids))
            rows2 = con.execute(
                f"SELECT subject_id, predicate, object_id FROM relations WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})",
                tuple(ids) + tuple(ids),
            ).fetchall()
            for row in rows2:
                if row["subject_id"] not in ids or row["object_id"] not in ids:
                    edges.append({"subject": row["subject_id"], "predicate": row["predicate"], "object": row["object_id"]})
                    ids.add(row["subject_id"])
                    ids.add(row["object_id"])
        if not ids:
            return {"entity": entity, "found": True, "nodes": [], "edges": []}
        placeholders_n = ",".join("?" * len(ids))
        nodes_raw = con.execute(
            f"SELECT id, name, type FROM entities WHERE id IN ({placeholders_n}) LIMIT ?",
            tuple(ids) + (limit,),
        ).fetchall()
        nodes = [dict(r) for r in nodes_raw]
        return {"entity": entity, "found": True, "nodes": nodes, "edges": edges[: limit * 4]}
    finally:
        con.close()


def keyword_search(query: str, limit: int = 6) -> list[dict[str, Any]]:
    """FTS trigram substring search over ingested chunks; falls back to LIKE."""
    con = _connect()
    try:
        if len(query.strip()) >= 2:
            try:
                match = '"' + query.strip().replace('"', "") + '"'
                rows = con.execute(
                    "SELECT c.rowid AS id, c.text FROM chunks_fts c "
                    "WHERE c MATCH ? ORDER BY rowid DESC LIMIT ?",
                    (match, limit),
                ).fetchall()
                docs: list[dict[str, Any]] = []
                for row in rows:
                    chunk = con.execute(
                        "SELECT c.id, c.doc_id, c.text, d.title FROM chunks c "
                        "JOIN documents d ON d.id = c.doc_id WHERE c.id=?",
                        (row["id"],),
                    ).fetchone()
                    if chunk:
                        docs.append(dict(chunk))
                if docs:
                    return docs
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                pass  # FTS not available; fall back to LIKE
        pattern = f"%{query.strip()}%"
        rows = con.execute(
            "SELECT c.id, c.doc_id, d.title AS title, c.text FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id "
            "WHERE c.text LIKE ? ORDER BY c.seq LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def entity_count() -> int:
    con = _connect()
    try:
        return int(con.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
    finally:
        con.close()


def relation_count() -> int:
    con = _connect()
    try:
        return int(con.execute("SELECT COUNT(*) FROM relations").fetchone()[0])
    finally:
        con.close()


def stats() -> dict[str, Any]:
    con = _connect()
    try:
        docs = int(con.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        return {"documents": docs, "entities": entity_count(), "relations": relation_count()}
    finally:
        con.close()


def export_graph() -> dict[str, Any]:
    """Dump the full graph for visualization and external tooling.

    Returns nodes (with degree for sizing), edges (with predicate labels), and
    overall counts. Read-only; used by the web panel's graph page and CLI.
    """
    con = _connect()
    try:
        nodes_raw = con.execute("SELECT id, name, type, canonical FROM entities ORDER BY id").fetchall()
        edges_raw = con.execute("SELECT subject_id, predicate, object_id FROM relations ORDER BY id").fetchall()
        degree: dict[int, int] = {int(row["id"]): 0 for row in nodes_raw}
        edges: list[dict[str, Any]] = []
        for row in edges_raw:
            subject_id = int(row["subject_id"])
            object_id = int(row["object_id"])
            degree[subject_id] = degree.get(subject_id, 0) + 1
            degree[object_id] = degree.get(object_id, 0) + 1
            edges.append({"source": subject_id, "target": object_id, "predicate": row["predicate"]})
        nodes = [
            {"id": int(row["id"]), "name": row["name"], "type": row["type"], "degree": degree.get(int(row["id"]), 0)}
            for row in nodes_raw
        ]
        doc_count = int(con.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {"documents": doc_count, "entities": len(nodes), "relations": len(edges)},
        }
    finally:
        con.close()


def _degree_of(con: sqlite3.Connection, eid: int) -> int:
    row = con.execute(
        "SELECT COUNT(*) FROM relations WHERE subject_id=? OR object_id=?", (eid, eid)
    ).fetchone()
    return int(row[0])


def _merge_into(con: sqlite3.Connection, keep_id: int, dup_id: int) -> None:
    """Reassign dup's relations onto keep, dropping self-loops and duplicates."""
    existing = {
        (int(r["subject_id"]), r["predicate"], int(r["object_id"]))
        for r in con.execute("SELECT subject_id, predicate, object_id FROM relations").fetchall()
    }
    rels = con.execute(
        "SELECT id, subject_id, predicate, object_id FROM relations WHERE subject_id=? OR object_id=?",
        (dup_id, dup_id),
    ).fetchall()
    for r in rels:
        sid = keep_id if int(r["subject_id"]) == dup_id else int(r["subject_id"])
        oid = keep_id if int(r["object_id"]) == dup_id else int(r["object_id"])
        if sid == oid:
            con.execute("DELETE FROM relations WHERE id=?", (r["id"],))
            continue
        key = (sid, r["predicate"], oid)
        if key in existing:
            con.execute("DELETE FROM relations WHERE id=?", (r["id"],))
            continue
        existing.add(key)
        con.execute("UPDATE relations SET subject_id=?, object_id=? WHERE id=?", (sid, oid, r["id"]))
    con.execute("DELETE FROM entities WHERE id=?", (dup_id,))


def dedupe_aliases() -> dict[str, Any]:
    """Merge existing entities that share an alias base (e.g. 肉鸽/肉鸽玩法/Roguelike).

    Re-canonicalizes every entity under :func:`_alias_base`, then folds each
    collision group into the most-connected member. Returns a summary of what
    was merged; safe to run idempotently.
    """
    con = _connect()
    try:
        rows = con.execute("SELECT id, name, canonical FROM entities").fetchall()
        for r in rows:
            base = _alias_base(r["name"])
            # Namespaced identities (qq:/group:) are authoritative — renaming
            # a person must never re-derive their canonical from the nickname.
            if ":" in r["canonical"] and r["canonical"] != base:
                continue
            con.execute("UPDATE entities SET canonical=? WHERE id=?", (base, r["id"]))
        rows = con.execute("SELECT id, name, type, canonical FROM entities").fetchall()
        groups: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            groups.setdefault(r["canonical"], []).append(r)
        merged: list[dict[str, str]] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda m: (-_degree_of(con, int(m["id"])), int(m["id"])))
            keep = members[0]
            for dup in members[1:]:
                _merge_into(con, int(keep["id"]), int(dup["id"]))
                merged.append({"alias": dup["name"], "canonical": keep["name"]})
        con.commit()
        return {"merged": len(merged), "pairs": merged}
    finally:
        con.close()