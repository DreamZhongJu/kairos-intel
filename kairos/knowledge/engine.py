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


def upsert_entity(name: str, etype: str = "entity") -> int:
    """Insert an entity deduplicated by normalized name."""
    canonical = _normalize(name)
    con = _connect()
    try:
        row = con.execute("SELECT id FROM entities WHERE canonical=? LIMIT 1", (canonical,)).fetchone()
        if row:
            return int(row["id"])
        cur = con.execute(
            "INSERT INTO entities (name, type, canonical, created_at) VALUES (?,?,?,?)",
            (name, etype, canonical, _now()),
        )
        con.commit()
        return int(cur.lastrowid)
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
    canonical = _normalize(name)
    con = _connect()
    try:
        row = con.execute("SELECT * FROM entities WHERE canonical=?", (canonical,)).fetchone()
        return dict(row) if row else None
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