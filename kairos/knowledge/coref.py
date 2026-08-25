"""Person coreference resolution across entity-creation paths.

Root cause this fixes: deterministic speaker nodes get canonical="qq:<uid>"
while LLM-extracted person mentions fall back to canonical=<raw name>, so the
same human splits into several nodes. The registry below maps every known
alias to its authoritative canonical so upsert can route mentions to the
existing person node instead of forking a new one.

Sources (highest priority first):
1. ``data/coref_overrides.json`` — manual assertions {"alias": "qq:<uid>"}
2. ``data/nickmap.json``          — nickname -> uid evidence from exports
3. entities table                 — current name of every qq:* person node
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kairos.infrastructure.settings import DATA_DIR

_CACHE_TTL = 300.0
_registry: dict[str, str] | None = None
_registry_at = 0.0


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _build_registry() -> dict[str, str]:
    reg: dict[str, str] = {}
    # weakest: current display names of deterministic person nodes
    try:
        from kairos.knowledge import engine

        con = engine._connect()
        try:
            rows = con.execute(
                "SELECT name, canonical FROM entities WHERE type='人名' AND canonical LIKE 'qq:%'"
            ).fetchall()
        finally:
            con.close()
        for r in rows:
            if r["name"]:
                reg[r["name"].strip()] = r["canonical"]
    except Exception:  # noqa: BLE001
        pass
    # stronger: export-derived nickname evidence
    nickmap_path = Path(__file__).resolve().parents[2] / "data" / "nickmap.json"
    container_fallback = DATA_DIR / "nickmap.json"
    for p in (nickmap_path, container_fallback):
        for nick, uid in _read_json(p).items():
            nick = str(nick).strip()
            if nick and str(uid).strip():
                reg[nick] = f"qq:{str(uid).strip()}"
    # strongest: manual assertions win over everything
    for alias, canon in _read_json(DATA_DIR / "coref_overrides.json").items():
        alias = str(alias).strip()
        canon = str(canon).strip()
        if alias and canon:
            reg[alias] = canon
    return reg


def _get_registry() -> dict[str, str]:
    global _registry, _registry_at
    now = time.time()
    if _registry is None or now - _registry_at > _CACHE_TTL:
        _registry = _build_registry()
        _registry_at = now
    return _registry


def resolve_person(name: str) -> str | None:
    """Return the authoritative canonical ('qq:<uid>') for a person mention."""
    n = (name or "").strip()
    if not n:
        return None
    canon = _get_registry().get(n)
    return canon or None


def merge_entity(old_id: int, target_id: int) -> dict[str, int]:
    """Re-point every relation of ``old_id`` to ``target_id`` and delete it."""
    from kairos.knowledge import engine

    old_id, target_id = int(old_id), int(target_id)
    if old_id == target_id:
        return {"moved": 0, "dropped_dupes": 0}
    con = engine._connect()
    moved = dropped = 0
    try:
        rows = con.execute(
            "SELECT id, subject_id, predicate, object_id FROM relations WHERE subject_id=? OR object_id=?",
            (old_id, old_id),
        ).fetchall()
        for r in rows:
            s = target_id if int(r["subject_id"]) == old_id else int(r["subject_id"])
            o = target_id if int(r["object_id"]) == old_id else int(r["object_id"])
            if s == o:
                con.execute("DELETE FROM relations WHERE id=?", (r["id"],))
                dropped += 1
                continue
            dup = con.execute(
                "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=? AND id!=?",
                (s, r["predicate"], o, r["id"]),
            ).fetchone()
            if dup:
                con.execute("DELETE FROM relations WHERE id=?", (r["id"],))
                dropped += 1
                continue
            con.execute(
                "UPDATE relations SET subject_id=?, object_id=? WHERE id=?",
                (s, o, r["id"]),
            )
            moved += 1
        con.execute("DELETE FROM entities WHERE id=?", (old_id,))
        con.commit()
    finally:
        con.close()
    return {"moved": moved, "dropped_dupes": dropped}


def sweep_bare_persons(dry_run: bool = False) -> dict[str, Any]:
    """Find person nodes whose canonical lacks the qq:/group: prefix and try
    to resolve each via the registry; merge hits into their canonical node."""
    from kairos.knowledge import engine

    con = engine._connect()
    try:
        rows = con.execute(
            "SELECT id, name, canonical FROM entities "
            "WHERE type='人名' AND canonical NOT LIKE 'qq:%' AND canonical NOT LIKE 'group:%'"
        ).fetchall()
    finally:
        con.close()

    report = {"scanned": len(rows), "merged": [], "unresolved": []}
    for r in rows:
        canon = resolve_person(r["name"]) or (
            resolve_person(r["canonical"]) if r["canonical"] else None
        )
        if not canon:
            report["unresolved"].append({"id": r["id"], "name": r["name"], "canonical": r["canonical"]})
            continue
        con = engine._connect()
        try:
            tgt = con.execute("SELECT id FROM entities WHERE canonical=? LIMIT 1", (canon,)).fetchone()
        finally:
            con.close()
        if not tgt:
            # no node yet for that canonical: promote this one in place
            if not dry_run:
                con = engine._connect()
                try:
                    con.execute("UPDATE entities SET canonical=? WHERE id=?", (canon, r["id"]))
                    con.execute("UPDATE entities SET name=? WHERE id=? AND name LIKE 'qq:%'",
                                (r["name"], r["id"]))
                    con.commit()
                finally:
                    con.close()
            report["merged"].append({
                "id": r["id"], "name": r["name"], "from": r["canonical"], "to": canon, "promoted": True,
            })
            continue
        if dry_run:
            report["merged"].append({
                "id": r["id"], "name": r["name"], "from": r["canonical"],
                "to": canon, "target_id": int(tgt["id"]), "promoted": False,
            })
            continue
        stats = merge_entity(r["id"], int(tgt["id"]))
        report["merged"].append({
            "id": r["id"], "name": r["name"], "from": r["canonical"], "to": canon,
            "target_id": int(tgt["id"]), **stats,
        })
    return report
