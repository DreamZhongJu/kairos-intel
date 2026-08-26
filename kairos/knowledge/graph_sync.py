"""Debounced Neo4j mirror refresh + periodic hyperedge mining.

Full ``sync_all()`` is cheap at this graph scale (~seconds for a few thousand
nodes), so the simplest correct incremental strategy is: after any ingest,
schedule one full re-sync, coalescing bursts into a single run.

Reasoning-chain hyperedges are re-mined on the same hook but far less often
(``_MINE_INTERVAL``): mining is an idempotent upsert (repeated chains only
increment their use counter), so a full pass is always safe — it just costs
a couple of minutes of SQLite joins.
"""

from __future__ import annotations

import threading
import time as _time
from time import monotonic

_lock = threading.Lock()
_scheduled_at = 0.0
_running = False
_MIN_INTERVAL = 120.0  # seconds between actual syncs

_mine_lock = threading.Lock()
_last_mine = 0.0
_MINE_INTERVAL = 1800.0  # seconds between hyperedge mining passes


def schedule_resync(force: bool = False) -> bool:
    """Request a background resync; returns True if one will run."""
    global _scheduled_at
    now = monotonic()
    with _lock:
        if _running:
            return False
        if not force and now - _scheduled_at < _MIN_INTERVAL:
            return False
        _scheduled_at = now
    t = threading.Thread(target=_run, name="neo4j-resync", daemon=True)
    t.start()
    return True


def _run() -> None:
    global _running
    with _lock:
        if _running:
            return
        _running = True
    try:
        from kairos.knowledge import graph_store

        if not graph_store.available():
            return
        try:
            graph_store.init_schema()
            graph_store.sync_all()
            _maybe_mine()
        except Exception:  # noqa: BLE001  — mirror failures must not break ingest
            pass
    finally:
        with _lock:
            _running = False


def _maybe_mine() -> None:
    """Re-mine reasoning chains into HyperEdges if the cooldown has elapsed."""
    global _last_mine
    now = monotonic()
    with _mine_lock:
        if now - _last_mine < _MINE_INTERVAL:
            return
        _last_mine = now
    t = threading.Thread(target=_mine_run, name="hyperedge-mining", daemon=True)
    t.start()


def _mine_run() -> None:
    try:
        from kairos.knowledge import graph_store, hypergraph

        if not graph_store.available():
            return
        with graph_store._driver().session() as s:
            s.run("CREATE CONSTRAINT hedge_id IF NOT EXISTS FOR (h:HyperEdge) REQUIRE h.id IS UNIQUE")
        chains = hypergraph.mine_chains(min_confidence=0.45)
        stars = hypergraph.mine_event_stars()
        if chains:
            hypergraph.save_hyper_edges(chains, source="auto_resync")
        if stars:
            hypergraph.save_hyper_edges(stars, source="event_star_resync")
    except Exception:  # noqa: BLE001  — mining failures must not break ingest
        pass


def force_mine_now() -> bool:
    """Manual trigger (tools/CLI); bypasses the cooldown."""
    global _last_mine
    with _mine_lock:
        _last_mine = monotonic()
    t = threading.Thread(target=_mine_run, name="hyperedge-mining", daemon=True)
    t.start()
    return True
