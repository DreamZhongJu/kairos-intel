"""Debounced Neo4j mirror refresh.

Full ``sync_all()`` is cheap at this graph scale (~seconds for a few thousand
nodes), so the simplest correct incremental strategy is: after any ingest,
schedule one full re-sync, coalescing bursts into a single run.
"""

from __future__ import annotations

import threading
from time import monotonic

_lock = threading.Lock()
_scheduled_at = 0.0
_running = False
_MIN_INTERVAL = 120.0  # seconds between actual syncs


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
        except Exception:  # noqa: BLE001  — mirror failures must not break ingest
            pass
    finally:
        with _lock:
            _running = False
