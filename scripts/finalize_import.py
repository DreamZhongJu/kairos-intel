#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-import finalizer: Neo4j sync, hyperedge mining, QA report."""
import json
import sys
import time

sys.path.insert(0, "/app")
out = open("/tmp/_finalize.out", "w", buffering=1)

def w(msg):
    out.write(msg + "\n")

try:
    t0 = time.time()
    from kairos.knowledge import engine, graph_store, hypergraph
    engine.init()
    con = engine._connect()

    docs = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    ents = con.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"]
    rels = con.execute("SELECT COUNT(*) c FROM relations").fetchone()["c"]
    anchored = con.execute(
        "SELECT COUNT(*) c FROM entities WHERE canonical LIKE 'qq:%'").fetchone()["c"]
    chunked = con.execute(
        "SELECT COUNT(*) c FROM relations WHERE source_chunk_id IS NOT NULL").fetchone()["c"]
    w(f"SQLITE docs={docs} ents={ents} rels={rels} anchored_ents={anchored} "
      f"rels_with_chunk={chunked}")

    w("syncing neo4j ...")
    graph_store.sync_all(batch_size=1000)
    w(f"neo4j synced ({time.time()-t0:.0f}s)")

    chains = hypergraph.mine_chains()
    hypergraph.save_hyper_edges(chains)
    w(f"chains mined={len(chains)}")
    stars = hypergraph.mine_event_stars()
    hypergraph.save_hyper_edges(stars, source="event_star")
    w(f"stars mined={len(stars)}")

    w("--- top stars ---")
    for s in sorted(stars, key=lambda x: -x["confidence"])[:10]:
        parts = ", ".join(s["names"][1:7])
        w(f"STAR {s['names'][0]} conf={s['confidence']:.2f} n={len(s['names'])-1} "
          f"since={s.get('since')} until={s.get('until')}: {parts}")

    w("--- hub persons ---")
    hubs = con.execute("""
        SELECT e.name, e.canonical,
          (SELECT COUNT(*) FROM relations r WHERE r.subject_id=e.id OR r.object_id=e.id) deg
        FROM entities e WHERE e.type='人名' AND e.canonical LIKE 'qq:%'
        ORDER BY deg DESC LIMIT 10""").fetchall()
    for h in hubs:
        w(f"HUB {h['name']} [{h['canonical']}] deg={h['deg']}")

    w("--- identity spot checks ---")
    for name in ("梦时空", "东北雨姐", "居委会（暴揍zblyyzxzjsdhrx版）",
                 "群底层fvv", "小刘", "老蒯#3773", "大师中的大师"):
        r = con.execute(
            "SELECT name, canonical FROM entities WHERE name=? OR canonical=? LIMIT 1",
            (name, name)).fetchone()
        if r:
            w(f"ID {r['name']} -> {r['canonical']}")
        else:
            w(f"ID {name} -> ABSENT")

    w(f"FINALIZE-DONE mins={(time.time()-t0)/60:.1f}")
except Exception as e:
    import traceback
    w(f"ERROR: {type(e).__name__}: {e}")
    w(traceback.format_exc()[:1500])
out.close()
