"""Neo4j AI retrieval demo against live data (runs inside kairos container)."""
import sys
sys.path.insert(0, "/app")

from kairos.knowledge import graph_store
from kairos.knowledge import ingest as kg_ingest

print("== 高连接度实体 ==")
with graph_store._driver().session() as s:
    rows = s.run(
        "MATCH (e:Entity) WITH e, count { (e)--() } AS deg "
        "RETURN e.name AS n, e.type AS t, deg ORDER BY deg DESC LIMIT 6"
    ).data()
for r in rows:
    print(f"  {r['n']} [{r['t']}] 度={r['deg']}")

kw = sys.argv[1] if len(sys.argv) > 1 else "东方"
print(f"\n== 实体检索 '{kw}' ==")
hits = graph_store.search(kw, limit=4)
for h in hits:
    ctx = "; ".join(f"{c['other']}←{c['pred']}" for c in h["ctx"])
    print(f"  {h['name']} ({h['type']}, 度={h['deg']}) {ctx[:100]}")

print(f"\n== 多跳子图 (2跳) 围绕 '{kw}' ==")
nb = graph_store.neighborhood(kw, hops=2, limit=12)
for e in nb.get("edges", [])[:10]:
    print(f"  {e['src']} --[{e['pred']}]--> {e['dst']}")

q = kg_ingest.query_knowledge(entity=kw)
print("\n== query_knowledge 端到端（机器人实际调用路径）==")
print(q.get("answer", "")[:600])
