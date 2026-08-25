"""Community detection + interactive visualization for the Kairós KG.

Builds a high-confidence subgraph from SQLite, runs Louvain community
detection, then emits:
  /app/data/graph_viz.html        self-contained interactive vis (vis.js CDN)
  /app/data/graph_communities.json  node->community map, per-community top
                                    members, and cross-community bridge nodes

Usage (inside the kairos container): python scripts/graph_analytics.py
Requires: networkx (pip install networkx). Re-run after major imports.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/app")

import networkx as nx  # noqa: E402
from networkx.algorithms.community import louvain_communities  # noqa: E402

from kairos.knowledge import engine  # noqa: E402

NOISE = "('提及','提到','回复','询问','艾特','呼叫','调侃','评论','引用','@','活跃于')"
MIN_CONF = 7
OUT_VIZ = "/app/data/graph_viz.html"
OUT_JSON = "/app/data/graph_communities.json"
PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c",
    "#e67e22", "#34495e", "#fd79a8", "#00cec9", "#6c5ce7", "#d63031",
    "#0984e3", "#b8e994", "#fab1a0",
]

TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Kairos KG - Louvain</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>html,body{height:100%;margin:0}#m{width:100%;height:100%}
#lg{position:fixed;top:10px;left:10px;background:rgba(255,255,255,.92);padding:8px 12px;border-radius:8px;font:13px sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.15)}
#hint{position:fixed;bottom:10px;left:10px;background:rgba(255,255,255,.85);padding:6px 10px;border-radius:8px;font:12px sans-serif}</style></head><body>
<div id="m"></div>
<div id="lg"><b>Louvain 社区发现</b><span id="nc"></span> 个社区分别着色 | 悬停看关系,点击聚焦</div>
<div id="hint">滚轮缩放 / 拖拽平移 | 节点大小=关系数 颜色=圈子 | 悬停节点显示详情</div>
<script>
var nodes = new vis.DataSet(NODES_JSON);
var edges = new vis.DataSet(EDGES_JSON);
// physics disabled: layout is pre-computed server-side (spring_layout),
// so rendering is static and instant — no CPU burn on load or drag.
var net = new vis.Network(document.getElementById("m"), {nodes:nodes, edges:edges},
 {physics:{enabled:false},
  interaction:{hover:true, tooltipDelay:150, navigationButtons:true, keyboard:true},
  edges:{selectionWidth:2}});
net.on("click", function(p){ if(p.nodes.length){ net.focus(p.nodes[0], {scale:1.6, animation:{duration:300}}); } });
document.getElementById("nc").textContent = __NCOMM__;
</script></body></html>"""


def build_graph():
    engine.init()
    con = engine._connect()
    rows = con.execute(
        f"""
        SELECT r.subject_id sid, e1.name sn, e1.type st,
               r.predicate p, r.object_id oid, e2.name oname, e2.type ot,
               r.confidence c
        FROM relations r
        JOIN entities e1 ON e1.id = r.subject_id
        JOIN entities e2 ON e2.id = r.object_id
        WHERE r.confidence >= {MIN_CONF} AND r.predicate NOT IN {NOISE}
          AND e1.type != '群组' AND e2.type != '群组'
        """
    ).fetchall()
    con.close()
    G = nx.Graph()
    for r in rows:
        if G.has_edge(r["sid"], r["oid"]):
            G[r["sid"]][r["oid"]]["w"] += int(r["c"])
        else:
            G.add_node(r["sid"], name=r["sn"], type=r["st"])
            G.add_node(r["oid"], name=r["oname"], type=r["ot"])
            G.add_edge(r["sid"], r["oid"], w=int(r["c"]), p=r["p"])
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if len(G) > 3200:
        keep = [n for n, d in dict(G.degree()).items() if d >= 2]
        G = G.subgraph(keep).copy()
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def main() -> None:
    G = build_graph()
    print(f"graph: {len(G)} nodes, {G.number_of_edges()} edges")

    comms = louvain_communities(G, weight="w", resolution=1.0, seed=42)
    comms = sorted(comms, key=len, reverse=True)
    cid = {n: i for i, cset in enumerate(comms) for n in cset}
    print(f"communities: {len(comms)}, sizes: {[len(c) for c in comms[:12]]}")

    # Pre-compute layout server-side so the browser renders a static scene:
    # zero startup physics => instant load, smooth pan/zoom even on phones.
    pos = nx.spring_layout(G, weight="w", iterations=120, seed=42)

    # bridge score: approximate betweenness — people connecting communities
    bridges = sorted(
        nx.betweenness_centrality(G, k=min(200, len(G)), weight="w", seed=42).items(),
        key=lambda x: -x[1],
    )[:20]

    nj = [
        {
            "id": str(n),
            "x": float(pos[n][0]) * 2400,
            "y": float(pos[n][1]) * 1600,
            "shape": "dot",
            "size": 4 + min(G.degree(n), 26),
            "label": G.nodes[n]["name"][:14] if G.degree(n) >= 4 else "",
            "title": f"{G.nodes[n]['name']} [{G.nodes[n]['type']}] 度={G.degree(n)} 社区#{cid[n] + 1}",
            "color": PALETTE[cid[n] % len(PALETTE)],
        }
        for n in G.nodes
    ]
    ej = [
        {
            "from": str(u), "to": str(v),
            "title": G[u][v]["p"],
            "width": 0.4,
            "color": {"color": "#97b8c8", "highlight": "#e74c3c", "hover": "#e67e22"},
        }
        for u, v in G.edges
    ]
    html = (
        TEMPLATE.replace("__NCOMM__", str(len(comms)))
        .replace("NODES_JSON", json.dumps(nj, ensure_ascii=False))
        .replace("EDGES_JSON", json.dumps(ej, ensure_ascii=False))
    )
    with open(OUT_VIZ, "w", encoding="utf-8") as f:
        f.write(html)

    payload = {
        "generated_nodes": len(G),
        "communities": [
            {
                "id": i + 1,
                "size": len(cset),
                "top_members": [
                    G.nodes[n]["name"]
                    for n, _ in sorted(G.subgraph(cset).degree, key=lambda x: -x[1])[:8]
                ],
            }
            for i, cset in enumerate(comms)
        ],
        "bridges": [{"name": G.nodes[n]["name"], "score": round(s, 4)} for n, s in bridges],
        "node_community": {str(n): cid[n] + 1 for n in G.nodes},
        "node_names": {str(n): G.nodes[n]["name"] for n in G.nodes},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"wrote {OUT_VIZ} ({os.path.getsize(OUT_VIZ)} bytes)")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
