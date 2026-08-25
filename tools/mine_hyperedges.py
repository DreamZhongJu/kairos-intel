"""Mine reasoning chains from the KG and persist them as Neo4j hyperedges.

Usage (inside kairos container):
    python /app/tools/mine_hyperedges.py            # mine + save + report
    python /app/tools/mine_hyperedges.py --dry      # distribution only
"""
import sys

sys.path.insert(0, "/app")

from kairos.knowledge import graph_store, hypergraph  # noqa: E402


def main() -> None:
    dry = "--dry" in sys.argv
    if not graph_store.available():
        print("Neo4j unavailable")
        return

    print("mining two-hop reasoning chains ...")
    chains = hypergraph.mine_chains(min_confidence=0.45)
    print(f"candidate chains: {len(chains)}")
    if not chains:
        return

    buckets = {"strong": 0}
    hist = [0] * 6  # 0.45-0.55-0.65-0.75-0.85-1.0
    for c in chains:
        idx = min(5, int((c["confidence"] - 0.45) / 0.1))
        hist[idx] += 1
    print("confidence histogram:", {f"{0.45 + i * 0.1:.2f}+": n for i, n in enumerate(hist)})

    for c in chains[:8]:
        chain_str = " -> ".join(
            f"{c['names'][i]} --[{c['predicates'][i]}]--> {c['names'][i + 1]}"
            for i in range(len(c["predicates"]))
        )
        print(f"  [{c['confidence']:.3f}] {chain_str}")

    if dry:
        return

    graph_store.init_schema()
    with graph_store._driver().session() as s:
        s.run("CREATE CONSTRAINT hedge_id IF NOT EXISTS FOR (h:HyperEdge) REQUIRE h.id IS UNIQUE")
    n = hypergraph.save_hyper_edges(chains)
    print(f"saved {n} hyperedges")
    with graph_store._driver().session() as s:
        total = s.run("MATCH (h:HyperEdge) RETURN count(h) AS c").single()["c"]
    print(f"total hyperedges in neo4j: {total}")


if __name__ == "__main__":
    main()
