"""``ppr_local``: local push PageRank agrees with the whole-graph power iteration on chunk
ranking, reads only the neighbourhood it needs, and is deterministic."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import InMemoryGraph, Node, Triple, ppr_chunk_scores, ppr_local, ppr_seeds  # noqa: E402

NODES = [Node(f"n{i}", f"노드{i}", kind="instance") for i in range(12)]
# a chain n0-n1-n2-n3 plus a star around n5 and an unrelated far component n8..n11
TRIPLES = ([Triple("n0", "관련", "n1"), Triple("n1", "관련", "n2"), Triple("n2", "관련", "n3"),
            Triple("n5", "관련", "n1"), Triple("n5", "관련", "n6"), Triple("n5", "관련", "n7")]
           + [Triple("n8", "관련", "n9"), Triple("n9", "관련", "n10"), Triple("n10", "관련", "n11")])
MENTIONS = {"n0": ["k0"], "n1": ["k1"], "n2": ["k2"], "n3": ["k3"], "n5": ["k5"], "n6": ["k6"],
            "n7": ["k7"], "n8": ["k8"], "n11": ["k11"]}
IDS = sorted({ck for cks in MENTIONS.values() for ck in cks})


class RankGraph(InMemoryGraph):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fetched: list[list[str]] = []

    def node_specificity(self, ids):
        return {u: 1.0 for u in ids}

    def entities_of_chunks(self, chunk_ids, limit=400):
        out = {}
        for u, cks in MENTIONS.items():
            for ck in cks:
                if ck in chunk_ids:
                    out.setdefault(ck, []).append(u)
        return out

    def all_edges(self, limit=80000):
        return [(t.s, t.o) for t in TRIPLES][:limit]

    def walk_neighbors_many(self, ids):
        self.fetched.append(list(ids))
        adj = {}
        for t in TRIPLES:
            adj.setdefault(t.s, []).append(t.o)
            adj.setdefault(t.o, []).append(t.s)
        return {u: adj.get(u, []) for u in ids}


def test_local_push_ranks_chunks_like_power_iteration():
    g = RankGraph(NODES, TRIPLES)
    seeds = {"n0": 1.0}
    full = ppr_chunk_scores(g, seeds, IDS, alpha=0.5, iters=12)
    local = ppr_local(g, seeds, IDS, alpha=0.5, tol=1e-6)
    order_full = sorted((k for k in full), key=lambda k: -full[k])
    order_local = sorted((k for k in local), key=lambda k: -local[k])
    assert order_full[:4] == order_local[:4] and order_full[:2] == ["k0", "k1"]
    assert all(abs(full[k] - local[k]) < 1e-3 for k in full)
    # the far component is never reached by either
    assert "k8" not in full and "k8" not in local and "k11" not in local


def test_reads_only_the_neighbourhood_it_needs():
    g = RankGraph(NODES, TRIPLES)
    ppr_local(g, {"n0": 1.0}, IDS, alpha=0.5, tol=1e-3)
    touched = {u for batch in g.fetched for u in batch}
    assert "n0" in touched and "n1" in touched
    assert not touched & {"n8", "n9", "n10", "n11"}       # never asked for the far component


def test_deterministic_and_falls_back_to_neighbor_ids():
    g = RankGraph(NODES, TRIPLES)
    a = ppr_local(g, {"n0": 0.7, "n5": 0.3}, IDS)
    b = ppr_local(g, {"n5": 0.3, "n0": 0.7}, IDS)
    assert a == b and max(a.values()) == 1.0

    class NoBatch:                                        # a store without walk_neighbors_many
        def __init__(self, inner):
            self.inner = inner
        def neighbor_ids(self, u, *, limit=100, direction="both"):
            return self.inner.neighbor_ids(u, limit=limit, direction=direction)
        def entities_of_chunks(self, ids, limit=400):
            return RankGraph(NODES, TRIPLES).entities_of_chunks(ids, limit)
    c = ppr_local(NoBatch(InMemoryGraph(NODES, TRIPLES)), {"n0": 0.7, "n5": 0.3}, IDS)
    assert sorted(c, key=lambda k: -c[k])[:3] == sorted(a, key=lambda k: -a[k])[:3]


def test_seeds_helper_still_feeds_it():
    g = RankGraph(NODES, TRIPLES)
    seeds = ppr_seeds(g, "노드0")
    assert seeds and ppr_local(g, seeds, IDS)


def test_seed_without_walkable_edge_carries_no_mass_like_power_iteration():
    g = RankGraph(NODES, TRIPLES)
    assert ppr_local(g, {"n4": 1.0}, IDS) == {} == ppr_chunk_scores(g, {"n4": 1.0}, IDS)   # n4 has no edge
    mixed = ppr_local(g, {"n4": 0.9, "n0": 0.1}, IDS)
    assert sorted(mixed, key=lambda k: -mixed[k])[:2] == ["k0", "k1"]                    # n4 ignored, n0 carries all
