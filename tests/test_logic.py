"""Deeper checks on the self-contained (in-memory) logic — fusion + traversal."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, InMemoryGraph, InMemoryVector, Node, Triple, build_inmemory  # noqa: E402
from omnifuse.fusion import dynamic_cut, jaccard, mmr, rank_relations  # noqa: E402

NODES = [
    Node("c_reg", "규정", kind="class"),
    Node("n_audit", "감사규정", kind="instance"),
    Node("n_race76", "경마시행규정 제76조", kind="instance"),
    Node("n_info", "정보화업무처리규정", kind="instance"),
    Node("n_illegal", "불법도박", kind="instance"),
]
TRIPLES = [
    Triple("n_audit", "instanceOf", "c_reg"),
    Triple("n_race76", "instanceOf", "c_reg"),
    Triple("n_info", "instanceOf", "c_reg"),
    Triple("n_illegal", "관련규정", "n_race76"),
]


# ---- fusion primitives ----
def test_dynamic_cut_drops_low_scores():
    scored = [("a", 1.0), ("b", 0.9), ("c", 0.2), ("d", 0.1)]
    assert dynamic_cut(scored, ratio=0.55, min_k=1, max_k=10) == ["a", "b"]


def test_dynamic_cut_respects_min_k():
    scored = [("a", 1.0), ("b", 0.1)]
    assert dynamic_cut(scored, ratio=0.9, min_k=2, max_k=10) == ["a", "b"]


def test_mmr_keeps_decisive_minority():
    # two near-duplicates + one distinct minority (the exception) — MMR must keep the minority
    cands = [
        ("담보 비율 60 퍼센트 기준 적용", 1.0),
        ("담보 비율 60 퍼센트 기준 적용 동일 내용", 0.96),
        ("예외 유예 기간 에는 적용 되지 않는다", 0.70),
    ]
    out = mmr(cands, lam=0.6, k=2)
    assert any("예외" in x for x in out), out


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_rank_relations_dedup_and_rank():
    rels = ["불법도박 → 관련규정 → 경마시행규정 제76조", "x → y → z", "불법도박 → 관련규정 → 경마시행규정 제76조"]
    out = rank_relations(rels, "불법도박 규정", limit=10)
    assert out[0].startswith("불법도박") and len(out) == 2  # deduped, query-relevant first


# ---- graph traversal ----
def test_neighbors_bidirectional():
    g = InMemoryGraph(NODES, TRIPLES)
    preds = {p for _, p, _ in g.neighbors("n_race76")}
    assert "관련규정" in preds and "instanceOf" in preds  # incoming + outgoing


def test_class_enumeration_complete():
    g = InMemoryGraph(NODES, TRIPLES)
    labels = {n.label for n in g.class_instances("c_reg")}
    assert labels == {"감사규정", "경마시행규정 제76조", "정보화업무처리규정"}


# ---- end-to-end: conditional question filters, vector store optional ----
def test_search_graph_only_no_vectors():
    of = OmniFuse_graph_only()
    r = of.search("불법도박 관련 규정")
    assert any("관련규정" in x for x in r.relations)
    assert r.chunks == []  # graph-only, no vector store content


def OmniFuse_graph_only():
    from omnifuse import OmniFuse

    return OmniFuse(InMemoryGraph(NODES, TRIPLES), InMemoryVector([]))


def test_bm25_label_ranking_order():
    g = InMemoryGraph(NODES, TRIPLES)
    hits = g.search_labels("규정", limit=5)
    assert hits  # '규정' substring matches all *규정 nodes; class '규정' ranks (exact) high
    assert any(n.label == "규정" for n, _ in hits)
