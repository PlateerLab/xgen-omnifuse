"""Retrieval-side improvements — field-weighted (title) BM25 + graph-companion fusion.

Two properties the ranking must have:
  * a query term hitting a chunk's short ``title`` field lifts it above a chunk
    that only mentions it deep in a long body (BM25F);
  * a passage that shares no query vocabulary but is *referenced* by a strong
    lexical seed is surfaced in one shot, no LLM (graph-companion fusion).
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, OmniFuse, build_inmemory  # noqa: E402
from omnifuse.backends.memory import InMemoryGraph, InMemoryVector  # noqa: E402
from omnifuse.text import BM25F, tokenize  # noqa: E402


def test_bm25f_title_beats_deep_body_mention():
    docs = [
        {"title": tokenize("과태료 부과 기준"), "body": tokenize("이 조는 과태료 부과 기준을 정한다")},
        {"title": tokenize("목적"), "body": tokenize("장문 " * 200 + " 과태료 " + "기타 " * 200)},
    ]
    bm = BM25F(docs, {"title": 4.0, "body": 1.0})
    ranked = bm.search("과태료 부과 기준", limit=2)
    assert ranked[0][0] == 0  # title-matching doc wins over deep body mention


def test_title_field_used_by_vector_store():
    chunks = [
        Chunk("a", text="본문에 과태료 라는 단어가 한 번 나온다", title="과태료 부과"),
        Chunk("b", text="완전히 다른 내용의 긴 본문 " * 30, title="목적"),
    ]
    v = InMemoryVector(chunks)
    from omnifuse.text import BM25F
    assert isinstance(v._bm25, BM25F)  # titles present -> field-weighted index active
    hits = v.search("과태료", limit=2)
    assert hits[0][0].id == "a"


def test_graph_companion_fusion_surfaces_cited_passage():
    # A is lexically retrievable; B shares no query words but A references B.
    chunks = [
        Chunk("A", text="신고 의무를 위반한 경우 제32조에 따라 처벌한다", title="위반 신고"),
        Chunk("B", text="제32조 벌칙 금액은 일억원 이하 과태료로 정한다", title="벌칙"),
        Chunk("C", text="무관한 다른 조문 내용", title="기타"),
    ]
    nodes = [("A", "위반 신고"), ("B", "벌칙"), ("C", "기타")]
    triples = [("A", "references", "B")]
    of = build_inmemory(nodes, triples, chunks)  # graph_fusion on by default

    ids_fused = [c.id for c, _ in of.retrieve("신고 의무 위반")]
    assert "A" in ids_fused and "B" in ids_fused  # B pulled in via A's reference

    of_off = OmniFuse(InMemoryGraph([__import__("omnifuse").Node(n, l) for n, l in nodes],
                                    [__import__("omnifuse").Triple(*t) for t in triples]),
                      InMemoryVector(chunks), graph_fusion=False)
    ids_off = [c.id for c, _ in of_off.retrieve("신고 의무 위반")]
    assert "B" not in ids_off  # without fusion, B is unreachable from the query


def test_neighbor_ids_distinct_bidirectional():
    from omnifuse import Node, Triple

    g = InMemoryGraph([Node("A", "a"), Node("B", "b"), Node("C", "c")],
                      [Triple("A", "ref", "B"), Triple("C", "ref", "A")])
    ids = set(g.neighbor_ids("A"))
    assert ids == {"B", "C"}  # outgoing (B) + incoming (C)


def test_fusion_is_opt_outable_and_backward_compatible():
    # No titles, no graph edges -> behaves exactly like plain BM25 retrieval.
    chunks = [Chunk("x", text="담보 한도는 5억"), Chunk("y", text="무관")]
    of = build_inmemory([], [], chunks)
    hits = of.retrieve("담보 한도")
    assert hits and hits[0][0].id == "x"


def test_neighbor_ids_direction():
    from omnifuse import Node, Triple

    g = InMemoryGraph([Node("A", "a"), Node("B", "b"), Node("C", "c")],
                      [Triple("A", "ref", "B"), Triple("C", "ref", "A")])
    assert g.neighbor_ids("A", direction="out") == ["B"]   # A cites B
    assert g.neighbor_ids("A", direction="in") == ["C"]    # C cites A
    assert set(g.neighbor_ids("A", direction="both")) == {"B", "C"}
    with pytest.raises(ValueError):
        g.neighbor_ids("A", direction="sideways")


def test_fusion_follows_out_edges_not_the_citing_crowd():
    """A seed surfaces what it *references*; the crowd that references the seed is not
    evidence. On finreg, expanding the in-direction drops multi-hop to 24/120."""
    from omnifuse import Node, Triple

    nodes = [Node(i, i) for i in ("SEED", "CITED", "CITER")]
    triples = [Triple("SEED", "references", "CITED"), Triple("CITER", "references", "SEED")]
    chunks = [
        Chunk("SEED", text="신고 의무 위반에 관한 조항"),
        Chunk("CITED", text="전혀 다른 어휘의 본문"),   # shares no query vocabulary
        Chunk("CITER", text="또 다른 무관한 본문"),
    ]
    ids = lambda of: [c.id for c, _ in of.retrieve("신고 의무 위반")]  # noqa: E731

    out = ids(build_inmemory(nodes, triples, chunks))          # default: out-edges
    assert out[0] == "SEED"
    assert "CITED" in out and "CITER" not in out

    both = ids(build_inmemory(nodes, triples, chunks, fusion_direction="both"))
    assert "CITED" in both and "CITER" in both                 # the crowd leaks back in
