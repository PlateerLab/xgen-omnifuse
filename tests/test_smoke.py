import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, InMemoryGraph, Node, Triple, build_inmemory  # noqa: E402

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
CHUNKS = [
    Chunk("ch1", "불법도박은 한국마사회법 제48조에 따라 단속되며 경마시행규정 제76조가 적용된다.",
          entities=["n_illegal", "n_race76"]),
    Chunk("ch2", "감사규정은 내부 감사 절차를 규정한다.", entities=["n_audit"]),
]


def test_search_runs_zero_infra():
    of = build_inmemory(NODES, TRIPLES, CHUNKS)
    r = of.search("불법도박 관련 규정")
    assert r.answer
    assert "규정" in r.class_seed
    assert any("관련규정" in x for x in r.relations)
    assert "경마시행규정 제76조" in r.evidence_nodes


def test_bm25_label_search():
    g = InMemoryGraph(NODES, TRIPLES)
    hits = g.search_labels("감사", limit=3)
    assert hits and hits[0][0].label == "감사규정"


def test_class_enumeration():
    g = InMemoryGraph(NODES, TRIPLES)
    insts = g.class_instances("c_reg")
    assert len(insts) == 3
    assert g.count_class("c_reg") == 3
