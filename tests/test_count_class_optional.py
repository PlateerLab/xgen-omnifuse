"""A GraphStore written against the 0.7.x protocol (no ``count_class``) must still search."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, InMemoryGraph, InMemoryVector, Node, OmniFuse, Triple  # noqa: E402
from omnifuse.llm import EchoLLM  # noqa: E402

NODES = [
    Node("c_reg", "규정", kind="class"),
    Node("n_audit", "감사규정", kind="instance"),
    Node("n_info", "정보화업무처리규정", kind="instance"),
    Node("n_race", "경마시행규정", kind="instance"),
]
TRIPLES = [
    Triple("n_audit", "instanceOf", "c_reg"),
    Triple("n_info", "instanceOf", "c_reg"),
    Triple("n_race", "instanceOf", "c_reg"),
]
CHUNKS = [
    Chunk("ch1", "감사규정은 내부 감사 절차를 규정한다.", entities=["n_audit"]),
    Chunk("ch2", "경마시행규정은 경마의 시행을 규정한다.", entities=["n_race"]),
]


class LegacyGraph:
    """Delegates everything except ``count_class`` to the in-memory graph."""

    def __init__(self, inner):
        self._inner = inner

    def search_labels(self, query, *, limit=30):
        return self._inner.search_labels(query, limit=limit)

    def class_instances(self, class_id, *, limit=1000):
        return self._inner.class_instances(class_id, limit=limit)

    def neighbors(self, node_id, *, hops=1, limit=100):
        return self._inner.neighbors(node_id, hops=hops, limit=limit)

    def neighbor_ids(self, node_id, *, limit=100, direction="both"):
        return self._inner.neighbor_ids(node_id, limit=limit, direction=direction)


def _fuse(graph):
    return OmniFuse(graph, InMemoryVector(CHUNKS), llm=EchoLLM(), graph_fusion=False)


def test_search_without_count_class_uses_enumerated_size():
    inner = InMemoryGraph(NODES, TRIPLES)
    assert not hasattr(LegacyGraph(inner), "count_class")
    r = _fuse(LegacyGraph(inner)).search("규정 전부 나열", synthesize=False)
    assert "has 3 instances" in r.class_seed


def test_search_with_count_class_uses_exact_total():
    class Counting(LegacyGraph):
        def count_class(self, class_id):
            return 1234   # exact total beyond the enumeration limit

    r = _fuse(Counting(InMemoryGraph(NODES, TRIPLES))).search("규정 전부 나열", synthesize=False)
    assert "has 1234 instances" in r.class_seed and "(+1231 more)" in r.class_seed
