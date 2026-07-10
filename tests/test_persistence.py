"""Index persistence — a built in-memory index survives a round-trip to disk, so a
process can `load_index` instead of paying the build cost again. Rankings must be
identical, and the non-portable bits (LLM, embedder callable) must not be persisted.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import build_inmemory, load_index, save_index  # noqa: E402
from omnifuse.backends.memory import InMemoryVector  # noqa: E402
from omnifuse.llm import EchoLLM  # noqa: E402

TRIPLES = [("담보", "instanceOf", "규정"), ("담보", "근거", "제3조")]
CHUNKS = [
    {"id": "c1", "title": "담보 한도", "text": "담보 한도는 5억원이다"},
    {"id": "c2", "title": "제3조", "text": "근거 규정을 정한다"},
    {"id": "c3", "title": "기타", "text": "무관한 본문 내용"},
]


def _built():
    return build_inmemory([], TRIPLES, CHUNKS)


def test_round_trip_preserves_ranking(tmp_path):
    of = _built()
    q = "담보 한도"
    before = [(c.id, round(s, 12)) for c, s in of.retrieve(q, limit=5)]

    p = tmp_path / "idx.pkl"
    save_index(of, p)
    loaded = load_index(p)

    after = [(c.id, round(s, 12)) for c, s in loaded.retrieve(q, limit=5)]
    assert before == after
    assert before, "expected a non-empty ranking"


def test_loaded_index_keeps_graph_and_chunks(tmp_path):
    of = _built()
    p = tmp_path / "idx.pkl"
    save_index(of, p)
    loaded = load_index(p)

    assert loaded.graph.neighbor_ids("담보") == of.graph.neighbor_ids("담보")
    assert [c.id for c in loaded.vector.fetch(["c1", "c2"])] == ["c1", "c2"]


def test_embedder_not_persisted_and_reattachable(tmp_path):
    emb = lambda s: [1.0, 0.0]  # noqa: E731 — a closure is not portable
    vec = InMemoryVector([], embedder=emb)
    assert vec.embedder is emb

    of = _built()
    p = tmp_path / "idx.pkl"
    save_index(of, p)

    loaded = load_index(p)
    assert loaded.vector.embedder is None      # dropped on save
    assert loaded.vector._dense is False

    loaded2 = load_index(p, embedder=emb)
    assert loaded2.vector.embedder is emb      # re-attached


def test_llm_not_persisted_and_default_echo(tmp_path):
    of = _built()
    p = tmp_path / "idx.pkl"
    save_index(of, p)
    assert isinstance(load_index(p).llm, EchoLLM)


def test_rejects_non_inmemory_backend(tmp_path):
    of = _built()

    class NotInMemory:
        pass

    of.vector = NotInMemory()
    with pytest.raises(TypeError):
        save_index(of, tmp_path / "x.pkl")


def test_rejects_unknown_format(tmp_path):
    import pickle

    p = tmp_path / "bad.pkl"
    p.write_bytes(pickle.dumps({"format": 999, "graph": None, "vector": None}))
    with pytest.raises(ValueError):
        load_index(p)
