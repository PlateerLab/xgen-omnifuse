from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import pickle
from threading import Event

import pytest
from omnifuse import Chunk, Feedback
from omnifuse._compact_postings import CompactPostingsSnapshot
from omnifuse.backends.memory import InMemoryVector
from omnifuse.text import BM25, BM25F, tokenize


def _ranked(vector: InMemoryVector, query: str, limit: int = 20):
    return [(chunk.id, score.hex()) for chunk, score in vector.search(query, limit=limit)]


def test_plain_store_defers_only_materialization_and_matches_static_oracle():
    chunks = [Chunk("a", "alpha beta"), Chunk("b", "beta"), Chunk("c", "gamma")]
    vector = InMemoryVector(chunks)
    oracle = BM25((tokenize(chunk.text) for chunk in chunks))

    assert vector._bm25 is None
    expected = [
        (chunks[index].id, score.hex())
        for index, score in oracle.search("alpha beta", limit=3)
    ]
    assert _ranked(vector, "alpha beta", 3) == expected
    assert isinstance(vector._bm25, CompactPostingsSnapshot)
    assert vector._bm25._retains_reverse is False
    assert vector._lexical_source is None


def test_fielded_store_preserves_exact_title_body_scores():
    chunks = [
        Chunk("a", "long body " * 20, title="alpha"),
        Chunk("b", "alpha in body", title="other"),
    ]
    vector = InMemoryVector(chunks)
    oracle = BM25F(
        (
            {"title": tokenize(chunk.title), "body": tokenize(chunk.text)}
            for chunk in chunks
        ),
        {"title": 4.0, "body": 1.0},
    )

    assert vector._bm25 is None
    expected = [
        (chunks[index].id, score.hex())
        for index, score in oracle.search("alpha", limit=2)
    ]
    assert _ranked(vector, "alpha", 2) == expected
    assert isinstance(vector._bm25, CompactPostingsSnapshot)
    assert vector._bm25.mode == "bm25f"
    assert vector._bm25._retains_reverse is False


def test_first_search_builds_once_under_contention(monkeypatch):
    vector = InMemoryVector([Chunk(str(index), "alpha beta") for index in range(40)])
    original = vector._build_lexical
    entered = Event()
    release = Event()
    builds = 0

    def counted_build():
        nonlocal builds
        builds += 1
        entered.set()
        assert release.wait(timeout=2)
        return original()

    monkeypatch.setattr(vector, "_build_lexical", counted_build)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_ranked, vector, "alpha", 5) for _ in range(8)]
        assert entered.wait(timeout=2)
        release.set()
        rankings = [future.result(timeout=2) for future in futures]

    assert builds == 1
    assert all(ranking == rankings[0] for ranking in rankings)


def test_construction_snapshot_isolated_from_later_chunk_mutation():
    source = Chunk("a", "alpha", title="original")
    vector = InMemoryVector([source])
    source.text = "changed"
    source.title = "changed"

    assert _ranked(vector, "alpha")
    assert _ranked(vector, "changed") == []


def test_unmaterialized_and_materialized_pickle_roundtrips_are_exact():
    vector = InMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    lazy = pickle.loads(pickle.dumps(vector))

    assert vector._bm25 is None
    assert lazy._bm25 is None
    assert hasattr(lazy, "_lock")
    expected = _ranked(lazy, "alpha")
    materialized = pickle.loads(pickle.dumps(lazy))
    assert isinstance(materialized._bm25, CompactPostingsSnapshot)
    assert _ranked(materialized, "alpha") == expected

def test_materialized_pickle_uses_validated_forward_only_state():
    vector = InMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    expected = _ranked(vector, "alpha")

    state = vector.__getstate__()
    assert set(state["_bm25"]) == {"kind", "state"}
    restored = InMemoryVector.__new__(InMemoryVector)
    restored.__setstate__(state)

    assert restored._bm25._retains_reverse is False
    assert _ranked(restored, "alpha") == expected

    corrupt = deepcopy(state)
    corrupt["_bm25"]["state"]["state_version"] = -1
    with pytest.raises(ValueError, match="packed vector forward"):
        InMemoryVector.__new__(InMemoryVector).__setstate__(corrupt)

    invalid_envelope = deepcopy(state)
    invalid_envelope["_bm25"]["extra"] = True
    with pytest.raises(ValueError, match="snapshot envelope"):
        InMemoryVector.__new__(InMemoryVector).__setstate__(invalid_envelope)


def test_legacy_materialized_state_restores_without_new_fields():
    vector = InMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    expected = _ranked(vector, "alpha")
    state = vector.__getstate__()
    state["_bm25"] = BM25([tokenize("alpha"), tokenize("beta")])
    state.pop("_title_weight")
    state.pop("_idf_pow")
    state.pop("_lexical_source")
    legacy = InMemoryVector.__new__(InMemoryVector)

    legacy.__setstate__(state)

    assert hasattr(legacy, "_lock")
    assert _ranked(legacy, "alpha") == expected


def test_feedback_store_remains_eager_and_incremental():
    feedback = Feedback()
    feedback.remember("remembered", ["a"])
    vector = InMemoryVector(
        [Chunk("a", "alpha"), Chunk("b", "beta")], feedback=feedback
    )

    assert isinstance(vector._bm25, BM25F)
    assert vector._lexical_source is None
    assert _ranked(vector, "remembered")[0][0] == "a"
    vector.remember("second", ["b"])
    assert _ranked(vector, "second")[0][0] == "b"
    vector.forget("second", ["b"])
    assert _ranked(vector, "second") == []


def test_lazy_plain_and_eager_feedback_keep_content_idf_equal():
    chunks = [Chunk("a", "alpha shared"), Chunk("b", "beta shared")]
    feedback = Feedback()
    feedback.remember("shared memory", ["a"])
    plain = InMemoryVector(chunks)
    warm = InMemoryVector(chunks, feedback=feedback)

    assert plain._bm25 is None
    _ranked(plain, "shared")
    assert plain._bm25.idf["shared"] == warm._bm25.idf["shared"]


def test_dense_only_stays_unmaterialized_and_hybrid_builds_on_demand():
    def embedder(_query):
        return [1.0, 0.0]

    dense = InMemoryVector(
        [Chunk("a", "", embedding=[1.0, 0.0])], embedder=embedder
    )
    hybrid = InMemoryVector(
        [Chunk("a", "alpha", embedding=[1.0, 0.0])], embedder=embedder
    )

    assert dense._bm25 is None
    assert _ranked(dense, "query")
    assert dense._bm25 is None
    assert hybrid._bm25 is None
    assert _ranked(hybrid, "alpha")
    assert isinstance(hybrid._bm25, CompactPostingsSnapshot)
