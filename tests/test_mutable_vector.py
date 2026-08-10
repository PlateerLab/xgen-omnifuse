"""Exactness and lifecycle tests for the opt-in mutable passage store."""

from __future__ import annotations

from copy import copy, deepcopy
import gzip
import pathlib
import pickle
import random
import sys
from threading import Event, Thread

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import (  # noqa: E402
    Chunk,
    Feedback,
    InMemoryGraph,
    InMemoryVector,
    MutableInMemoryVector,
    MutableVectorStore,
    Node,
    OmniFuse,
    Triple,
    build_inmemory,
    load_index,
    save_index,
)
from omnifuse._compact_mutable import CompactMutableBM25  # noqa: E402
from omnifuse._compact_mutable_fielded import CompactMutableBM25F  # noqa: E402
from omnifuse.text import tokenize  # noqa: E402


QUERIES = ["alpha", "beta beta", "gamma delta", "missing"]
LEGACY_VECTOR_KEYS = {
    "embedder",
    "lexical_weight",
    "dense_weight",
    "_pool",
    "_title_weight",
    "_idf_pow",
    "_feedback",
    "_revision",
    "_next_slot",
    "_chunks",
    "_slot_by_id",
    "_title_count",
    "_lexical_count",
    "_embedding_count",
    "_fielded",
    "_bm25",
    "_lexical",
    "_dense",
}


def _ranked(vector, query: str, *, limit: int = 20):
    return [
        (chunk.id, score.hex()) for chunk, score in vector.search(query, limit=limit)
    ]


def _assert_static_oracle(vector, queries=QUERIES, **kwargs):
    rebuilt = InMemoryVector(vector.chunks, **kwargs)
    for query in queries:
        assert _ranked(vector, query) == _ranked(rebuilt, query)


def test_lexical_index_is_lazy_and_pre_search_work_is_coalesced(monkeypatch):
    import omnifuse.backends.memory as memory

    tokenized: list[str] = []
    original_tokenize = memory.tokenize

    def counted_tokenize(text: str):
        tokenized.append(text)
        return original_tokenize(text)

    monkeypatch.setattr(memory, "tokenize", counted_tokenize)
    vector = MutableInMemoryVector(
        [Chunk("a", "alpha"), Chunk("b", "beta"), Chunk("c", "gamma")],
        feedback=Feedback(),
    )

    inserted = vector.upsert_chunks([Chunk("d", "delta")])
    updated = vector.upsert_chunks([Chunk("b", "beta updated")])
    deleted = vector.delete_chunks(["c"])
    unchanged = vector.upsert_chunks([Chunk("a", "alpha")])
    vector.remember("remembered beta", ["b"])

    assert vector._bm25 is None
    assert tokenized == []
    assert all(
        (result.incremental, result.rebuilt, result.reindexed) == (True, False, 0)
        for result in (inserted, updated, deleted, unchanged)
    )
    before_revision = vector.revision

    assert _ranked(vector, "remembered beta")[0][0] == "b"
    assert vector.revision == before_revision
    assert isinstance(vector._bm25, CompactMutableBM25F)
    assert len(tokenized) == 9  # title, body and memory for each of three live chunks
    materialized_calls = list(tokenized)
    _ranked(vector, "alpha")
    assert tokenized == materialized_calls


def test_korean_character_fallback_preserves_sparse_mutable_slots():
    vector = MutableInMemoryVector(
        [
            Chunk("compound", "황갈색입니다."),
            Chunk("deleted", "파란색입니다."),
            Chunk("longer", "세 가지 색상을 모두 제공합니다."),
        ]
    )
    vector.delete_chunks(["deleted"])

    assert [chunk.id for chunk, _score in vector.search("색", limit=3)] == [
        "compound",
        "longer",
    ]


def test_pre_search_crud_builds_final_stable_order_once():
    vector = MutableInMemoryVector(
        [Chunk("a", "same"), Chunk("b", "same"), Chunk("c", "same")]
    )
    results = [
        vector.delete_chunks(["b"]),
        vector.upsert_chunks([Chunk("d", "same")]),
        vector.upsert_chunks([Chunk("a", "same updated")]),
        vector.upsert_chunks([Chunk("b", "same")]),
        vector.upsert_chunks([Chunk("c", "same")]),
    ]

    assert vector._bm25 is None
    assert [chunk.id for chunk in vector.chunks] == ["a", "c", "d", "b"]
    assert all(result.incremental and result.reindexed == 0 for result in results)
    oracle = InMemoryVector(vector.chunks)
    for query in ("same", "updated"):
        assert _ranked(vector, query) == _ranked(oracle, query)
    assert isinstance(vector._bm25, CompactMutableBM25)
    assert vector._bm25._mutation_version == 0


def test_mutation_batches_are_idempotent_and_exact_to_full_rebuild():
    vector = MutableInMemoryVector(
        [
            Chunk("a", "alpha alpha beta"),
            Chunk("b", "beta gamma"),
            Chunk("c", "delta"),
        ]
    )
    _assert_static_oracle(vector)

    result = vector.upsert_chunks(
        [Chunk("b", "alpha gamma gamma"), Chunk("d", "beta delta")]
    )
    assert (result.revision, result.inserted, result.updated, result.changed) == (
        1,
        1,
        1,
        2,
    )
    _assert_static_oracle(vector)

    result = vector.upsert_chunks(
        [Chunk("b", "alpha gamma gamma"), Chunk("d", "beta delta")]
    )
    assert (result.revision, result.unchanged, result.changed) == (1, 2, 0)

    result = vector.delete_chunks(["a", "absent"])
    assert (result.revision, result.deleted, result.missing) == (2, 1, 1)
    _assert_static_oracle(vector)

    result = vector.delete_chunks(["a", "absent"])
    assert (result.revision, result.deleted, result.missing) == (2, 0, 2)


def test_title_topology_transitions_match_static_bm25_and_bm25f():
    vector = MutableInMemoryVector([Chunk("a", "alpha body"), Chunk("b", "beta body")])
    assert vector._fielded is False
    _assert_static_oracle(vector)

    result = vector.upsert_chunks([Chunk("a", "alpha body", title="gamma alpha")])
    assert vector._fielded is True
    assert (result.incremental, result.reindexed) == (False, 2)
    _assert_static_oracle(vector)

    lazy = MutableInMemoryVector([Chunk("only", "body", title="title")])
    result = lazy.delete_chunks(["only"])
    assert (result.incremental, result.rebuilt, result.reindexed) == (True, False, 0)
    assert lazy._bm25 is None

    last = MutableInMemoryVector([Chunk("only", "body", title="title")])
    _ranked(last, "title")
    result = last.delete_chunks(["only"])
    assert (result.incremental, result.rebuilt, result.reindexed) == (False, True, 0)
    assert last.chunks == []

    result = vector.upsert_chunks([Chunk("a", "alpha body")])
    assert vector._fielded is False
    assert (result.incremental, result.reindexed) == (False, 2)
    _assert_static_oracle(vector)


def test_deleted_slots_are_not_reused_and_ties_keep_corpus_order():
    vector = MutableInMemoryVector(
        [Chunk("a", "same"), Chunk("b", "same"), Chunk("c", "same")]
    )
    vector.delete_chunks(["b"])
    vector.upsert_chunks([Chunk("b", "same")])

    assert [chunk.id for chunk in vector.chunks] == ["a", "c", "b"]
    assert [chunk_id for chunk_id, _score in _ranked(vector, "same")] == [
        "a",
        "c",
        "b",
    ]
    _assert_static_oracle(vector, ["same"])


def test_mutable_store_detaches_caller_and_result_objects():
    entities = ["entity-a"]
    embedding = [1.0, 0.0]
    meta = {"nested": {"value": 1}}
    source = Chunk("a", "alpha", entities, embedding, meta)
    vector = MutableInMemoryVector([source], embedder=lambda _query: [1.0, 0.0])

    entities.append("caller-change")
    embedding[0] = 0.0
    meta["nested"]["value"] = 2
    source.text = "caller-change"

    stored = vector.fetch(["a"])[0]
    assert stored.text == "alpha"
    assert stored.entities == ["entity-a"]
    assert stored.embedding == [1.0, 0.0]
    assert stored.meta == {"nested": {"value": 1}}

    stored.text = "result-change"
    stored.entities.append("result-change")
    stored.meta["nested"]["value"] = 3
    assert vector.fetch(["a"])[0].text == "alpha"
    assert vector.fetch(["a"])[0].entities == ["entity-a"]
    assert vector.fetch(["a"])[0].meta == {"nested": {"value": 1}}


def test_internal_chunks_are_slot_frozen_and_share_empty_payloads():
    import omnifuse.backends.memory as memory

    vector = MutableInMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    clones = [
        vector,
        copy(vector),
        deepcopy(vector),
        pickle.loads(pickle.dumps(vector)),
    ]

    for clone in clones:
        first, second = clone._chunks.values()
        assert not hasattr(clone, "_order")
        assert not hasattr(first, "__dict__")
        assert isinstance(first, memory._PackedChunk)
        assert first.entities is second.entities
        assert first.embedding is second.embedding is None
        assert first.meta is second.meta is memory._EMPTY_META
        with pytest.raises(AttributeError):
            first.text = "changed"


@pytest.mark.parametrize(
    ("entities", "embedding"),
    [
        ([], None),
        ([], []),
        (["entity-a"], [1.0, 0.0]),
        (("entity-a",), (1.0, 0.0)),
    ],
)
def test_public_chunk_payload_types_roundtrip_across_copy_and_pickle(
    entities, embedding
):
    source = Chunk(
        "a",
        "alpha",
        entities=entities,
        embedding=embedding,
        meta={"nested": {"value": 1}},
    )
    vector = MutableInMemoryVector([source])
    clones = [
        vector,
        copy(vector),
        deepcopy(vector),
        pickle.loads(pickle.dumps(vector)),
    ]

    for clone in clones:
        fetched = clone.fetch(["a"])[0]
        assert type(fetched.entities) is list
        assert fetched.entities == list(entities)
        if embedding is None:
            assert fetched.embedding is None
        else:
            assert type(fetched.embedding) is list
            assert fetched.embedding == list(embedding)
        assert fetched.meta == {"nested": {"value": 1}}
        fetched.entities.append("detached")
        fetched.meta["nested"]["value"] = 2
        assert clone.fetch(["a"])[0].meta == {"nested": {"value": 1}}


def test_noop_upsert_validates_without_deepcopy_or_retaining_input(monkeypatch):
    import omnifuse.backends.memory as memory

    vector = MutableInMemoryVector(
        [
            Chunk(
                "a",
                "alpha",
                entities=["entity-a"],
                embedding=[1.0, 0.0],
                meta={"nested": {"value": 1}},
            )
        ]
    )
    incoming = Chunk(
        "a",
        "alpha",
        entities=["entity-a"],
        embedding=[1.0, 0.0],
        meta={"nested": {"value": 1}},
    )
    copied: list[object] = []
    original_deepcopy = memory.deepcopy

    def counted_deepcopy(value):
        copied.append(value)
        return original_deepcopy(value)

    monkeypatch.setattr(memory, "deepcopy", counted_deepcopy)
    result = vector.upsert_chunks([incoming])

    assert (result.revision, result.unchanged, result.changed) == (0, 1, 0)
    assert copied == []

    incoming.text = "caller mutation"
    incoming.entities.append("caller mutation")
    incoming.embedding[0] = 0.0
    incoming.meta["nested"]["value"] = 2
    monkeypatch.setattr(memory, "deepcopy", original_deepcopy)
    stored = vector.fetch(["a"])[0]
    assert stored.text == "alpha"
    assert stored.entities == ["entity-a"]
    assert stored.embedding == [1.0, 0.0]
    assert stored.meta == {"nested": {"value": 1}}


def test_tuple_payload_upsert_is_value_equal_noop_without_deepcopy(monkeypatch):
    import omnifuse.backends.memory as memory

    vector = MutableInMemoryVector(
        [Chunk("a", "alpha", entities=["entity-a"], embedding=[1.0, 0.0])]
    )
    copied: list[object] = []
    original_deepcopy = memory.deepcopy

    def counted_deepcopy(value):
        copied.append(value)
        return original_deepcopy(value)

    monkeypatch.setattr(memory, "deepcopy", counted_deepcopy)
    result = vector.upsert_chunks(
        [
            Chunk(
                "a",
                "alpha",
                entities=("entity-a",),
                embedding=(1.0, 0.0),
            )
        ]
    )

    assert (result.revision, result.unchanged, result.changed) == (0, 1, 0)
    assert vector.revision == 0
    assert copied == []


def test_one_shot_iterable_payload_is_frozen_once_and_can_be_a_noop():
    vector = MutableInMemoryVector(
        [Chunk("a", "alpha", entities=["entity-a"], embedding=[1.0, 0.0])]
    )
    candidate = Chunk(
        "a",
        "alpha",
        entities=iter(["entity-a"]),
        embedding=iter([1.0, 0.0]),
    )

    result = vector.upsert_chunks([candidate])

    assert (result.revision, result.unchanged, result.changed) == (0, 1, 0)
    assert vector.fetch(["a"])[0] == Chunk(
        "a", "alpha", entities=["entity-a"], embedding=[1.0, 0.0]
    )

    changed = vector.upsert_chunks(
        [
            Chunk(
                "a",
                "changed",
                entities=iter(["entity-b"]),
                embedding=iter([0.0, 1.0]),
            )
        ]
    )

    assert (changed.revision, changed.updated, changed.unchanged) == (1, 1, 0)
    assert vector.fetch(["a"])[0] == Chunk(
        "a", "changed", entities=["entity-b"], embedding=[0.0, 1.0]
    )


def test_none_and_empty_embedding_remain_distinct_mutation_values():
    vector = MutableInMemoryVector([Chunk("a", "alpha", embedding=None)])

    result = vector.upsert_chunks([Chunk("a", "alpha", embedding=())])

    assert (result.revision, result.updated, result.unchanged) == (1, 1, 0)
    assert vector.fetch(["a"])[0].embedding == []


def test_changed_upsert_detaches_mutable_caller_fields():
    vector = MutableInMemoryVector([Chunk("a", "alpha")])
    changed = Chunk(
        "a",
        "updated",
        entities=["entity-a"],
        embedding=[1.0, 0.0],
        meta={"nested": {"value": 1}},
    )

    result = vector.upsert_chunks([changed])
    changed.text = "caller mutation"
    changed.entities.append("caller mutation")
    changed.embedding[0] = 0.0
    changed.meta["nested"]["value"] = 2

    assert (result.revision, result.updated) == (1, 1)
    stored = vector.fetch(["a"])[0]
    assert stored.text == "updated"
    assert stored.entities == ["entity-a"]
    assert stored.embedding == [1.0, 0.0]
    assert stored.meta == {"nested": {"value": 1}}


def test_invalid_later_upsert_item_is_prevalidated_before_any_copy_or_change(
    monkeypatch,
):
    import omnifuse.backends.memory as memory

    vector = MutableInMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    before_chunks = vector.chunks
    before_rankings = [_ranked(vector, query) for query in QUERIES]
    copied: list[object] = []
    original_deepcopy = memory.deepcopy

    def counted_deepcopy(value):
        copied.append(value)
        return original_deepcopy(value)

    monkeypatch.setattr(memory, "deepcopy", counted_deepcopy)
    with pytest.raises(TypeError, match="expected Chunk"):
        vector.upsert_chunks(
            [Chunk("a", "changed", meta={"nested": {"value": 1}}), object()]
        )

    assert copied == []
    assert vector.revision == 0
    monkeypatch.setattr(memory, "deepcopy", original_deepcopy)
    assert vector.chunks == before_chunks
    assert [_ranked(vector, query) for query in QUERIES] == before_rankings


def test_invalid_batches_leave_revision_payload_and_rankings_unchanged():
    vector = MutableInMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    before_revision = vector.revision
    before_chunks = vector.chunks
    before_rankings = [_ranked(vector, query) for query in QUERIES]

    with pytest.raises(ValueError, match="duplicate"):
        vector.upsert_chunks([Chunk("c", "gamma"), Chunk("c", "delta")])
    with pytest.raises(ValueError, match="duplicate"):
        vector.delete_chunks(["a", "a"])
    with pytest.raises(ValueError, match="non-empty"):
        vector.upsert_chunks([Chunk("", "gamma")])
    with pytest.raises(TypeError, match="expected Chunk"):
        vector.upsert_chunks([object()])

    assert vector.revision == before_revision
    assert vector.chunks == before_chunks
    assert [_ranked(vector, query) for query in QUERIES] == before_rankings


def test_dense_and_hybrid_mutations_match_full_rebuild():
    def embed(query: str) -> list[float]:
        return {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }.get(query, [0.6, 0.4])

    dense = MutableInMemoryVector(
        [Chunk("a", "", embedding=[1.0, 0.0]), Chunk("b", "", embedding=[0.0, 1.0])],
        embedder=embed,
    )
    _assert_static_oracle(dense, ["alpha", "beta"], embedder=embed)
    dense.upsert_chunks([Chunk("a", "", embedding=[0.8, 0.2])])
    dense.upsert_chunks([Chunk("c", "", embedding=[0.4, 0.6])])
    dense.delete_chunks(["b"])
    _assert_static_oracle(dense, ["alpha", "beta"], embedder=embed)

    hybrid = MutableInMemoryVector(
        [
            Chunk("a", "alpha alpha", embedding=[0.9, 0.1]),
            Chunk("b", "beta", embedding=[0.2, 0.8]),
            Chunk("c", "gamma", embedding=[0.5, 0.5]),
        ],
        embedder=embed,
    )
    _assert_static_oracle(hybrid, ["alpha", "beta"], embedder=embed)
    hybrid.upsert_chunks([Chunk("b", "alpha beta", embedding=[0.7, 0.3])])
    hybrid.delete_chunks(["c"])
    _assert_static_oracle(hybrid, ["alpha", "beta"], embedder=embed)


@pytest.mark.parametrize("fielded", [False, True])
def test_randomized_vector_mutations_match_every_full_rebuild(fielded):
    rng = random.Random(20260722 + fielded)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
    active: dict[str, Chunk] = {}
    order: list[str] = []
    retired: list[str] = []
    next_id = 0

    def make_chunk(chunk_id: str) -> Chunk:
        words = [rng.choice(vocabulary) for _ in range(rng.randint(0, 8))]
        title = f"title {rng.choice(vocabulary)}" if fielded else ""
        return Chunk(chunk_id, " ".join(words), title=title)

    for _ in range(5):
        chunk_id = f"d{next_id}"
        next_id += 1
        active[chunk_id] = make_chunk(chunk_id)
        order.append(chunk_id)
    vector = MutableInMemoryVector([active[chunk_id] for chunk_id in order])
    expected_revision = 0

    for _step in range(60):
        action = rng.choice(["insert", "insert", "update", "delete", "noop"])
        if action == "insert" or not order:
            if retired and rng.random() < 0.3:
                chunk_id = retired.pop(rng.randrange(len(retired)))
            else:
                chunk_id = f"d{next_id}"
                next_id += 1
            chunk = make_chunk(chunk_id)
            result = vector.upsert_chunks([chunk])
            active[chunk_id] = chunk
            order.append(chunk_id)
            assert result.inserted == 1
            expected_revision += 1
        elif action == "update":
            chunk_id = rng.choice(order)
            chunk = make_chunk(chunk_id)
            result = vector.upsert_chunks([chunk])
            active[chunk_id] = chunk
            assert result.updated == 1
            expected_revision += 1
        elif action == "delete":
            chunk_id = rng.choice(order)
            result = vector.delete_chunks([chunk_id])
            del active[chunk_id]
            order.remove(chunk_id)
            retired.append(chunk_id)
            assert result.deleted == 1
            expected_revision += 1
        else:
            chunk_id = rng.choice(order)
            result = vector.upsert_chunks([active[chunk_id]])
            assert result.unchanged == 1

        assert vector.revision == expected_revision
        assert [chunk.id for chunk in vector.chunks] == order
        oracle = InMemoryVector([active[chunk_id] for chunk_id in order])
        for query in QUERIES:
            assert _ranked(vector, query) == _ranked(oracle, query)


def test_feedback_is_preserved_on_update_dropped_on_delete_and_not_resurrected():
    feedback = Feedback()
    feedback.remember("remembered phrase", ["a"])
    vector = MutableInMemoryVector(
        [Chunk("a", "alpha"), Chunk("b", "beta")], feedback=feedback
    )
    assert vector.feedback is not feedback
    assert vector.feedback.queries("a") == ["remembered phrase"]

    feedback.remember("caller-side mutation", ["a"])
    assert vector.feedback.queries("a") == ["remembered phrase"]
    exposed = vector.feedback
    exposed.remember("snapshot-side mutation", ["a"])
    assert vector.feedback.queries("a") == ["remembered phrase"]

    vector.upsert_chunks([Chunk("a", "alpha updated")])
    assert vector.feedback.queries("a") == ["remembered phrase"]
    assert _ranked(vector, "remembered phrase")[0][0] == "a"

    vector.remember("second phrase", ["a"])
    assert vector.feedback.queries("a") == ["remembered phrase", "second phrase"]
    vector.forget("second phrase", ["a"])
    assert vector.feedback.queries("a") == ["remembered phrase"]

    vector.delete_chunks(["a"])
    assert vector.feedback.queries("a") == []
    vector.upsert_chunks([Chunk("a", "alpha reinserted")])
    assert vector.feedback.queries("a") == []
    assert _ranked(vector, "remembered phrase") == []


def test_feedback_changes_before_first_search_materialize_final_memory_only():
    feedback = Feedback()
    feedback.remember("obsolete", ["a"])
    vector = MutableInMemoryVector(
        [Chunk("a", "alpha"), Chunk("b", "beta")], feedback=feedback
    )

    vector.remember("remembered", ["a"])
    vector.forget("obsolete", ["a"])

    assert vector._bm25 is None
    assert vector.feedback.queries("a") == ["remembered"]
    assert _ranked(vector, "obsolete") == []
    assert _ranked(vector, "remembered")[0][0] == "a"
    _assert_static_oracle(
        vector,
        ["alpha", "obsolete", "remembered"],
        feedback=vector.feedback,
    )


def test_unmaterialized_index_survives_pickle_and_remains_mutable():
    vector = MutableInMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    vector.upsert_chunks([Chunk("c", "gamma")])

    loaded = pickle.loads(pickle.dumps(vector))

    assert vector._bm25 is None
    assert loaded._bm25 is None
    assert loaded.revision == 1
    assert loaded.chunks == vector.chunks
    result = loaded.delete_chunks(["b"])
    assert (result.revision, result.incremental, result.reindexed) == (2, True, 0)
    _assert_static_oracle(loaded)
    assert vector._bm25 is None


def test_materialized_deepcopy_keeps_storage_and_mutations_independent():
    vector = MutableInMemoryVector([Chunk("a", "alpha"), Chunk("b", "beta")])
    before = [_ranked(vector, query) for query in QUERIES]

    cloned = deepcopy(vector)

    assert not hasattr(cloned, "_order")
    assert cloned.chunks == vector.chunks
    assert [_ranked(cloned, query) for query in QUERIES] == before
    cloned.upsert_chunks([Chunk("a", "changed")])
    cloned.delete_chunks(["b"])
    assert [chunk.id for chunk in cloned.chunks] == ["a"]
    assert [chunk.id for chunk in vector.chunks] == ["a", "b"]
    assert [_ranked(vector, query) for query in QUERIES] == before


@pytest.mark.parametrize("order_kind", ["list", "dict"])
def test_exact_legacy_order_pickle_migrates_holes_without_load_rebuild(
    order_kind,
):
    source = MutableInMemoryVector(
        [
            Chunk("a", "same"),
            Chunk("b", "same"),
            Chunk("c", "same"),
            Chunk("d", "same"),
        ]
    )
    source.delete_chunks(["b"])
    source.upsert_chunks([Chunk("e", "same")])
    expected_slots = list(source._chunks)
    public_by_slot = dict(zip(expected_slots, source.chunks, strict=True))
    reverse_backed = CompactMutableBM25(
        (slot, tokenize(chunk.text)) for slot, chunk in source._chunks.items()
    )
    state = {
        key: source.__dict__[key]
        for key in LEGACY_VECTOR_KEYS
        if key not in {"embedder", "_dense", "_bm25", "_chunks"}
    }
    state.update(
        {
            "embedder": None,
            "_dense": False,
            "_bm25": reverse_backed,
            "_chunks": {
                slot: public_by_slot[slot] for slot in reversed(expected_slots)
            },
            "_order": (
                expected_slots
                if order_kind == "list"
                else dict.fromkeys(expected_slots)
            ),
        }
    )
    assert set(state) == LEGACY_VECTOR_KEYS | {"_order"}
    legacy = MutableInMemoryVector.__new__(MutableInMemoryVector)
    legacy.__dict__.update(state)

    builds = 0
    original_build = MutableInMemoryVector._build_lexical

    def counted_build(self, fielded, chunks):
        nonlocal builds
        builds += 1
        return original_build(self, fielded, chunks)

    MutableInMemoryVector._build_lexical = counted_build
    try:
        loaded = pickle.loads(pickle.dumps(legacy))
        assert builds == 0
        assert _ranked(loaded, "same")
        assert builds == 0
    finally:
        MutableInMemoryVector._build_lexical = original_build

    assert not hasattr(loaded, "_order")
    assert list(loaded._chunks) == expected_slots == [0, 2, 3, 4]
    assert [chunk.id for chunk in loaded.chunks] == ["a", "c", "d", "e"]
    assert loaded._bm25._base._retains_reverse is False
    result = loaded.upsert_chunks([Chunk("c", "same", title="title")])
    assert (result.rebuilt, result.reindexed) == (True, 4)
    assert list(loaded._chunks) == expected_slots
    assert [chunk.id for chunk in loaded.chunks] == ["a", "c", "d", "e"]
    _assert_static_oracle(loaded, ["same", "title"])


def test_concurrent_first_search_materializes_once_without_revision_change(monkeypatch):
    vector = MutableInMemoryVector(
        [Chunk("a", "alpha alpha"), Chunk("b", "alpha beta")]
    )
    started = Event()
    release = Event()
    calls: list[tuple[bool, tuple[int, ...]]] = []
    rankings: list[list[tuple[str, str]]] = []
    errors: list[BaseException] = []
    original_build = vector._build_lexical

    def synchronized_build(fielded, chunks):
        calls.append((fielded, tuple(chunks)))
        started.set()
        assert release.wait(timeout=2)
        return original_build(fielded, chunks)

    monkeypatch.setattr(vector, "_build_lexical", synchronized_build)

    def search():
        try:
            rankings.append(_ranked(vector, "alpha"))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = Thread(target=search)
    second = Thread(target=search)
    first.start()
    assert started.wait(timeout=2)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert calls == [(False, (0, 1))]
    assert rankings[0] == rankings[1]
    assert vector.revision == 0


def test_builder_protocol_facade_and_mutable_persistence(tmp_path):
    mutable = build_inmemory([], [], [Chunk("a", "alpha")], mutable=True)
    static = build_inmemory([], [], [Chunk("a", "alpha")])

    assert type(static.vector) is InMemoryVector
    assert isinstance(mutable.vector, MutableInMemoryVector)
    assert isinstance(mutable.vector, MutableVectorStore)
    assert not isinstance(static.vector, MutableVectorStore)
    with pytest.raises(TypeError, match="does not support"):
        static.upsert_chunks([Chunk("b", "beta")])

    mutable.upsert_chunks([Chunk("b", "beta")])
    path = tmp_path / "mutable-index.pkl.gz"
    save_index(mutable, path)
    loaded = load_index(path)

    assert isinstance(loaded.vector, MutableInMemoryVector)
    assert loaded.vector.revision == 1
    assert [_ranked(loaded.vector, query) for query in QUERIES] == [
        _ranked(mutable.vector, query) for query in QUERIES
    ]
    result = loaded.delete_chunks(["a"])
    assert (result.revision, result.deleted) == (2, 1)
    _assert_static_oracle(loaded.vector)


def test_mutable_feedback_persistence_remains_exact_and_teachable(tmp_path):
    omnifuse = build_inmemory(
        [],
        [],
        [Chunk("a", "alpha"), Chunk("b", "beta")],
        feedback=Feedback(),
        mutable=True,
    )
    omnifuse.remember("remembered phrase", ["a"])
    path = tmp_path / "mutable-feedback.pkl.gz"
    save_index(omnifuse, path)
    loaded = load_index(path)

    assert _ranked(loaded.vector, "remembered phrase")[0][0] == "a"
    loaded.upsert_chunks([Chunk("a", "alpha updated")])
    assert _ranked(loaded.vector, "remembered phrase")[0][0] == "a"
    _assert_static_oracle(
        loaded.vector,
        ["alpha", "remembered phrase"],
        feedback=loaded.vector.feedback,
    )

    loaded.delete_chunks(["a"])
    loaded.upsert_chunks([Chunk("a", "alpha reinserted")])
    assert loaded.vector.feedback.queries("a") == []
    assert _ranked(loaded.vector, "remembered phrase") == []


def test_partial_embeddings_toggle_dense_mode_and_reattach_after_load(tmp_path):
    def embed(query: str) -> list[float]:
        return [1.0, 0.0] if query == "alpha" else [0.0, 1.0]

    vector = MutableInMemoryVector(
        [
            Chunk("a", "alpha", embedding=[1.0, 0.0]),
            Chunk("b", "beta", embedding=[0.0, 1.0]),
        ],
        embedder=embed,
    )
    assert vector._dense is True
    vector.upsert_chunks([Chunk("b", "beta")])
    assert vector._dense is False
    _assert_static_oracle(vector, ["alpha", "beta"], embedder=embed)

    vector.upsert_chunks([Chunk("b", "beta", embedding=[0.0, 1.0])])
    assert vector._dense is True
    omnifuse = OmniFuse(InMemoryGraph([], []), vector)
    path = tmp_path / "mutable-dense.pkl.gz"
    save_index(omnifuse, path)

    cold = load_index(path)
    assert cold.vector._dense is False
    warm = load_index(path, embedder=embed)
    assert warm.vector._dense is True
    _assert_static_oracle(warm.vector, ["alpha", "beta"], embedder=embed)


def test_retrieve_holds_one_revision_across_search_and_graph_fetch():
    graph = InMemoryGraph(
        [Node("seed", "seed"), Node("linked", "linked")],
        [Triple("seed", "references", "linked")],
    )
    vector = MutableInMemoryVector(
        [Chunk("seed", "alpha"), Chunk("linked", "unrelated")]
    )
    omnifuse = OmniFuse(graph, vector, vector_k=1, fusion_expand_top=1)
    searched = Event()
    mutation_attempted = Event()
    mutation_done = Event()
    original_search = vector.search

    def synchronized_search(query: str, *, limit: int = 20):
        hits = original_search(query, limit=limit)
        searched.set()
        assert mutation_attempted.wait(timeout=2)
        return hits

    vector.search = synchronized_search

    def delete_linked():
        assert searched.wait(timeout=2)
        mutation_attempted.set()
        vector.delete_chunks(["linked"])
        mutation_done.set()

    thread = Thread(target=delete_linked)
    thread.start()
    result = omnifuse.retrieve("alpha", limit=2)
    thread.join(timeout=2)

    assert [chunk.id for chunk, _score in result] == ["seed", "linked"]
    assert mutation_done.is_set()
    assert vector.fetch(["linked"]) == []


def test_save_holds_one_mutable_revision(monkeypatch, tmp_path):
    omnifuse = build_inmemory([], [], [Chunk("a", "alpha")], mutable=True)
    vector = omnifuse.vector
    serializing = Event()
    mutation_attempted = Event()
    mutation_done = Event()
    original_getstate = MutableInMemoryVector.__getstate__

    def synchronized_getstate(self):
        serializing.set()
        assert mutation_attempted.wait(timeout=2)
        return original_getstate(self)

    monkeypatch.setattr(MutableInMemoryVector, "__getstate__", synchronized_getstate)

    def insert_during_save():
        assert serializing.wait(timeout=2)
        mutation_attempted.set()
        vector.upsert_chunks([Chunk("b", "beta")])
        mutation_done.set()

    thread = Thread(target=insert_during_save)
    thread.start()
    path = tmp_path / "snapshot.pkl.gz"
    save_index(omnifuse, path)
    thread.join(timeout=2)

    assert mutation_done.is_set()
    assert [chunk.id for chunk in vector.chunks] == ["a", "b"]
    snapshot = load_index(path)
    assert [chunk.id for chunk in snapshot.vector.chunks] == ["a"]
    assert snapshot.vector.revision == 0


def test_failed_save_preserves_existing_index_and_removes_temporary(
    monkeypatch, tmp_path
):
    import omnifuse.facade as facade

    omnifuse = build_inmemory([], [], [Chunk("a", "alpha")], mutable=True)
    path = tmp_path / "durable.pkl.gz"
    save_index(omnifuse, path)
    before = path.read_bytes()

    def fail_dump(*_args, **_kwargs):
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(facade.pickle, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="injected"):
        save_index(omnifuse, path)

    assert path.read_bytes() == before
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("bad_format", [True, 1.0, "1", None])
def test_load_rejects_non_exact_format_types(tmp_path, bad_format):
    static = build_inmemory([], [], [Chunk("a", "alpha")])
    path = tmp_path / f"bad-format-{bad_format!r}.pkl.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(
            {"format": bad_format, "graph": static.graph, "vector": static.vector},
            handle,
        )
    with pytest.raises(ValueError, match="unsupported index format"):
        load_index(path)


@pytest.mark.parametrize(
    ("index_format", "vector_kind", "message"),
    [
        (1, "mutable", "static index format"),
        (2, "static", "mutable index format"),
        (1, "missing", "in-memory graph/vector"),
    ],
)
def test_load_rejects_format_backend_mismatches(
    tmp_path, index_format, vector_kind, message
):
    static = build_inmemory([], [], [Chunk("a", "alpha")])
    mutable = build_inmemory([], [], [Chunk("a", "alpha")], mutable=True)
    vector = {
        "static": static.vector,
        "mutable": mutable.vector,
        "missing": None,
    }[vector_kind]
    path = tmp_path / f"bad-backend-{index_format}-{vector_kind}.pkl.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(
            {"format": index_format, "graph": static.graph, "vector": vector},
            handle,
        )
    with pytest.raises(ValueError, match=message):
        load_index(path)
