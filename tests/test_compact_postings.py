from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
import pickle
import random
import weakref

import pytest

import omnifuse._compact_postings as compact_postings
from omnifuse._compact_postings import (
    _UINT64_MAX,
    CompactPostingsSnapshot,
    _WeightCacheBuilder,
    _append_adaptive_uint,
    _decode_uvarint,
    _encode_uvarint,
    _freeze_posting_streams,
    _minimal_uint_array,
    _unsigned_typecode,
)
from omnifuse.text import BM25, BM25F, _MutableBM25, _MutableBM25F, tokenize


def _hex_rank(ranked):
    return [(doc_id, score.hex()) for doc_id, score in ranked]


def _restore(state: dict) -> CompactPostingsSnapshot:
    restored = CompactPostingsSnapshot.__new__(CompactPostingsSnapshot)
    restored.__setstate__(state)
    return restored


@pytest.mark.parametrize(
    "value",
    [0, 1, 127, 128, 255, 16_384, 2**40, 2**64, 2**200 + 17],
)
def test_unsigned_leb128_roundtrip_is_canonical_and_unbounded(value):
    encoded = _encode_uvarint(value)

    decoded, position = _decode_uvarint(encoded, 0, len(encoded))

    assert decoded == value
    assert position == len(encoded)
    if len(encoded) > 1:
        assert encoded[-1] != 0


@pytest.mark.parametrize("encoded", [b"", b"\x80", b"\x80\x00", b"\x81\x00"])
def test_unsigned_leb128_rejects_truncated_and_noncanonical_encodings(encoded):
    with pytest.raises(ValueError):
        _decode_uvarint(encoded, 0, len(encoded))


@pytest.mark.parametrize(
    ("data", "position", "end", "error"),
    [
        (bytearray(b"\x00"), 0, 1, TypeError),
        (b"\x00", -1, 1, ValueError),
        (b"\x00", 1, 0, ValueError),
        (b"\x00", 0, 2, ValueError),
    ],
)
def test_unsigned_leb128_rejects_invalid_bounds(data, position, end, error):
    with pytest.raises(error):
        _decode_uvarint(data, position, end)


@pytest.mark.parametrize(
    ("value", "typecode"),
    [
        (0, "B"),
        (255, "B"),
        (256, "H"),
        (65_535, "H"),
        (65_536, "I"),
        (2**32 - 1, "I"),
        (2**32, "Q"),
        (_UINT64_MAX, "Q"),
    ],
)
def test_unsigned_width_selection_is_lossless_at_every_boundary(value, typecode):
    assert _unsigned_typecode(value) == typecode
    packed = _minimal_uint_array((0, value))
    assert packed.typecode == typecode
    assert tuple(packed) == (0, value)


@pytest.mark.parametrize("value", [-1, True, 1.5, 2**64])
def test_unsigned_width_selection_rejects_non_uint64_values(value):
    with pytest.raises(ValueError):
        _unsigned_typecode(value)


def test_streaming_unsigned_array_promotes_only_when_a_boundary_is_crossed():
    values = array("B")

    same = _append_adaptive_uint(values, 255)
    assert same is values and values.typecode == "B"
    values = _append_adaptive_uint(values, 256)
    assert values.typecode == "H" and tuple(values) == (255, 256)
    same = _append_adaptive_uint(values, 65_535)
    assert same is values
    values = _append_adaptive_uint(values, 65_536)
    assert values.typecode == "I"
    values = _append_adaptive_uint(values, 2**32)
    assert values.typecode == "Q"
    assert tuple(values) == (255, 256, 65_535, 65_536, 2**32)


def test_posting_stream_freeze_is_exact_immutable_and_releases_staged_payloads():
    streams = [
        bytearray(b"\x00\x01"),
        bytearray(),
        bytearray(b"\x80\x01\xff"),
    ]

    blob, offsets = _freeze_posting_streams(streams)

    assert type(blob) is bytes
    assert blob == b"\x00\x01\x80\x01\xff"
    assert tuple(offsets) == (0, 2, 2, 5)
    assert streams == [bytearray(), bytearray(), bytearray()]
    with pytest.raises(TypeError):
        blob[0] = 0xFF


def test_posting_offsets_promote_during_freeze_without_final_repack():
    streams = [bytearray(b"x" * 255), bytearray(b"y")]

    blob, offsets = _freeze_posting_streams(streams)

    assert blob == b"x" * 255 + b"y"
    assert offsets.typecode == "H"
    assert tuple(offsets) == (0, 255, 256)
    assert all(not stream for stream in streams)


def test_packed_vocabulary_preserves_sequence_and_mapping_contracts():
    terms = (
        "",
        "alpha",
        "한글",
        "nul\0inside",
        "é",
        "e\u0301",
        "😀",
        "\ud800",
        "\udc00",
        "\ud800\udc00",
    )
    snapshot = CompactPostingsSnapshot.from_bm25([(0, terms)])

    assert isinstance(snapshot.terms, Sequence)
    assert type(snapshot.terms) is not tuple
    assert snapshot.terms == terms
    assert terms == snapshot.terms
    assert tuple(snapshot.terms) == terms
    assert repr(snapshot.terms) == repr(terms)
    assert hash(snapshot.terms) == hash(terms)
    assert snapshot.terms[0] == ""
    assert snapshot.terms[-1] == "\ud800\udc00"
    assert snapshot.terms[1:8:2] == terms[1:8:2]
    assert tuple(reversed(snapshot.terms)) == tuple(reversed(terms))
    assert snapshot.terms.count("한글") == 1
    assert snapshot.terms.count("missing") == 0
    assert "\ud800" in snapshot.terms
    assert "missing" not in snapshot.terms
    with pytest.raises(IndexError):
        _ = snapshot.terms[len(terms)]

    assert snapshot.terms.index("alpha") == 1
    assert snapshot.terms.index("alpha", -len(terms), 2) == 1
    with pytest.raises(ValueError):
        snapshot.terms.index("alpha", -1)
    with pytest.raises(ValueError):
        snapshot.terms.index("alpha", 0, -len(terms))
    with pytest.raises(TypeError):
        snapshot.terms.index("alpha", 0, None)

    assert isinstance(snapshot._term_ids, Mapping)
    assert type(snapshot._term_ids) is not dict
    assert len(snapshot._term_ids) == len(terms)
    assert list(snapshot._term_ids) == list(terms)
    assert dict(snapshot._term_ids) == {
        term: term_id for term_id, term in enumerate(terms)
    }
    assert snapshot._term_ids["\udc00"] == 8
    assert snapshot._term_ids.get("missing") is None
    assert snapshot._term_ids.get("missing", 99) == 99
    assert "e\u0301" in snapshot._term_ids
    with pytest.raises(KeyError):
        _ = snapshot._term_ids["missing"]

    class StringSubclass(str):
        pass

    subclass = StringSubclass("alpha")
    assert subclass in snapshot.terms
    assert snapshot.terms.count(subclass) == 1
    assert snapshot.terms.index(subclass) == 1
    assert snapshot._term_ids[subclass] == 1
    assert 42 not in snapshot.terms
    assert snapshot.terms.count(42) == 0
    assert snapshot._term_ids.get(42) is None
    with pytest.raises(ValueError):
        snapshot.terms.index(42)
    with pytest.raises(KeyError):
        _ = snapshot._term_ids[42]

    snapshot._validate_storage()


def test_packed_vocabulary_byte_checks_forced_hash_collisions(monkeypatch):
    monkeypatch.setattr(
        compact_postings,
        "hash",
        lambda _term: 0xA5A5A5A500000001,
        raising=False,
    )
    terms = tuple(f"collision-{index}" for index in range(512)) + (
        "",
        "한글",
        "é",
        "e\u0301",
        "\ud800",
        "\udc00",
        "\ud800\udc00",
        "nul\0inside",
        "😀",
    )

    snapshot = CompactPostingsSnapshot.from_bm25([(0, terms)])

    assert tuple(snapshot.terms) == terms
    assert [snapshot._term_ids[term] for term in terms] == list(range(len(terms)))
    assert snapshot._term_ids.get("collision-absent") is None
    snapshot._validate_storage()


@pytest.mark.parametrize(
    ("encoded_size", "typecode"),
    [(0, "B"), (255, "B"), (256, "H"), (65_535, "H"), (65_536, "I")],
)
def test_packed_vocabulary_offsets_use_minimal_lossless_width(encoded_size, typecode):
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["x" * encoded_size])])

    assert snapshot._vocabulary._offsets.typecode == typecode
    assert tuple(snapshot._vocabulary._offsets) == (0, encoded_size)
    snapshot._validate_storage()


@pytest.mark.parametrize(("term_count", "typecode"), [(255, "B"), (256, "H")])
def test_packed_vocabulary_slots_use_term_id_width(term_count, typecode):
    snapshot = CompactPostingsSnapshot.from_bm25(
        [(0, (f"term-{term_id}" for term_id in range(term_count)))]
    )

    assert snapshot._vocabulary._slots.typecode == typecode
    assert len(snapshot._vocabulary._slots) == compact_postings._packed_slot_capacity(
        term_count
    )
    assert tuple(snapshot.terms) == tuple(
        f"term-{term_id}" for term_id in range(term_count)
    )
    snapshot._validate_storage()


@pytest.mark.parametrize(("term_count", "typecode"), [(65_535, "H"), (65_536, "I")])
def test_packed_vocabulary_slot_width_high_boundaries(term_count, typecode):
    terms = tuple(f"term-{term_id}" for term_id in range(term_count))

    vocabulary = compact_postings._PackedVocabulary.from_validated_terms(terms)

    assert vocabulary._slots.typecode == typecode
    assert vocabulary.find(terms[0]) == 0
    assert vocabulary.find(terms[-1]) == term_count - 1
    assert vocabulary.find("missing") is None
    vocabulary._validate_storage()


def test_empty_packed_vocabulary_has_canonical_sequence_lookup_and_state():
    snapshot = CompactPostingsSnapshot.from_bm25([])

    assert snapshot.terms == ()
    assert tuple(snapshot._term_ids) == ()
    assert snapshot._vocabulary._blob == b""
    assert tuple(snapshot._vocabulary._offsets) == (0,)
    assert snapshot._vocabulary._offsets.typecode == "B"
    assert snapshot._vocabulary._slots.typecode == "B"
    assert len(snapshot._vocabulary._slots) == 8
    assert not any(snapshot._vocabulary._slots)
    assert type(snapshot._doc_positions) is compact_postings._IdentityDocPositions
    assert len(snapshot._doc_positions) == 0
    assert type(snapshot.__getstate__()["terms"]) is tuple
    snapshot._validate_storage()


def test_dense_document_positions_use_a_dict_compatible_identity_view():
    snapshot = CompactPostingsSnapshot.from_bm25(
        [(0, ["alpha"]), (1, ["beta"]), (2, ["gamma"])],
        max_doc_id=2**100,
    )
    positions = snapshot._doc_positions

    assert type(positions) is compact_postings._IdentityDocPositions
    assert isinstance(positions, Mapping)
    assert len(positions) == 3
    assert tuple(positions) == (0, 1, 2)
    assert dict(positions) == {0: 0, 1: 1, 2: 2}
    assert positions[0] == 0
    assert positions.get(2) == 2
    with pytest.raises(AttributeError, match="immutable"):
        positions._count = 4
    for absent in (-1, 3, 2**100):
        assert absent not in positions
        assert positions.get(absent) is None
        with pytest.raises(KeyError):
            _ = positions[absent]
    assert snapshot._document_record(1) is not None
    assert snapshot._document_record(3) is None
    snapshot._validate_storage()


def test_sparse_and_huge_document_ids_keep_the_exact_position_dict():
    doc_ids = (0, 2, 2**100)
    snapshot = CompactPostingsSnapshot.from_bm25(
        [(doc_id, ["term"]) for doc_id in doc_ids]
    )

    assert type(snapshot._doc_positions) is dict
    assert snapshot._doc_positions == {0: 0, 2: 1, 2**100: 2}
    assert _hex_rank(snapshot.search_tokens(["term"])) == _hex_rank(
        _MutableBM25([(doc_id, ["term"]) for doc_id in doc_ids]).search("term")
    )
    snapshot._validate_storage()


def test_storage_audit_rejects_noncanonical_dense_and_sparse_position_views():
    dense = CompactPostingsSnapshot.from_bm25(
        [(0, ["alpha"]), (1, ["beta"]), (2, ["gamma"])]
    )
    dense._doc_positions = {0: 0, 1: 1, 2: 2}
    with pytest.raises(ValueError, match="document lookup"):
        dense._validate_storage()

    sparse = CompactPostingsSnapshot.from_bm25([(0, ["alpha"]), (2, ["beta"])])
    sparse._doc_positions = compact_postings._IdentityDocPositions(2)
    with pytest.raises(ValueError, match="document lookup"):
        sparse._validate_storage()


def test_packed_lookup_lazily_caches_only_resolved_terms():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["alpha", "beta"])])
    lookup = snapshot._term_ids

    assert lookup._resolved == {}
    assert lookup.get("alpha") == 0
    assert lookup._resolved == {"alpha": 0}
    assert lookup.get("alpha") == 0
    assert lookup.get("missing") is None
    assert lookup.get(42) is None
    assert lookup._resolved == {"alpha": 0}

    class StringSubclass(str):
        pass

    assert lookup.get(StringSubclass("beta")) == 1
    assert lookup._resolved == {"alpha": 0, "beta": 1}

    class DisplayOverride(str):
        def __str__(self):
            return "not-the-underlying-value"

    assert lookup.get(DisplayOverride("beta")) == 1
    assert lookup._resolved == {"alpha": 0, "beta": 1}
    cached = lookup._resolved
    snapshot._validate_storage()
    assert lookup._resolved is cached
    assert lookup._resolved == {"alpha": 0, "beta": 1}

    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored._term_ids._resolved == {}


def test_storage_audit_rejects_corrupt_resolved_term_cache():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["alpha", "beta"])])
    snapshot._term_ids._resolved["alpha"] = 1

    with pytest.raises(ValueError, match="lookup cache"):
        snapshot._validate_storage()


def test_last_streamed_document_is_released_before_posting_freeze(monkeypatch):
    class WeakMapping(dict):
        pass

    class WeakTokens(list):
        pass

    references = {}

    def docs():
        tokens = WeakTokens(f"last-{index}" for index in range(20_000))
        raw_fields = WeakMapping(body=tokens)
        references["tokens"] = weakref.ref(tokens)
        references["fields"] = weakref.ref(raw_fields)
        yield 2**100, raw_fields
        del raw_fields, tokens

    original = compact_postings._freeze_posting_streams

    def audited_freeze(streams):
        assert references["fields"]() is None
        assert references["tokens"]() is None
        return original(streams)

    monkeypatch.setattr(compact_postings, "_freeze_posting_streams", audited_freeze)

    snapshot = CompactPostingsSnapshot.from_bm25f(docs(), {"body": 1.0})

    assert snapshot.N == 1
    assert references["fields"]() is None
    assert references["tokens"]() is None


def test_compact_bm25_matches_mutable_random_oracle_exactly():
    rng = random.Random(20260722)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    doc_ids = [0, 3, 9, 27, 2**40 + 5]
    docs = [
        (doc_id, [rng.choice(vocabulary) for _ in range(rng.randrange(0, 28))])
        for doc_id in doc_ids
    ]
    mutable = _MutableBM25(docs)
    compact = CompactPostingsSnapshot.from_bm25(docs, max_doc_id=2**80 + 9)

    assert compact.N == mutable.N
    assert compact.avgdl.hex() == mutable.avgdl.hex()
    assert compact.max_doc_id == 2**80 + 9
    assert {term: value.hex() for term, value in compact.idf.items()} == {
        term: value.hex() for term, value in mutable.idf.items()
    }
    for query in [
        "alpha",
        "alpha alpha beta",
        "zeta gamma alpha",
        "missing",
        "",
    ]:
        tokens = tokenize(query)
        assert _hex_rank(compact.search_tokens(tokens, limit=20)) == _hex_rank(
            mutable.search(query, limit=20)
        )
        for doc_id in doc_ids:
            assert (
                compact.score_tokens(tokens, doc_id).hex()
                == mutable.score(tokens, doc_id).hex()
            )


def test_compact_bm25_preserves_search_multiply_and_score_repeat_add():
    docs = [(0, ["alpha"]), (2, ["alpha", "alpha"])]
    compact = CompactPostingsSnapshot.from_bm25(docs)
    mutable = _MutableBM25(docs)
    tokens = ["alpha"] * 17

    assert _hex_rank(compact.search_tokens(tokens)) == _hex_rank(
        mutable.search(" ".join(tokens))
    )
    for doc_id in (0, 2):
        assert (
            compact.score_tokens(tokens, doc_id).hex()
            == mutable.score(tokens, doc_id).hex()
        )


def test_compact_bm25_matches_static_reference_float_for_float():
    documents = [
        ["alpha", "beta", "alpha"],
        ["beta", "gamma"],
        [],
        ["alpha", "delta", "gamma", "gamma"],
    ]
    static = BM25(documents)
    compact = CompactPostingsSnapshot.from_bm25(enumerate(documents))

    for query in ["alpha", "gamma alpha", "beta beta gamma", "missing"]:
        tokens = tokenize(query)
        assert _hex_rank(compact.search_tokens(tokens)) == _hex_rank(
            static.search(query)
        )
        for doc_id in range(len(documents)):
            assert (
                compact.score_tokens(tokens, doc_id).hex()
                == static.score(tokens, doc_id).hex()
            )


def test_compact_public_search_preserves_korean_query_coordination():
    documents = [
        tokenize("외주화 무엇"),
        tokenize("외주화"),
        tokenize("무엇"),
    ]
    static = BM25(documents)
    compact = CompactPostingsSnapshot._from_bm25_for_vector(
        enumerate(documents)
    )
    query = "외주화란 무엇인가요?"

    assert _hex_rank(compact.search(query, limit=10)) == _hex_rank(
        static.search(query, limit=10)
    )
    assert compact.score(tokenize("외주화"), 1).hex() == static.score(
        tokenize("외주화"), 1
    ).hex()
    assert compact._retains_reverse is False


def test_compact_bm25f_matches_mutable_random_oracle_and_df_dfe_exactly():
    rng = random.Random(42)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
    weights = {"title": 2.0, "body": 1.0, "memory": 1.0}
    evidence = {"memory"}
    docs = []
    for doc_id in [1, 4, 8, 19, 2**45]:
        fields = {
            "title": [rng.choice(vocabulary) for _ in range(rng.randrange(5))],
            "body": [rng.choice(vocabulary) for _ in range(rng.randrange(18))],
            "memory": [rng.choice(vocabulary) for _ in range(rng.randrange(4))],
            "ignored": ["not-indexed"],
        }
        docs.append((doc_id, fields))
    mutable = _MutableBM25F(docs, weights, evidence_fields=evidence)
    compact = CompactPostingsSnapshot.from_bm25f(
        docs,
        weights,
        evidence_fields=evidence | {"ghost"},
        max_doc_id=2**90,
    )

    assert compact.N == mutable.N
    assert compact.max_doc_id == 2**90
    assert {field: value.hex() for field, value in compact.avglen.items()} == {
        field: value.hex() for field, value in mutable.avglen.items()
    }
    assert {term: value.hex() for term, value in compact.idf.items()} == {
        term: value.hex() for term, value in mutable.idf.items()
    }
    assert dict(zip(compact.terms, compact._df)) == mutable._df
    assert {
        term: value for term, value in zip(compact.terms, compact._dfe) if value
    } == mutable._dfe
    for query in ["alpha", "alpha alpha beta", "gamma delta", "missing", ""]:
        tokens = tokenize(query)
        assert _hex_rank(compact.search_tokens(tokens)) == _hex_rank(
            mutable.search(query)
        )
        for doc_id, _fields in docs:
            assert (
                compact.score_tokens(tokens, doc_id).hex()
                == mutable.score(tokens, doc_id).hex()
            )


def test_content_and_evidence_presence_are_counted_independently():
    docs = [
        (0, {"body": ["shared"], "memory": ["shared"]}),
        (1, {"body": [], "memory": ["shared"]}),
    ]
    compact = CompactPostingsSnapshot.from_bm25f(
        docs, {"body": 1.0, "memory": 1.0}, evidence_fields={"memory"}
    )
    term_id = compact.terms.index("shared")

    assert compact._df[term_id] == 1
    assert compact._dfe[term_id] == 2
    assert (
        compact.idf["shared"].hex()
        == _MutableBM25F(
            docs,
            {"body": 1.0, "memory": 1.0},
            evidence_fields={"memory"},
        )
        .idf["shared"]
        .hex()
    )


def test_compact_bm25f_matches_static_reference_float_for_float():
    documents = [
        {"title": ["alpha"], "body": ["beta", "alpha"], "memory": []},
        {"title": ["beta"], "body": ["gamma"], "memory": ["alpha"]},
        {"title": [], "body": [], "memory": ["gamma", "gamma"]},
    ]
    weights = {"title": 2.0, "body": 1.0, "memory": 1.0}
    evidence = {"memory"}
    static = BM25F(documents, weights, evidence_fields=evidence)
    compact = CompactPostingsSnapshot.from_bm25f(
        enumerate(documents), weights, evidence_fields=evidence
    )

    for query in ["alpha", "gamma alpha", "gamma gamma beta", "missing"]:
        tokens = tokenize(query)
        assert _hex_rank(compact.search_tokens(tokens)) == _hex_rank(
            static.search(query)
        )
        for doc_id in range(len(documents)):
            assert (
                compact.score_tokens(tokens, doc_id).hex()
                == static.score(tokens, doc_id).hex()
            )


def test_high_term_frequency_and_arbitrary_doc_id_remain_exact():
    doc_id = 2**100 + 123
    docs = [(doc_id, ["term"] * 20_000)]
    compact = CompactPostingsSnapshot.from_bm25(docs, max_doc_id=2**140)
    mutable = _MutableBM25(docs)

    assert _hex_rank(compact.search_tokens(["term"])) == _hex_rank(
        mutable.search("term")
    )
    assert (
        compact.score_tokens(["term"], doc_id).hex()
        == mutable.score(["term"], doc_id).hex()
    )


def test_weight_cache_builder_streams_common_values_into_final_arrays():
    builder = _WeightCacheBuilder()
    initial_ids = builder._ids
    initial_weights = builder._weights
    source = [
        (0, float.fromhex("0x1.999999999999ap-4")),
        (17, -0.0),
        (_UINT64_MAX, float.fromhex("0x1.fffffffffffffp+100")),
    ]

    for doc_id, weight in source:
        builder.append(doc_id, weight)
    ids, weights = builder.finish()

    assert ids is not initial_ids
    assert initial_ids.typecode == "B" and tuple(initial_ids) == (0, 17)
    assert weights is initial_weights
    assert isinstance(ids, array) and ids.typecode == "Q"
    assert isinstance(weights, array) and weights.typecode == "d"
    assert tuple(ids) == tuple(doc_id for doc_id, _weight in source)
    assert [weight.hex() for weight in weights] == [
        weight.hex() for _doc_id, weight in source
    ]


def test_weight_cache_ids_promote_at_boundaries_then_fallback_beyond_uint64():
    builder = _WeightCacheBuilder()

    for doc_id, typecode in [
        (255, "B"),
        (256, "H"),
        (65_535, "H"),
        (65_536, "I"),
        (2**32 - 1, "I"),
        (2**32, "Q"),
        (_UINT64_MAX, "Q"),
    ]:
        builder.append(doc_id, float(doc_id))
        assert isinstance(builder._ids, array)
        assert builder._ids.typecode == typecode

    builder.append(2**64, 1.0)
    builder.append(2**100, 2.0)
    ids, weights = builder.finish()

    assert type(ids) is tuple
    assert ids == (255, 256, 65_535, 65_536, 2**32 - 1, 2**32, 2**64 - 1, 2**64, 2**100)
    assert len(weights) == len(ids)


def test_weight_cache_builder_switches_once_to_exact_arbitrary_int_ids():
    builder = _WeightCacheBuilder()
    source = [
        (7, float.fromhex("0x1.0000000000001p-3")),
        (_UINT64_MAX + 1, float.fromhex("0x1.fffffffffffffp-2")),
        (2**100 + 9, float.fromhex("0x1.23456789abcdep+5")),
    ]

    for doc_id, weight in source:
        builder.append(doc_id, weight)
    ids, weights = builder.finish()

    assert type(ids) is tuple
    assert ids == tuple(doc_id for doc_id, _weight in source)
    assert isinstance(weights, array) and weights.typecode == "d"
    assert [weight.hex() for weight in weights] == [
        weight.hex() for _doc_id, weight in source
    ]


def test_derived_weight_cache_packs_common_ids_without_narrowing_large_ids():
    compact = CompactPostingsSnapshot.from_bm25(
        [
            (2, ["small", "mixed"]),
            (5, ["small"]),
            (2**100, ["mixed", "huge"]),
        ]
    )

    compact.search_tokens(["small", "mixed", "huge"])
    small_ids, small_weights = compact._weight_cache[compact._term_ids["small"]]
    mixed_ids, mixed_weights = compact._weight_cache[compact._term_ids["mixed"]]
    huge_ids, huge_weights = compact._weight_cache[compact._term_ids["huge"]]

    assert isinstance(small_ids, array) and small_ids.typecode == "B"
    assert isinstance(small_weights, array) and small_weights.typecode == "d"
    assert tuple(small_ids) == (2, 5)
    assert type(mixed_ids) is tuple and mixed_ids == (2, 2**100)
    assert isinstance(mixed_weights, array) and mixed_weights.typecode == "d"
    assert type(huge_ids) is tuple and huge_ids == (2**100,)
    assert isinstance(huge_weights, array) and huge_weights.typecode == "d"


def test_adaptive_weight_cache_is_exact_under_concurrent_cold_queries():
    doc_ids = (1, 256, 65_536, 2**32, _UINT64_MAX, 2**64)
    snapshot = CompactPostingsSnapshot.from_bm25(
        (doc_id, ["shared", f"term-{index}"]) for index, doc_id in enumerate(doc_ids)
    )
    expected = _hex_rank(snapshot.search_tokens(["shared"]))
    snapshot._weight_cache.clear()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: _hex_rank(snapshot.search_tokens(["shared"])),
                range(64),
            )
        )

    assert all(result == expected for result in results)
    ids, weights = snapshot._weight_cache[snapshot._term_ids["shared"]]
    assert type(ids) is tuple and ids == doc_ids
    assert isinstance(weights, array) and weights.typecode == "d"


def test_construction_is_deterministic_across_equivalent_input_containers():
    docs = [
        (7, {"title": ["z", "a", "z"], "body": ["b"]}),
        (2, {"title": ["a"], "body": ["c", "b"]}),
        (19, {"title": [], "body": []}),
    ]
    weights = {"title": 2.0, "body": 1.0}

    second_docs = [
        (2, {"body": ["c", "b"], "title": ["a"]}),
        (7, {"body": ["b"], "title": ["z", "a", "z"]}),
        (19, {"body": [], "title": []}),
    ]
    first = CompactPostingsSnapshot.from_bm25f(sorted(docs), weights, max_doc_id=99)
    second = CompactPostingsSnapshot.from_bm25f(second_docs, weights, max_doc_id=99)

    assert first.__getstate__() == second.__getstate__()
    assert first._posting_blob == second._posting_blob
    assert first._reverse_blob == second._reverse_blob


@pytest.mark.parametrize(
    ("length", "typecode"),
    [(255, "B"), (256, "H"), (65_535, "H"), (65_536, "I")],
)
def test_document_lengths_use_the_minimal_lossless_width(length, typecode):
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["term"] * length)])

    assert snapshot._lengths[0].typecode == typecode
    assert snapshot._totals.typecode == typecode
    assert snapshot._lengths[0][0] == length
    assert snapshot._totals[0] == length


def test_fields_choose_widths_independently_and_totals_do_not_follow_columns():
    snapshot = CompactPostingsSnapshot.from_bm25f(
        [
            (0, {"title": ["title"] * 200, "body": ["body"]}),
            (1, {"title": ["title"] * 200, "body": ["body"]}),
        ],
        {"title": 2.0, "body": 1.0},
    )

    assert [column.typecode for column in snapshot._lengths] == ["B", "B"]
    assert snapshot._totals.typecode == "H"
    assert tuple(snapshot._totals) == (400, 2)


def test_frequency_and_offset_tables_promote_independently_at_256_documents():
    content = CompactPostingsSnapshot.from_bm25(
        (doc_id, ["term"]) for doc_id in range(256)
    )
    evidence = CompactPostingsSnapshot.from_bm25f(
        ((doc_id, {"body": (), "memory": ("term",)}) for doc_id in range(256)),
        {"body": 1.0, "memory": 1.0},
        evidence_fields={"memory"},
    )

    assert content._df.typecode == "H" and tuple(content._df) == (256,)
    assert content._dfe.typecode == "B"
    assert content._lengths[0].typecode == "B"
    assert content._totals.typecode == "H"
    assert content._posting_offsets.typecode == "H"
    assert content._reverse_offsets.typecode == "H"
    assert evidence._df.typecode == "B" and tuple(evidence._df) == (0,)
    assert evidence._dfe.typecode == "H" and tuple(evidence._dfe) == (256,)


def test_fresh_snapshot_passes_the_same_cross_stream_audit_as_pickle():
    snapshot = CompactPostingsSnapshot.from_bm25f(
        [
            (0, {"body": ["alpha", "beta"], "memory": ["alpha"]}),
            (9, {"body": ["gamma"], "memory": []}),
        ],
        {"body": 1.0, "memory": 1.0},
        evidence_fields={"memory"},
        max_doc_id=99,
    )

    snapshot._validate_storage()


def test_pickle_roundtrip_preserves_canonical_state_but_not_derived_cache():
    snapshot = CompactPostingsSnapshot.from_bm25f(
        [
            (0, {"title": ["alpha"], "body": ["beta"]}),
            (4, {"title": ["beta"], "body": ["alpha"]}),
        ],
        {"title": 2.0, "body": 1.0},
        max_doc_id=100,
    )
    expected = _hex_rank(snapshot.search_tokens(["alpha", "beta"]))
    assert snapshot._weight_cache

    restored = pickle.loads(pickle.dumps(snapshot))
    shallow = copy(snapshot)
    deep = deepcopy(snapshot)

    assert restored.__getstate__() == snapshot.__getstate__()
    assert restored._weight_cache == {}
    assert restored.max_doc_id == 100
    assert _hex_rank(restored.search_tokens(["alpha", "beta"])) == expected
    for cloned in (shallow, deep):
        assert cloned is not snapshot
        assert cloned.__getstate__() == snapshot.__getstate__()
        assert cloned._vocabulary is not snapshot._vocabulary
        assert cloned.terms._vocabulary is cloned._vocabulary
        assert cloned._term_ids._vocabulary is cloned._vocabulary
        assert cloned._weight_cache == {}
        assert _hex_rank(cloned.search_tokens(["alpha", "beta"])) == expected


def test_v1_state_stays_tuple_based_and_restores_old_wide_runtime_arrays_minimally():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["alpha"]), (1, ["beta"])])
    snapshot.search_tokens(["alpha"])
    assert snapshot._term_ids._resolved == {"alpha": 0}
    assert type(snapshot._doc_positions) is compact_postings._IdentityDocPositions
    array_names = (
        "_totals",
        "_df",
        "_dfe",
        "_posting_offsets",
        "_reverse_offsets",
    )
    for name in array_names:
        setattr(snapshot, name, array("Q", getattr(snapshot, name)))
    snapshot._lengths = tuple(array("Q", column) for column in snapshot._lengths)

    snapshot._validate_storage()
    state = snapshot.__getstate__()
    restored = pickle.loads(pickle.dumps(snapshot))
    copied = deepcopy(snapshot)

    assert state["state_version"] == 1
    assert not {"term_blob", "term_offsets", "term_slots"} & state.keys()
    for key in (
        "terms",
        "totals",
        "lengths",
        "df",
        "dfe",
        "posting_offsets",
        "reverse_offsets",
    ):
        assert type(state[key]) is tuple
    assert restored.__getstate__() == state
    assert copied.__getstate__() == state
    assert restored._totals.typecode == "B"
    assert all(column.typecode == "B" for column in restored._lengths)
    assert restored._df.typecode == "B"
    assert restored._posting_offsets.typecode == "B"
    assert restored._reverse_offsets.typecode == "B"
    assert restored._term_ids._resolved == {}
    assert copied._term_ids._resolved == {}
    assert type(restored._doc_positions) is compact_postings._IdentityDocPositions
    assert type(copied._doc_positions) is compact_postings._IdentityDocPositions


def test_empty_snapshots_and_oov_queries_do_not_grow_cache():
    plain = CompactPostingsSnapshot.from_bm25([], max_doc_id=5)
    fielded = CompactPostingsSnapshot.from_bm25f(
        [(8, {"ignored": ["term"]})], {}, evidence_fields={"ghost"}
    )

    assert plain.N == 0
    assert plain.max_doc_id == 5
    assert plain.search_tokens(["missing"]) == []
    assert plain._weight_cache == {}
    assert fielded.N == 1
    assert fielded.terms == ()
    assert fielded.search_tokens(["missing"]) == []
    assert fielded.score_tokens(["missing"], 8) == 0.0


def test_limit_and_tie_contract_matches_mutable_index():
    docs = [(9, ["same"]), (2**70, ["same"]), (2**80, ["same"])]
    compact = CompactPostingsSnapshot.from_bm25(docs)
    mutable = _MutableBM25(sorted(docs))

    for limit in (-1, 0, 1, 2, 20):
        assert _hex_rank(compact.search_tokens(["same"], limit=limit)) == _hex_rank(
            mutable.search("same", limit=limit)
        )


@pytest.mark.parametrize("bad_id", [-1, True, 1.5, "1"])
def test_invalid_document_ids_are_rejected(bad_id):
    with pytest.raises(ValueError):
        CompactPostingsSnapshot.from_bm25([(bad_id, ["term"])])


def test_invalid_later_token_does_not_publish_a_partial_object():
    consumed = []

    def docs():
        consumed.append(0)
        yield 0, ["valid"]
        consumed.append(1)
        yield 1, ["valid", object()]

    with pytest.raises(TypeError):
        CompactPostingsSnapshot.from_bm25(docs())
    assert consumed == [0, 1]


def test_plain_stream_consumes_a_reused_token_buffer_before_advancing():
    shared = []

    def docs():
        shared[:] = ["alpha"]
        yield 0, shared
        shared[:] = ["beta", "beta"]
        yield 1, shared

    compact = CompactPostingsSnapshot.from_bm25(docs())

    assert [doc_id for doc_id, _score in compact.search_tokens(["alpha"])] == [0]
    assert [doc_id for doc_id, _score in compact.search_tokens(["beta"])] == [1]


def test_fielded_stream_consumes_a_reused_mapping_before_advancing():
    shared = {"body": [], "memory": []}

    def docs():
        shared["body"][:] = ["alpha"]
        shared["memory"][:] = ["proof"]
        yield 0, shared
        shared["body"][:] = ["beta", "beta"]
        shared["memory"][:] = []
        yield 1, shared

    compact = CompactPostingsSnapshot.from_bm25f(
        docs(), {"body": 1.0, "memory": 1.0}, evidence_fields={"memory"}
    )

    assert [doc_id for doc_id, _score in compact.search_tokens(["alpha"])] == [0]
    assert [doc_id for doc_id, _score in compact.search_tokens(["proof"])] == [0]
    assert [doc_id for doc_id, _score in compact.search_tokens(["beta"])] == [1]


def test_pickle_state_rejects_schema_and_high_water_corruption():
    snapshot = CompactPostingsSnapshot.from_bm25([(2, ["term"])], max_doc_id=9)

    wrong_version = snapshot.__getstate__()
    wrong_version["state_version"] = 999
    with pytest.raises(ValueError, match="unsupported"):
        _restore(wrong_version)

    low_high_water = snapshot.__getstate__()
    low_high_water["max_doc_id"] = 1
    with pytest.raises(ValueError, match="highest"):
        _restore(low_high_water)


def test_failed_pickle_restore_leaves_an_existing_snapshot_unchanged():
    target = CompactPostingsSnapshot.from_bm25([(0, ["kept"])], max_doc_id=7)
    before = target.__getstate__()
    corrupt = CompactPostingsSnapshot.from_bm25([(2, ["other"])]).__getstate__()
    corrupt["df"] = (0,)

    with pytest.raises(ValueError):
        target.__setstate__(corrupt)

    assert target.__getstate__() == before
    assert target.search_tokens(["kept"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.update(posting_offsets=(1, len(state["posting_blob"]))),
        lambda state: state.update(posting_offsets=(0, len(state["posting_blob"]) + 1)),
        lambda state: state.update(reverse_offsets=(0, 0)),
        lambda state: state.update(df=(0,)),
        lambda state: state.update(totals=(2,)),
        lambda state: state.update(lengths=((2,),)),
    ],
)
def test_pickle_state_rejects_offset_and_stat_corruption(mutate):
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["term"])])
    state = deepcopy(snapshot.__getstate__())
    mutate(state)

    with pytest.raises(ValueError):
        _restore(state)


def test_pickle_state_rejects_noncanonical_and_cross_stream_corruption():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["term"])])

    noncanonical = snapshot.__getstate__()
    noncanonical["posting_blob"] = b"\x80\x00\x01"
    noncanonical["posting_offsets"] = (0, 3)
    with pytest.raises(ValueError, match="non-canonical"):
        _restore(noncanonical)

    mismatched_tf = snapshot.__getstate__()
    mismatched_tf["posting_blob"] = b"\x00\x02"
    with pytest.raises(ValueError, match="forward and reverse"):
        _restore(mismatched_tf)

    trailing_reverse = snapshot.__getstate__()
    trailing_reverse["reverse_blob"] += b"\x00"
    trailing_reverse["reverse_offsets"] = (0, len(trailing_reverse["reverse_blob"]))
    with pytest.raises(ValueError, match="trailing"):
        _restore(trailing_reverse)


def test_pickle_state_rejects_a_vocabulary_term_without_postings():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["term"])])
    state = snapshot.__getstate__()
    state["terms"] += ("ghost",)
    state["df"] += (0,)
    state["dfe"] += (0,)
    state["posting_offsets"] += (state["posting_offsets"][-1],)

    with pytest.raises(ValueError, match="require a posting"):
        _restore(state)


def test_pickle_state_still_rejects_duplicate_untrusted_terms():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["alpha", "beta"])])
    state = snapshot.__getstate__()
    state["terms"] = ("alpha", "alpha")

    with pytest.raises(ValueError, match="unique strings"):
        _restore(state)


def test_installed_metadata_validation_does_not_iterate_the_public_term_view(
    monkeypatch,
):
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["alpha", "beta"])])

    def reject_iteration(_terms):
        raise AssertionError("installed metadata validation decoded terms twice")

    monkeypatch.setattr(compact_postings._PackedTerms, "__iter__", reject_iteration)

    snapshot._validate_installed_metadata()


@pytest.mark.parametrize(
    "attribute",
    [
        ("_term_ids", {"alpha": 1, "beta": 0}),
        ("_doc_positions", {2: 1, 5: 0}),
    ],
)
def test_storage_audit_rejects_corrupt_derived_lookups(attribute):
    snapshot = CompactPostingsSnapshot.from_bm25([(2, ["alpha"]), (5, ["beta"])])
    name, value = attribute
    setattr(snapshot, name, value)

    with pytest.raises(ValueError, match="lookup"):
        snapshot._validate_storage()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda vocabulary: setattr(vocabulary, "_blob", bytearray(vocabulary._blob)),
        lambda vocabulary: setattr(vocabulary, "_offsets", array("b", [0, 5, 9])),
        lambda vocabulary: setattr(vocabulary, "_slots", array("I", vocabulary._slots)),
        lambda vocabulary: setattr(vocabulary, "_mask", 0),
        lambda vocabulary: vocabulary._slots.__setitem__(
            next(index for index, entry in enumerate(vocabulary._slots) if entry), 0
        ),
        lambda vocabulary: vocabulary._slots.__setitem__(
            next(index for index, entry in enumerate(vocabulary._slots) if not entry),
            next(entry for entry in vocabulary._slots if entry),
        ),
    ],
)
def test_storage_audit_rejects_corrupt_packed_vocabulary(mutate):
    snapshot = CompactPostingsSnapshot.from_bm25(
        [(0, ["alpha", "beta"]), (4, ["beta"])]
    )
    mutate(snapshot._vocabulary)

    with pytest.raises(ValueError, match="vocabulary|offset|lookup"):
        snapshot._validate_storage()


def test_packed_lookup_bounds_probe_on_a_corrupt_full_table():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["alpha", "beta"])])
    snapshot._vocabulary._slots = array(
        snapshot._vocabulary._slots.typecode,
        [1] * len(snapshot._vocabulary._slots),
    )

    with pytest.raises(ValueError, match="terminating empty slot"):
        snapshot._vocabulary.find("missing")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda snapshot: setattr(snapshot, "max_doc_id", 1),
        lambda snapshot: (
            setattr(snapshot, "doc_ids", (2, 2)),
            setattr(snapshot, "_doc_positions", {2: 1}),
        ),
        lambda snapshot: (
            setattr(snapshot, "terms", (42, "beta")),
            setattr(snapshot, "_term_ids", {42: 0, "beta": 1}),
        ),
        lambda snapshot: (
            setattr(snapshot, "terms", ("alpha", "alpha")),
            setattr(snapshot, "_term_ids", {"alpha": 1}),
        ),
        lambda snapshot: setattr(snapshot, "_posting_blob", bytearray()),
        lambda snapshot: setattr(snapshot, "_reverse_blob", bytearray()),
        lambda snapshot: setattr(snapshot, "_totals", array("i", [2])),
        lambda snapshot: setattr(snapshot, "_posting_offsets", array("L", [0, 2, 4])),
        lambda snapshot: setattr(snapshot, "fields", ["body"]),
        lambda snapshot: setattr(snapshot, "_df", array("Q", [3, 1])),
    ],
)
def test_storage_audit_revalidates_all_installed_metadata(mutate):
    snapshot = CompactPostingsSnapshot.from_bm25([(2, ["alpha"]), (5, ["beta"])])
    mutate(snapshot)

    with pytest.raises(ValueError):
        snapshot._validate_storage()


@pytest.mark.parametrize("typecode", ["b", "h", "i", "q", "L"])
def test_storage_audit_rejects_signed_and_platform_variable_numeric_arrays(typecode):
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["term"])])
    snapshot._totals = array(typecode, snapshot._totals)

    with pytest.raises(ValueError, match="numeric arrays"):
        snapshot._validate_storage()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("totals", (True,)),
        ("lengths", ((True,),)),
        ("df", (True,)),
        ("posting_offsets", (False, 2)),
        ("reverse_offsets", (False, 4)),
    ],
)
def test_pickle_state_rejects_boolean_numeric_metadata(key, value):
    state = CompactPostingsSnapshot.from_bm25([(0, ["term"])]).__getstate__()
    state[key] = value

    with pytest.raises(ValueError):
        _restore(state)


def test_storage_metadata_audit_is_read_only_and_preserves_warm_cache():
    snapshot = CompactPostingsSnapshot.from_bm25([(0, ["term"])])
    snapshot.search_tokens(["term"])
    before = snapshot.__dict__.copy()
    cache = snapshot._weight_cache
    cached = dict(cache)

    snapshot._validate_storage()

    assert snapshot._weight_cache is cache
    assert snapshot._weight_cache == cached
    assert snapshot.__dict__ == before


def test_non_increasing_ids_and_non_string_tokens_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        CompactPostingsSnapshot.from_bm25([(0, ["a"]), (0, ["b"])])
    with pytest.raises(ValueError, match="strictly increasing"):
        CompactPostingsSnapshot.from_bm25([(2, ["a"]), (1, ["b"])])
    with pytest.raises(TypeError, match="str"):
        CompactPostingsSnapshot.from_bm25([(0, ["a", 1])])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_non_finite_scoring_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        CompactPostingsSnapshot.from_bm25([(0, ["term"])], k1=value)
    with pytest.raises(ValueError, match="finite"):
        CompactPostingsSnapshot.from_bm25f([(0, {"body": ["term"]})], {"body": value})
