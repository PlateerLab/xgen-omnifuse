from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
import copy
import pickle
import random

import pytest

from omnifuse._compact_mutable_fielded import CompactMutableBM25F
from omnifuse._compact_postings import CompactPostingsSnapshot
from omnifuse.text import _MutableBM25F, tokenize


WEIGHTS = {"title": 4.0, "body": 1.0, "memory": 1.0}
EVIDENCE = {"memory"}
QUERIES = ["alpha", "beta beta gamma", "delta alpha beta", "missing", ""]


class _NoIterTerms(Sequence):
    def __init__(self, values):
        self._values = values
        self.iterations = 0

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        self.iterations += 1
        raise AssertionError("mutation must not iterate the base vocabulary")


class _GetOnlyTermIds(Mapping):
    def __init__(self, values):
        self._values = values
        self.get_calls = 0

    def get(self, key, default=None):
        self.get_calls += 1
        return self._values.get(key, default)

    def __getitem__(self, key):
        raise AssertionError("term lookup must use Mapping.get")

    def __iter__(self):
        raise AssertionError("mutation must not iterate the term lookup")

    def __len__(self):
        raise AssertionError("mutation must not size the term lookup")


def _hex_rank(ranked):
    return [(doc_id, score.hex()) for doc_id, score in ranked]


def _assert_lazy_base_metadata(compact):
    assert type(compact._terms) is not dict
    assert type(compact._df) is not dict
    assert type(compact._dfe) is not dict
    assert compact._terms._base is compact._base
    assert compact._df._base is compact._base
    assert compact._dfe._base is compact._base
    assert compact._df._values is compact._base._df
    assert compact._dfe._values is compact._base._dfe
    assert compact._terms == {term: term for term in compact._base.terms}
    assert compact._df == {
        term: int(compact._base._df[term_id])
        for term_id, term in enumerate(compact._base.terms)
        if compact._base._df[term_id]
    }
    assert compact._dfe == {
        term: int(compact._base._dfe[term_id])
        for term_id, term in enumerate(compact._base.terms)
        if compact._base._dfe[term_id]
    }


def _assert_oracle(compact, mutable, state, queries=QUERIES):
    assert compact.N == mutable.N == len(state)
    assert {field: value.hex() for field, value in compact.avglen.items()} == {
        field: value.hex() for field, value in mutable.avglen.items()
    }
    assert compact._df == mutable._df
    assert compact._dfe == mutable._dfe
    assert set(compact.idf) == set(mutable.idf)
    for term, expected in mutable.idf.items():
        assert compact.idf[term].hex() == expected.hex()
    for query in queries:
        tokens = tokenize(query)
        for limit in (-1, 0, 1, 3, 20):
            expected = _hex_rank(mutable.search(query, limit=limit))
            assert _hex_rank(compact.search_tokens(tokens, limit=limit)) == expected
            assert _hex_rank(compact.search(query, limit=limit)) == expected
        for doc_id in state:
            expected = mutable.score(tokens, doc_id).hex()
            assert compact.score_tokens(tokens, doc_id).hex() == expected
            assert compact.score(tokens, doc_id).hex() == expected


def _mutable(state, weights=WEIGHTS, evidence=EVIDENCE):
    return _MutableBM25F(state.items(), weights, evidence_fields=evidence)


def test_initial_fielded_state_is_float_exact_with_sparse_large_ids():
    state = {
        2: {
            "title": ["alpha", "beta"],
            "body": ["alpha", "alpha"],
            "memory": [],
            "ignored": ["not-indexed"],
        },
        8: {"title": [], "body": ["beta", "gamma"], "memory": ["alpha"]},
        13: {},
        2**70: {"title": ["delta"], "body": ["alpha"], "memory": ["gamma"]},
    }
    compact = CompactMutableBM25F(
        state.items(), WEIGHTS, k1=1.2, b=0.63, idf_pow=1.0, evidence_fields=EVIDENCE
    )
    mutable = _MutableBM25F(
        state.items(), WEIGHTS, k1=1.2, b=0.63, idf_pow=1.0, evidence_fields=EVIDENCE
    )

    _assert_oracle(compact, mutable, state)
    _assert_lazy_base_metadata(compact)
    assert compact.storage_stats()["override_docs"] == 0
    assert compact.storage_stats()["base_docs"] == len(state)


def test_first_small_fielded_mutation_does_not_iterate_or_copy_base_vocabulary():
    docs = [(doc_id, {"body": [f"term-{doc_id}"]}) for doc_id in range(512)]
    compact = CompactMutableBM25F(docs, WEIGHTS, evidence_fields=EVIDENCE)
    original_terms = compact._base.terms
    original_term_ids = compact._base._term_ids
    base_term_id = original_term_ids["term-0"]
    guarded_terms = _NoIterTerms(original_terms)
    guarded_term_ids = _GetOnlyTermIds(original_term_ids)
    compact._base.terms = guarded_terms
    compact._base._term_ids = guarded_term_ids

    assert compact.upsert(0, {"body": ["replacement"]})

    assert guarded_terms.iterations == 0
    assert guarded_term_ids.get_calls == 3
    assert compact._metadata._base_df_delta == {base_term_id: -1}
    assert compact._metadata._base_dfe_delta == {}
    assert set(compact._metadata._extras) == {"replacement"}
    assert compact.storage_stats()["base_df_deltas"] == 1
    assert compact.storage_stats()["base_dfe_deltas"] == 0
    assert compact.storage_stats()["extra_terms"] == 1


def test_fielded_layered_views_track_independent_df_and_dfe_transitions():
    compact = CompactMutableBM25F(
        [
            (1, {"body": ["pivot"], "memory": ["pivot"]}),
            (4, {"body": ["stable"]}),
        ],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )

    assert compact.upsert(1, {"memory": ["pivot"]})
    assert isinstance(compact._terms, Mapping)
    assert isinstance(compact._df, Mapping)
    assert isinstance(compact._dfe, Mapping)
    assert list(compact._terms) == ["pivot", "stable"]
    assert compact._terms == {"pivot": "pivot", "stable": "stable"}
    assert compact._df == {"stable": 1}
    assert compact._dfe == {"pivot": 1}
    assert compact._df.get("pivot") is None
    assert compact._dfe.get("pivot") == 1
    with pytest.raises(KeyError):
        compact._df["pivot"]
    with pytest.raises(TypeError):
        compact._dfe["pivot"] = 2
    assert compact._metadata._base_df_delta
    assert compact._metadata._base_dfe_delta == {}

    assert compact.upsert(1, {})
    assert "pivot" not in compact._terms
    assert "pivot" not in compact._df
    assert "pivot" not in compact._dfe
    assert compact.storage_stats()["base_df_deltas"] == 1
    assert compact.storage_stats()["base_dfe_deltas"] == 1

    assert compact.upsert(1, {"memory": ["pivot"]})
    assert "pivot" in compact._terms
    assert compact._df.get("pivot") is None
    assert compact._dfe["pivot"] == 1
    assert compact.storage_stats()["base_dfe_deltas"] == 0

    assert compact.upsert(1, {"body": ["pivot"], "memory": ["pivot"]})
    assert compact._overrides == {}
    assert compact.storage_stats()["base_df_deltas"] == 0
    assert compact.storage_stats()["base_dfe_deltas"] == 0
    assert compact.storage_stats()["extra_terms"] == 0
    _assert_lazy_base_metadata(compact)


def test_fielded_mapping_orders_track_each_stat_reactivation_independently():
    compact = CompactMutableBM25F(
        [
            (1, {"body": ["pivot"], "memory": ["pivot"]}),
            (4, {"body": ["stable"]}),
        ],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )
    assert compact.upsert(7, {"body": ["content-extra"], "memory": ["evidence-extra"]})

    assert compact.upsert(1, {"memory": ["pivot"]})
    assert list(compact._terms) == [
        "pivot",
        "stable",
        "content-extra",
        "evidence-extra",
    ]
    assert list(compact._df) == ["stable", "content-extra"]
    assert list(compact._dfe) == ["pivot", "evidence-extra"]

    assert compact.upsert(1, {"body": ["pivot"], "memory": ["pivot"]})
    assert list(compact._terms) == [
        "pivot",
        "stable",
        "content-extra",
        "evidence-extra",
    ]
    assert list(compact._df) == ["stable", "content-extra", "pivot"]
    assert list(compact._dfe) == ["pivot", "evidence-extra"]

    assert compact.upsert(1, {})
    assert list(compact._terms) == ["stable", "content-extra", "evidence-extra"]
    assert compact.upsert(1, {"body": ["pivot"], "memory": ["pivot"]})
    assert list(compact._terms) == [
        "stable",
        "content-extra",
        "evidence-extra",
        "pivot",
    ]
    assert list(compact._df) == ["stable", "content-extra", "pivot"]
    assert list(compact._dfe) == ["evidence-extra", "pivot"]
    assert compact.storage_stats()["base_df_deltas"] == 0
    assert compact.storage_stats()["base_dfe_deltas"] == 0


def test_new_fielded_terms_preserve_first_input_order_across_pickle():
    compact = CompactMutableBM25F(
        [(0, {"body": ["seed"]})], WEIGHTS, evidence_fields=EVIDENCE
    )
    new_terms = ["zeta", "alpha", "gamma", "mu", "beta", "omega", "delta"]
    assert compact.upsert(
        1,
        {
            "title": new_terms[:2],
            "body": new_terms[2:5],
            "memory": new_terms[5:],
        },
    )
    expected = ["seed", *new_terms]

    assert list(compact._terms) == expected
    assert list(compact._df) == ["seed", *new_terms[:5]]
    assert list(compact._dfe) == new_terms[5:]
    loaded = pickle.loads(pickle.dumps(compact))
    assert list(loaded._terms) == expected
    assert list(loaded._df) == ["seed", *new_terms[:5]]
    assert list(loaded._dfe) == new_terms[5:]


def test_evidence_content_transitions_revert_and_term_reintroduction_are_exact():
    state = {
        1: {"body": ["alpha"], "memory": ["rare"]},
        4: {"body": ["beta"], "memory": []},
    }
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _mutable(state)

    transitions = [
        {"body": ["rare"], "memory": ["rare"]},
        {"body": ["rare"], "memory": []},
        {"body": [], "memory": ["rare"]},
        {"body": ["alpha"], "memory": ["rare"]},
    ]
    for fields in transitions:
        assert compact.upsert(1, fields) == mutable.upsert(1, fields)
        state[1] = fields
        _assert_oracle(compact, mutable, state)

    assert compact._overrides == {}
    _assert_lazy_base_metadata(compact)
    loaded = pickle.loads(pickle.dumps(compact))
    _assert_lazy_base_metadata(loaded)
    _assert_oracle(loaded, mutable, state)
    assert compact.upsert(1, {"body": ["alpha"], "memory": []})
    assert type(compact._terms) is not dict
    assert type(compact._df) is not dict
    assert type(compact._dfe) is not dict
    assert compact.storage_stats()["base_dfe_deltas"] == 1
    assert mutable.upsert(1, {"body": ["alpha"], "memory": []})
    state[1] = {"body": ["alpha"], "memory": []}
    assert "rare" not in compact._terms
    _assert_oracle(compact, mutable, state)

    assert compact.upsert(9, {"memory": ["rare", "rare"]})
    assert mutable.upsert(9, {"memory": ["rare", "rare"]})
    state[9] = {"memory": ["rare", "rare"]}
    _assert_oracle(compact, mutable, state)


def test_fielded_exact_batch_restore_bypasses_sparse_reverse_patch(monkeypatch):
    state = {
        2: {"title": ["alpha"], "body": ["beta"], "memory": []},
        7: {"title": [], "body": ["gamma"], "memory": ["evidence"]},
    }
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _mutable(state)
    replacements = {
        2: {"body": ["changed"], "memory": ["new-evidence"]},
        7: {"title": ["other"]},
    }
    assert compact.upsert_many(replacements.items()) == 2
    assert mutable.upsert(2, replacements[2])
    assert mutable.upsert(7, replacements[7])
    compact.search_tokens(["changed", "other"])
    cache = compact._weight_cache
    metadata = compact._metadata
    version = compact._mutation_version

    def must_not_reverse_patch(*_args, **_kwargs):
        raise AssertionError("exact full-base restore must not reverse sparse metadata")

    monkeypatch.setattr(type(metadata), "prepare_patch", must_not_reverse_patch)
    assert compact.upsert_many(state.items()) == 2
    assert mutable.upsert(2, state[2])
    assert mutable.upsert(7, state[7])

    assert compact._overrides == {}
    assert compact._delta_postings == {}
    assert compact._metadata is not metadata
    assert compact._weight_cache is cache
    assert compact._weight_cache == {}
    assert compact._mutation_version == version + 1
    _assert_lazy_base_metadata(compact)
    _assert_oracle(compact, mutable, state)


def test_fielded_partial_tombstone_restore_keeps_sparse_path_and_high_water(
    monkeypatch,
):
    original_state = {
        1: {"body": ["alpha"], "memory": ["evidence"]},
        4: {"body": ["beta"]},
    }
    state = copy.deepcopy(original_state)
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _mutable(state)
    replacements = {
        1: {"body": ["changed"]},
        4: {"memory": ["other"]},
    }
    assert compact.upsert_many(replacements.items()) == 2
    assert mutable.upsert(1, replacements[1])
    assert mutable.upsert(4, replacements[4])
    state.update(copy.deepcopy(replacements))

    prepare_calls = []
    metadata_type = type(compact._metadata)
    original_prepare = metadata_type.prepare_patch

    def tracked_prepare(metadata, *args, **kwargs):
        prepare_calls.append((tuple(args[0]), tuple(args[1])))
        return original_prepare(metadata, *args, **kwargs)

    monkeypatch.setattr(metadata_type, "prepare_patch", tracked_prepare)
    assert compact.upsert_many([(1, original_state[1])]) == 1
    assert mutable.upsert(1, original_state[1])
    state[1] = copy.deepcopy(original_state[1])
    assert prepare_calls
    assert set(compact._overrides) == {4}
    _assert_oracle(compact, mutable, state)

    assert compact.upsert(1, {"memory": ["changed-again"]})
    assert mutable.upsert(1, {"memory": ["changed-again"]})
    state[1] = {"memory": ["changed-again"]}
    assert compact.delete(4)
    assert mutable.delete(4)
    del state[4]
    call_count = len(prepare_calls)
    assert compact.upsert_many([(1, original_state[1])]) == 1
    assert mutable.upsert(1, original_state[1])
    state[1] = copy.deepcopy(original_state[1])
    assert len(prepare_calls) == call_count + 1
    assert compact._overrides == {4: None}
    _assert_oracle(compact, mutable, state)

    assert compact.upsert(10, {"body": ["temporary"]})
    assert mutable.upsert(10, {"body": ["temporary"]})
    state[10] = {"body": ["temporary"]}
    assert compact.delete(10)
    assert mutable.delete(10)
    del state[10]
    assert compact._max_doc_id == 10
    assert compact.compact()
    assert compact._base.max_doc_id == 10
    _assert_oracle(compact, mutable, state)
    with pytest.raises(ValueError, match="greater than"):
        compact.upsert(10, {"body": ["reused"]})


def test_zero_and_negative_field_weights_preserve_mutable_float_order():
    weights = {"title": 2.0, "body": 0.0, "memory": -0.25}
    state = {
        0: {"title": ["alpha"], "body": ["beta"], "memory": []},
        3: {"title": ["beta"], "body": ["alpha"], "memory": ["gamma"]},
        7: {"title": ["gamma"], "body": [], "memory": []},
    }
    compact = CompactMutableBM25F(state.items(), weights, evidence_fields=EVIDENCE)
    mutable = _MutableBM25F(state.items(), weights, evidence_fields=EVIDENCE)

    _assert_oracle(compact, mutable, state, ["alpha", "beta", "gamma alpha"])


def test_fielded_base_lengths_are_indexed_directly_across_sparse_mutations():
    huge_id = 2**80 + 7
    state = {
        2: {"title": ["target"], "body": ["target"], "memory": []},
        9: {"title": [], "body": ["target", "other"], "memory": []},
        huge_id: {
            "title": ["target"],
            "body": ["target", "target"],
            "memory": ["target"],
        },
    }
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _MutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    replacement = {
        "title": ["target", "updated"],
        "body": [],
        "memory": ["target"],
    }
    inserted = {
        "title": [],
        "body": ["target", "inserted"],
        "memory": ["target"],
    }
    assert compact.upsert(9, replacement) == mutable.upsert(9, replacement)
    assert compact.delete(2) == mutable.delete(2)
    assert compact.upsert(huge_id + 10, inserted) == mutable.upsert(
        huge_id + 10, inserted
    )

    original_columns = compact._base._lengths
    indexed_fields: list[int] = []

    class _IndexOnlyLengthColumns(Sequence):
        def __len__(self):
            return len(original_columns)

        def __getitem__(self, index):
            indexed_fields.append(index)
            return original_columns[index]

        def __iter__(self):
            raise AssertionError("base scoring must not materialize a length row")

    compact._base._lengths = _IndexOnlyLengthColumns()
    expected = _hex_rank(mutable.search("target"))
    assert _hex_rank(compact.search_tokens(["target"])) == expected
    assert indexed_fields == [0, 1]
    ids, weights = compact._weight_cache[compact._terms["target"]]
    assert type(ids) is tuple and tuple(ids) == (9, huge_id, huge_id + 10)
    expected_scores = dict(expected)
    assert [weight.hex() for weight in weights] == [
        expected_scores[doc_id] for doc_id in ids
    ]


def test_direct_noop_and_repeated_override_churn_are_bounded():
    state = {
        0: {"title": ["seed"], "body": [], "memory": []},
        5: {"title": [], "body": ["other"], "memory": []},
    }
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _mutable(state)
    compact.search_tokens(["seed"])
    cache = compact._weight_cache
    cached = dict(cache)
    terms = compact._terms
    df = compact._df
    dfe = compact._dfe
    version = compact._mutation_version

    assert compact.upsert(0, {"title": ["seed"], "ignored": ["x"]}) is False
    assert compact.delete(999) is False
    assert compact._mutation_version == version
    assert compact._weight_cache is cache
    assert compact._weight_cache == cached
    assert compact._terms is terms
    assert compact._df is df
    assert compact._dfe is dfe
    _assert_lazy_base_metadata(compact)

    vocabulary = ["alpha", "beta", "gamma", "delta"]
    for step in range(160):
        term = vocabulary[step % len(vocabulary)]
        fields = {
            "title": [term] * (1 + step % 3),
            "body": ["body", term],
            "memory": [term] if step % 2 else [],
        }
        assert compact.upsert(0, fields) == mutable.upsert(0, fields)
        if step == 0:
            assert type(compact._terms) is not dict
            assert type(compact._df) is not dict
            assert type(compact._dfe) is not dict
        state[0] = fields
        stats = compact.storage_stats()
        assert stats["override_docs"] == 1
        assert stats["live_overrides"] == 1
        assert stats["delta_postings"] <= 3

    _assert_oracle(compact, mutable, state)


def test_fielded_query_cache_uses_compact_numeric_storage():
    compact = CompactMutableBM25F(
        [(0, {"body": ["term"]}), (3, {"body": ["term"]})],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )

    compact.search_tokens(["term"])
    ids, weights = compact._weight_cache[compact._terms["term"]]

    assert ids.typecode == "B"
    assert weights.typecode == "d"
    assert tuple(ids) == (0, 3)


def test_fielded_query_cache_preserves_exact_mixed_width_document_ids():
    huge_id = 2**80 + 7
    state = {
        0: {"body": ["term"]},
        3: {"title": ["term"], "body": ["term", "term"]},
        huge_id: {"memory": ["term"]},
    }
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _MutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)

    expected = _hex_rank(mutable.search("term"))
    assert _hex_rank(compact.search_tokens(["term"])) == expected
    ids, weights = compact._weight_cache[compact._terms["term"]]

    assert type(ids) is tuple and ids == (0, 3, huge_id)
    assert isinstance(weights, array) and weights.typecode == "d"
    expected_weights = {
        doc_id: mutable.score(["term"], doc_id).hex() for doc_id in state
    }
    assert [weight.hex() for weight in weights] == [
        expected_weights[doc_id] for doc_id in ids
    ]


def test_batch_and_source_validation_failures_are_atomic():
    compact = CompactMutableBM25F(
        [(2, {"body": ["alpha"]}), (5, {"body": ["beta"]})],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )
    compact.search_tokens(["alpha"])
    before = pickle.dumps(compact.__getstate__())
    cache = compact._weight_cache
    entries = dict(cache)
    terms = compact._terms
    df = compact._df
    dfe = compact._dfe

    with pytest.raises(TypeError, match="only str"):
        compact.upsert_many([(2, {"body": ["changed"]}), (8, {"memory": [object()]})])
    with pytest.raises(TypeError, match="field mappings"):
        compact.upsert_many([(8, ["not-a-mapping"])])
    with pytest.raises(ValueError, match="greater than"):
        compact.upsert_many([(8, {"body": ["first"]}), (7, {"body": ["descending"]})])
    with pytest.raises(ValueError, match="duplicate batch"):
        compact.upsert_many([(2, {"body": ["first"]}), (2, {"body": ["duplicate"]})])
    with pytest.raises(ValueError, match="duplicate batch"):
        compact.delete_many([2, 2])

    def interrupted():
        yield 2, {"body": ["changed"]}
        raise RuntimeError("source interrupted")

    with pytest.raises(RuntimeError, match="source interrupted"):
        compact.upsert_many(interrupted())

    assert pickle.dumps(compact.__getstate__()) == before
    assert compact._weight_cache is cache
    assert compact._weight_cache == entries
    assert compact._terms is terms
    assert compact._df is df
    assert compact._dfe is dfe
    _assert_lazy_base_metadata(compact)


def test_compaction_preserves_version_warm_cache_and_high_water():
    state = {
        1: {"title": ["alpha"], "body": [], "memory": []},
        4: {"title": [], "body": ["beta"], "memory": []},
    }
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _mutable(state)
    assert compact.upsert(1, {"body": ["alpha", "changed"]})
    assert mutable.upsert(1, {"body": ["alpha", "changed"]})
    state[1] = {"body": ["alpha", "changed"]}
    assert compact.delete(4)
    assert mutable.delete(4)
    del state[4]
    assert compact.upsert(9, {"title": ["inserted"], "memory": ["alpha"]})
    assert mutable.upsert(9, {"title": ["inserted"], "memory": ["alpha"]})
    state[9] = {"title": ["inserted"], "memory": ["alpha"]}

    expected = _hex_rank(compact.search_tokens(["alpha", "inserted"]))
    cache = compact._weight_cache
    entries = dict(cache)
    version = compact._mutation_version
    layout = compact._layout_epoch
    assert compact.compact()

    assert compact._mutation_version == version
    assert compact._layout_epoch == layout + 1
    assert compact._weight_cache is cache
    for term, cached in entries.items():
        assert compact._weight_cache[term] is cached
    assert _hex_rank(compact.search_tokens(["alpha", "inserted"])) == expected
    assert compact._base.max_doc_id == 9
    assert compact.storage_stats()["override_docs"] == 0
    _assert_lazy_base_metadata(compact)
    assert compact.compact() is False
    _assert_oracle(compact, mutable, state)

    loaded = pickle.loads(pickle.dumps(compact))
    _assert_lazy_base_metadata(loaded)
    with pytest.raises(ValueError, match="greater than"):
        loaded.upsert(8, {"body": ["reserved-gap"]})
    assert loaded.upsert(10, {"body": ["valid"]})


def test_compaction_candidate_failure_is_atomic(monkeypatch):
    compact = CompactMutableBM25F(
        [(1, {"body": ["alpha"]})], WEIGHTS, evidence_fields=EVIDENCE
    )
    assert compact.upsert(1, {"body": ["changed"]})
    compact.search_tokens(["changed"])
    before = pickle.dumps(compact.__getstate__())
    cache = compact._weight_cache
    entries = dict(cache)

    def explode(*_args, **_kwargs):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(CompactPostingsSnapshot, "from_bm25f", explode)
    with pytest.raises(RuntimeError, match="candidate failed"):
        compact.compact()

    assert pickle.dumps(compact.__getstate__()) == before
    assert compact._weight_cache is cache
    assert compact._weight_cache == entries


def test_fielded_v1_restore_rebuilds_only_overridden_documents(monkeypatch):
    state = {doc_id: {"body": [f"term-{doc_id}"]} for doc_id in range(40)}
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    assert compact.upsert(3, {"memory": ["changed"]})
    assert compact.delete(17)
    assert compact.upsert(100, {"body": ["inserted"]})
    serialized = compact.__getstate__()
    assert set(serialized) == {
        "state_version",
        "k1",
        "b",
        "idf_pow",
        "weights",
        "evidence_fields",
        "base",
        "overrides",
        "max_doc_id",
        "mutation_version",
        "layout_epoch",
    }

    calls = []
    original = CompactPostingsSnapshot._document_record

    def tracked_document_record(snapshot, doc_id):
        calls.append(doc_id)
        return original(snapshot, doc_id)

    monkeypatch.setattr(
        CompactPostingsSnapshot, "_document_record", tracked_document_record
    )
    restored = CompactMutableBM25F._from_state(serialized)

    assert calls == [3, 17, 100]
    assert restored.__getstate__()["overrides"] == serialized["overrides"]
    assert restored._terms == compact._terms
    assert restored._df == compact._df
    assert restored._dfe == compact._dfe
    assert restored.storage_stats()["base_df_deltas"] == 2
    assert restored.storage_stats()["base_dfe_deltas"] == 0
    assert restored.storage_stats()["extra_terms"] == 2


def test_fielded_copy_variants_rebuild_independent_sparse_metadata():
    compact = CompactMutableBM25F(
        [(1, {"body": ["alpha"]}), (4, {"memory": ["beta"]})],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )
    assert compact.upsert(1, {"body": ["alpha"], "memory": ["extra"]})
    assert compact.delete(4)
    original_state = pickle.dumps(compact.__getstate__())

    for cloned in (copy.copy(compact), copy.deepcopy(compact)):
        cloned_state = cloned.__getstate__()
        compact_state = compact.__getstate__()
        assert cloned is not compact
        assert cloned._metadata is not compact._metadata
        assert cloned._terms == compact._terms
        assert cloned._df == compact._df
        assert cloned._dfe == compact._dfe
        assert {key: value for key, value in cloned_state.items() if key != "base"} == {
            key: value for key, value in compact_state.items() if key != "base"
        }
        assert cloned._base.__getstate__() == compact._base.__getstate__()
        assert cloned.upsert(8, {"body": ["clone-only"]})
        assert "clone-only" in cloned._terms
        assert "clone-only" not in compact._terms
        assert pickle.dumps(compact.__getstate__()) == original_state


def test_pickle_roundtrip_with_override_tombstone_and_unknown_evidence_is_exact():
    evidence = EVIDENCE | {"future-field"}
    state = {
        2: {"title": ["alpha"], "body": [], "memory": []},
        5: {"title": [], "body": ["beta"], "memory": []},
    }
    compact = CompactMutableBM25F(state.items(), WEIGHTS, evidence_fields=evidence)
    mutable = _MutableBM25F(state.items(), WEIGHTS, evidence_fields=evidence)
    assert compact.upsert(2, {"body": ["alpha", "changed"]})
    assert mutable.upsert(2, {"body": ["alpha", "changed"]})
    state[2] = {"body": ["alpha", "changed"]}
    assert compact.delete(5)
    assert mutable.delete(5)
    del state[5]
    assert compact.upsert(9, {"memory": ["inserted"]})
    assert mutable.upsert(9, {"memory": ["inserted"]})
    state[9] = {"memory": ["inserted"]}
    compact.search_tokens(["alpha", "inserted"])

    loaded = pickle.loads(pickle.dumps(compact))

    assert loaded._weight_cache == {}
    assert loaded.evidence_fields == evidence
    assert loaded._mutation_version == compact._mutation_version
    assert loaded._max_doc_id == compact._max_doc_id
    assert loaded.storage_stats() == compact.storage_stats()
    _assert_oracle(loaded, mutable, state)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.update(state_version=999),
        lambda state: state.update(max_doc_id=-1),
        lambda state: state.update(mutation_version=-1),
        lambda state: state.update(weights={"body": 1.0, "title": 4.0, "memory": 1.0}),
        lambda state: state.update(evidence_fields=frozenset({"body"})),
        lambda state: state.update(overrides={1: ((1, 0, 0), ({"x": 1}, {}, {}))}),
        lambda state: state.update(overrides={7: ((1,), ({"x": 1},))}),
        lambda state: state.update(overrides={7: ((1, 0, 0), ({"x": 0}, {}, {}))}),
    ],
)
def test_pickle_state_corruption_is_rejected_failure_atomically(mutate):
    target = CompactMutableBM25F(
        [(2, {"body": ["kept"]})], WEIGHTS, evidence_fields=EVIDENCE
    )
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    mutate(state)

    with pytest.raises(ValueError):
        target.__setstate__(state)

    assert pickle.dumps(target.__getstate__()) == before
    assert target.search_tokens(["kept"])


def test_pickle_state_revalidates_nested_base_and_derived_lookups():
    target = CompactMutableBM25F(
        [(2, {"body": ["alpha"]}), (5, {"memory": ["beta"]})],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )
    before = pickle.dumps(target.__getstate__())

    for attribute, value in [
        ("_term_ids", {"alpha": 1, "beta": 0}),
        ("_doc_positions", {2: 1, 5: 0}),
    ]:
        state = copy.deepcopy(target.__getstate__())
        setattr(state["base"], attribute, value)
        with pytest.raises(ValueError, match="lookup"):
            target.__setstate__(state)
        assert pickle.dumps(target.__getstate__()) == before

    state = copy.deepcopy(target.__getstate__())
    state["base"]._posting_blob = b"\x02\x02"
    with pytest.raises(ValueError):
        target.__setstate__(state)
    assert pickle.dumps(target.__getstate__()) == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda base: setattr(base, "max_doc_id", 0),
        lambda base: (
            setattr(base, "doc_ids", (2, 2)),
            setattr(base, "_doc_positions", {2: 1}),
        ),
        lambda base: (
            setattr(base, "terms", (42, "beta")),
            setattr(base, "_term_ids", {42: 0, "beta": 1}),
        ),
        lambda base: (
            setattr(base, "terms", ("alpha", "alpha")),
            setattr(base, "_term_ids", {"alpha": 1}),
        ),
        lambda base: setattr(base, "_posting_blob", bytearray(base._posting_blob)),
        lambda base: setattr(base, "_reverse_blob", bytearray(base._reverse_blob)),
    ],
)
def test_pickle_state_rejects_nested_base_metadata_atomically(mutate):
    target = CompactMutableBM25F(
        [(2, {"body": ["alpha"]}), (5, {"memory": ["beta"]})],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    mutate(state["base"])

    with pytest.raises(ValueError):
        target.__setstate__(state)

    assert pickle.dumps(target.__getstate__()) == before
    assert target.search_tokens(["alpha"])


@pytest.mark.parametrize("value", [2**64, 2**80])
def test_pickle_rejects_field_lengths_and_frequencies_above_storage_domain(value):
    target = CompactMutableBM25F([], WEIGHTS, evidence_fields=EVIDENCE)
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    state["overrides"] = {0: ((value, 0, 0), ({"huge": value}, {}, {}))}
    state["max_doc_id"] = 0

    with pytest.raises(ValueError, match="record|frequencies"):
        target.__setstate__(state)
    assert pickle.dumps(target.__getstate__()) == before


def test_aggregate_field_overflow_is_rejected_on_restore_and_before_mutation():
    target = CompactMutableBM25F([], WEIGHTS, evidence_fields=EVIDENCE)
    before = pickle.dumps(target.__getstate__())
    half = 2**63
    state = copy.deepcopy(target.__getstate__())
    state["overrides"] = {
        0: ((half, 0, 0), ({"a": half}, {}, {})),
        1: ((half, 0, 0), ({"b": half}, {}, {})),
    }
    state["max_doc_id"] = 1
    with pytest.raises(ValueError, match="field total"):
        target.__setstate__(state)
    assert pickle.dumps(target.__getstate__()) == before

    maximum = 2**64 - 1
    state = copy.deepcopy(target.__getstate__())
    state["overrides"] = {0: ((maximum, 0, 0), ({"a": maximum}, {}, {}))}
    state["max_doc_id"] = 0
    target.__setstate__(state)
    before_max = pickle.dumps(target.__getstate__())
    with pytest.raises(ValueError, match="field total"):
        target.upsert(1, {"title": ["b"]})
    assert pickle.dumps(target.__getstate__()) == before_max


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_non_finite_fielded_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        CompactMutableBM25F([], {"body": value})
    with pytest.raises(ValueError, match="finite"):
        CompactMutableBM25F([], {"body": 1.0}, k1=value)


def test_randomized_mutations_compaction_and_pickle_match_mutable_oracle():
    rng = random.Random(20260722)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
    state: dict[int, dict[str, list[str]]] = {}
    compact = CompactMutableBM25F([], WEIGHTS, evidence_fields=EVIDENCE)
    mutable = _MutableBM25F([], WEIGHTS, evidence_fields=EVIDENCE)
    next_id = 0

    def document():
        return {
            field: [rng.choice(vocabulary) for _ in range(rng.randint(0, maximum))]
            for field, maximum in (("title", 4), ("body", 10), ("memory", 3))
        }

    for step in range(180):
        action = rng.choice(["insert", "insert", "update", "delete", "noop"])
        if action == "insert" or not state:
            doc_id = next_id
            next_id += rng.randint(1, 4)
            fields = document()
            assert compact.upsert(doc_id, fields) == mutable.upsert(doc_id, fields)
            state[doc_id] = fields
        elif action == "update":
            doc_id = rng.choice(list(state))
            fields = document()
            assert compact.upsert(doc_id, fields) == mutable.upsert(doc_id, fields)
            state[doc_id] = fields
        elif action == "delete":
            doc_id = rng.choice(list(state))
            assert compact.delete(doc_id) == mutable.delete(doc_id)
            del state[doc_id]
        else:
            doc_id = rng.choice(list(state))
            equivalent = {
                field: list(reversed(tokens)) for field, tokens in state[doc_id].items()
            }
            version = compact._mutation_version
            cache = compact._weight_cache
            assert compact.upsert(doc_id, equivalent) is False
            assert compact._mutation_version == version
            assert compact._weight_cache is cache

        _assert_oracle(compact, mutable, state)
        if step % 11 == 0 and compact._overrides:
            version = compact._mutation_version
            assert compact.compact()
            assert compact._mutation_version == version
            _assert_oracle(compact, mutable, state)
        if step % 17 == 0:
            compact = pickle.loads(pickle.dumps(compact))
            assert compact._weight_cache == {}
            _assert_oracle(compact, mutable, state)
