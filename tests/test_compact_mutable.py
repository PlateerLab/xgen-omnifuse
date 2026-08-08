from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
import copy
import pickle
import random

import pytest

from omnifuse._compact_mutable import CompactMutableBM25
from omnifuse._compact_postings import CompactPostingsSnapshot
from omnifuse.text import _MutableBM25, tokenize


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
    assert compact._terms._base is compact._base
    assert compact._df._base is compact._base
    assert compact._df._values is compact._base._df
    assert compact._terms == {term: term for term in compact._base.terms}
    assert compact._df == {
        term: int(compact._base._df[term_id])
        for term_id, term in enumerate(compact._base.terms)
        if compact._base._df[term_id]
    }


def _assert_oracle(compact, mutable, state, queries=QUERIES):
    assert compact.N == mutable.N == len(state)
    assert compact.avgdl.hex() == mutable.avgdl.hex()
    assert set(compact.idf) == set(mutable.idf)
    for term, expected in mutable.idf.items():
        assert compact.idf[term].hex() == expected.hex()
    for query in queries:
        tokens = tokenize(query)
        for limit in (-1, 0, 1, 3, 20):
            assert _hex_rank(compact.search_tokens(tokens, limit=limit)) == _hex_rank(
                mutable.search(query, limit=limit)
            )
        for doc_id in state:
            assert (
                compact.score_tokens(tokens, doc_id).hex()
                == mutable.score(tokens, doc_id).hex()
            )


def test_compact_mutable_bm25_initial_state_is_exact():
    state = {
        2: ["alpha", "beta", "alpha"],
        8: ["beta", "gamma"],
        13: [],
        2**70: ["delta", "alpha"],
    }
    compact = CompactMutableBM25(state.items(), k1=1.2, b=0.63, idf_pow=1.0)
    mutable = _MutableBM25(state.items(), k1=1.2, b=0.63, idf_pow=1.0)

    _assert_oracle(compact, mutable, state)
    _assert_lazy_base_metadata(compact)
    assert compact.storage_stats()["override_docs"] == 0
    assert compact.storage_stats()["base_docs"] == len(state)


def test_first_small_mutation_does_not_iterate_or_copy_base_vocabulary():
    docs = [(doc_id, [f"term-{doc_id}"]) for doc_id in range(512)]
    compact = CompactMutableBM25(docs)
    original_terms = compact._base.terms
    original_term_ids = compact._base._term_ids
    base_term_id = original_term_ids["term-0"]
    guarded_terms = _NoIterTerms(original_terms)
    guarded_term_ids = _GetOnlyTermIds(original_term_ids)
    compact._base.terms = guarded_terms
    compact._base._term_ids = guarded_term_ids

    assert compact.upsert(0, ["replacement"])

    assert guarded_terms.iterations == 0
    assert guarded_term_ids.get_calls == 3
    assert compact._metadata._base_df_delta == {base_term_id: -1}
    assert set(compact._metadata._extras) == {"replacement"}
    assert compact.storage_stats()["base_df_deltas"] == 1
    assert compact.storage_stats()["extra_terms"] == 1


def test_layered_metadata_views_follow_mapping_contract_and_retire_sparse_state():
    compact = CompactMutableBM25([(1, ["rare", "stable"]), (4, ["stable"])])

    assert compact.upsert(1, ["stable", "new"])
    assert isinstance(compact._terms, Mapping)
    assert isinstance(compact._df, Mapping)
    assert list(compact._terms) == ["stable", "new"]
    assert dict(compact._terms) == {"stable": "stable", "new": "new"}
    assert compact._terms == {"stable": "stable", "new": "new"}
    assert compact._df == {"stable": 2, "new": 1}
    assert compact._terms.get("new") == "new"
    assert compact._df.get("new") == 1
    assert compact._terms.get("missing") is None
    assert compact._df.get("rare") is None
    assert "rare" not in compact._terms
    assert "rare" not in compact._df
    assert len(compact._terms) == 2
    assert len(compact._df) == 2
    with pytest.raises(KeyError):
        compact._terms["rare"]
    with pytest.raises(KeyError):
        compact._df["rare"]
    with pytest.raises(TypeError):
        compact._terms["other"] = "other"
    with pytest.raises(TypeError):
        del compact._df["stable"]

    assert compact.upsert(1, ["stable"])
    assert compact.storage_stats()["extra_terms"] == 0
    assert "new" not in compact._terms
    assert compact.upsert(1, ["rare", "stable"])
    assert compact._overrides == {}
    assert compact.storage_stats()["base_df_deltas"] == 0
    assert compact.storage_stats()["extra_terms"] == 0
    _assert_lazy_base_metadata(compact)


def test_reactivated_base_term_moves_to_mapping_tail_without_dense_state():
    compact = CompactMutableBM25([(1, ["alpha"]), (4, ["beta"])])
    assert compact.upsert(7, ["persistent"])
    assert compact.upsert(1, [])
    assert list(compact._terms) == ["beta", "persistent"]
    assert list(compact._df) == ["beta", "persistent"]

    assert compact.upsert(1, ["alpha"])

    assert set(compact._overrides) == {7}
    assert list(compact._terms) == ["beta", "persistent", "alpha"]
    assert list(compact._df) == ["beta", "persistent", "alpha"]
    assert compact.storage_stats()["base_df_deltas"] == 0
    assert len(compact._metadata._term_tail) == 2
    assert len(compact._metadata._df_tail) == 2


def test_plain_metadata_rejects_evidence_statistics_without_mutation():
    compact = CompactMutableBM25([(1, ["alpha"])])
    before = compact._metadata.sparse_counts()

    with pytest.raises(ValueError, match="cannot contain evidence"):
        compact._metadata.prepare_patch({}, {"alpha": 1})

    assert compact._metadata.sparse_counts() == before
    assert compact._metadata._dfe_tail == {}


def test_extra_terms_share_the_first_canonical_string_object():
    compact = CompactMutableBM25([])
    first = "term-" + "x" * 300
    equal_but_distinct = first.encode().decode()
    assert equal_but_distinct == first
    assert equal_but_distinct is not first

    assert compact.upsert_many([(1, [first]), (2, [equal_but_distinct])]) == 2

    canonical = compact._terms[first]
    assert canonical is first
    assert next(iter(compact._overrides[1][1])) is canonical
    assert next(iter(compact._overrides[2][1])) is canonical
    assert compact._metadata._extras[first].term is canonical


def test_base_override_revert_and_direct_noop_have_distinct_versions():
    state = {2: ["alpha", "beta"], 7: ["gamma"]}
    compact = CompactMutableBM25(state.items())
    mutable = _MutableBM25(state.items())

    assert compact.upsert(2, ["alpha", "changed"])
    assert type(compact._terms) is not dict
    assert type(compact._df) is not dict
    assert compact.storage_stats()["base_df_deltas"] == 1
    assert compact.storage_stats()["extra_terms"] == 1
    assert mutable.upsert(2, ["alpha", "changed"])
    state[2] = ["alpha", "changed"]
    _assert_oracle(compact, mutable, state)
    assert set(compact._overrides) == {2}

    before_version = compact._mutation_version
    assert compact.upsert(2, ["alpha", "beta"])
    assert mutable.upsert(2, ["alpha", "beta"])
    state[2] = ["alpha", "beta"]
    assert compact._mutation_version == before_version + 1
    assert compact._overrides == {}
    _assert_lazy_base_metadata(compact)
    _assert_oracle(compact, mutable, state)

    loaded = pickle.loads(pickle.dumps(compact))
    _assert_lazy_base_metadata(loaded)
    _assert_oracle(loaded, mutable, state)

    compact.search_tokens(["alpha"])
    cache = compact._weight_cache
    cached_alpha = cache[compact._terms["alpha"]]
    terms = compact._terms
    df = compact._df
    version = compact._mutation_version
    assert compact.upsert(2, ["beta", "alpha"]) is False
    assert compact.delete(999) is False
    assert compact._weight_cache is cache
    assert compact._weight_cache[compact._terms["alpha"]] is cached_alpha
    assert compact._mutation_version == version
    assert compact._terms is terms
    assert compact._df is df


def test_exact_batch_restore_to_base_bypasses_sparse_reverse_patch(monkeypatch):
    state = {2: ["alpha", "beta"], 7: ["gamma"]}
    compact = CompactMutableBM25(state.items())
    mutable = _MutableBM25(state.items())
    assert compact.upsert_many([(2, ["changed"]), (7, ["other"])]) == 2
    assert mutable.upsert(2, ["changed"])
    assert mutable.upsert(7, ["other"])
    compact.search_tokens(["changed", "other"])
    cache = compact._weight_cache
    metadata = compact._metadata
    version = compact._mutation_version

    def must_not_reverse_patch(*_args, **_kwargs):
        raise AssertionError("exact full-base restore must not reverse sparse metadata")

    monkeypatch.setattr(type(metadata), "prepare_patch", must_not_reverse_patch)
    assert compact.upsert_many([(2, state[2]), (7, state[7])]) == 2
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


def test_partial_and_tombstoned_restores_keep_sparse_path_and_high_water(monkeypatch):
    original_state = {1: ["alpha"], 4: ["beta"]}
    state = copy.deepcopy(original_state)
    compact = CompactMutableBM25(state.items())
    mutable = _MutableBM25(state.items())
    assert compact.upsert_many([(1, ["changed"]), (4, ["other"])]) == 2
    assert mutable.upsert(1, ["changed"])
    assert mutable.upsert(4, ["other"])
    state.update({1: ["changed"], 4: ["other"]})

    prepare_calls = []
    metadata_type = type(compact._metadata)
    original_prepare = metadata_type.prepare_patch

    def tracked_prepare(metadata, *args, **kwargs):
        prepare_calls.append(tuple(args[0]))
        return original_prepare(metadata, *args, **kwargs)

    monkeypatch.setattr(metadata_type, "prepare_patch", tracked_prepare)
    assert compact.upsert_many([(1, original_state[1])]) == 1
    assert mutable.upsert(1, original_state[1])
    state[1] = original_state[1]
    assert prepare_calls
    assert set(compact._overrides) == {4}
    _assert_oracle(compact, mutable, state)

    assert compact.upsert(1, ["changed-again"])
    assert mutable.upsert(1, ["changed-again"])
    state[1] = ["changed-again"]
    assert compact.delete(4)
    assert mutable.delete(4)
    del state[4]
    call_count = len(prepare_calls)
    assert compact.upsert_many([(1, original_state[1])]) == 1
    assert mutable.upsert(1, original_state[1])
    state[1] = original_state[1]
    assert len(prepare_calls) == call_count + 1
    assert compact._overrides == {4: None}
    _assert_oracle(compact, mutable, state)

    assert compact.upsert(10, ["temporary"])
    assert mutable.upsert(10, ["temporary"])
    state[10] = ["temporary"]
    assert compact.delete(10)
    assert mutable.delete(10)
    del state[10]
    assert compact._max_doc_id == 10
    assert compact.compact()
    assert compact._base.max_doc_id == 10
    _assert_oracle(compact, mutable, state)
    with pytest.raises(ValueError, match="greater than"):
        compact.upsert(10, ["reused"])


def test_retired_base_term_can_be_reintroduced_without_ghost_postings():
    state = {1: ["rare", "alpha"], 4: ["rare", "beta"]}
    compact = CompactMutableBM25(state.items())
    mutable = _MutableBM25(state.items())

    assert compact.upsert(1, ["alpha"])
    assert mutable.upsert(1, ["alpha"])
    state[1] = ["alpha"]
    assert compact.delete(4)
    assert mutable.delete(4)
    del state[4]
    assert "rare" not in compact._terms
    assert compact.search_tokens(["rare"]) == []
    assert "rare" not in compact._weight_cache
    _assert_oracle(compact, mutable, state)

    assert compact.upsert(9, ["rare", "rare"])
    assert mutable.upsert(9, ["rare", "rare"])
    state[9] = ["rare", "rare"]
    _assert_oracle(compact, mutable, state)
    assert [doc_id for doc_id, _score in compact.search_tokens(["rare"])] == [9]

    assert compact.delete(9)
    assert mutable.delete(9)
    del state[9]
    assert compact.compact()
    assert "rare" not in compact._terms
    assert compact.search_tokens(["rare"]) == []
    _assert_oracle(compact, mutable, state)


def test_insert_delete_compact_and_pickle_preserve_high_water():
    compact = CompactMutableBM25([(2, ["base"])])
    assert compact.upsert(10, ["temporary"])
    assert compact.delete(10)
    assert compact._max_doc_id == 10
    assert compact.compact()
    assert compact._base.max_doc_id == 10
    _assert_lazy_base_metadata(compact)

    loaded = pickle.loads(pickle.dumps(compact))
    assert loaded._max_doc_id == 10
    _assert_lazy_base_metadata(loaded)
    with pytest.raises(ValueError, match="greater than"):
        loaded.upsert(9, ["reserved-gap"])
    with pytest.raises(ValueError, match="greater than"):
        loaded.upsert(10, ["reused"])
    assert loaded.upsert(11, ["valid"])


def test_repeated_override_churn_stays_bounded():
    compact = CompactMutableBM25([(0, ["seed"]), (5, ["other"])])
    vocabulary = ["alpha", "beta", "gamma", "delta"]

    for step in range(200):
        tokens = [vocabulary[step % len(vocabulary)]] * (1 + step % 4)
        assert compact.upsert(0, tokens)
        stats = compact.storage_stats()
        assert stats["override_docs"] == 1
        assert stats["live_overrides"] == 1
        assert stats["delta_postings"] == 1
        assert stats["delta_terms"] == 1

    assert len(compact._overrides) == 1
    assert sum(len(posting) for posting in compact._delta_postings.values()) == 1


def test_term_cache_lazily_merges_filtered_base_and_unsorted_delta(monkeypatch):
    state = {
        1: ["term"],
        4: ["term"],
        7: ["term"],
        10: ["term"],
    }
    compact = CompactMutableBM25(state.items())
    assert compact.upsert(10, ["term", "term"])
    state[10] = ["term", "term"]
    assert compact.upsert(4, ["term", "term", "term"])
    state[4] = ["term", "term", "term"]
    assert compact.delete(7)
    del state[7]
    huge_id = 2**70
    assert compact.upsert(huge_id, ["term"])
    state[huge_id] = ["term"]

    original_decode = compact._base._decode_posting
    decoded_ids = []

    def tracked_decode(term_id):
        for posting in original_decode(term_id):
            decoded_ids.append(posting[0])
            yield posting

    monkeypatch.setattr(compact._base, "_decode_posting", tracked_decode)
    frequencies = compact._term_frequencies("term")

    assert decoded_ids == []
    iterator = iter(frequencies)
    assert next(iterator) == (1, 1)
    assert decoded_ids == [1]
    assert list(iterator) == [(4, 3), (10, 2), (huge_id, 1)]

    mutable = _MutableBM25(state.items())
    assert _hex_rank(compact.search_tokens(["term"])) == _hex_rank(
        mutable.search("term")
    )
    ids, weights = compact._weight_cache[compact._terms["term"]]
    assert type(ids) is tuple and ids == (1, 4, 10, huge_id)
    assert isinstance(weights, array) and weights.typecode == "d"


def test_compaction_preserves_logical_version_and_warm_cache_identity():
    compact = CompactMutableBM25([(1, ["alpha"]), (4, ["beta"])])
    assert compact.upsert(1, ["alpha", "changed"])
    assert compact.delete(4)
    assert compact.upsert(9, ["inserted", "alpha"])
    expected = _hex_rank(compact.search_tokens(["alpha", "inserted"]))
    cache = compact._weight_cache
    cache_entries = dict(cache)
    version = compact._mutation_version
    layout = compact._layout_epoch

    assert compact.compact()

    assert compact._mutation_version == version
    assert compact._layout_epoch == layout + 1
    assert compact._weight_cache is cache
    for term, cached in cache_entries.items():
        assert compact._weight_cache[term] is cached
    assert _hex_rank(compact.search_tokens(["alpha", "inserted"])) == expected
    assert compact.storage_stats()["override_docs"] == 0
    _assert_lazy_base_metadata(compact)
    assert compact.compact() is False


def test_compaction_candidate_failure_is_atomic(monkeypatch):
    compact = CompactMutableBM25([(1, ["alpha"])])
    assert compact.upsert(1, ["changed"])
    compact.search_tokens(["changed"])
    before_state = pickle.dumps(compact.__getstate__())
    cache = compact._weight_cache
    cache_entries = dict(cache)
    terms = compact._terms
    df = compact._df

    def explode(*_args, **_kwargs):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(CompactPostingsSnapshot, "from_bm25", explode)
    with pytest.raises(RuntimeError, match="candidate failed"):
        compact.compact()

    assert pickle.dumps(compact.__getstate__()) == before_state
    assert compact._weight_cache is cache
    assert compact._weight_cache == cache_entries
    assert compact._terms is terms
    assert compact._df is df


def test_batch_validation_and_source_failures_are_atomic():
    class ExplodingToken(str):
        def __hash__(self):
            raise RuntimeError("must never hash")

    compact = CompactMutableBM25([(2, ["alpha"]), (5, ["beta"])])
    compact.search_tokens(["alpha"])
    before_state = pickle.dumps(compact.__getstate__())
    cache = compact._weight_cache
    cache_entries = dict(cache)
    terms = compact._terms
    df = compact._df

    with pytest.raises(TypeError, match="only str"):
        compact.upsert_many([(2, ["changed"]), (8, [ExplodingToken("boom")])])
    with pytest.raises(ValueError, match="greater than"):
        compact.upsert_many([(8, ["first"]), (7, ["second-invalid"])])
    with pytest.raises(ValueError, match="duplicate batch"):
        compact.upsert_many([(2, ["first"]), (2, ["duplicate"])])
    with pytest.raises(ValueError, match="duplicate batch"):
        compact.delete_many([2, 2])
    with pytest.raises(ValueError, match="non-negative int"):
        compact.delete_many([2, True])

    def interrupted():
        yield 2, ["changed"]
        raise RuntimeError("source interrupted")

    with pytest.raises(RuntimeError, match="source interrupted"):
        compact.upsert_many(interrupted())

    assert pickle.dumps(compact.__getstate__()) == before_state
    assert compact._weight_cache is cache
    assert compact._weight_cache == cache_entries
    assert compact._terms is terms
    assert compact._df is df
    _assert_lazy_base_metadata(compact)


def test_sparse_metadata_prepare_failure_is_atomic(monkeypatch):
    compact = CompactMutableBM25([(2, ["alpha"]), (5, ["beta"])])
    compact.search_tokens(["alpha"])
    before = pickle.dumps(compact.__getstate__())
    metadata = compact._metadata
    terms = compact._terms
    df = compact._df
    cache = compact._weight_cache
    cache_entries = dict(cache)

    def fail_prepare(*_args, **_kwargs):
        raise RuntimeError("injected sparse metadata failure")

    monkeypatch.setattr(type(metadata), "prepare_patch", fail_prepare)
    with pytest.raises(RuntimeError, match="injected"):
        compact.upsert(2, ["changed"])

    assert pickle.dumps(compact.__getstate__()) == before
    assert compact._metadata is metadata
    assert compact._terms is terms
    assert compact._df is df
    assert compact._weight_cache is cache
    assert compact._weight_cache == cache_entries


def test_v1_restore_rebuilds_metadata_from_only_overridden_documents(monkeypatch):
    state = {doc_id: [f"term-{doc_id}"] for doc_id in range(40)}
    compact = CompactMutableBM25(state.items())
    assert compact.upsert(3, ["changed"])
    assert compact.delete(17)
    assert compact.upsert(100, ["inserted"])
    serialized = compact.__getstate__()
    assert set(serialized) == {
        "state_version",
        "k1",
        "b",
        "idf_pow",
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
    restored = CompactMutableBM25._from_state(serialized)

    assert calls == [3, 17, 100]
    assert restored.__getstate__()["overrides"] == serialized["overrides"]
    assert restored._df == compact._df
    assert restored._terms == compact._terms
    assert restored.storage_stats()["base_df_deltas"] == 2
    assert restored.storage_stats()["extra_terms"] == 2


def test_copy_variants_rebuild_independent_sparse_metadata():
    compact = CompactMutableBM25([(1, ["alpha"]), (4, ["beta"])])
    assert compact.upsert(1, ["alpha", "extra"])
    assert compact.delete(4)
    original_state = pickle.dumps(compact.__getstate__())

    for cloned in (copy.copy(compact), copy.deepcopy(compact)):
        cloned_state = cloned.__getstate__()
        compact_state = compact.__getstate__()
        assert cloned is not compact
        assert cloned._metadata is not compact._metadata
        assert cloned._terms == compact._terms
        assert cloned._df == compact._df
        assert {key: value for key, value in cloned_state.items() if key != "base"} == {
            key: value for key, value in compact_state.items() if key != "base"
        }
        assert cloned._base.__getstate__() == compact._base.__getstate__()
        assert cloned.upsert(8, ["clone-only"])
        assert "clone-only" in cloned._terms
        assert "clone-only" not in compact._terms
        assert pickle.dumps(compact.__getstate__()) == original_state


def test_pickle_roundtrip_with_live_overrides_and_tombstones_is_exact():
    state = {2: ["alpha"], 5: ["beta"]}
    compact = CompactMutableBM25(state.items())
    mutable = _MutableBM25(state.items())
    assert compact.upsert(2, ["alpha", "changed"])
    assert mutable.upsert(2, ["alpha", "changed"])
    state[2] = ["alpha", "changed"]
    assert compact.delete(5)
    assert mutable.delete(5)
    del state[5]
    assert compact.upsert(9, ["inserted"])
    assert mutable.upsert(9, ["inserted"])
    state[9] = ["inserted"]
    compact.search_tokens(["alpha", "inserted"])

    loaded = pickle.loads(pickle.dumps(compact))

    assert loaded._weight_cache == {}
    assert loaded._mutation_version == compact._mutation_version
    assert loaded._max_doc_id == compact._max_doc_id
    assert loaded.storage_stats() == compact.storage_stats()
    _assert_oracle(loaded, mutable, state)
    assert loaded.upsert(12, ["continued"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.update(state_version=999),
        lambda state: state.update(max_doc_id=-1),
        lambda state: state.update(mutation_version=-1),
        lambda state: state.update(overrides={1: (2, {"term": 1})}),
        lambda state: state.update(overrides={7: (2, {"term": 1})}),
        lambda state: state.update(overrides={7: (1, {"term": 0})}),
    ],
)
def test_pickle_state_corruption_is_rejected_failure_atomically(mutate):
    target = CompactMutableBM25([(2, ["kept"])])
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    mutate(state)

    with pytest.raises(ValueError):
        target.__setstate__(state)

    assert pickle.dumps(target.__getstate__()) == before
    assert target.search_tokens(["kept"])


def test_pickle_state_revalidates_the_nested_base_failure_atomically():
    target = CompactMutableBM25([(2, ["kept"])])
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    state["base"]._posting_blob = b"\x02\x02"

    with pytest.raises(ValueError, match="forward and reverse"):
        target.__setstate__(state)

    assert pickle.dumps(target.__getstate__()) == before
    assert target.search_tokens(["kept"])


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("_term_ids", {"alpha": 1, "beta": 0}),
        ("_doc_positions", {2: 1, 5: 0}),
    ],
)
def test_pickle_state_rejects_corrupt_base_derived_lookups(attribute, value):
    target = CompactMutableBM25([(2, ["alpha"]), (5, ["beta"])])
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    setattr(state["base"], attribute, value)

    with pytest.raises(ValueError, match="lookup"):
        target.__setstate__(state)

    assert pickle.dumps(target.__getstate__()) == before


@pytest.mark.parametrize("value", [2**64, 2**80])
def test_pickle_state_rejects_lengths_and_frequencies_above_storage_domain(value):
    target = CompactMutableBM25([])
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    state["overrides"] = {0: (value, {"huge": value})}
    state["max_doc_id"] = 0

    with pytest.raises(ValueError, match="record|frequencies"):
        target.__setstate__(state)

    assert pickle.dumps(target.__getstate__()) == before


def test_pickle_state_rejects_aggregate_length_overflow():
    target = CompactMutableBM25([])
    before = pickle.dumps(target.__getstate__())
    state = copy.deepcopy(target.__getstate__())
    half = 2**63
    state["overrides"] = {0: (half, {"a": half}), 1: (half, {"b": half})}
    state["max_doc_id"] = 1

    with pytest.raises(ValueError, match="total length"):
        target.__setstate__(state)

    assert pickle.dumps(target.__getstate__()) == before


def test_upsert_rejects_aggregate_length_overflow_before_mutation():
    target = CompactMutableBM25([])
    state = copy.deepcopy(target.__getstate__())
    maximum = 2**64 - 1
    state["overrides"] = {0: (maximum, {"a": maximum})}
    state["max_doc_id"] = 0
    target.__setstate__(state)
    before = pickle.dumps(target.__getstate__())

    with pytest.raises(ValueError, match="total length"):
        target.upsert(1, ["b"])

    assert pickle.dumps(target.__getstate__()) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_non_finite_scoring_parameters_are_rejected_before_persistence(value):
    with pytest.raises(ValueError, match="finite"):
        CompactMutableBM25([(0, ["term"])], k1=value)


def test_randomized_mutations_and_frequent_compaction_match_mutable_oracle():
    rng = random.Random(20260722)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
    state: dict[int, list[str]] = {}
    compact = CompactMutableBM25([])
    mutable = _MutableBM25([])
    next_id = 0

    for step in range(180):
        action = rng.choice(["insert", "insert", "update", "delete", "noop"])
        if action == "insert" or not state:
            doc_id = next_id
            next_id += rng.randint(1, 4)
            tokens = [rng.choice(vocabulary) for _ in range(rng.randint(0, 12))]
            assert compact.upsert(doc_id, tokens) == mutable.upsert(doc_id, tokens)
            state[doc_id] = tokens
        elif action == "update":
            doc_id = rng.choice(list(state))
            tokens = [rng.choice(vocabulary) for _ in range(rng.randint(0, 12))]
            assert compact.upsert(doc_id, tokens) == mutable.upsert(doc_id, tokens)
            state[doc_id] = tokens
        elif action == "delete":
            doc_id = rng.choice(list(state))
            assert compact.delete(doc_id) == mutable.delete(doc_id)
            del state[doc_id]
        else:
            doc_id = rng.choice(list(state))
            version = compact._mutation_version
            cache = compact._weight_cache
            assert compact.upsert(doc_id, list(reversed(state[doc_id]))) is False
            assert compact._mutation_version == version
            assert compact._weight_cache is cache

        _assert_oracle(compact, mutable, state)
        if step % 11 == 0 and compact._overrides:
            version = compact._mutation_version
            expected = {
                query: _hex_rank(compact.search_tokens(tokenize(query)))
                for query in QUERIES
            }
            assert compact.compact()
            assert compact._mutation_version == version
            for query in QUERIES:
                assert (
                    _hex_rank(compact.search_tokens(tokenize(query))) == expected[query]
                )
            _assert_oracle(compact, mutable, state)
