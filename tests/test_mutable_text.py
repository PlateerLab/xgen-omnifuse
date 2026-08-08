"""Exactness contract for the opt-in mutable lexical indexes."""

from __future__ import annotations

import copy
import pickle
import random
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse.text import BM25, BM25F, _MutableBM25, _MutableBM25F, tokenize  # noqa: E402


def _hex_ranked(ranked):
    return [(doc_id, score.hex()) for doc_id, score in ranked]


def _fresh_term(value: str) -> str:
    fresh = (value + "\0")[:-1]
    assert fresh == value and fresh is not value
    return fresh


def _assert_owned_term_identity(index) -> None:
    assert set(index._terms) == set(index._postings)
    for term in index._terms:
        assert index._terms[term] is term
    for mapping in (index._postings, index._df, index._weight_cache):
        for term in mapping:
            assert term is index._terms[term]
    for term in getattr(index, "_dfe", {}):
        assert term is index._terms[term]

    doc_terms = []
    for frozen in index._docs.values():
        counts = frozen[1]
        field_counts = counts if isinstance(counts, tuple) else (counts,)
        for field in field_counts:
            for term in field:
                assert term is index._terms[term]
                doc_terms.append(term)
    assert len({id(term) for term in doc_terms}) == len(set(doc_terms))


def _assert_bm25_oracle(index, state, queries, *, k1=1.5, b=0.75, idf_pow=1.2):
    stable_ids = sorted(state)
    rebuilt = BM25(
        [state[doc_id] for doc_id in stable_ids],
        k1=k1,
        b=b,
        idf_pow=idf_pow,
    )
    assert index.N == rebuilt.N == len(stable_ids)
    assert index.avgdl.hex() == rebuilt.avgdl.hex()
    assert set(index.idf) == set(rebuilt.idf)
    for term, expected in rebuilt.idf.items():
        assert index.idf[term].hex() == expected.hex()
    for query in queries:
        expected = [
            (stable_ids[dense_id], score) for dense_id, score in rebuilt.search(query)
        ]
        assert _hex_ranked(index.search(query)) == _hex_ranked(expected)
        query_tokens = tokenize(query)
        for dense_id, stable_id in enumerate(stable_ids):
            assert (
                index.score(query_tokens, stable_id).hex()
                == rebuilt.score(query_tokens, dense_id).hex()
            )


def _assert_bm25f_oracle(
    index,
    state,
    queries,
    weights,
    *,
    evidence_fields=(),
    k1=1.5,
    b=0.75,
    idf_pow=1.2,
):
    stable_ids = sorted(state)
    rebuilt = BM25F(
        [state[doc_id] for doc_id in stable_ids],
        weights,
        k1=k1,
        b=b,
        idf_pow=idf_pow,
        evidence_fields=set(evidence_fields),
    )
    assert index.N == rebuilt.N == len(stable_ids)
    assert set(index.avglen) == set(rebuilt.avglen)
    for field, expected in rebuilt.avglen.items():
        assert index.avglen[field].hex() == expected.hex()
    assert set(index.idf) == set(rebuilt.idf)
    for term, expected in rebuilt.idf.items():
        assert index.idf[term].hex() == expected.hex()
    for query in queries:
        expected = [
            (stable_ids[dense_id], score) for dense_id, score in rebuilt.search(query)
        ]
        assert _hex_ranked(index.search(query)) == _hex_ranked(expected)
        query_tokens = tokenize(query)
        for dense_id, stable_id in enumerate(stable_ids):
            assert (
                index.score(query_tokens, stable_id).hex()
                == rebuilt.score(query_tokens, dense_id).hex()
            )


def test_mutable_bm25_exact_query_multiplicity_and_stable_ties():
    state = {
        3: ["alpha", "alpha", "beta"],
        8: ["alpha", "alpha", "beta"],
        13: ["beta", "gamma"],
        21: [],
    }
    index = _MutableBM25(state.items(), k1=1.2, b=0.63, idf_pow=1.0)

    _assert_bm25_oracle(
        index,
        state,
        ["alpha", "alpha alpha beta", "beta alpha alpha", "missing"],
        k1=1.2,
        b=0.63,
        idf_pow=1.0,
    )
    assert [doc_id for doc_id, _score in index.search("alpha", limit=2)] == [3, 8]

    assert index.upsert(3, ["alpha", "alpha", "beta", "delta"])
    assert index.upsert(3, state[3])
    assert [doc_id for doc_id, _score in index.search("alpha", limit=2)] == [3, 8]


def test_mutable_bm25_idempotency_and_cache_invalidation():
    state = {2: ["alpha", "beta", "alpha"], 7: ["gamma"]}
    index = _MutableBM25(state.items())
    assert index._weight_cache == {}
    index.search("alpha missing")
    assert set(index._weight_cache) == {"alpha"}
    alpha_cache = index._weight_cache["alpha"]
    version = index._mutation_version

    assert (
        index.upsert(
            2, [_fresh_term("beta"), _fresh_term("alpha"), _fresh_term("alpha")]
        )
        is False
    )
    assert index.delete(99) is False
    assert index._weight_cache["alpha"] is alpha_cache
    assert "missing" not in index._weight_cache
    assert index._mutation_version == version

    assert index.upsert(2, ["alpha", "delta"]) is True
    state[2] = ["alpha", "delta"]
    assert index._weight_cache == {}
    assert index._mutation_version == version + 1
    _assert_bm25_oracle(index, state, ["alpha", "beta", "delta"])


def test_mutable_bm25_delete_empty_and_monotonic_ids():
    state = {4: [], 9: ["alpha"], 12: ["beta"]}
    index = _MutableBM25(state.items())

    assert index.delete(4) is True
    del state[4]
    assert index.delete(4) is False
    _assert_bm25_oracle(index, state, ["alpha", "beta"])

    with pytest.raises(ValueError, match="greater than"):
        index.upsert(4, ["reused"])
    with pytest.raises(ValueError, match="non-negative int"):
        index.upsert(True, ["invalid"])

    assert index.upsert(20, []) is True
    state[20] = []
    _assert_bm25_oracle(index, state, ["alpha", "missing"])


def test_mutable_bm25_rejects_invalid_tokens_before_commit():
    class ExplodingToken(str):
        def __hash__(self):
            raise RuntimeError("must never hash")

    index = _MutableBM25([(2, ["alpha"])])
    index.search("alpha")
    cache = index._weight_cache["alpha"]
    version = index._mutation_version

    with pytest.raises(TypeError, match="only str"):
        index.upsert(5, [ExplodingToken("boom")])

    assert index.N == 1
    assert index._max_doc_id == 2
    assert set(index._docs) == {2}
    assert set(index._postings) == {"alpha"}
    assert index._weight_cache["alpha"] is cache
    assert index._mutation_version == version


def test_mutable_bm25_batch_prevalidates_every_item_without_partial_commit():
    class ExplodingToken(str):
        def __hash__(self):
            raise RuntimeError("must never hash")

    class ExplodingId(int):
        def __hash__(self):
            raise RuntimeError("must never hash")

    index = _MutableBM25([(2, ["alpha"]), (5, ["beta"])])
    index.search("alpha missing")

    before = copy.deepcopy(index.__dict__)
    with pytest.raises(TypeError, match="only str"):
        index.upsert_many([(2, ["changed"]), (8, ["valid", ExplodingToken("boom")])])
    assert index.__dict__ == before
    assert set(index._terms) == {"alpha", "beta"}

    with pytest.raises(ValueError, match="greater than"):
        index.upsert_many([(8, ["first"]), (7, ["second-invalid"])])
    assert index.__dict__ == before

    with pytest.raises(ValueError, match="duplicate batch"):
        index.upsert_many([(2, ["first"]), (2, ["duplicate"])])
    assert index.__dict__ == before

    with pytest.raises(ValueError, match="non-negative int"):
        index.delete_many([2, ExplodingId(5)])
    assert index.__dict__ == before

    def interrupted_batch():
        yield 2, ["changed"]
        raise RuntimeError("source interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        index.upsert_many(interrupted_batch())
    assert index.__dict__ == before


def test_mutable_bm25_batch_mixed_upsert_and_delete_match_static_rebuild():
    state = {
        2: ["alpha", "beta"],
        5: ["gamma"],
        9: ["delta", "delta"],
    }
    index = _MutableBM25(state.items())
    index.search("alpha gamma")
    version = index._mutation_version

    changed = index.upsert_many(
        [
            (2, ["alpha", "updated"]),
            (5, ["gamma"]),
            (12, ["inserted", "alpha"]),
            (15, []),
        ]
    )
    assert changed == 3
    assert index._mutation_version == version + 1
    assert index._weight_cache == {}
    state[2] = ["alpha", "updated"]
    state[12] = ["inserted", "alpha"]
    state[15] = []
    _assert_bm25_oracle(index, state, ["alpha", "gamma", "inserted", "delta"])

    version = index._mutation_version
    assert index.delete_many([999, 5, 12]) == 2
    assert index._mutation_version == version + 1
    del state[5], state[12]
    _assert_bm25_oracle(index, state, ["alpha", "gamma", "inserted", "delta"])

    before = copy.deepcopy(index.__dict__)
    with pytest.raises(ValueError, match="duplicate batch"):
        index.delete_many([2, 2])
    assert index.__dict__ == before


def test_mutable_bm25_owns_canonical_terms_and_retires_churned_vocabulary():
    index = _MutableBM25(
        [
            (1, [_fresh_term("shared"), _fresh_term("retired")]),
            (4, [_fresh_term("shared")]),
        ]
    )
    _assert_owned_term_identity(index)
    assert len(index._terms) == 2

    index.search(_fresh_term("shared"))
    assert next(iter(index._weight_cache)) is index._terms["shared"]
    loaded = pickle.loads(pickle.dumps(index))
    _assert_owned_term_identity(loaded)
    assert _hex_ranked(loaded.search("shared")) == _hex_ranked(index.search("shared"))
    legacy_state = copy.deepcopy(index.__dict__)
    del legacy_state["_terms"]
    migrated = object.__new__(_MutableBM25)
    migrated.__setstate__(legacy_state)
    _assert_owned_term_identity(migrated)
    assert _hex_ranked(migrated.search("shared")) == _hex_ranked(index.search("shared"))

    assert index.upsert(1, [_fresh_term("shared"), _fresh_term("new")])
    assert set(index._terms) == {"shared", "new"}
    _assert_owned_term_identity(index)
    assert index.delete(1)
    assert set(index._terms) == {"shared"}
    assert index.delete(4)
    assert index._terms == {}
    assert index._postings == {}


def test_mutable_bm25_randomized_mutations_match_every_static_rebuild():
    rng = random.Random(20260722)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]
    state: dict[int, list[str]] = {}
    index = _MutableBM25([])
    next_id = 0
    queries = ["alpha", "beta beta gamma", "delta alpha beta", "missing"]

    for _step in range(80):
        action = rng.choice(["insert", "insert", "update", "delete"])
        if action == "insert" or not state:
            doc_id = next_id
            next_id += rng.randint(1, 3)
            tokens = [rng.choice(vocabulary) for _ in range(rng.randint(0, 9))]
            assert index.upsert(doc_id, tokens)
            state[doc_id] = tokens
        elif action == "update":
            doc_id = rng.choice(sorted(state))
            tokens = [rng.choice(vocabulary) for _ in range(rng.randint(0, 9))]
            changed = index.upsert(doc_id, tokens)
            if changed:
                state[doc_id] = tokens
        else:
            doc_id = rng.choice(sorted(state))
            assert index.delete(doc_id)
            del state[doc_id]
        _assert_bm25_oracle(index, state, queries)


WEIGHTS = {"title": 4.0, "body": 1.0, "memory": 1.0}
EVIDENCE = {"memory"}


def test_mutable_bm25f_exact_fields_normalization_and_unique_query_terms():
    state = {
        1: {
            "title": ["alpha", "alpha"],
            "body": ["beta", "gamma", "gamma"],
            "memory": ["answer", "alpha"],
        },
        6: {"title": ["beta"], "body": [], "memory": ["answer", "answer"]},
        10: {"title": [], "body": ["alpha", "delta"], "memory": []},
        17: {},
    }
    index = _MutableBM25F(
        state.items(),
        WEIGHTS,
        k1=1.35,
        b=0.68,
        idf_pow=1.0,
        evidence_fields=EVIDENCE,
    )

    _assert_bm25f_oracle(
        index,
        state,
        ["alpha", "alpha alpha beta", "answer", "gamma alpha", "missing"],
        WEIGHTS,
        evidence_fields=EVIDENCE,
        k1=1.35,
        b=0.68,
        idf_pow=1.0,
    )


def test_mutable_bm25f_evidence_only_content_transitions_are_exact():
    state = {
        2: {"title": [], "body": [], "memory": ["rare", "rare"]},
        5: {"title": ["common"], "body": ["common"], "memory": []},
    }
    index = _MutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    assert "rare" not in index._df
    assert index._dfe["rare"] == 1
    _assert_bm25f_oracle(
        index, state, ["rare", "common"], WEIGHTS, evidence_fields=EVIDENCE
    )

    state[2] = {"title": [], "body": ["rare"], "memory": []}
    assert index.upsert(2, state[2])
    assert index._df["rare"] == 1
    assert "rare" not in index._dfe
    _assert_bm25f_oracle(
        index, state, ["rare", "common"], WEIGHTS, evidence_fields=EVIDENCE
    )

    state[9] = {"title": [], "body": [], "memory": ["rare"]}
    assert index.upsert(9, state[9])
    assert index._df["rare"] == 1
    assert index._dfe["rare"] == 1
    _assert_bm25f_oracle(index, state, ["rare"], WEIGHTS, evidence_fields=EVIDENCE)

    assert index.delete(2)
    del state[2]
    assert "rare" not in index._df
    assert index._dfe["rare"] == 1
    _assert_bm25f_oracle(index, state, ["rare"], WEIGHTS, evidence_fields=EVIDENCE)


def test_mutable_bm25f_idempotency_empty_docs_and_cache_invalidation():
    state = {
        3: {"title": ["alpha"], "body": ["beta"], "memory": []},
        11: {},
    }
    index = _MutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    assert index._weight_cache == {}
    index.search("alpha missing")
    assert set(index._weight_cache) == {"alpha"}
    alpha_cache = index._weight_cache["alpha"]
    version = index._mutation_version

    same_scoring_fields = {
        "body": ["beta"],
        "title": ["alpha"],
        "memory": [],
        "ignored": ["not-indexed"],
    }
    same_scoring_fields = {
        field: [_fresh_term(term) for term in terms]
        for field, terms in same_scoring_fields.items()
    }
    assert index.upsert(3, same_scoring_fields) is False
    assert index.delete(999) is False
    assert index._weight_cache["alpha"] is alpha_cache
    assert "missing" not in index._weight_cache
    assert index._mutation_version == version

    state[3] = {"title": ["alpha"], "body": ["delta"], "memory": ["beta"]}
    assert index.upsert(3, state[3])
    assert index._weight_cache == {}
    assert index._mutation_version == version + 1
    _assert_bm25f_oracle(
        index,
        state,
        ["alpha", "beta", "delta", "missing"],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )


def test_mutable_bm25f_ids_ties_and_invalid_tokens_are_stable():
    class ExplodingToken(str):
        def __hash__(self):
            raise RuntimeError("must never hash")

    state = {
        4: {"title": ["alpha"], "body": [], "memory": []},
        9: {"title": ["alpha"], "body": [], "memory": []},
    }
    index = _MutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    assert [doc_id for doc_id, _score in index.search("alpha")] == [4, 9]
    cache = index._weight_cache["alpha"]
    version = index._mutation_version

    with pytest.raises(TypeError, match="only str"):
        index.upsert(12, {"title": [ExplodingToken("boom")]})

    assert index.N == 2
    assert index._max_doc_id == 9
    assert set(index._docs) == {4, 9}
    assert index._weight_cache["alpha"] is cache
    assert index._mutation_version == version
    with pytest.raises(ValueError, match="greater than"):
        index.upsert(3, {"title": ["stale"]})
    with pytest.raises(ValueError, match="non-negative int"):
        index.delete(False)
    _assert_bm25f_oracle(
        index, state, ["alpha", "missing"], WEIGHTS, evidence_fields=EVIDENCE
    )


def test_mutable_bm25f_batch_prevalidation_and_mixed_mutations_are_exact():
    class ExplodingToken(str):
        def __hash__(self):
            raise RuntimeError("must never hash")

    state = {
        3: {"title": ["alpha"], "body": ["old"], "memory": []},
        7: {"title": ["beta"], "body": [], "memory": ["answer"]},
        11: {"title": [], "body": ["delete-me"], "memory": []},
    }
    index = _MutableBM25F(state.items(), WEIGHTS, evidence_fields=EVIDENCE)
    index.search("alpha missing")
    before = copy.deepcopy(index.__dict__)

    with pytest.raises(TypeError, match="only str"):
        index.upsert_many(
            [
                (3, {"title": ["changed"]}),
                (14, {"memory": [ExplodingToken("boom")]}),
            ]
        )
    assert index.__dict__ == before
    assert set(index._terms) == {"alpha", "answer", "beta", "delete-me", "old"}

    with pytest.raises(ValueError, match="greater than"):
        index.upsert_many(
            [(14, {"title": ["first"]}), (13, {"title": ["second-invalid"]})]
        )
    assert index.__dict__ == before

    version = index._mutation_version
    assert (
        index.upsert_many(
            [
                (3, {"title": ["alpha"], "body": ["new"], "memory": []}),
                (7, state[7]),
                (14, {"title": ["inserted"], "memory": ["answer", "rare"]}),
            ]
        )
        == 2
    )
    assert index._mutation_version == version + 1
    state[3] = {"title": ["alpha"], "body": ["new"], "memory": []}
    state[14] = {"title": ["inserted"], "memory": ["answer", "rare"]}
    _assert_bm25f_oracle(
        index,
        state,
        ["alpha", "new", "answer", "rare", "delete-me"],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )

    version = index._mutation_version
    assert index.delete_many([404, 7, 11]) == 2
    assert index._mutation_version == version + 1
    del state[7], state[11]
    _assert_bm25f_oracle(
        index,
        state,
        ["alpha", "new", "answer", "rare", "delete-me"],
        WEIGHTS,
        evidence_fields=EVIDENCE,
    )

    before = copy.deepcopy(index.__dict__)
    with pytest.raises(ValueError, match="duplicate batch"):
        index.delete_many([3, 3])
    assert index.__dict__ == before


def test_mutable_bm25f_owns_canonical_terms_and_retires_churned_vocabulary():
    weights = {"title": 4.0, "body": 1.0}
    index = _MutableBM25F(
        [
            (
                2,
                {
                    "title": [_fresh_term("shared")],
                    "body": [_fresh_term("retired")],
                },
            ),
            (7, {"body": [_fresh_term("shared")]}),
        ],
        weights,
    )
    _assert_owned_term_identity(index)
    assert len(index._terms) == 2

    index.search(_fresh_term("shared"))
    assert next(iter(index._weight_cache)) is index._terms["shared"]
    loaded = pickle.loads(pickle.dumps(index))
    _assert_owned_term_identity(loaded)
    assert _hex_ranked(loaded.search("shared")) == _hex_ranked(index.search("shared"))
    legacy_state = copy.deepcopy(index.__dict__)
    del legacy_state["_terms"]
    migrated = object.__new__(_MutableBM25F)
    migrated.__setstate__(legacy_state)
    _assert_owned_term_identity(migrated)
    assert _hex_ranked(migrated.search("shared")) == _hex_ranked(index.search("shared"))

    assert index.upsert(
        2,
        {
            "title": [_fresh_term("shared")],
            "body": [_fresh_term("new")],
        },
    )
    assert set(index._terms) == {"shared", "new"}
    _assert_owned_term_identity(index)
    assert index.delete(2)
    assert set(index._terms) == {"shared"}
    assert index.delete(7)
    assert index._terms == {}
    assert index._postings == {}


def test_mutable_indexes_do_not_cache_oov_query_terms():
    indexes = [
        _MutableBM25([(1, ["known"])]),
        _MutableBM25F([(1, {"title": ["known"]})], WEIGHTS, evidence_fields=EVIDENCE),
    ]
    for index in indexes:
        index.search("known")
        cache_size = len(index._weight_cache)
        for number in range(10_000):
            assert index.search(f"unseen{number}") == []
        assert len(index._weight_cache) == cache_size
        assert set(index._weight_cache) == {"known"}


def test_mutable_bm25f_randomized_mutations_match_every_static_rebuild():
    rng = random.Random(836536)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    state: dict[int, dict[str, list[str]]] = {}
    index = _MutableBM25F([], WEIGHTS, evidence_fields=EVIDENCE)
    next_id = 1
    queries = ["alpha", "beta beta gamma", "delta answer", "epsilon zeta", "missing"]

    def random_fields():
        return {
            field: [rng.choice(vocabulary) for _ in range(rng.randint(0, 5))]
            for field in WEIGHTS
        }

    for _step in range(70):
        action = rng.choice(["insert", "insert", "update", "delete"])
        if action == "insert" or not state:
            doc_id = next_id
            next_id += rng.randint(1, 4)
            fields = random_fields()
            assert index.upsert(doc_id, fields)
            state[doc_id] = fields
        elif action == "update":
            doc_id = rng.choice(sorted(state))
            fields = random_fields()
            changed = index.upsert(doc_id, fields)
            if changed:
                state[doc_id] = fields
        else:
            doc_id = rng.choice(sorted(state))
            assert index.delete(doc_id)
            del state[doc_id]
        _assert_bm25f_oracle(
            index,
            state,
            queries,
            WEIGHTS,
            evidence_fields=EVIDENCE,
        )
