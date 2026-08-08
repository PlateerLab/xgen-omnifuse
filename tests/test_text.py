"""Exact-semantics tests for token normalization and bounded text ranking."""

from __future__ import annotations

import pathlib
import random
import os
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse.text import (  # noqa: E402
    BM25,
    BM25F,
    _KO_SUFFIX,
    _KO_SUFFIX_BY_LAST,
    _ko_stem,
    _top_k_scores,
    tokenize,
    tokenize_query,
)


def _reference_ko_stem(word: str) -> str:
    changed = True
    while changed and len(word) >= 3:
        changed = False
        for suffix in _KO_SUFFIX:
            if len(word) - len(suffix) >= 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                changed = True
                break
    return word


def _reference_top_k(scores: dict[int, float], limit: int) -> list[tuple[int, float]]:
    ranked = [(doc_id, score) for doc_id, score in scores.items() if score > 0]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def _reference_bm25_search(
    index: BM25, query: str, limit: int
) -> list[tuple[int, float]]:
    qtf: dict[str, int] = {}
    for term in tokenize_query(query):
        qtf[term] = qtf.get(term, 0) + 1
    scores: dict[int, float] = {}
    for term, query_count in qtf.items():
        postings = index._pd.get(term)
        if postings is None:
            continue
        weights = index._pw[term]
        for doc_id, weight in zip(postings, weights):
            scores[doc_id] = scores.get(doc_id, 0.0) + query_count * weight
    return _reference_top_k(scores, limit)


def _reference_bm25f_search(
    index: BM25F, query: str, limit: int
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for term in dict.fromkeys(tokenize_query(query)):
        postings = index._pd.get(term)
        if postings is None:
            continue
        for doc_id, weight in zip(postings, index._pw[term]):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight
    return _reference_top_k(scores, limit)


def test_ascii_tokenizer_fast_path_preserves_the_word_pattern_contract() -> None:
    separators = "".join(
        chr(codepoint)
        for codepoint in range(128)
        if not chr(codepoint).isalnum()
    )

    assert tokenize(f"ALPHAS{separators}beta2{separators}GAMMA") == [
        "alpha",
        "beta2",
        "gamma",
    ]


def test_suffix_index_partitions_the_ordered_suffix_contract() -> None:
    assert set(_KO_SUFFIX_BY_LAST) == {suffix[-1] for suffix in _KO_SUFFIX}
    for last, bucket in _KO_SUFFIX_BY_LAST.items():
        assert bucket == tuple(suffix for suffix in _KO_SUFFIX if suffix[-1] == last)


def test_query_tokenizer_removes_korean_query_operators_without_losing_subject() -> None:
    tokens = tokenize_query("오픈뱅킹의 법제화에 대해 설명해주세요.")

    assert "#오픈뱅킹" in tokens
    assert "#법제" in tokens
    assert "#대해" not in tokens
    assert not any("설명" in token for token in tokens)


def test_query_tokenizer_normalizes_definition_copula_and_has_safe_fallback() -> None:
    tokens = tokenize_query("스키밍이란 무엇인가요?")
    assert "#스키밍" in tokens
    assert "#무엇" not in tokens
    assert tokenize_query("무엇인가요?") == tokenize("무엇인가요?")
    assert tokenize_query("HTTP APIs") == tokenize("HTTP APIs")


def test_query_coordination_excludes_operator_score_and_operator_only_hit() -> None:
    index = BM25(
        [
            tokenize("외주화 무엇"),
            tokenize("외주화"),
            tokenize("무엇"),
        ]
    )

    ranked = index.search("외주화란 무엇인가요?", limit=10)

    assert [doc_id for doc_id, _score in ranked] == [1, 0]
    assert ranked[0][1] > ranked[1][1]


def test_query_coordination_drops_partial_ngram_tail_when_complete_words_match() -> None:
    documents = [
        {"title": tokenize("휴대 전화"), "body": []},
        {"title": tokenize("휴대전화"), "body": []},
    ]
    index = BM25F(documents, {"title": 1.0, "body": 1.0})

    assert [doc_id for doc_id, _score in index.search("휴대 전화", limit=10)] == [0]


def test_query_coordination_does_not_confuse_content_with_grammar() -> None:
    assert "#임진왜란" in tokenize_query("임진왜란이 끝난 날은 언제인가?")
    assert "#설명회" in tokenize_query("설명회 일정은?")


def test_bm25f_scores_are_stable_across_hash_seeds() -> None:
    source_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    code = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
from omnifuse.text import BM25F, _MutableBM25F, tokenize

documents = [
    {{"title": ["alpha", "alpha", "beta"], "body": ["gamma", "delta", "epsilon", "epsilon"]}},
    {{"title": ["alpha"], "body": ["zeta"]}},
    {{"title": [], "body": ["beta", "gamma"]}},
]
weights = {{"title": 4.0, "body": 1.0}}
query = "alpha beta gamma delta epsilon"
tokens = tokenize(query)
static = BM25F(documents, weights)
mutable = _MutableBM25F(enumerate(documents), weights)
print(" ".join((
    static.search(query, limit=1)[0][1].hex(),
    static.score(tokens, 0).hex(),
    mutable.search(query, limit=1)[0][1].hex(),
    mutable.score(tokens, 0).hex(),
)))
"""
    outputs = []
    for seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0] == outputs[1]
    assert len(set(outputs[0].split())) == 1


def test_suffix_index_matches_full_scan_for_generated_hangul_words() -> None:
    rng = random.Random(20260722)
    suffixes = list(_KO_SUFFIX)
    for _ in range(10_000):
        stem = "".join(
            chr(rng.randrange(0xAC00, 0xD7A4)) for _ in range(rng.randrange(1, 13))
        )
        word = stem + "".join(rng.choice(suffixes) for _ in range(rng.randrange(4)))
        assert _ko_stem(word) == _reference_ko_stem(word)


def test_bounded_top_k_matches_full_sort_for_scores_limits_and_ties() -> None:
    rng = random.Random(20260722)
    score_values = (
        float("nan"),
        -2.0,
        -0.0,
        0.0,
        0.125,
        0.5,
        0.5,
        1.0,
        3.25,
    )
    for size in (0, 1, 2, 5, 20, 101, 1_000):
        items = [(doc_id, rng.choice(score_values)) for doc_id in range(size)]
        rng.shuffle(items)
        scores = dict(items)
        for limit in (-3, -1, 0, 1, 2, 7, 20, size, size + 3):
            assert _top_k_scores(scores, limit) == _reference_top_k(scores, limit)


def test_bounded_top_k_tie_order_is_independent_of_candidate_insertion() -> None:
    forward = {doc_id: 1.0 for doc_id in range(100)}
    reverse = dict(reversed(list(forward.items())))
    expected = [(doc_id, 1.0) for doc_id in range(20)]
    assert _top_k_scores(forward, 20) == expected
    assert _top_k_scores(reverse, 20) == expected


def test_bm25_search_is_bit_identical_to_full_sort_reference() -> None:
    rng = random.Random(20260722)
    vocabulary = [f"term{index}" for index in range(40)]
    documents = [
        [rng.choice(vocabulary) for _ in range(rng.randrange(1, 30))]
        for _ in range(300)
    ]
    index = BM25(documents)
    for _ in range(50):
        query = " ".join(rng.choice(vocabulary) for _ in range(rng.randrange(1, 8)))
        for limit in (0, 1, 5, 20, 500):
            assert index.search(query, limit=limit) == _reference_bm25_search(
                index, query, limit
            )


def test_bm25f_search_is_bit_identical_to_full_sort_reference() -> None:
    rng = random.Random(20260722)
    vocabulary = [f"term{index}" for index in range(40)]
    documents = [
        {
            "title": [rng.choice(vocabulary) for _ in range(rng.randrange(1, 6))],
            "body": [rng.choice(vocabulary) for _ in range(rng.randrange(1, 30))],
        }
        for _ in range(300)
    ]
    index = BM25F(documents, {"title": 4.0, "body": 1.0})
    for _ in range(50):
        query = " ".join(rng.choice(vocabulary) for _ in range(rng.randrange(1, 8)))
        for limit in (0, 1, 5, 20, 500):
            assert index.search(query, limit=limit) == _reference_bm25f_search(
                index, query, limit
            )


def test_bm25_and_bm25f_keep_lowest_document_id_for_equal_scores() -> None:
    bm25 = BM25([["same"], ["same"], ["same"]])
    bm25f = BM25F(
        [{"title": ["same"], "body": []} for _ in range(3)],
        {"title": 4.0, "body": 1.0},
    )
    assert [doc_id for doc_id, _score in bm25.search("same", limit=2)] == [0, 1]
    assert [doc_id for doc_id, _score in bm25f.search("same", limit=2)] == [0, 1]
