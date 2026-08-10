"""Candidate-local lexical reranking shared by in-memory vector stores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from collections import Counter
import math
import re
from typing import Protocol

from .models import Chunk
from .settings import (
    DEFAULT_IDF_POW,
    DEFAULT_LEXICAL_B,
    DEFAULT_LEXICAL_K1,
)
from .text import (
    _HANGUL,
    _ko_query_stem,
    _ko_stem,
    _top_k_scores,
    tokenize,
    tokenize_query,
)

_MIN_QUERY_TERMS = 3
_ORDERED_BIGRAM_BOOST = 0.5
_PRIMARY_RANK_SHARE = 0.4
_TITLE_QUERY_WEIGHT = 3.0
_TITLE_TERM_WEIGHT = 2.0
_BODY_TERM_WEIGHT = 1.0
_ORDERED_SURFACE_PAIR_WEIGHT = 1.5
_HIGH_COVERAGE = 0.8
_PARTIAL_COVERAGE = 0.5
_HIGH_COVERAGE_WEIGHT = 1.5
_PARTIAL_COVERAGE_WEIGHT = 0.5

_PERSONAL_MEMORY_MARKERS = frozenset(
    {
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "me",
        "mine",
        "my",
        "myself",
        "our",
        "ours",
        "ourselves",
        "us",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
    }
)
_ENGLISH_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


class _TextCandidate(Protocol):
    title: str
    text: str


def _korean_characters(text: str) -> frozenset[str]:
    return frozenset(
        character
        for word in _HANGUL.findall((text or "").casefold())
        for character in _ko_stem(word)
    )


def rank_korean_character_fallback(
    query: str,
    chunks: Sequence[_TextCandidate] | Mapping[int, _TextCandidate],
    *,
    limit: int,
) -> list[tuple[int, float]]:
    """Rank Korean character evidence when normal lexical retrieval has no hit."""
    query_characters = frozenset(
        character
        for word in _HANGUL.findall((query or "").casefold())
        for stem in (_ko_query_stem(word),)
        if stem is not None
        for character in stem
    )
    if not query_characters or not chunks or limit <= 0:
        return []

    chunk_items = chunks.items() if isinstance(chunks, Mapping) else enumerate(chunks)
    documents = [
        (document_id, _korean_characters(f"{chunk.title} {chunk.text}"))
        for document_id, chunk in chunk_items
    ]
    average_length = sum(
        len(characters) for _document_id, characters in documents
    ) / len(documents)
    frequencies = Counter(
        character
        for _document_id, document in documents
        for character in query_characters.intersection(document)
    )
    document_count = len(documents)
    scores: dict[int, float] = {}
    for document_id, document in documents:
        matched = query_characters.intersection(document)
        if not matched:
            continue
        length_normalization = (
            1.0
            - DEFAULT_LEXICAL_B
            + DEFAULT_LEXICAL_B * ((len(document) or 1) / (average_length or 1.0))
        )
        denominator = 1.0 + DEFAULT_LEXICAL_K1 * length_normalization
        score = 0.0
        for character in matched:
            frequency = frequencies[character]
            inverse_document_frequency = (
                math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
                ** DEFAULT_IDF_POW
            )
            score += (
                inverse_document_frequency * (DEFAULT_LEXICAL_K1 + 1.0) / denominator
            )
        scores[document_id] = score
    return _top_k_scores(scores, limit)


def _ordered_bigram_fraction(
    query_pairs: frozenset[tuple[str, str]], chunk: Chunk
) -> float:
    if not query_pairs:
        return 0.0
    matched_pairs: set[tuple[str, str]] = set()
    document_terms = tokenize(f"{chunk.title} {chunk.text}")
    for pair in zip(document_terms, document_terms[1:]):
        if pair not in query_pairs:
            continue
        matched_pairs.add(pair)
        if len(matched_pairs) == len(query_pairs):
            return 1.0
    return len(matched_pairs) / len(query_pairs)


def _surface_candidate_score(
    surface_query: str, terms: tuple[str, ...], chunk: Chunk
) -> float:
    if not terms:
        return 0.0
    title = chunk.title.casefold()
    body = chunk.text.casefold()
    score = _TITLE_QUERY_WEIGHT * len(terms) if surface_query in title else 0.0
    matched = 0
    for term in terms:
        in_title = term in title
        in_body = term in body
        if not (in_title or in_body):
            continue
        matched += 1
        score += _TITLE_TERM_WEIGHT * in_title + _BODY_TERM_WEIGHT * in_body
    for left, right in zip(terms, terms[1:]):
        if f"{left} {right}" in title or f"{left} {right}" in body:
            score += _ORDERED_SURFACE_PAIR_WEIGHT
    coverage = matched / len(terms)
    if coverage >= _HIGH_COVERAGE:
        score += _HIGH_COVERAGE_WEIGHT * len(terms)
    elif coverage >= _PARTIAL_COVERAGE:
        score += _PARTIAL_COVERAGE_WEIGHT * len(terms)
    return score


def is_personal_memory_query(query: str) -> bool:
    """Return whether a query explicitly refers to the speaker or their group."""
    words = _ENGLISH_WORD.findall(query.casefold().replace("’", "'"))
    return any(word in _PERSONAL_MEMORY_MARKERS for word in words)


def rerank_lexical_candidates(
    query: str,
    ranked: list[tuple[int, float]],
    chunk_at: Callable[[int], Chunk],
) -> list[tuple[int, float]]:
    """Rerank an admitted BM25 frontier without scanning the corpus."""
    query_terms = tokenize_query(query)
    if len(ranked) < 2 or len(query_terms) < _MIN_QUERY_TERMS:
        return ranked

    top_score = ranked[0][1]
    query_pairs = frozenset(zip(query_terms, query_terms[1:]))
    personal_memory = is_personal_memory_query(query)
    cached_chunks = (
        {document_id: chunk_at(document_id) for document_id, _score in ranked}
        if personal_memory
        else None
    )

    def resolve(document_id: int) -> Chunk:
        return (
            chunk_at(document_id)
            if cached_chunks is None
            else cached_chunks[document_id]
        )

    phrase_ranked = sorted(
        ranked,
        key=lambda item: (
            -(
                item[1]
                + top_score
                * _ORDERED_BIGRAM_BOOST
                * _ordered_bigram_fraction(query_pairs, resolve(item[0]))
            ),
            item[0],
        ),
    )
    if not personal_memory:
        return phrase_ranked

    phrase_positions = {
        document_id: rank for rank, (document_id, _score) in enumerate(phrase_ranked)
    }
    surface_query = query.casefold()
    surface_terms = tuple(surface_query.split())
    surface_ranked = sorted(
        phrase_ranked,
        key=lambda item: (
            -_surface_candidate_score(surface_query, surface_terms, resolve(item[0])),
            phrase_positions[item[0]],
        ),
    )
    fused = {
        document_id: _PRIMARY_RANK_SHARE / rank
        for rank, (document_id, _score) in enumerate(phrase_ranked, 1)
    }
    surface_rank_share = 1.0 - _PRIMARY_RANK_SHARE
    for rank, (document_id, _score) in enumerate(surface_ranked, 1):
        fused[document_id] += surface_rank_share / rank
    maximum = max(fused.values()) or 1.0
    return sorted(
        (
            (document_id, top_score * score / maximum)
            for document_id, score in fused.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
