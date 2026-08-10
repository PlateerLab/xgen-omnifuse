"""Deterministic document links derived from title mentions in passage text."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import NamedTuple

from .models import Chunk, Triple
from .text import tokenize

_QUALIFIER_BOUNDARIES = frozenset({"from", "in", "of"})
_TERMINAL_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")
_MIN_DISAMBIGUATED_ALIAS_LENGTH = 5
_EXACT_TITLE_AFFINITY = 1.0
_FUZZY_TITLE_AFFINITY = 0.95
_TARGET = "\0"


class TitleQueryMatch(NamedTuple):
    affinity: float
    offset: int


def _title_aliases(title: str) -> tuple[tuple[str, ...], ...]:
    """Return conservative multi-token aliases for one document title.

    The full title is always preferred. A terminal disambiguator (for example,
    ``"Andrea Carroll (soprano)"``) is removed, and a leading name before a
    location qualifier (for example, ``"Philip V"`` in ``"Philip V of Spain"``)
    is accepted. Single-token aliases are intentionally excluded because they
    create noisy links in ordinary prose.
    """
    base = _TERMINAL_PARENTHETICAL.sub("", title).strip()
    tokens = tuple(tokenize(base))
    if len(tokens) < 2:
        return ()

    aliases = [tokens]
    for index, token in enumerate(tokens):
        if token in _QUALIFIER_BOUNDARIES and index >= 2:
            aliases.append(tokens[:index])
            break
    return tuple(dict.fromkeys(aliases))


def _query_title_aliases(title: str) -> tuple[tuple[str, ...], ...]:
    aliases = list(_title_aliases(title))
    without_disambiguator = _TERMINAL_PARENTHETICAL.sub("", title).strip()
    tokens = tuple(tokenize(without_disambiguator))
    if (
        len(tokens) == 1
        and without_disambiguator != title.strip()
        and len(tokens[0]) >= _MIN_DISAMBIGUATED_ALIAS_LENGTH
    ):
        aliases.append(tokens)
    return tuple(dict.fromkeys(aliases))


def _within_one_edit(left: str, right: str) -> bool:
    """Whether two non-empty tokens differ by at most one insert/delete/replace."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(left) == len(right):
            i += 1
        j += 1
    return edits + (j < len(right)) <= 1


def title_query_match(question: str, title: str) -> TitleQueryMatch | None:
    """Find an exact or single-typo multi-token title mention."""
    query_tokens = tuple(tokenize(question))
    best: TitleQueryMatch | None = None
    for alias in _query_title_aliases(title):
        width = len(alias)
        for start in range(len(query_tokens) - width + 1):
            candidate = query_tokens[start : start + width]
            if candidate == alias:
                match = TitleQueryMatch(_EXACT_TITLE_AFFINITY, start)
                if best is None or (-match.affinity, match.offset) < (
                    -best.affinity,
                    best.offset,
                ):
                    best = match
                continue
            if width < 2 or not any(a == b for a, b in zip(alias, candidate)):
                continue
            changed = sum(a != b for a, b in zip(alias, candidate))
            if changed == 1 and all(
                _within_one_edit(a, b) for a, b in zip(alias, candidate)
            ):
                match = TitleQueryMatch(_FUZZY_TITLE_AFFINITY, start)
                if best is None or (-match.affinity, match.offset) < (
                    -best.affinity,
                    best.offset,
                ):
                    best = match
    return best


def derive_title_links(chunks: Iterable[Chunk]) -> list[Triple]:
    """Create directed ``references`` edges for unambiguous title mentions.

    Aliases shared by multiple documents are discarded rather than guessed.
    A token trie keeps construction proportional to the corpus text plus the
    matched title depth, avoiding a document-by-title Cartesian scan.
    """
    materialized = list(chunks)
    owners: dict[tuple[str, ...], list[str]] = {}
    for chunk in materialized:
        for alias in _title_aliases(chunk.title):
            owners.setdefault(alias, []).append(chunk.id)

    trie: dict = {}
    for alias, chunk_ids in owners.items():
        unique_ids = tuple(dict.fromkeys(chunk_ids))
        if len(unique_ids) != 1:
            continue
        branch = trie
        for token in alias:
            branch = branch.setdefault(token, {})
        branch.setdefault(_TARGET, []).append(unique_ids[0])

    links: list[Triple] = []
    for chunk in materialized:
        tokens = tokenize(chunk.text)
        seen: set[str] = set()
        for start in range(len(tokens)):
            branch = trie
            position = start
            while position < len(tokens) and tokens[position] in branch:
                branch = branch[tokens[position]]
                position += 1
                for target_id in branch.get(_TARGET, ()):
                    if target_id == chunk.id or target_id in seen:
                        continue
                    seen.add(target_id)
                    links.append(Triple(chunk.id, "references", target_id))
    return links
