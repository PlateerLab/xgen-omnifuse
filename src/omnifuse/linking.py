"""Deterministic document links derived from title mentions in passage text."""

from __future__ import annotations

from collections.abc import Iterable
import re

from .models import Chunk, Triple
from .text import tokenize

_QUALIFIER_BOUNDARIES = frozenset({"from", "in", "of"})
_TERMINAL_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")
_TARGET = "\0"


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
