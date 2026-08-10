"""Queries confirmed to answer a chunk, indexed as BM25F evidence.

Evidence terms score only the chunk that owns them. They do not enter document frequency
or length normalization, so unrelated chunks and the collection's content IDF stay fixed.
"""

from __future__ import annotations

import json
from typing import Iterable


class Feedback:
    """Per-chunk memory of the queries it was confirmed to answer. Zero dependencies."""

    __slots__ = ("_mem",)

    def __init__(self) -> None:
        self._mem: dict[str, list[str]] = {}

    def remember(self, query: str, doc_ids: Iterable[str]) -> None:
        """Record that ``query`` was answered by each of ``doc_ids``."""
        q = (query or "").strip()
        if not q:
            return
        for doc_id in doc_ids:
            seen = self._mem.setdefault(doc_id, [])
            if q not in seen:
                seen.append(q)

    def forget(self, query: str, doc_ids: Iterable[str]) -> None:
        """Withdraw a remembered ``(query -> document)`` pair. Unknown pairs are a no-op."""
        q = (query or "").strip()
        for doc_id in doc_ids:
            seen = self._mem.get(doc_id)
            if seen and q in seen:
                seen.remove(q)
                if not seen:
                    del self._mem[doc_id]

    def drop(self, doc_ids: Iterable[str]) -> int:
        """Remove every remembered query for deleted documents."""
        removed = 0
        for doc_id in doc_ids:
            if self._mem.pop(doc_id, None) is not None:
                removed += 1
        return removed

    def copy(self) -> "Feedback":
        """Return a detached snapshot suitable for one index owner."""
        duplicate = Feedback()
        duplicate._mem = {
            doc_id: list(queries) for doc_id, queries in self._mem.items()
        }
        return duplicate

    def _replace_queries(self, doc_id: str, queries: Iterable[str]) -> None:
        values = list(queries)
        if values:
            self._mem[doc_id] = values
        else:
            self._mem.pop(doc_id, None)

    def observe_ranked(
        self, retrieved: Iterable[str], relevant: Iterable[str], query: str
    ) -> None:
        """Record a judged result list: only the confirmed-relevant chunks remember it."""
        rel = set(relevant)
        self.remember(query, [d for d in retrieved if d in rel])

    def queries(self, doc_id: str) -> list[str]:
        """Queries this chunk is known to answer (empty if never confirmed)."""
        return list(self._mem.get(doc_id, ()))

    def text(self, doc_id: str) -> str:
        return " ".join(self._mem.get(doc_id, ()))

    def __len__(self) -> int:
        return len(self._mem)

    def __bool__(self) -> bool:
        return bool(self._mem)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._mem, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Feedback":
        fb = cls()
        with open(path, encoding="utf-8") as fh:
            fb._mem = {str(k): list(v) for k, v in json.load(fh).items()}
        return fb
