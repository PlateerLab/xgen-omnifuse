"""VectorStore over candidates a platform has already retrieved and scored.

Wraps dense scores from an external index (Qdrant, pgvector, ...) so OmniFuse
does not re-embed or re-query; only the lexical index is built here, on the
candidate set, and fused with the given dense scores per query
(``dense_weight * minmax(dense) + lexical_weight * minmax(lexical)``).
"""
from __future__ import annotations

from ..fusion import minmax
from ..models import Chunk
from .memory import InMemoryVector


class PreloadedVectorStore:
    """``chunks`` with their dense ``scores`` ({chunk id: score}) from the caller's index."""

    def __init__(
        self,
        chunks: list[Chunk],
        scores: dict[str, float],
        *,
        lexical_weight: float = 0.8,
        dense_weight: float = 1.0,
    ):
        self._chunks = list(chunks)
        self._scores = scores
        self._by_id = {c.id: c for c in self._chunks}
        self._lexical = InMemoryVector(
            self._chunks, lexical_weight=lexical_weight, dense_weight=dense_weight
        )

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        pos = {c.id: i for i, c in enumerate(self._chunks)}
        dense = minmax([(i, self._scores.get(c.id, 0.0)) for i, c in enumerate(self._chunks)])
        lex_raw = {i: 0.0 for i in range(len(self._chunks))}  # lexical search skips non-matches
        try:
            for c, s in self._lexical.search(query, limit=len(self._chunks)):
                if c.id in pos:
                    lex_raw[pos[c.id]] = s
        except Exception:
            pass
        lexical = minmax(list(lex_raw.items()))
        fused = {
            i: self._lexical.dense_weight * dense.get(i, 0.0)
            + self._lexical.lexical_weight * lexical.get(i, 0.0)
            for i in range(len(self._chunks))
        }
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return [(self._chunks[i], s) for i, s in ranked[:limit]]

    def fetch(self, ids: list[str]) -> list[Chunk]:
        """Only chunk ids resolve; graph node ids are not passages."""
        return [self._by_id[i] for i in ids if i in self._by_id]
