"""Memory — what a document has *proven* to answer, not just what it says.

synaptic-memory learns by reinforcing graph nodes and edges on co-activation (Hebbian).
That is a **query-independent** prior: "this document tends to be relevant". Measured on a
held-out split of NFCorpus it does not help — it is neutral on one split (−0.0002) and
harmful on the other (−0.0174). Relevance is not a property of a document; it is a property
of a *(query, document)* pair, so a query-independent prior injects noise.

We tried query-independent priors too, and they were worse: multiplying scores by a Beta
posterior odds `(hits+1)/(misses+1)` cost **−0.0384**, positive-only **−0.0175**, and
empirical-Bayes shrinkage to the base rate **−0.0489**. The bias is structural — every
judged document sinks relative to never-shown ones.

So memory here is **query-conditional and stored as text**: when a query is confirmed to be
answered by a document, that query becomes part of what the document is *about*. It is
indexed as its own BM25F field, so it neither dilutes the body's length normalization nor
inherits the title's boost. Nothing is tuned: an unremembered document has an empty memory
field, which is *bit-identical* to having no memory field at all.

    fb = Feedback()
    fb.remember("statin side effects", ["doc7"])       # a user confirmed doc7 answered it
    of = build_inmemory(nodes, triples, chunks, feedback=fb)

Measured on held-out queries (feedback replayed on one half, evaluated on the other):
NFCorpus **+0.0019 / +0.0076**, MIRACL-ko **+0.0618 / +0.0729** across both splits, versus
synaptic's −0.0002 / −0.0174.
"""
from __future__ import annotations

import json
from typing import Iterable


class Feedback:
    """Per-document memory of the queries it was confirmed to answer. Zero dependencies."""

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

    def observe_ranked(self, retrieved: Iterable[str], relevant: Iterable[str], query: str) -> None:
        """Record a judged result list: only the confirmed-relevant documents remember it."""
        rel = set(relevant)
        self.remember(query, [d for d in retrieved if d in rel])

    def queries(self, doc_id: str) -> list[str]:
        """Queries this document is known to answer (empty if never confirmed)."""
        return self._mem.get(doc_id, [])

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
