"""Memory — the queries a chunk was confirmed to answer, indexed as *evidence*.

A user asks the same thing in different words. The document that answered them last time
does not contain the new phrasing; it never will. What connects them is the *earlier
query*. So memory stores that query, and retrieval matches against it.

The trap is that memory must not become content. Our first attempt appended the remembered
query to the document body and appeared to win big — until the placebo controls showed the
gain survived shuffling the (query, document) pairing, and even lifted queries whose
relevant documents remembered nothing. Injecting query text raises the document frequency
of query vocabulary and deflates its IDF corpus-wide; the "memory" was an accidental,
uncontrolled `idf_pow` reduction. See `eval/results/adaptive_memory.json`.

So memory is an **evidence field** (`BM25F(evidence_fields=…)`): its terms score a
document but never enter document frequency, and they are not length-normalized. Three
properties follow, none of them tuned:

* a chunk that remembers nothing has an empty evidence field, so a cold store ranks
  **bit-identically** to one built with no feedback at all;
* the collection's IDF is untouched, so memory cannot move a query whose relevant
  documents remember nothing (measured: **Δ = +0.0000**);
* remembering a second query does not dilute the first.

    fb = Feedback()
    fb.remember("statin side effects", ["doc7"])       # a user confirmed doc7 answered it
    of = build_inmemory(nodes, triples, chunks, feedback=fb)

Measured against synaptic's Hebbian reinforcement on the same corpus, queries and scorer
(paraphrased re-queries, held-out): ΔMRR@10 **+0.4167 vs +0.0093** on the chunks that
remember, with placebos at +0.024 (shuffled) and +0.080 (random query). On *unrelated*
queries — a different question, not a rephrasing — memory correctly does nothing (+0.0006).
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

    def observe_ranked(self, retrieved: Iterable[str], relevant: Iterable[str], query: str) -> None:
        """Record a judged result list: only the confirmed-relevant chunks remember it."""
        rel = set(relevant)
        self.remember(query, [d for d in retrieved if d in rel])

    def queries(self, doc_id: str) -> list[str]:
        """Queries this chunk is known to answer (empty if never confirmed)."""
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
