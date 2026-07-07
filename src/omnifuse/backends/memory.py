"""Zero-infra in-memory backends — dict + BM25, pure Python (no DB, no numpy).

These make ``pip install xgen-omnifuse`` run the full algorithm with zero
infrastructure. For scale, swap in Fuseki/Qdrant adapters that match the same
protocols (see omnifuse.protocols).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from ..models import Chunk, Node, Triple
from ..text import BM25, BM25F, tokenize

_ISA = {"instanceOf", "type", "subClassOf", "rdf:type"}
# Title weighted above body in fielded lexical retrieval — a short heading is
# a far stronger relevance signal per token than the passage it heads.
_TITLE_WEIGHT = 4.0


class InMemoryGraph:
    """Triples + node labels, indexed for BM25 label search and 1-hop traversal."""

    def __init__(self, nodes: list[Node], triples: list[Triple]):
        self.nodes: dict[str, Node] = {n.id: n for n in nodes}
        self.triples = triples
        # adjacency: node_id -> list of (subj_label, predicate, obj_label)
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        # class_id -> [instance node ids]   (via instanceOf/type/subClassOf)
        self._members: dict[str, list[str]] = {}
        # node_id -> [neighbor node ids]   (for retrieval-time graph fusion)
        self._adj_ids: dict[str, list[str]] = {}
        for t in triples:
            sl = self._label(t.s)
            ol = self._label(t.o)
            self._adj.setdefault(t.s, []).append((sl, t.p, ol))
            self._adj.setdefault(t.o, []).append((sl, t.p, ol))
            self._adj_ids.setdefault(t.s, []).append(t.o)
            self._adj_ids.setdefault(t.o, []).append(t.s)
            if t.p in _ISA:
                self._members.setdefault(t.o, []).append(t.s)
        self._ids = list(self.nodes.keys())
        self._bm25 = BM25([tokenize(self.nodes[i].label) for i in self._ids])

    def _label(self, nid: str) -> str:
        n = self.nodes.get(nid)
        return n.label if n else nid

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        return [(self.nodes[self._ids[i]], s) for i, s in self._bm25.search(query, limit=limit)]

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        ids = self._members.get(class_id, [])
        return [self.nodes[i] for i in ids[:limit] if i in self.nodes]

    def neighbor_ids(self, node_id: str, *, limit: int = 100) -> list[str]:
        """Distinct neighbor node ids of ``node_id`` (for retrieval-time fusion)."""
        out: list[str] = []
        seen = {node_id}
        for other in self._adj_ids.get(node_id, ()):
            if other not in seen:
                seen.add(other)
                out.append(other)
                if len(out) >= limit:
                    break
        return out

    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]:
        out = list(self._adj.get(node_id, []))[:limit]
        if hops > 1:
            seen = {node_id}
            frontier = {t[2] for t in out} | {t[0] for t in out}
            for _ in range(hops - 1):
                nxt: set[str] = set()
                for lbl in list(frontier):
                    nid = self._by_label(lbl)
                    if nid and nid not in seen:
                        seen.add(nid)
                        out.extend(self._adj.get(nid, [])[:limit])
                frontier = nxt
        return out[:limit]

    def count_class(self, class_id: str) -> int:
        return len(self._members.get(class_id, []))

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def _by_label(self, label: str) -> Optional[str]:
        for nid, n in self.nodes.items():
            if n.label == label:
                return nid
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class InMemoryVector:
    """Passage store with three retrieval modes, chosen by what the chunks carry:

    - **hybrid** — embeddings *and* text present: dense cosine and lexical BM25(F)
      are fused with Reciprocal Rank Fusion (dense recovers paraphrase; lexical
      nails exact terms — each covers the other's blind spot).
    - **dense** — embeddings only: cosine.
    - **lexical** — text only (zero embeddings): field-weighted BM25 over
      title/body, else plain BM25.
    """

    def __init__(self, chunks: list[Chunk], *, embedder: Optional[Callable[[str], list[float]]] = None,
                 title_weight: float = _TITLE_WEIGHT, rrf_k: int = 60,
                 lexical_weight: float = 1.0, dense_weight: float = 1.0):
        self.chunks = chunks
        self._by_id = {c.id: c for c in chunks}
        self.embedder = embedder
        self.rrf_k, self.lexical_weight, self.dense_weight = rrf_k, lexical_weight, dense_weight
        self._dense = embedder is not None and bool(chunks) and all(c.embedding for c in chunks)
        self._lexical = any((c.text or c.title) for c in chunks)
        if self._lexical:
            if any(c.title for c in chunks):
                docs = [{"title": tokenize(c.title), "body": tokenize(c.text)} for c in chunks]
                self._bm25 = BM25F(docs, {"title": title_weight, "body": 1.0})
            else:
                self._bm25 = BM25([tokenize(c.text) for c in chunks])

    def _dense_ranked(self, query: str, limit: int) -> list[tuple[int, float]]:
        q = self.embedder(query)  # type: ignore[misc]
        scored = [(i, _cosine(q, c.embedding)) for i, c in enumerate(self.chunks) if c.embedding]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        if self._dense and self._lexical:
            pool = max(limit, 40)
            dense = self._dense_ranked(query, pool)
            lex = self._bm25.search(query, limit=pool)
            fused: dict[int, float] = {}
            for r, (i, _) in enumerate(dense):
                fused[i] = fused.get(i, 0.0) + self.dense_weight / (self.rrf_k + r)
            for r, (i, _) in enumerate(lex):
                fused[i] = fused.get(i, 0.0) + self.lexical_weight / (self.rrf_k + r)
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
            return [(self.chunks[i], s) for i, s in ranked]
        if self._dense:
            return [(self.chunks[i], s) for i, s in self._dense_ranked(query, limit)]
        if self._lexical:
            return [(self.chunks[i], s) for i, s in self._bm25.search(query, limit=limit)]
        return []

    def fetch(self, ids: list[str]) -> list[Chunk]:
        return [self._by_id[i] for i in ids if i in self._by_id]
