"""Zero-infra in-memory backends — dict + BM25, pure Python (no DB, no numpy).

These make ``pip install xgen-omnifuse`` run the full algorithm with zero
infrastructure. For scale, swap in Fuseki/Qdrant adapters that match the same
protocols (see omnifuse.protocols).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from ..models import Chunk, Node, Triple
from ..text import BM25, tokenize

_ISA = {"instanceOf", "type", "subClassOf", "rdf:type"}


class InMemoryGraph:
    """Triples + node labels, indexed for BM25 label search and 1-hop traversal."""

    def __init__(self, nodes: list[Node], triples: list[Triple]):
        self.nodes: dict[str, Node] = {n.id: n for n in nodes}
        self.triples = triples
        # adjacency: node_id -> list of (subj_label, predicate, obj_label)
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        # class_id -> [instance node ids]   (via instanceOf/type/subClassOf)
        self._members: dict[str, list[str]] = {}
        for t in triples:
            sl = self._label(t.s)
            ol = self._label(t.o)
            self._adj.setdefault(t.s, []).append((sl, t.p, ol))
            self._adj.setdefault(t.o, []).append((sl, t.p, ol))
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
    """Passage store. Uses cosine when embeddings + an embedder are present,
    otherwise pure-Python BM25 over chunk text — so it works with zero embeddings."""

    def __init__(self, chunks: list[Chunk], *, embedder: Optional[Callable[[str], list[float]]] = None):
        self.chunks = chunks
        self._by_id = {c.id: c for c in chunks}
        self.embedder = embedder
        self._use_vec = embedder is not None and all(c.embedding for c in chunks) and bool(chunks)
        if not self._use_vec:
            self._bm25 = BM25([tokenize(c.text) for c in chunks])

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        if self._use_vec:
            q = self.embedder(query)  # type: ignore[misc]
            scored = [(c, _cosine(q, c.embedding)) for c in self.chunks if c.embedding]
            scored.sort(key=lambda x: -x[1])
            return scored[:limit]
        return [(self.chunks[i], s) for i, s in self._bm25.search(query, limit=limit)]

    def fetch(self, ids: list[str]) -> list[Chunk]:
        return [self._by_id[i] for i in ids if i in self._by_id]
