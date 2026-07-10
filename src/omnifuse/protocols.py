"""Pluggable backend interfaces — structural typing, no inheritance required.

The OmniFuse algorithm only ever talks to these Protocols, never to a concrete
DB/LLM. Ship the in-memory backends (zero infra) by default; swap in Fuseki /
Qdrant / any LLM by passing objects that match these shapes.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .models import Chunk, Node


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge-graph operations the algorithm needs (label search + traversal)."""

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        """Full-text search over node labels. Returns (node, score) desc."""

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        """Enumerate all instances of a class (the structural 'count/list' power)."""

    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]:
        """1-hop (or N-hop) relations around a node as (subj_label, predicate, obj_label)."""

    def neighbor_ids(self, node_id: str, *, limit: int = 100, direction: str = "both") -> list[str]:
        """Neighbor node ids of ``node_id`` — used to fuse graph structure into
        the retrieval ranking (surface a cited/linked passage alongside its seed).

        ``direction`` is ``"out"`` (what this node points at), ``"in"``, or ``"both"``.
        Retrieval fusion asks for ``"out"``: the passage a seed *references*."""

    def count_class(self, class_id: str) -> int:
        """Number of instances of a class."""

    def get_node(self, node_id: str) -> Optional[Node]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense/lexical passage retrieval."""

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        """Return (chunk, score) desc — cosine if embeddings present, else BM25."""

    def fetch(self, ids: list[str]) -> list[Chunk]:
        ...


@runtime_checkable
class LLM(Protocol):
    """Language model used once, for final synthesis over the fused evidence."""

    def generate(self, prompt: str, *, system: str = "", timeout: Optional[float] = None) -> str:
        ...
