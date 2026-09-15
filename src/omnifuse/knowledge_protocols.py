"""Snapshot-bound interfaces for hierarchy, embeddings and graph data.

Providers apply scope BEFORE ranking/traversal/counting. They own authorization
and storage transactions; the retrieval library never opens an application DB.
Methods are synchronous to compose with OmniFuse's existing backends; async
hosts use KnowledgeSearch.asearch with cooperative OperationContext cancellation.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from .knowledge import (
    EmbeddingProfile,
    Evidence,
    GraphEntity,
    GraphFact,
    OperationContext,
    Resource,
    SourceChunk,
)


class UnsupportedCapability(ValueError):
    pass


@dataclass(frozen=True)
class ReadScope:
    root_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] | None = None  # None = all; () = empty
    recursive: bool = True

    def __post_init__(self):
        if type(self.recursive) is not bool:
            raise ValueError("recursive must be boolean")
        for name in ("root_ids", "resource_ids"):
            value = getattr(self, name)
            if value is None and name == "resource_ids":
                continue
            if isinstance(value, str) or value is None:
                raise ValueError(f"{name} must be a sequence of resource IDs")
            value = tuple(value)
            if any(not isinstance(i, str) or not i for i in value):
                raise ValueError(f"{name} must be a sequence of resource IDs")
            object.__setattr__(self, name, tuple(dict.fromkeys(value)))


@dataclass(frozen=True)
class Page:
    items: tuple
    next_cursor: str | None
    total: int
    # Reference store is exact; external adapters may explicitly return False.
    exact: bool = True


class HierarchyStore(Protocol):
    def search_resources(self, query: str, *, limit: int = 20) -> list[tuple[Resource, float]]: ...
    def resolve_path(self, parts: tuple[str, ...], *, parent_id: str | None = None) -> list[Resource]: ...
    def get_resources(self, ids: list[str]) -> list[Resource]: ...
    def children(self, parent_id: str | None, *, cursor: str | None = None, limit: int = 100) -> Page: ...
    def ancestors(self, resource_id: str) -> list[Resource]: ...
    def descendants(self, root_id: str, *, cursor: str | None = None, limit: int = 100) -> Page: ...


class ContentStore(Protocol):
    def get_chunks(self, ids: list[str]) -> list[SourceChunk]: ...
    def get_evidence(self, ids: list[str]) -> list[Evidence]: ...
    def search_text(self, query: str, *, limit: int) -> list[tuple[str, float]]: ...


class EmbeddingStore(Protocol):
    def profiles(self) -> list[EmbeddingProfile]: ...
    def search_vector(self, values: list[float], *, profile_id: str, limit: int) -> list[tuple[str, float]]: ...


class KnowledgeGraphStore(Protocol):
    def search_entities(self, query: str, *, limit: int) -> list[tuple[GraphEntity, float]]: ...
    def get_entities(self, ids: list[str]) -> list[GraphEntity]: ...
    def neighbors(self, entity_id: str, *, direction: str = "both", limit: int = 100) -> list[GraphFact]: ...
    def class_members(self, class_id: str, *, cursor: str | None = None, limit: int = 100) -> Page: ...
    def entities_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[str]]: ...
    def entity_chunk_counts(self, entity_ids: list[str]) -> dict[str, int]: ...
    def chunks_for_entities(self, entity_ids: list[str], *, limit: int = 1000) -> list[str]: ...


class KnowledgeView(Protocol):
    corpus_id: str
    snapshot_id: str
    capabilities: frozenset[str]
    hierarchy: HierarchyStore | None
    content: ContentStore
    embeddings: EmbeddingStore | None
    graph: KnowledgeGraphStore | None

    def check(self) -> None:
        """Verify cancellation, view lifetime and current access policy."""


class KnowledgeProvider(Protocol):
    def open_view(self, *, scope: ReadScope, snapshot_id: str | None,
                  context: OperationContext) -> AbstractContextManager[KnowledgeView]: ...
