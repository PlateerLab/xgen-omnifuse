"""Core data models for OmniFuse (plain dataclasses, zero deps)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Node:
    """A graph node. ``kind`` is one of class | instance | property."""

    id: str
    label: str
    kind: str = "instance"


@dataclass
class Triple:
    """A directed edge: subject -> predicate -> object (node ids)."""

    s: str
    p: str
    o: str


@dataclass
class Chunk:
    """A text passage. ``entities`` are node ids this chunk mentions.

    ``title`` is an optional short high-signal field (heading, doc title, article
    number) weighted above the body in lexical retrieval. ``embedding`` is
    optional — when absent, vector stores fall back to BM25.
    """

    id: str
    text: str
    entities: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    meta: dict[str, Any] = field(default_factory=dict)
    title: str = ""


@dataclass
class SearchResult:
    """Result of an OmniFuse search."""

    answer: str
    question: str
    chunks: list[Chunk] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)  # "s → p → o" strings used
    evidence_nodes: list[str] = field(default_factory=list)  # node labels the answer cites
    class_seed: str = ""
    mode: str = "omnifuse"
