"""Convenience builder — wire the in-memory backends + a default LLM in one call."""
from __future__ import annotations

from typing import Callable, Optional

from .backends.memory import InMemoryGraph, InMemoryVector
from .llm import EchoLLM
from .models import Chunk, Node, Triple
from .oneshot import OmniFuse


def build_inmemory(
    nodes: list[Node],
    triples: list[Triple],
    chunks: list[Chunk],
    *,
    llm=None,
    embedder: Optional[Callable[[str], list[float]]] = None,
    **kwargs,
) -> OmniFuse:
    """Build an OmniFuse over zero-infra in-memory backends.

    Pass ``llm`` (anything with ``generate(prompt, system=, timeout=)``) for real
    synthesis; defaults to EchoLLM so it runs with no API. Pass ``embedder`` to
    enable cosine vector search (else BM25 lexical).
    """
    graph = InMemoryGraph(nodes, triples)
    vector = InMemoryVector(chunks, embedder=embedder)
    return OmniFuse(graph, vector, llm or EchoLLM(), **kwargs)
