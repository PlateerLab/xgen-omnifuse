"""Convenience builders — give a file / Fuseki / loose tuples and search right away."""
from __future__ import annotations

import pickle
from typing import Callable, Optional

from .backends.memory import InMemoryGraph, InMemoryVector
from .feedback import Feedback
from .llm import EchoLLM
from .loaders import (
    derive_nodes,
    read_chunks_csv,
    read_jsonl,
    read_triples_csv,
    to_chunk,
    to_node,
    to_triple,
)
from .models import Chunk, Node, Triple
from .oneshot import OmniFuse


def build_inmemory(nodes, triples, chunks, *, llm=None,
                   embedder: Optional[Callable[[str], list[float]]] = None,
                   feedback: Optional[Feedback] = None,
                   vector_kwargs: Optional[dict] = None, **kwargs) -> OmniFuse:
    """Build an OmniFuse over zero-infra in-memory backends from Node/Triple/Chunk lists.

    ``vector_kwargs`` tunes the passage store (``title_weight`` for field-weighted
    BM25; ``lexical_weight``/``dense_weight`` for the hybrid dense+lexical fusion).
    """
    graph = InMemoryGraph([to_node(n) for n in nodes], [to_triple(t) for t in triples])
    vector = InMemoryVector([to_chunk(c) for c in chunks], embedder=embedder,
                            feedback=feedback, **(vector_kwargs or {}))
    return OmniFuse(graph, vector, llm or EchoLLM(), **kwargs)


_INDEX_FORMAT = 1


def save_index(of: OmniFuse, path) -> None:
    """Persist a built in-memory index (graph + passage store) so the next process can
    ``load_index`` it instead of paying the build cost again. Stdlib pickle, zero deps.

    The LLM and the embedder callable are *not* persisted — pass them to ``load_index``.
    Only the in-memory backends are supported (a Fuseki graph lives in its own store).

    .. warning:: pickle executes arbitrary code on load. Only load indexes you produced.
    """
    if not isinstance(of.graph, InMemoryGraph) or not isinstance(of.vector, InMemoryVector):
        raise TypeError("save_index supports the in-memory backends only "
                        f"(got {type(of.graph).__name__}/{type(of.vector).__name__})")
    with open(path, "wb") as fh:
        pickle.dump({"format": _INDEX_FORMAT, "graph": of.graph, "vector": of.vector},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path, *, llm=None, embedder: Optional[Callable[[str], list[float]]] = None,
               **kwargs) -> OmniFuse:
    """Rebuild an OmniFuse from an index written by :func:`save_index`.

    .. warning:: pickle executes arbitrary code on load. Only load indexes you produced.
    """
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    fmt = blob.get("format")
    if fmt != _INDEX_FORMAT:
        raise ValueError(f"unsupported index format {fmt!r} (expected {_INDEX_FORMAT})")
    vector = blob["vector"]
    if embedder is not None:
        vector.attach_embedder(embedder)
    return OmniFuse(blob["graph"], vector, llm or EchoLLM(), **kwargs)


def from_triples(triples, chunks=None, *, nodes=None, labels=None, llm=None,
                 embedder=None, **kwargs) -> OmniFuse:
    """Build from loose ``(s, p, o)`` tuples/dicts/Triples. Nodes are inferred if omitted.

        of = from_triples([("담보", "instanceOf", "규정"), ("담보", "한도", "5억")],
                          chunks=[("c1", "담보 한도는 5억원이다", ["담보"])])
    """
    trs = [to_triple(t) for t in triples]
    chs = [to_chunk(c) for c in (chunks or [])]
    nds = [to_node(n) for n in nodes] if nodes else derive_nodes(trs, labels)
    return build_inmemory(nds, trs, chs, llm=llm, embedder=embedder, **kwargs)


def from_jsonl(triples=None, *, nodes=None, chunks=None, **kwargs) -> OmniFuse:
    """Build from JSONL files (one JSON object per line)."""
    return from_triples(read_jsonl(triples), read_jsonl(chunks),
                        nodes=(read_jsonl(nodes) or None), **kwargs)


def from_csv(triples=None, *, chunks=None, **kwargs) -> OmniFuse:
    """Build from CSV files. triples: s/p/o (or subject/predicate/object); chunks: id,text,entities(|-sep)."""
    return from_triples(read_triples_csv(triples) if triples else [],
                        read_chunks_csv(chunks) if chunks else [], **kwargs)


def from_fuseki(query_url, graph_uri=None, *, user=None, password=None,
                vector=None, llm=None, **kwargs) -> OmniFuse:
    """Build over an Apache Jena Fuseki (or any SPARQL 1.1) endpoint — graph-only by default."""
    from .backends.fuseki import FusekiGraph

    graph = FusekiGraph(query_url, graph_uri, user=user, password=password)
    return OmniFuse(graph, vector if vector is not None else InMemoryVector([]), llm, **kwargs)
