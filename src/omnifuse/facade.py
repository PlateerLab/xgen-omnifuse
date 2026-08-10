"""Convenience builders — give a file / Fuseki / loose tuples and search right away."""

from __future__ import annotations

from contextlib import nullcontext
import gzip
import os
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

from .backends.memory import InMemoryGraph, InMemoryVector, MutableInMemoryVector
from .feedback import Feedback
from .llm import EchoLLM
from .linking import derive_title_links
from .loaders import (
    derive_nodes,
    read_chunks_csv,
    read_jsonl,
    read_triples_csv,
    to_chunk,
    to_node,
    to_triple,
)
from .oneshot import OmniFuse


def build_inmemory(
    nodes,
    triples,
    chunks,
    *,
    llm=None,
    embedder: Optional[Callable[[str], list[float]]] = None,
    feedback: Optional[Feedback] = None,
    mutable: bool = False,
    auto_link_titles: bool = False,
    vector_kwargs: Optional[dict] = None,
    **kwargs,
) -> OmniFuse:
    """Build an OmniFuse over zero-infra in-memory backends from Node/Triple/Chunk lists.

    Set ``mutable=True`` to opt into exact incremental chunk upserts and deletes.
    Set ``auto_link_titles=True`` on an immutable corpus to derive directed graph
    edges when one passage mentions another passage's unambiguous title. This
    supplies text-only corpora with deterministic multi-hop retrieval without an
    extraction model.
    ``vector_kwargs`` tunes the passage store (``title_weight`` for field-weighted
    BM25; ``lexical_weight``/``dense_weight`` for the hybrid dense+lexical fusion).
    """
    if auto_link_titles and mutable:
        raise ValueError(
            "auto_link_titles requires an immutable corpus because mutable chunk "
            "updates would otherwise leave derived graph edges stale"
        )
    normalized_nodes = [to_node(node) for node in nodes]
    normalized_triples = [to_triple(triple) for triple in triples]
    vector_type = MutableInMemoryVector if mutable else InMemoryVector
    normalized_chunks = (
        map(to_chunk, chunks) if mutable else [to_chunk(c) for c in chunks]
    )
    if auto_link_titles:
        chunk_nodes = {node.id for node in normalized_nodes}
        normalized_nodes.extend(
            to_node((chunk.id, chunk.title or chunk.id))
            for chunk in normalized_chunks
            if chunk.id not in chunk_nodes
        )
        existing = {(triple.s, triple.p, triple.o) for triple in normalized_triples}
        normalized_triples.extend(
            triple
            for triple in derive_title_links(normalized_chunks)
            if (triple.s, triple.p, triple.o) not in existing
        )
    graph = InMemoryGraph(normalized_nodes, normalized_triples)
    vector = vector_type(
        normalized_chunks,
        embedder=embedder,
        feedback=feedback,
        **(vector_kwargs or {}),
    )
    return OmniFuse(graph, vector, llm or EchoLLM(), **kwargs)


_STATIC_INDEX_FORMAT = 1
_MUTABLE_INDEX_FORMAT = 2
_INDEX_FORMATS = {_STATIC_INDEX_FORMAT, _MUTABLE_INDEX_FORMAT}
GZIP_MAGIC = bytes([0x1F, 0x8B])


def save_index(of: OmniFuse, path) -> None:
    """Persist a built in-memory index (graph + passage store) so the next process can
    ``load_index`` it instead of paying the build cost again. Stdlib pickle + gzip, zero deps.

    gzip is lossless, so a loaded index scores bit-identically to the one saved. Compression
    trades a smaller artifact for decompression work; ``load_index`` also reads legacy
    pre-gzip files.

    The LLM and the embedder callable are *not* persisted — pass them to ``load_index``.
    Only the in-memory backends are supported (a Fuseki graph lives in its own store).

    .. warning:: pickle executes arbitrary code on load. Only load indexes you produced.
    """
    if not isinstance(of.graph, InMemoryGraph) or not isinstance(
        of.vector, InMemoryVector
    ):
        raise TypeError(
            "save_index supports the in-memory backends only "
            f"(got {type(of.graph).__name__}/{type(of.vector).__name__})"
        )
    index_format = (
        _MUTABLE_INDEX_FORMAT
        if isinstance(of.vector, MutableInMemoryVector)
        else _STATIC_INDEX_FORMAT
    )
    target = Path(path)
    read_view = getattr(of.vector, "read_view", None)
    view = read_view() if callable(read_view) else nullcontext()
    temporary = None
    try:
        with view:
            of.vector._prepare_for_persistence()
            with NamedTemporaryFile(
                mode="w+b",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as raw:
                temporary = Path(raw.name)
                with gzip.GzipFile(
                    filename="", mode="wb", compresslevel=6, fileobj=raw
                ) as fh:
                    pickle.dump(
                        {
                            "format": index_format,
                            "graph": of.graph,
                            "vector": of.vector,
                        },
                        fh,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                raw.flush()
                os.fsync(raw.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_index(
    path, *, llm=None, embedder: Optional[Callable[[str], list[float]]] = None, **kwargs
) -> OmniFuse:
    """Rebuild an OmniFuse from an index written by :func:`save_index`.

    .. warning:: pickle executes arbitrary code on load. Only load indexes you produced.
    """
    with open(path, "rb") as fh:
        magic = fh.read(2)
    opener = gzip.open if magic == GZIP_MAGIC else open  # pre-gzip indexes still load
    with opener(path, "rb") as fh:
        blob = pickle.load(fh)
    if type(blob) is not dict:
        raise ValueError("index payload must be a dict")
    fmt = blob.get("format")
    if type(fmt) is not int or fmt not in _INDEX_FORMATS:
        expected = ", ".join(str(value) for value in sorted(_INDEX_FORMATS))
        raise ValueError(
            f"unsupported index format {fmt!r} (expected one of {expected})"
        )
    graph = blob.get("graph")
    vector = blob.get("vector")
    if not isinstance(graph, InMemoryGraph) or not isinstance(vector, InMemoryVector):
        raise ValueError(
            "index payload does not contain the in-memory graph/vector backends"
        )
    if fmt == _MUTABLE_INDEX_FORMAT and not isinstance(vector, MutableInMemoryVector):
        raise ValueError("mutable index format does not contain a mutable vector store")
    if fmt == _STATIC_INDEX_FORMAT and isinstance(vector, MutableInMemoryVector):
        raise ValueError("static index format contains a mutable vector store")
    if embedder is not None:
        vector.attach_embedder(embedder)
    return OmniFuse(graph, vector, llm or EchoLLM(), **kwargs)


def from_triples(
    triples, chunks=None, *, nodes=None, labels=None, llm=None, embedder=None, **kwargs
) -> OmniFuse:
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
    return from_triples(
        read_jsonl(triples),
        read_jsonl(chunks),
        nodes=(read_jsonl(nodes) or None),
        **kwargs,
    )


def from_csv(triples=None, *, chunks=None, **kwargs) -> OmniFuse:
    """Build from CSV files. triples: s/p/o (or subject/predicate/object); chunks: id,text,entities(|-sep)."""
    return from_triples(
        read_triples_csv(triples) if triples else [],
        read_chunks_csv(chunks) if chunks else [],
        **kwargs,
    )


def from_fuseki(
    query_url,
    graph_uri=None,
    *,
    user=None,
    password=None,
    vector=None,
    llm=None,
    **kwargs,
) -> OmniFuse:
    """Build over an Apache Jena Fuseki (or any SPARQL 1.1) endpoint — graph-only by default."""
    from .backends.fuseki import FusekiGraph

    graph = FusekiGraph(query_url, graph_uri, user=user, password=password)
    return OmniFuse(
        graph, vector if vector is not None else InMemoryVector([]), llm, **kwargs
    )
