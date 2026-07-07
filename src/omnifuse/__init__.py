"""OmniFuse — backend-agnostic one-shot GraphRAG (vector + graph fusion).

Quickstart (zero infra, zero API):

    from omnifuse import build_inmemory, Node, Triple, Chunk
    of = build_inmemory(nodes, triples, chunks)
    print(of.search("불법도박 관련 규정").answer)
"""
from .backends.fuseki import FusekiGraph
from .backends.memory import InMemoryGraph, InMemoryVector
from .facade import build_inmemory, from_csv, from_fuseki, from_jsonl, from_triples
from .llm import CallableLLM, EchoLLM
from .models import Chunk, Node, SearchResult, Triple
from .vault import Vault
from .oneshot import OmniFuse
from .protocols import LLM, GraphStore, VectorStore
from .text import BM25, tokenize

__version__ = "0.5.0"
__all__ = [
    "OmniFuse",
    "Vault",
    "build_inmemory",
    "from_triples",
    "from_jsonl",
    "from_csv",
    "from_fuseki",
    "Node",
    "Triple",
    "Chunk",
    "SearchResult",
    "GraphStore",
    "VectorStore",
    "LLM",
    "InMemoryGraph",
    "InMemoryVector",
    "FusekiGraph",
    "EchoLLM",
    "CallableLLM",
    "BM25",
    "tokenize",
]
