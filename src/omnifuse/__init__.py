"""OmniFuse — backend-agnostic one-shot GraphRAG."""

from importlib import import_module


__version__ = "0.5.0"

_EXPORTS = {
    "OmniFuse": (".oneshot", "OmniFuse"),
    "Vault": (".vault", "Vault"),
    "build_inmemory": (".facade", "build_inmemory"),
    "Feedback": (".feedback", "Feedback"),
    "save_index": (".facade", "save_index"),
    "load_index": (".facade", "load_index"),
    "build_sqlite_index": (".backends.sqlite_snapshot", "build_sqlite_index"),
    "save_sqlite_index": (".backends.sqlite_snapshot", "save_sqlite_index"),
    "open_sqlite_index": (".backends.sqlite_snapshot", "open_sqlite_index"),
    "from_triples": (".facade", "from_triples"),
    "from_jsonl": (".facade", "from_jsonl"),
    "from_csv": (".facade", "from_csv"),
    "from_fuseki": (".facade", "from_fuseki"),
    "Node": (".models", "Node"),
    "Triple": (".models", "Triple"),
    "Chunk": (".models", "Chunk"),
    "ChunkMutationResult": (".models", "ChunkMutationResult"),
    "SearchResult": (".models", "SearchResult"),
    "GraphStore": (".protocols", "GraphStore"),
    "VectorStore": (".protocols", "VectorStore"),
    "MutableVectorStore": (".protocols", "MutableVectorStore"),
    "LLM": (".protocols", "LLM"),
    "InMemoryGraph": (".backends.memory", "InMemoryGraph"),
    "InMemoryVector": (".backends.memory", "InMemoryVector"),
    "MutableInMemoryVector": (".backends.memory", "MutableInMemoryVector"),
    "SQLiteSnapshotGraph": (".backends.sqlite_snapshot", "SQLiteSnapshotGraph"),
    "SQLiteSnapshotVector": (".backends.sqlite_snapshot", "SQLiteSnapshotVector"),
    "FusekiGraph": (".backends.fuseki", "FusekiGraph"),
    "EchoLLM": (".llm", "EchoLLM"),
    "CallableLLM": (".llm", "CallableLLM"),
    "BM25": (".text", "BM25"),
    "tokenize": (".text", "tokenize"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
