"""Backend exports loaded on first use."""

from importlib import import_module


_EXPORTS = {
    "InMemoryGraph": (".memory", "InMemoryGraph"),
    "InMemoryVector": (".memory", "InMemoryVector"),
    "MutableInMemoryVector": (".memory", "MutableInMemoryVector"),
    "SQLiteSnapshotGraph": (".sqlite_snapshot", "SQLiteSnapshotGraph"),
    "SQLiteSnapshotVector": (".sqlite_snapshot", "SQLiteSnapshotVector"),
    "build_sqlite_index": (".sqlite_snapshot", "build_sqlite_index"),
    "open_sqlite_index": (".sqlite_snapshot", "open_sqlite_index"),
    "save_sqlite_index": (".sqlite_snapshot", "save_sqlite_index"),
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
