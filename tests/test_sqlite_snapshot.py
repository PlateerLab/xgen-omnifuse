from __future__ import annotations

import sqlite3

import pytest

from omnifuse import (
    Chunk,
    Node,
    Triple,
    build_inmemory,
    build_sqlite_index,
    open_sqlite_index,
    save_sqlite_index,
)
from omnifuse.backends.sqlite_snapshot import (
    SQLiteSnapshotGraph,
    SQLiteSnapshotVector,
)
from omnifuse.backends import sqlite_snapshot


def _ranked(of, query: str, limit: int = 20):
    return [(chunk.id, score.hex()) for chunk, score in of.retrieve(query, limit=limit)]


def test_posting_uvarint_fast_path_preserves_canonical_validation():
    decode = sqlite_snapshot._decode_posting_uvarint
    assert decode(b"\x7f", 0, 1) == (127, 1)
    assert decode(b"\x80\x01", 0, 2) == (128, 2)
    with pytest.raises(ValueError, match="non-canonical"):
        decode(b"\x80\x00", 0, 2)
    with pytest.raises(ValueError, match="truncated"):
        decode(b"\x80", 0, 1)
    with pytest.raises(ValueError, match="truncated"):
        decode(b"", 0, 0)

@pytest.mark.parametrize(
    ("chunks", "query"),
    [
        ([Chunk("a", "alpha beta"), Chunk("b", "beta")], "alpha alpha beta"),
        (
            [
                Chunk("a", "long body " * 20, title="alpha"),
                Chunk("b", "alpha body", title="other"),
            ],
            "alpha",
        ),
        (
            [
                Chunk("a", "외주화 무엇"),
                Chunk("b", "외주화"),
                Chunk("c", "무엇"),
            ],
            "외주화란 무엇인가요?",
        ),
    ],
)
def test_snapshot_preserves_exact_bm25_and_bm25f_scores(tmp_path, chunks, query):
    source = build_inmemory([], [], chunks)
    assert source.vector._bm25 is None
    path = tmp_path / "index.sqlite"

    save_sqlite_index(source, path)
    with sqlite3.connect(path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM snapshot_meta"))
        assert metadata["vector_mode"] == (
            "raw_bm25f" if any(chunk.title for chunk in chunks) else "raw_bm25"
        )
        assert connection.execute(
            "SELECT mode FROM lexical_config WHERE scope = 'vector'"
        ).fetchone() == (metadata["vector_mode"].removeprefix("raw_"),)
    loaded = open_sqlite_index(path)
    try:
        assert isinstance(loaded.vector, SQLiteSnapshotVector)
        assert _ranked(loaded, query) == _ranked(source, query)
    finally:
        loaded.close()


def test_snapshot_preserves_rankings_without_blobopen(tmp_path, monkeypatch):
    chunks = [Chunk("a", "alpha beta"), Chunk("b", "beta gamma")]
    source = build_inmemory([], [], chunks)
    path = tmp_path / "index.sqlite"
    save_sqlite_index(source, path)
    monkeypatch.setattr(sqlite_snapshot, "_BLOB_OPEN_AVAILABLE", False)

    with open_sqlite_index(path) as loaded:
        assert _ranked(loaded, "alpha beta") == _ranked(source, "alpha beta")


def test_snapshot_fetch_preserves_entities_metadata_and_input_order(tmp_path):
    chunks = [
        Chunk("a", "alpha", entities=["n1"], meta={"page": 2}, title="A"),
        Chunk("b", "beta"),
    ]
    path = tmp_path / "index.sqlite"
    save_sqlite_index(build_inmemory([], [], chunks), path)

    with open_sqlite_index(path) as loaded:
        fetched = loaded.vector.fetch(["b", "missing", "a"])

    assert [chunk.id for chunk in fetched] == ["b", "a"]
    assert fetched[1].entities == ["n1"]
    assert fetched[1].meta == {"page": 2}
    assert fetched[1].title == "A"


def test_direct_sqlite_builder_matches_inmemory_without_retaining_source(tmp_path):
    chunks = [
        Chunk("a", "alpha beta", title="A"),
        Chunk("b", "beta gamma", title="B"),
    ]
    expected = build_inmemory([], [], chunks, vector_kwargs={"idf_pow": 1.0})
    path = tmp_path / "direct.sqlite"

    build_sqlite_index(
        [], [], chunks, path, vector_kwargs={"idf_pow": 1.0}
    )

    with open_sqlite_index(path) as loaded:
        assert _ranked(loaded, "alpha beta") == _ranked(expected, "alpha beta")


def test_direct_sqlite_builder_preserves_plain_bm25_query_multiplicity(tmp_path):
    chunks = [Chunk("a", "alpha beta"), Chunk("b", "beta beta")]
    expected = build_inmemory([], [], chunks, vector_kwargs={"idf_pow": 1.0})
    path = tmp_path / "plain.sqlite"

    build_sqlite_index([], [], iter(chunks), path, vector_kwargs={"idf_pow": 1.0})

    with open_sqlite_index(path) as loaded:
        assert _ranked(loaded, "alpha alpha beta") == _ranked(
            expected, "alpha alpha beta"
        )


def test_direct_sqlite_builder_streams_raw_postings_in_bounded_blocks(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sqlite_snapshot, "_POSTING_BLOCK_BYTES", 24)
    chunks = [
        Chunk(str(index), f"shared body{index}", title=f"title{index}")
        for index in range(40)
    ]
    expected = build_inmemory([], [], chunks, vector_kwargs={"idf_pow": 1.0})
    path = tmp_path / "blocked.sqlite"

    build_sqlite_index([], [], iter(chunks), path, vector_kwargs={"idf_pow": 1.0})

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM vector_payload").fetchone()[0] > 1
        assert connection.execute("SELECT MAX(block_id) FROM vector_terms").fetchone()[0] > 0
    finally:
        connection.close()
    with open_sqlite_index(path) as loaded:
        assert _ranked(loaded, "shared body17 title17") == _ranked(
            expected, "shared body17 title17"
        )


def test_direct_sqlite_builder_preserves_graph_scores_and_rdf_type(tmp_path):
    nodes = [Node("class", "Policy class"), Node("a", "Alpha policy")]
    triples = [Triple("a", "rdf:type", "class")]
    expected = build_inmemory(nodes, triples, [])
    path = tmp_path / "graph.sqlite"

    build_sqlite_index(iter(nodes), iter(triples), [], path)

    with open_sqlite_index(path) as loaded:
        actual = [
            (node.id, score.hex())
            for node, score in loaded.graph.search_labels("policy policy")
        ]
        reference = [
            (node.id, score.hex())
            for node, score in expected.graph.search_labels("policy policy")
        ]
        assert actual == reference
        assert [node.id for node in loaded.graph.class_instances("class")] == ["a"]
        assert loaded.graph.count_class("class") == 1


def test_snapshot_graph_matches_labels_classes_and_directions(tmp_path):
    nodes = [
        Node("rule", "Refund rule", "class"),
        Node("a", "Alpha policy"),
        Node("b", "Beta exception"),
    ]
    triples = [
        Triple("a", "instanceOf", "rule"),
        Triple("a", "references", "b"),
    ]
    source = build_inmemory(nodes, triples, [Chunk("a", "alpha"), Chunk("b", "beta")])
    path = tmp_path / "index.sqlite"
    save_sqlite_index(source, path)
    with sqlite3.connect(path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM snapshot_meta"))
        assert metadata["graph_mode"] == "raw_bm25"
        assert connection.execute(
            "SELECT mode FROM lexical_config WHERE scope = 'graph'"
        ).fetchone() == ("bm25",)

    with open_sqlite_index(path) as loaded:
        assert isinstance(loaded.graph, SQLiteSnapshotGraph)
        assert [node.id for node, _score in loaded.graph.search_labels("refund")] == [
            node.id for node, _score in source.graph.search_labels("refund")
        ]
        assert [node.id for node in loaded.graph.class_instances("rule")] == ["a"]
        assert loaded.graph.count_class("rule") == 1
        assert loaded.graph.neighbor_ids("a", direction="out") == ["rule", "b"]
        assert loaded.graph.neighbor_ids("b", direction="in") == ["a"]
        assert loaded.graph.neighbor_ids("a", direction="both") == ["rule", "b"]
        assert loaded.graph.get_node("b") == nodes[2]
        assert loaded.graph.neighbors("a") == source.graph.neighbors("a")


def test_snapshot_supports_empty_lexical_and_graph_state(tmp_path):
    path = tmp_path / "empty.sqlite"
    save_sqlite_index(build_inmemory([], [], [Chunk("empty", "")]), path)

    with open_sqlite_index(path) as loaded:
        assert loaded.retrieve("anything") == []
        assert loaded.graph.search_labels("anything") == []
        assert [chunk.id for chunk in loaded.vector.fetch(["empty"])] == ["empty"]


def test_snapshot_is_sqlite_and_closing_shared_backends_is_idempotent(tmp_path):
    path = tmp_path / "index.sqlite"
    save_sqlite_index(build_inmemory([], [], [Chunk("a", "alpha")]), path)
    assert path.read_bytes()[:16] == b"SQLite format 3\x00"
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA page_size").fetchone()[0] == 4096
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    assert connection.execute(
        "SELECT value FROM snapshot_meta WHERE key = 'text_codec'"
    ).fetchone()[0] == "raw-or-zlib-v1"
    connection.close()

    loaded = open_sqlite_index(path)
    loaded.close()
    loaded.close()
    with pytest.raises(RuntimeError, match="closed"):
        loaded.vector.fetch(["a"])


def test_snapshot_text_codec_selects_shorter_form_and_is_lossless(tmp_path):
    path = tmp_path / "text.sqlite"
    chunks = [Chunk("short", "x"), Chunk("long", "반복 문장 " * 500, title="제목")]
    save_sqlite_index(build_inmemory([], [], chunks), path)

    connection = sqlite3.connect(path)
    rows = connection.execute("SELECT chunk_id, title, text FROM chunks").fetchall()
    connection.close()

    stored = {chunk_id: (title, text) for chunk_id, title, text in rows}
    assert stored["short"][1] == b"\x00x"
    assert stored["long"][1].startswith(b"\x01")
    with open_sqlite_index(path) as loaded:
        assert loaded.vector.fetch(["short", "long"]) == chunks


def test_open_preserves_legacy_v2_raw_text_compatibility(tmp_path):
    path = tmp_path / "legacy.sqlite"
    save_sqlite_index(build_inmemory([], [], [Chunk("a", "legacy", title="old")]), path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE chunks SET title = 'old', text = 'legacy'")
    connection.execute("DELETE FROM snapshot_meta WHERE key = 'text_codec'")
    connection.execute(
        "UPDATE snapshot_meta SET value = '2' WHERE key = 'schema_version'"
    )
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    with open_sqlite_index(path) as loaded:
        assert loaded.vector.fetch(["a"]) == [Chunk("a", "legacy", title="old")]


def test_snapshot_rejects_corrupt_compressed_text(tmp_path):
    path = tmp_path / "corrupt.sqlite"
    save_sqlite_index(build_inmemory([], [], [Chunk("a", "text " * 100)]), path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE chunks SET text = ?", (b"\x01not-zlib",))
    connection.commit()
    connection.close()

    with open_sqlite_index(path) as loaded, pytest.raises(
        ValueError, match="compressed text"
    ):
        loaded.vector.fetch(["a"])


def test_snapshot_rejects_mutable_and_dense_stores(tmp_path):
    mutable = build_inmemory([], [], [Chunk("a", "alpha")], mutable=True)
    with pytest.raises(TypeError, match="immutable"):
        save_sqlite_index(mutable, tmp_path / "mutable.sqlite")

    dense = build_inmemory(
        [],
        [],
        [Chunk("a", "alpha", embedding=[1.0, 0.0])],
        embedder=lambda _text: [1.0, 0.0],
    )
    with pytest.raises(TypeError, match="dense"):
        save_sqlite_index(dense, tmp_path / "dense.sqlite")


def test_failed_snapshot_write_preserves_existing_target(tmp_path):
    path = tmp_path / "index.sqlite"
    path.write_bytes(b"existing")
    invalid = build_inmemory([], [], [Chunk("a", "alpha", meta={"bad": object()})])

    with pytest.raises(TypeError):
        save_sqlite_index(invalid, path)

    assert path.read_bytes() == b"existing"
    assert list(tmp_path.glob(".index.sqlite.*.tmp")) == []


def test_open_rejects_unknown_snapshot_schema(tmp_path):
    path = tmp_path / "unknown.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE snapshot_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO snapshot_meta VALUES (?, ?)",
        [("schema", "other"), ("schema_version", "1")],
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="schema"):
        open_sqlite_index(path)
