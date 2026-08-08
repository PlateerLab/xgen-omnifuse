"""Read-only, disk-queryable SQLite snapshots for static OmniFuse indexes."""
from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile
from threading import RLock
from urllib.parse import quote
import zlib

from .._compact_postings import (
    CompactPostingsSnapshot,
    _RawPostingsSnapshot,
)
from ..llm import EchoLLM
from ..loaders import to_chunk, to_node, to_triple
from ..models import Chunk, Node
from ..oneshot import OmniFuse
from ..settings import DEFAULT_IDF_POW, DEFAULT_TITLE_WEIGHT
from ..text import (
    BM25,
    BM25F,
    _analyze_query,
    _coordinate_query_scores,
    tokenize,
)


_SCHEMA = "omnifuse.sqlite_snapshot"
_SCHEMA_VERSION = 3
_READABLE_SCHEMA_VERSIONS = frozenset({2, _SCHEMA_VERSION})
_APPLICATION_ID = 1330005587
_INT_TYPECODE = "i"
_FLOAT_TYPECODE = "d"
_POSTING_BLOCK_BYTES = 64 * 1024
_ISA = frozenset({"instanceOf", "type", "subClassOf", "rdf:type"})
_SQL = """
PRAGMA page_size = 4096;
PRAGMA application_id = 1330005587;
PRAGMA user_version = 3;
BEGIN EXCLUSIVE;
CREATE TABLE snapshot_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE chunks (
    slot INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    title BLOB NOT NULL,
    text BLOB NOT NULL,
    entities_json TEXT,
    meta_json TEXT
);
CREATE TABLE vector_terms (
    term TEXT PRIMARY KEY,
    block_id INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL,
    byte_count INTEGER NOT NULL,
    doc_count INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE vector_payload (
    block_id INTEGER PRIMARY KEY,
    data BLOB NOT NULL
);
CREATE TABLE graph_nodes (
    slot INTEGER PRIMARY KEY,
    node_id TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    kind TEXT NOT NULL
);
CREATE TABLE graph_terms (
    term TEXT PRIMARY KEY,
    block_id INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL,
    byte_count INTEGER NOT NULL,
    doc_count INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE graph_payload (
    block_id INTEGER PRIMARY KEY,
    data BLOB NOT NULL
);
CREATE TABLE lexical_config (
    scope TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    k1 REAL NOT NULL,
    b REAL NOT NULL,
    idf_pow REAL NOT NULL,
    document_count INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE lexical_fields (
    scope TEXT NOT NULL,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    weight REAL NOT NULL,
    evidence INTEGER NOT NULL,
    total INTEGER NOT NULL,
    length_typecode TEXT NOT NULL,
    lengths BLOB NOT NULL,
    PRIMARY KEY (scope, position)
) WITHOUT ROWID;
CREATE TABLE graph_edges (
    position INTEGER PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL
);
CREATE INDEX graph_edges_subject ON graph_edges(subject_id, position);
CREATE INDEX graph_edges_object ON graph_edges(object_id, position);
CREATE INDEX graph_edges_predicate_object
    ON graph_edges(predicate, object_id, position);
"""

if array(_INT_TYPECODE).itemsize != 4 or array(_FLOAT_TYPECODE).itemsize != 8:
    raise RuntimeError("SQLite snapshot requires 32-bit int and 64-bit float arrays")


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _wire_bytes(values: array) -> bytes:
    if sys.byteorder == "little":
        return values.tobytes()
    normalized = array(values.typecode, values)
    if normalized.itemsize > 1:
        normalized.byteswap()
    return normalized.tobytes()


def _wire_array(typecode: str, payload: bytes) -> array:
    values = array(typecode)
    values.frombytes(payload)
    if sys.byteorder != "little" and values.itemsize > 1:
        values.byteswap()
    return values


def _decode_posting_uvarint(
    data: bytes, position: int, end: int
) -> tuple[int, int]:
    if position >= end:
        raise ValueError("truncated unsigned LEB128")
    byte = data[position]
    position += 1
    if byte < 0x80:
        return byte, position
    value = byte & 0x7F
    shift = 7
    while position < end:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            if byte == 0:
                raise ValueError("non-canonical unsigned LEB128")
            return value, position
        shift += 7
    raise ValueError("truncated unsigned LEB128")

def _insert_packed_postings(
    connection: sqlite3.Connection,
    directory_table: str,
    payload_table: str,
    postings: Iterable[tuple[str, array, array]],
) -> None:
    terms: list[str] = []
    block_ids = array(_INT_TYPECODE)
    offsets = array(_INT_TYPECODE)
    byte_counts = array(_INT_TYPECODE)
    counts = array(_INT_TYPECODE)

    def payload_rows():
        payload = bytearray()
        block_id = 0
        for term, doc_ids, weights in postings:
            if len(doc_ids) != len(weights):
                raise ValueError("lexical posting ids and weights differ in length")
            ids_payload = _wire_bytes(doc_ids)
            weight_payload = _wire_bytes(weights)
            if payload and (
                len(payload) + len(ids_payload) + len(weight_payload)
                > _POSTING_BLOCK_BYTES
            ):
                yield block_id, sqlite3.Binary(bytes(payload))
                payload.clear()
                block_id += 1
            offset = len(payload) + 1
            payload.extend(ids_payload)
            payload.extend(weight_payload)
            terms.append(term)
            block_ids.append(block_id)
            offsets.append(offset)
            byte_counts.append(len(ids_payload) + len(weight_payload))
            counts.append(len(doc_ids))
        yield block_id, sqlite3.Binary(bytes(payload))

    connection.executemany(
        f"INSERT INTO {payload_table} VALUES (?, ?)",
        payload_rows(),
    )
    connection.executemany(
        f"INSERT INTO {directory_table} VALUES (?, ?, ?, ?, ?)",
        zip(terms, block_ids, offsets, byte_counts, counts),
    )


def _bm25_postings(
    index: BM25 | BM25F | None, *, consume: bool
) -> Iterator[tuple[str, array, array]]:
    if index is None:
        return
    if not consume:
        for term, doc_ids in index._pd.items():
            yield term, doc_ids, index._pw[term]
        return
    for term in tuple(index._pd):
        doc_ids = index._pd.pop(term)
        yield term, doc_ids, index._pw.pop(term)
        index.idf.pop(term, None)


def _insert_raw_postings(
    connection: sqlite3.Connection,
    scope: str,
    directory_table: str,
    payload_table: str,
    index: CompactPostingsSnapshot | _RawPostingsSnapshot | None,
) -> None:
    if index is None:
        _insert_packed_postings(
            connection,
            directory_table,
            payload_table,
            (),
        )
        return

    if isinstance(index, _RawPostingsSnapshot):
        streams = index._posting_streams
        if len(streams) != len(index.terms):
            raise RuntimeError("raw posting stream count differs from the vocabulary")

        def payload_rows():
            block_id = 0
            payload = bytearray()
            for posting in streams:
                if payload and len(payload) + len(posting) > _POSTING_BLOCK_BYTES:
                    yield block_id, sqlite3.Binary(bytes(payload))
                    payload.clear()
                    block_id += 1
                payload.extend(posting)
            yield block_id, sqlite3.Binary(bytes(payload))

        def directory_rows():
            block_id = 0
            block_bytes = 0
            for term_id, (term, posting) in enumerate(
                zip(index.terms, streams, strict=True)
            ):
                byte_count = len(posting)
                if block_bytes and block_bytes + byte_count > _POSTING_BLOCK_BYTES:
                    block_id += 1
                    block_bytes = 0
                yield (
                    term,
                    block_id,
                    block_bytes + 1,
                    byte_count,
                    index._df[term_id],
                )
                block_bytes += byte_count

        connection.executemany(
            f"INSERT INTO {payload_table} VALUES (?, ?)", payload_rows()
        )
        connection.executemany(
            f"INSERT INTO {directory_table} VALUES (?, ?, ?, ?, ?)",
            directory_rows(),
        )
        for posting in streams:
            posting.clear()
        streams.clear()
    else:
        connection.execute(
            f"INSERT INTO {payload_table} VALUES (?, ?)",
            (0, sqlite3.Binary(index._posting_blob)),
        )
        connection.executemany(
            f"INSERT INTO {directory_table} VALUES (?, ?, ?, ?, ?)",
            (
                (
                    term,
                    0,
                    index._posting_offsets[term_id] + 1,
                    index._posting_offsets[term_id + 1]
                    - index._posting_offsets[term_id],
                    index._df[term_id] or index._dfe[term_id],
                )
                for term_id, term in enumerate(index.terms)
            ),
        )
    connection.execute(
        "INSERT INTO lexical_config VALUES (?, ?, ?, ?, ?, ?)",
        (scope, index.mode, index.k1, index.b, index.idf_pow, index.N),
    )
    connection.executemany(
        "INSERT INTO lexical_fields VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            (
                scope,
                position,
                field,
                index.weights[position],
                int(index.evidence[position]),
                index._totals[position],
                lengths.typecode,
                sqlite3.Binary(_wire_bytes(lengths)),
            )
            for position, (field, lengths) in enumerate(
                zip(index.fields, index._lengths, strict=True)
            )
        ),
    )


def _lexical_mode(
    index: BM25 | BM25F | CompactPostingsSnapshot | None,
) -> str:
    if index is None:
        return "none"
    if isinstance(index, CompactPostingsSnapshot):
        return f"raw_{index.mode}"
    if type(index) is BM25:
        return "bm25"
    if type(index) is BM25F:
        return "bm25f"
    raise TypeError(f"unsupported static lexical index {type(index).__name__}")


def _insert_snapshot_postings(
    connection: sqlite3.Connection,
    scope: str,
    directory_table: str,
    payload_table: str,
    index: BM25 | BM25F | CompactPostingsSnapshot | None,
    *,
    consume: bool,
) -> None:
    if isinstance(index, CompactPostingsSnapshot):
        _insert_raw_postings(
            connection,
            scope,
            directory_table,
            payload_table,
            index,
        )
        return
    _insert_packed_postings(
        connection,
        directory_table,
        payload_table,
        _bm25_postings(index, consume=consume),
    )


def _readonly_uri(path: Path) -> str:
    encoded = quote(path.resolve().as_posix(), safe="/:")
    return f"file:{encoded}?mode=ro&immutable=1"


class _SnapshotDatabase:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.lock = RLock()
        self.connection = sqlite3.connect(
            _readonly_uri(self.path),
            uri=True,
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA cache_size=-256")
        self.connection.execute("PRAGMA query_only=ON")
        try:
            rows = self.connection.execute(
                "SELECT key, value FROM snapshot_meta"
            ).fetchall()
            self.meta = dict(rows)
            version = self.meta.get("schema_version", "")
            if not version.isdecimal():
                raise ValueError("unsupported SQLite snapshot schema")
            self.schema_version = int(version)
            if (
                self.meta.get("schema") != _SCHEMA
                or self.schema_version not in _READABLE_SCHEMA_VERSIONS
            ):
                raise ValueError("unsupported SQLite snapshot schema")
            if self.schema_version == 3:
                if self.meta.get("text_codec") != "raw-or-zlib-v1":
                    raise ValueError("unsupported SQLite snapshot text codec")
            elif "text_codec" in self.meta:
                raise ValueError("invalid legacy SQLite snapshot text codec")
            if self.meta.get("byteorder") != "little":
                raise ValueError("unsupported SQLite snapshot byte order")
            if self.meta.get("vector_mode") not in {
                "none",
                "bm25",
                "bm25f",
                "raw_bm25",
                "raw_bm25f",
            }:
                raise ValueError("invalid SQLite snapshot vector mode")
            if self.meta.get("graph_mode") not in {"none", "bm25", "raw_bm25"}:
                raise ValueError("invalid SQLite snapshot graph mode")
            application_id = self.connection.execute(
                "PRAGMA application_id"
            ).fetchone()[0]
            user_version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if application_id != _APPLICATION_ID or user_version != self.schema_version:
                raise ValueError("invalid SQLite snapshot file identity")
            for key in ("chunk_count", "graph_node_count", "graph_edge_count"):
                value = self.meta.get(key, "")
                if not value.isdecimal() or str(int(value)) != value:
                    raise ValueError(f"invalid SQLite snapshot {key}")
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def execute(self, sql: str, parameters=()):
        if self.connection is None:
            raise RuntimeError("SQLite snapshot is closed")
        return self.connection.execute(sql, parameters)


def _query_postings(
    database: _SnapshotDatabase,
    table: str,
    payload_table: str,
    terms: list[str],
    *,
    repeated_terms: bool,
    anchors: frozenset[str],
    restricted: bool,
) -> list[tuple[int, float]]:
    ordered = list(dict.fromkeys(terms))
    if not ordered:
        return []
    rows: dict[str, tuple[bytes, bytes]] = {}
    for start in range(0, len(ordered), 500):
        batch = ordered[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        sql = f"""
            SELECT d.term,
                   substr(p.data, d.byte_offset, d.doc_count * 4),
                   substr(p.data, d.byte_offset + d.doc_count * 4, d.doc_count * 8)
              FROM {table} AS d JOIN {payload_table} AS p USING (block_id)
             WHERE d.term IN ({placeholders})
        """
        for term, doc_ids, weights in database.execute(sql, batch):
            rows[term] = (doc_ids, weights)

    counts: dict[str, int] = {}
    if repeated_terms:
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
    else:
        counts = dict.fromkeys(ordered, 1)

    scores: dict[int, float] = {}
    candidates: set[int] = set()
    complete_candidates: set[int] = set()
    for term in ordered:
        row = rows.get(term)
        if row is None:
            continue
        ids = _wire_array(_INT_TYPECODE, row[0])
        weights = _wire_array(_FLOAT_TYPECODE, row[1])
        if len(ids) != len(weights):
            raise ValueError("SQLite snapshot posting is truncated")
        if restricted and term in anchors:
            candidates.update(ids)
        if term in anchors and term.startswith("#"):
            complete_candidates.update(ids)
        multiplier = counts[term]
        if multiplier == 1:
            for doc_id, weight in zip(ids, weights):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight
        else:
            for doc_id, weight in zip(ids, weights):
                scores[doc_id] = scores.get(doc_id, 0.0) + multiplier * weight
    scores = _coordinate_query_scores(scores, candidates, complete_candidates)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class _RawSQLiteLexical:
    def __init__(self, database: _SnapshotDatabase, scope: str):
        self.database = database
        self.scope = scope
        self.directory_table = f"{scope}_terms"
        self.payload_table = f"{scope}_payload"
        row = database.execute(
            """SELECT mode, k1, b, idf_pow, document_count
                 FROM lexical_config WHERE scope = ?""",
            (scope,),
        ).fetchone()
        if row is None:
            raise ValueError(f"SQLite snapshot is missing {scope} lexical config")
        self.mode, self.k1, self.b, self.idf_pow, self.document_count = row
        if self.mode not in {"bm25", "bm25f"}:
            raise ValueError(f"SQLite snapshot has invalid {scope} raw mode")
        if (
            not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in (self.k1, self.b, self.idf_pow)
            )
            or self.k1 < 0
            or not 0 <= self.b <= 1
            or self.idf_pow <= 0
            or type(self.document_count) is not int
            or self.document_count < 0
        ):
            raise ValueError(f"SQLite snapshot has invalid {scope} scoring config")
        rows = database.execute(
            """SELECT position, name, weight, evidence, total,
                      length_typecode, lengths
                 FROM lexical_fields WHERE scope = ? ORDER BY position""",
            (scope,),
        ).fetchall()
        if not rows or [row[0] for row in rows] != list(range(len(rows))):
            raise ValueError(f"SQLite snapshot has invalid {scope} fields")
        self.weights: list[float] = []
        self.evidence: list[bool] = []
        self.totals: list[int] = []
        self.lengths: list[array] = []
        for _position, name, weight, evidence, total, typecode, payload in rows:
            if (
                type(name) is not str
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or evidence not in {0, 1}
                or type(total) is not int
                or total < 0
                or type(typecode) is not str
                or typecode not in {"B", "H", "I", "Q"}
                or not isinstance(payload, bytes)
            ):
                raise ValueError(f"SQLite snapshot has invalid {scope} field config")
            lengths = _wire_array(typecode, payload)
            if len(lengths) != self.document_count:
                raise ValueError(f"SQLite snapshot has invalid {scope} field lengths")
            self.weights.append(float(weight))
            self.evidence.append(bool(evidence))
            self.totals.append(total)
            self.lengths.append(lengths)
        if self.mode == "bm25" and (
            len(rows) != 1
            or rows[0][1] != "body"
            or self.weights != [1.0]
            or self.evidence != [False]
        ):
            raise ValueError(f"SQLite snapshot has invalid {scope} BM25 fields")
        self._averages = [
            (total / self.document_count if self.document_count else 0.0) or 1.0
            for total in self.totals
        ]
        self._k1_plus_one = self.k1 + 1

    def _posting_scores(self, payload: bytes, doc_count: int, idf: float):
        position = 0
        previous = 0
        posting_index = 0
        end = len(payload)
        decode = _decode_posting_uvarint
        document_count = self.document_count
        k1 = self.k1
        b = self.b
        k1_plus_one = self._k1_plus_one
        if self.mode == "bm25":
            lengths = self.lengths[0]
            average = self._averages[0]
            while position < end:
                delta, position = decode(payload, position, end)
                if posting_index and delta == 0:
                    raise ValueError(
                        "SQLite raw posting document ids are not increasing"
                    )
                doc_id = delta if not posting_index else previous + delta
                if doc_id >= document_count:
                    raise ValueError(
                        "SQLite raw posting references a missing document"
                    )
                frequency, position = decode(payload, position, end)
                if not frequency:
                    raise ValueError("SQLite raw posting has no term frequency")
                norm = k1 * (1 - b + b * (lengths[doc_id] or 1) / average)
                yield (
                    doc_id,
                    idf * k1_plus_one * frequency / (frequency + norm),
                )
                previous = doc_id
                posting_index += 1
            if posting_index != doc_count:
                raise ValueError("SQLite raw posting is truncated")
            return

        weights = self.weights
        evidence = self.evidence
        lengths = self.lengths
        averages = self._averages
        field_count = len(weights)
        while position < end:
            delta, position = decode(payload, position, end)
            if posting_index and delta == 0:
                raise ValueError("SQLite raw posting document ids are not increasing")
            doc_id = delta if not posting_index else previous + delta
            if doc_id >= document_count:
                raise ValueError("SQLite raw posting references a missing document")
            tfw = 0.0
            for field_index in range(field_count):
                frequency, position = decode(payload, position, end)
                if not frequency:
                    continue
                norm = (
                    1.0
                    if evidence[field_index]
                    else 1
                    - b
                    + b
                    * (lengths[field_index][doc_id] or 1)
                    / averages[field_index]
                )
                tfw += weights[field_index] * frequency / norm
            if not tfw:
                raise ValueError("SQLite raw posting has no term frequency")
            yield doc_id, idf * tfw * k1_plus_one / (k1 + tfw)
            previous = doc_id
            posting_index += 1
        if posting_index != doc_count:
            raise ValueError("SQLite raw posting is truncated")

    def search(
        self,
        terms: list[str],
        *,
        anchors: frozenset[str],
        restricted: bool,
    ) -> list[tuple[int, float]]:
        ordered = list(dict.fromkeys(terms))
        if not ordered:
            return []
        rows: dict[str, tuple[int, int, int, int]] = {}
        for start in range(0, len(ordered), 500):
            batch = ordered[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            sql = f"""
                SELECT term, doc_count, block_id, byte_offset, byte_count
                  FROM {self.directory_table}
                 WHERE term IN ({placeholders})
            """
            for term, doc_count, block_id, byte_offset, byte_count in self.database.execute(
                sql, batch
            ):
                rows[term] = (doc_count, block_id, byte_offset, byte_count)
        payloads: dict[str, bytes] = {}
        by_block: dict[int, list[tuple[str, int, int]]] = {}
        for term, (_doc_count, block_id, byte_offset, byte_count) in rows.items():
            by_block.setdefault(block_id, []).append(
                (term, byte_offset - 1, byte_count)
            )
        for block_id, entries in by_block.items():
            with self.database.connection.blobopen(
                self.payload_table,
                "data",
                block_id,
                readonly=True,
            ) as blob:
                for term, byte_offset, byte_count in entries:
                    if byte_offset < 0 or byte_count < 0:
                        raise ValueError("SQLite raw posting directory is invalid")
                    blob.seek(byte_offset)
                    payload = blob.read(byte_count)
                    if len(payload) != byte_count:
                        raise ValueError("SQLite raw posting payload is truncated")
                    payloads[term] = payload
        multipliers: dict[str, int]
        if self.mode == "bm25":
            multipliers = {}
            for term in terms:
                multipliers[term] = multipliers.get(term, 0) + 1
        else:
            multipliers = dict.fromkeys(ordered, 1)
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        for term in ordered:
            row = rows.get(term)
            if row is None:
                continue
            doc_count, _block_id, _byte_offset, _byte_count = row
            payload = payloads[term]
            if (
                type(doc_count) is not int
                or not 1 <= doc_count <= self.document_count
                or not isinstance(payload, bytes)
            ):
                raise ValueError("SQLite raw posting directory is invalid")
            idf = math.log(
                1
                + (self.document_count - doc_count + 0.5)
                / (doc_count + 0.5)
            ) ** self.idf_pow
            multiplier = multipliers[term]
            for doc_id, weight in self._posting_scores(payload, doc_count, idf):
                scores[doc_id] = scores.get(doc_id, 0.0) + multiplier * weight
                if restricted and term in anchors:
                    candidates.add(doc_id)
                if term in anchors and term.startswith("#"):
                    complete_candidates.add(doc_id)
        scores = _coordinate_query_scores(
            scores, candidates, complete_candidates
        )
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _pack_text(value: str) -> bytes:
    raw = value.encode("utf-8")
    compressed = zlib.compress(raw, level=zlib.Z_BEST_SPEED)
    if len(compressed) < len(raw):
        return b"\x01" + compressed
    return b"\x00" + raw


def _unpack_text(value: object) -> str:
    if not isinstance(value, bytes) or not value:
        raise ValueError("SQLite snapshot has invalid packed text")
    codec, payload = value[0], value[1:]
    if codec == 1:
        try:
            payload = zlib.decompress(payload)
        except zlib.error as exc:
            raise ValueError("SQLite snapshot has invalid compressed text") from exc
    elif codec != 0:
        raise ValueError("SQLite snapshot has an unknown text codec")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SQLite snapshot text is not valid UTF-8") from exc


def _decode_chunk(row: tuple, *, packed_text: bool) -> Chunk:
    _slot, chunk_id, title, text, entities_json, meta_json = row
    if packed_text:
        title = _unpack_text(title)
        text = _unpack_text(text)
    elif not isinstance(title, str) or not isinstance(text, str):
        raise ValueError("legacy SQLite snapshot has invalid chunk text")
    entities = [] if entities_json is None else json.loads(entities_json)
    meta = {} if meta_json is None else json.loads(meta_json)
    if not isinstance(entities, list) or not all(
        isinstance(value, str) for value in entities
    ):
        raise ValueError("SQLite snapshot chunk entities are invalid")
    if not isinstance(meta, dict):
        raise ValueError("SQLite snapshot chunk metadata is invalid")
    return Chunk(chunk_id, text, entities=entities, meta=meta, title=title)


class SQLiteSnapshotVector:
    """Static lexical vector store that reads only query-term postings from disk."""

    def __init__(self, database: _SnapshotDatabase):
        self._database = database
        self._mode = database.meta["vector_mode"]
        self._raw = (
            _RawSQLiteLexical(database, "vector")
            if self._mode.startswith("raw_")
            else None
        )

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        if limit <= 0 or self._mode == "none":
            return []
        with self._database.lock:
            analysis = _analyze_query(query)
            ranked = (
                self._raw.search(
                    analysis.terms,
                    anchors=analysis.anchors,
                    restricted=analysis.restricted,
                )
                if self._raw is not None
                else _query_postings(
                    self._database,
                    "vector_terms",
                    "vector_payload",
                    analysis.terms,
                    repeated_terms=self._mode == "bm25",
                    anchors=analysis.anchors,
                    restricted=analysis.restricted,
                )
            )[:limit]
            if not ranked:
                return []
            slots = [slot for slot, _score in ranked]
            placeholders = ",".join("?" for _ in slots)
            found = {
                row[0]: _decode_chunk(
                    row, packed_text=self._database.schema_version >= 3
                )
                for row in self._database.execute(
                    f"""SELECT slot, chunk_id, title, text, entities_json, meta_json
                          FROM chunks WHERE slot IN ({placeholders})""",
                    slots,
                )
            }
            if set(found) != set(slots):
                raise ValueError("SQLite snapshot ranking references a missing chunk")
            return [(found[slot], score) for slot, score in ranked]

    def fetch(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        with self._database.lock:
            found: dict[str, Chunk] = {}
            for start in range(0, len(ids), 500):
                batch = ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                for row in self._database.execute(
                    f"""SELECT slot, chunk_id, title, text, entities_json, meta_json
                          FROM chunks WHERE chunk_id IN ({placeholders})""",
                    batch,
                ):
                    chunk = _decode_chunk(
                        row, packed_text=self._database.schema_version >= 3
                    )
                    found[chunk.id] = chunk
            return [found[chunk_id] for chunk_id in ids if chunk_id in found]

    def close(self) -> None:
        self._database.close()


class SQLiteSnapshotGraph:
    """Read-only graph operations over the same immutable snapshot."""

    def __init__(self, database: _SnapshotDatabase):
        self._database = database
        self._mode = database.meta["graph_mode"]
        self._raw = (
            _RawSQLiteLexical(database, "graph")
            if self._mode.startswith("raw_")
            else None
        )

    @staticmethod
    def _node(row: tuple | None) -> Node | None:
        if row is None:
            return None
        return Node(row[0], row[1], row[2])

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        if limit <= 0 or self._mode == "none":
            return []
        with self._database.lock:
            analysis = _analyze_query(query)
            ranked = (
                self._raw.search(
                    analysis.terms,
                    anchors=analysis.anchors,
                    restricted=analysis.restricted,
                )
                if self._raw is not None
                else _query_postings(
                    self._database,
                    "graph_terms",
                    "graph_payload",
                    analysis.terms,
                    repeated_terms=True,
                    anchors=analysis.anchors,
                    restricted=analysis.restricted,
                )
            )[:limit]
            if not ranked:
                return []
            slots = [slot for slot, _score in ranked]
            placeholders = ",".join("?" for _ in slots)
            found = {
                row[0]: Node(row[1], row[2], row[3])
                for row in self._database.execute(
                    f"""SELECT slot, node_id, label, kind FROM graph_nodes
                          WHERE slot IN ({placeholders})""",
                    slots,
                )
            }
            return [(found[slot], score) for slot, score in ranked if slot in found]

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        placeholders = ",".join("?" for _ in _ISA)
        parameters = [class_id, *_ISA, limit]
        with self._database.lock:
            rows = self._database.execute(
                f"""SELECT n.node_id, n.label, n.kind
                      FROM graph_edges AS e JOIN graph_nodes AS n
                        ON n.node_id = e.subject_id
                     WHERE e.object_id = ? AND e.predicate IN ({placeholders})
                     ORDER BY e.position LIMIT ?""",
                parameters,
            ).fetchall()
            return [self._node(row) for row in rows]

    def neighbor_ids(
        self, node_id: str, *, limit: int = 100, direction: str = "both"
    ) -> list[str]:
        if direction not in {"out", "in", "both"}:
            raise ValueError(
                f"direction must be 'out', 'in' or 'both' (got {direction!r})"
            )
        if direction == "out":
            sql, parameters = (
                "SELECT object_id FROM graph_edges WHERE subject_id = ? ORDER BY position",
                (node_id,),
            )
        elif direction == "in":
            sql, parameters = (
                "SELECT subject_id FROM graph_edges WHERE object_id = ? ORDER BY position",
                (node_id,),
            )
        else:
            sql, parameters = (
                """SELECT CASE WHEN subject_id = ? THEN object_id ELSE subject_id END
                     FROM graph_edges WHERE subject_id = ? OR object_id = ?
                    ORDER BY position""",
                (node_id, node_id, node_id),
            )
        with self._database.lock:
            out: list[str] = []
            seen = {node_id}
            for (other,) in self._database.execute(sql, parameters):
                if other not in seen:
                    seen.add(other)
                    out.append(other)
                    if len(out) >= limit:
                        break
            return out

    def neighbors(
        self, node_id: str, *, hops: int = 1, limit: int = 100
    ) -> list[tuple[str, str, str]]:
        with self._database.lock:
            labels = {
                row[0]: row[1]
                for row in self._database.execute("SELECT node_id, label FROM graph_nodes")
            }
            out: list[tuple[str, str, str]] = []
            seen_nodes = {node_id}
            frontier = [node_id]
            for _depth in range(max(1, hops)):
                next_frontier: list[str] = []
                for current in frontier:
                    rows = self._database.execute(
                        """SELECT subject_id, predicate, object_id FROM graph_edges
                            WHERE subject_id = ? OR object_id = ? ORDER BY position""",
                        (current, current),
                    ).fetchall()
                    for subject, predicate, obj in rows:
                        triple = (
                            labels.get(subject, subject),
                            predicate,
                            labels.get(obj, obj),
                        )
                        if triple not in out:
                            out.append(triple)
                            if len(out) >= limit:
                                return out
                        other = obj if subject == current else subject
                        if other not in seen_nodes:
                            seen_nodes.add(other)
                            next_frontier.append(other)
                frontier = next_frontier
                if not frontier:
                    break
            return out

    def count_class(self, class_id: str) -> int:
        placeholders = ",".join("?" for _ in _ISA)
        with self._database.lock:
            row = self._database.execute(
                f"""SELECT COUNT(*) FROM graph_edges
                      WHERE object_id = ? AND predicate IN ({placeholders})""",
                [class_id, *_ISA],
            ).fetchone()
            return int(row[0])

    def get_node(self, node_id: str) -> Node | None:
        with self._database.lock:
            return self._node(
                self._database.execute(
                    "SELECT node_id, label, kind FROM graph_nodes WHERE node_id = ?",
                    (node_id,),
                ).fetchone()
            )

    def close(self) -> None:
        self._database.close()


def _write_snapshot(connection: sqlite3.Connection, of, *, consume: bool) -> None:
    vector = of.vector
    graph = of.graph
    vector._prepare_for_persistence()
    vector_index = vector._bm25
    graph_index = graph._bm25 if graph.nodes else None

    metadata = {
        "schema": _SCHEMA,
        "schema_version": str(_SCHEMA_VERSION),
        "text_codec": "raw-or-zlib-v1",
        "byteorder": "little",
        "vector_mode": _lexical_mode(vector_index),
        "graph_mode": _lexical_mode(graph_index),
        "chunk_count": str(len(vector.chunks)),
        "graph_node_count": str(len(graph.nodes)),
        "graph_edge_count": str(len(graph.triples)),
    }
    connection.executemany("INSERT INTO snapshot_meta VALUES (?, ?)", metadata.items())
    connection.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        (
            (
                slot,
                chunk.id,
                _pack_text(chunk.title),
                _pack_text(chunk.text),
                _json(chunk.entities) if chunk.entities else None,
                _json(chunk.meta) if chunk.meta else None,
            )
            for slot, chunk in enumerate(vector.chunks)
        ),
    )
    _insert_snapshot_postings(
        connection,
        "vector",
        "vector_terms",
        "vector_payload",
        vector_index,
        consume=consume,
    )
    connection.executemany(
        "INSERT INTO graph_nodes VALUES (?, ?, ?, ?)",
        (
            (slot, node.id, node.label, node.kind)
            for slot, node in enumerate(graph.nodes.values())
        ),
    )
    _insert_snapshot_postings(
        connection,
        "graph",
        "graph_terms",
        "graph_payload",
        graph_index,
        consume=consume,
    )
    connection.executemany(
        "INSERT INTO graph_edges VALUES (?, ?, ?, ?)",
        (
            (position, triple.s, triple.p, triple.o)
            for position, triple in enumerate(graph.triples)
        ),
    )


def _direct_vector_config(vector_kwargs) -> tuple[float, float]:
    options = dict(vector_kwargs or {})
    reserved = options.keys() & {"embedder", "feedback"}
    if reserved:
        names = ", ".join(sorted(reserved))
        raise TypeError(f"build_sqlite_index does not accept {names} in vector_kwargs")
    allowed = {
        "title_weight",
        "lexical_weight",
        "dense_weight",
        "pool",
        "idf_pow",
    }
    unknown = options.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unexpected vector_kwargs: {names}")
    title_weight = options.get("title_weight", DEFAULT_TITLE_WEIGHT)
    idf_pow = options.get("idf_pow", DEFAULT_IDF_POW)
    if (
        not isinstance(title_weight, (int, float))
        or not math.isfinite(title_weight)
        or not isinstance(idf_pow, (int, float))
        or not math.isfinite(idf_pow)
        or idf_pow <= 0
    ):
        raise ValueError(
            "SQLite title_weight must be finite and idf_pow must be finite and positive"
        )
    return float(title_weight), float(idf_pow)


def _write_direct_snapshot(
    connection: sqlite3.Connection,
    nodes,
    triples,
    chunks,
    *,
    vector_kwargs,
) -> None:
    title_weight, idf_pow = _direct_vector_config(vector_kwargs)
    chunk_count = 0
    has_title = False
    has_lexical = False
    lexical_titles: list[str] = []
    lexical_texts: list[str] = []

    def chunk_rows():
        nonlocal chunk_count, has_title, has_lexical
        for slot, raw_chunk in enumerate(chunks):
            chunk = to_chunk(raw_chunk)
            if chunk.embedding:
                raise TypeError("build_sqlite_index does not support dense embeddings")
            chunk_count = slot + 1
            has_title = has_title or bool(chunk.title)
            has_lexical = has_lexical or bool(chunk.title or chunk.text)
            lexical_titles.append(chunk.title)
            lexical_texts.append(chunk.text)
            yield (
                slot,
                chunk.id,
                _pack_text(chunk.title),
                _pack_text(chunk.text),
                _json(chunk.entities) if chunk.entities else None,
                _json(chunk.meta) if chunk.meta else None,
            )

    connection.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        chunk_rows(),
    )

    vector_index: _RawPostingsSnapshot | None = None
    if has_lexical:
        if has_title:
            documents = (
                (
                    slot,
                    {"title": tokenize(title), "body": tokenize(text)},
                )
                for slot, (title, text) in enumerate(
                    zip(lexical_titles, lexical_texts, strict=True)
                )
            )
            vector_index = CompactPostingsSnapshot._raw_bm25f_for_disk(
                documents,
                {"title": title_weight, "body": 1.0},
                idf_pow=idf_pow,
            )
        else:
            documents = (
                (slot, tokenize(text))
                for slot, text in enumerate(lexical_texts)
            )
            vector_index = CompactPostingsSnapshot._raw_bm25_for_disk(
                documents,
                idf_pow=idf_pow,
            )
    lexical_titles.clear()
    lexical_texts.clear()
    _insert_raw_postings(
        connection,
        "vector",
        "vector_terms",
        "vector_payload",
        vector_index,
    )

    normalized_nodes = {}
    for raw_node in nodes:
        node = to_node(raw_node)
        normalized_nodes[node.id] = node
    connection.executemany(
        "INSERT INTO graph_nodes VALUES (?, ?, ?, ?)",
        (
            (slot, node.id, node.label, node.kind)
            for slot, node in enumerate(normalized_nodes.values())
        ),
    )
    graph_index = (
        CompactPostingsSnapshot._raw_bm25_for_disk(
            (
                (slot, tokenize(node.label))
                for slot, node in enumerate(normalized_nodes.values())
            )
        )
        if normalized_nodes
        else None
    )
    _insert_raw_postings(
        connection,
        "graph",
        "graph_terms",
        "graph_payload",
        graph_index,
    )

    graph_edge_count = 0

    def edge_rows():
        nonlocal graph_edge_count
        for position, raw_triple in enumerate(triples):
            triple = to_triple(raw_triple)
            graph_edge_count = position + 1
            yield position, triple.s, triple.p, triple.o

    connection.executemany(
        "INSERT INTO graph_edges VALUES (?, ?, ?, ?)",
        edge_rows(),
    )
    metadata = {
        "schema": _SCHEMA,
        "schema_version": str(_SCHEMA_VERSION),
        "text_codec": "raw-or-zlib-v1",
        "byteorder": "little",
        "vector_mode": "none" if vector_index is None else f"raw_{vector_index.mode}",
        "graph_mode": "none" if graph_index is None else f"raw_{graph_index.mode}",
        "chunk_count": str(chunk_count),
        "graph_node_count": str(len(normalized_nodes)),
        "graph_edge_count": str(graph_edge_count),
    }
    connection.executemany("INSERT INTO snapshot_meta VALUES (?, ?)", metadata.items())


def _atomic_sqlite_snapshot(
    path, writer: Callable[[sqlite3.Connection], None]
) -> None:

    target = Path(path)
    temporary: Path | None = None
    connection: sqlite3.Connection | None = None
    try:
        with NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute("PRAGMA cache_size=-256")
        connection.executescript(_SQL)
        writer(connection)
        connection.commit()
        connection.close()
        connection = None
        with open(temporary, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(temporary, target)
    except BaseException:
        if connection is not None:
            with suppress(Exception):
                connection.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _save_sqlite_index(of, path, *, consume: bool) -> None:
    from .memory import InMemoryGraph, InMemoryVector, MutableInMemoryVector

    if not isinstance(of.graph, InMemoryGraph) or not isinstance(
        of.vector, InMemoryVector
    ):
        raise TypeError("save_sqlite_index supports the in-memory backends only")
    if isinstance(of.vector, MutableInMemoryVector):
        raise TypeError("save_sqlite_index requires an immutable vector store")
    if of.vector._dense or any(chunk.embedding for chunk in of.vector.chunks):
        raise TypeError("save_sqlite_index does not support dense embeddings")
    _atomic_sqlite_snapshot(
        path,
        lambda connection: _write_snapshot(connection, of, consume=consume),
    )


def save_sqlite_index(of, path) -> None:
    """Atomically persist a static lexical index as a disk-queryable SQLite file."""
    _save_sqlite_index(of, path, consume=False)


def build_sqlite_index(nodes, triples, chunks, path, *, vector_kwargs=None) -> None:
    """Build and atomically publish a static SQLite index without retaining a RAM copy."""
    _atomic_sqlite_snapshot(
        path,
        lambda connection: _write_direct_snapshot(
            connection,
            nodes,
            triples,
            chunks,
            vector_kwargs=vector_kwargs,
        ),
    )


def open_sqlite_index(path, *, llm=None, **kwargs):
    """Open a read-only SQLite snapshot without loading its corpus into RAM."""
    database = _SnapshotDatabase(Path(path))
    try:
        graph = SQLiteSnapshotGraph(database)
        vector = SQLiteSnapshotVector(database)
        return OmniFuse(graph, vector, llm or EchoLLM(), **kwargs)
    except BaseException:
        database.close()
        raise


__all__ = [
    "SQLiteSnapshotGraph",
    "SQLiteSnapshotVector",
    "build_sqlite_index",
    "open_sqlite_index",
    "save_sqlite_index",
]
