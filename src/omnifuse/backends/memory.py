"""Zero-infra in-memory backends — dict + BM25, pure Python (no DB, no numpy).

These make ``pip install xgen-omnifuse`` run the full algorithm with zero
infrastructure. For scale, swap in Fuseki/Qdrant adapters that match the same
protocols (see omnifuse.protocols).
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable
from copy import deepcopy
import hashlib
import hmac
from io import BytesIO
import math
import pickle
import pickletools
from threading import RLock
from typing import Callable, NamedTuple, Optional

from ..feedback import Feedback
from ..lexical_rerank import (
    rank_korean_character_fallback,
    rerank_lexical_candidates,
)
from ..models import Chunk, ChunkMutationResult, Node, Triple
from ..settings import DEFAULT_LEXICAL_B, DEFAULT_LEXICAL_K1, DEFAULT_TITLE_WEIGHT
from .._compact_mutable import CompactMutableBM25
from .._compact_mutable_fielded import CompactMutableBM25F
from .._compact_postings import CompactPostingsSnapshot
from ..text import (
    _IDF_POW,
    BM25,
    BM25F,
    _MutableBM25,
    _MutableBM25F,
    tokenize,
)

_ISA = {"instanceOf", "type", "subClassOf", "rdf:type"}
# Title weighted above body in fielded lexical retrieval — a short heading is
# a far stronger relevance signal per token than the passage it heads.
_TITLE_WEIGHT = DEFAULT_TITLE_WEIGHT
_LEXICAL_K1 = DEFAULT_LEXICAL_K1
_LEXICAL_B = DEFAULT_LEXICAL_B
_MUTABLE_VECTOR_STATE_VERSION = 5
_VECTOR_PAYLOAD_VERSION = 1
_VECTOR_PAYLOAD_DOMAIN = b"omnifuse.mutable-vector.bound-payload.v1\x00"
_VECTOR_PAYLOAD_PROTOCOL = 5
_VECTOR_PAYLOAD_PREFIX = b"\x80" + bytes((_VECTOR_PAYLOAD_PROTOCOL,))
_STATIC_FORWARD_SNAPSHOT_KIND = "compact-postings-packed-forward-v2"


def _pack_static_lexical(index: CompactPostingsSnapshot) -> dict:
    return {
        "kind": _STATIC_FORWARD_SNAPSHOT_KIND,
        "state": index._export_packed_forward_state(),
    }


def _unpack_static_lexical(value: object) -> object:
    if type(value) is not dict or value.get("kind") != _STATIC_FORWARD_SNAPSHOT_KIND:
        return value
    if set(value) != {"kind", "state"}:
        raise ValueError("invalid static lexical snapshot envelope")
    restored, captured = CompactPostingsSnapshot._from_packed_forward_state(
        value["state"]
    )
    if captured:
        raise ValueError("static lexical snapshot restored unexpected document records")
    return restored


_LEGACY_MUTABLE_VECTOR_KEYS = frozenset(
    {
        "embedder",
        "lexical_weight",
        "dense_weight",
        "_pool",
        "_title_weight",
        "_idf_pow",
        "_feedback",
        "_revision",
        "_next_slot",
        "_chunks",
        "_slot_by_id",
        "_title_count",
        "_lexical_count",
        "_embedding_count",
        "_fielded",
        "_bm25",
        "_lexical",
        "_dense",
    }
)


def _is_finite_builtin_number(value: object) -> bool:
    value_type = type(value)
    if value_type is not int and value_type is not float:
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _validate_vector_scoring_config(
    title_weight: object,
    lexical_weight: object,
    dense_weight: object,
    pool: object,
    idf_pow: object,
) -> None:
    if (
        any(
            not _is_finite_builtin_number(value)
            for value in (title_weight, lexical_weight, dense_weight, idf_pow)
        )
        or type(pool) is not int
    ):
        raise ValueError("mutable vector scoring configuration is invalid")


def _vector_payload_sha256(payload: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(_VECTOR_PAYLOAD_DOMAIN)
    digest.update(payload)
    return digest.digest()


def _decode_vector_payload(payload: bytes) -> object:
    try:
        stop_position = None
        for opcode, _argument, position in pickletools.genops(payload):
            if opcode.name == "STOP":
                stop_position = position
        if stop_position != len(payload) - 1:
            raise ValueError("mutable vector payload is not one complete pickle")
    except (IndexError, OverflowError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("mutable vector payload pickle is invalid") from exc
    stream = BytesIO(payload)
    try:
        decoded = pickle.Unpickler(stream).load()
    except (
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        OverflowError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError("mutable vector payload pickle is invalid") from exc
    if stream.tell() != len(payload):
        raise ValueError("mutable vector payload pickle has trailing bytes")
    return decoded


class InMemoryGraph:
    """Triples + node labels, indexed for BM25 label search and 1-hop traversal."""

    def __init__(self, nodes: list[Node], triples: list[Triple]):
        self.nodes: dict[str, Node] = {n.id: n for n in nodes}
        self.triples = triples
        # adjacency: node_id -> list of (subj_label, predicate, obj_label)
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        # class_id -> [instance node ids]   (via instanceOf/type/subClassOf)
        self._members: dict[str, list[str]] = {}
        # node_id -> [neighbor node ids]   (for retrieval-time graph fusion)
        self._adj_ids: dict[str, list[str]] = {}
        # direction matters: an edge (s -references-> o) means s *cites* o. Fusion wants
        # what a seed points AT, not the crowd of nodes pointing at it.
        self._out_ids: dict[str, list[str]] = {}
        self._in_ids: dict[str, list[str]] = {}
        for t in triples:
            sl = self._label(t.s)
            ol = self._label(t.o)
            self._adj.setdefault(t.s, []).append((sl, t.p, ol))
            self._adj.setdefault(t.o, []).append((sl, t.p, ol))
            self._adj_ids.setdefault(t.s, []).append(t.o)
            self._adj_ids.setdefault(t.o, []).append(t.s)
            self._out_ids.setdefault(t.s, []).append(t.o)
            self._in_ids.setdefault(t.o, []).append(t.s)
            if t.p in _ISA:
                self._members.setdefault(t.o, []).append(t.s)
        self._ids = list(self.nodes.keys())
        self._bm25 = CompactPostingsSnapshot._from_bm25_for_vector(
            (position, tokenize(self.nodes[node_id].label))
            for position, node_id in enumerate(self._ids)
        )
        # label -> first node id with that label (multi-hop traversal lookup)
        self._label_ix: dict[str, str] = {}
        for nid, n in self.nodes.items():
            self._label_ix.setdefault(n.label, nid)

    def _label(self, nid: str) -> str:
        n = self.nodes.get(nid)
        return n.label if n else nid

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        return [
            (self.nodes[self._ids[i]], s)
            for i, s in self._bm25.search(query, limit=limit)
        ]

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        ids = self._members.get(class_id, [])
        return [self.nodes[i] for i in ids[:limit] if i in self.nodes]

    def neighbor_ids(
        self, node_id: str, *, limit: int = 100, direction: str = "both"
    ) -> list[str]:
        """Distinct neighbor node ids of ``node_id`` (for retrieval-time fusion).

        ``direction`` selects edge orientation: ``"out"`` (nodes this one points at,
        e.g. the articles it cites), ``"in"`` (nodes pointing at it), or ``"both"``.
        """
        src = {"out": self._out_ids, "in": self._in_ids, "both": self._adj_ids}.get(
            direction
        )
        if src is None:
            raise ValueError(
                f"direction must be 'out', 'in' or 'both' (got {direction!r})"
            )
        out: list[str] = []
        seen = {node_id}
        for other in src.get(node_id, ()):
            if other not in seen:
                seen.add(other)
                out.append(other)
                if len(out) >= limit:
                    break
        return out

    def neighbors(
        self, node_id: str, *, hops: int = 1, limit: int = 100
    ) -> list[tuple[str, str, str]]:
        out = list(self._adj.get(node_id, []))[:limit]
        if hops > 1:
            seen = {node_id}
            frontier = {t[2] for t in out} | {t[0] for t in out}
            for _ in range(hops - 1):
                nxt: set[str] = set()
                for lbl in list(frontier):
                    nid = self._by_label(lbl)
                    if nid and nid not in seen:
                        seen.add(nid)
                        out.extend(self._adj.get(nid, [])[:limit])
                frontier = nxt
        return out[:limit]

    def count_class(self, class_id: str) -> int:
        return len(self._members.get(class_id, []))

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def _by_label(self, label: str) -> Optional[str]:
        return self._label_ix.get(label)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if isinstance(state.get("_bm25"), CompactPostingsSnapshot):
            state["_bm25"] = _pack_static_lexical(state["_bm25"])
        return state

    def __setstate__(self, state: dict) -> None:
        restored = dict(state)
        restored["_bm25"] = _unpack_static_lexical(restored.get("_bm25"))
        self.__dict__.update(restored)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _minmax(pairs: list[tuple[int, float]]) -> dict[int, float]:
    """Per-query [0,1] normalization so dense cosine and lexical BM25 (different
    scales) can be summed."""
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {i: (s - lo) / rng for i, s in pairs}


class InMemoryVector:
    """Passage store with three retrieval modes, chosen by what the chunks carry:

    - **hybrid** — embeddings *and* text present: dense cosine and lexical BM25(F)
      are min-max normalized per query and combined ``dense_weight*dense +
      lexical_weight*lexical`` (dense recovers paraphrase; lexical nails exact
      terms — each covers the other's blind spot).
    - **dense** — embeddings only: cosine.
    - **lexical** — text only (zero embeddings): field-weighted BM25 over
      title/body, else plain BM25.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embedder: Optional[Callable[[str], list[float]]] = None,
        title_weight: float = _TITLE_WEIGHT,
        lexical_weight: float = 0.8,
        dense_weight: float = 1.0,
        pool: int = 40,
        idf_pow: float = _IDF_POW,
        feedback: Optional[Feedback] = None,
    ):
        self.chunks = chunks
        self._ix: dict[str, int] | None = None
        self.embedder = embedder
        self.lexical_weight, self.dense_weight, self._pool = (
            lexical_weight,
            dense_weight,
            pool,
        )
        self._dense = (
            embedder is not None and bool(chunks) and all(c.embedding for c in chunks)
        )
        self._lexical = any((c.text or c.title) for c in chunks)
        # Own the feedback snapshot just like the mutable backend. This prevents a
        # caller-held Feedback from changing a deferred index behind the store's back.
        self.feedback = feedback.copy() if feedback is not None else None
        self._title_weight = title_weight
        self._idf_pow = idf_pow
        self._bm25: BM25 | BM25F | CompactPostingsSnapshot | None = None
        self._lexical_source = (
            tuple((chunk.title, chunk.text) for chunk in chunks)
            if self._lexical
            else None
        )
        self._lock = RLock()
        # All lexical modes materialize on first use. The scalar chunk source and owned
        # feedback snapshot preserve construction-time semantics until then.

    def _build_lexical(self) -> BM25F | CompactPostingsSnapshot:
        # Tokenization streams into the compact immutable index rather than retaining
        # precomputed Python score arrays for every term. Feedback remains on BM25F
        # because its evidence statistics are incrementally mutable.
        source = self._lexical_source
        if source is None:
            source = tuple((chunk.title, chunk.text) for chunk in self.chunks)
        if self.feedback is not None:

            def docs():
                return (
                    {
                        "title": tokenize(title),
                        "body": tokenize(text),
                        "memory": tokenize(self.feedback.text(chunk.id)),
                    }
                    for chunk, (title, text) in zip(self.chunks, source, strict=True)
                )

            return BM25F(
                docs,
                {"title": self._title_weight, "body": 1.0, "memory": 1.0},
                idf_pow=self._idf_pow,
                evidence_fields={"memory"},
            )
        if any(title for title, _text in source):

            def docs():
                return (
                    {"title": tokenize(title), "body": tokenize(text)}
                    for title, text in source
                )

            return CompactPostingsSnapshot._from_bm25f_for_vector(
                enumerate(docs()),
                {"title": self._title_weight, "body": 1.0},
                idf_pow=self._idf_pow,
            )

        def texts():
            return (tokenize(text) for _title, text in source)

        return CompactPostingsSnapshot._from_bm25_for_vector(
            enumerate(texts()), idf_pow=self._idf_pow
        )

    def _lexical_index(self) -> BM25F | CompactPostingsSnapshot:
        index = self._bm25
        if index is not None:
            return index
        with self._lock:
            if self._bm25 is None:
                if not self._lexical:
                    raise RuntimeError(
                        "lexical index requested for a non-lexical store"
                    )
                self._bm25 = self._build_lexical()
                self._lexical_source = None
            return self._bm25

    def _id_index(self) -> dict[str, int]:
        index = self._ix
        if index is not None:
            return index
        with self._lock:
            if self._ix is None:
                self._ix = {
                    chunk.id: position for position, chunk in enumerate(self.chunks)
                }
            return self._ix

    def _prepare_for_persistence(self) -> None:
        """Materialize deferred lexical state before writing a warm snapshot."""
        if self._lexical:
            self._lexical_index()

    def _dense_ranked(self, query: str, limit: int) -> list[tuple[int, float]]:
        q = self.embedder(query)  # type: ignore[misc]
        scored = [
            (i, _cosine(q, c.embedding))
            for i, c in enumerate(self.chunks)
            if c.embedding
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        if self._dense and self._lexical:
            pool = max(limit, self._pool)
            dn = _minmax(self._dense_ranked(query, pool))
            lexical = rerank_lexical_candidates(
                query,
                self._lexical_index().search(
                    query, limit=pool, recover_partial_outlier=True
                ),
                self.chunks.__getitem__,
            )
            ln = _minmax(lexical)
            fused = {
                i: self.dense_weight * dn.get(i, 0.0)
                + self.lexical_weight * ln.get(i, 0.0)
                for i in set(dn) | set(ln)
            }
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
            return [(self.chunks[i], s) for i, s in ranked]
        if self._dense:
            return [(self.chunks[i], s) for i, s in self._dense_ranked(query, limit)]
        if self._lexical:
            pool = max(limit, self._pool)
            candidates = self._lexical_index().search(
                query, limit=pool, recover_partial_outlier=True
            )
            if not candidates:
                candidates = rank_korean_character_fallback(
                    query, self.chunks, limit=pool
                )
            ranked = rerank_lexical_candidates(
                query,
                candidates,
                self.chunks.__getitem__,
            )
            return [(self.chunks[i], s) for i, s in ranked[:limit]]
        return []

    def remember(self, query: str, doc_ids: list[str]) -> None:
        """Fold a confirmed (query -> documents) pair into the live index, incrementally.

        Requires the store to have been built with a ``Feedback`` (an empty one is fine —
        an empty evidence field scores bit-identically to no field at all). Only the
        remembering chunks move: evidence never enters document frequency, so N and every
        content term's IDF are fixed. Cost is bounded by the memory, not the corpus.
        """
        index = self._lexical_index() if self.feedback is not None else None
        if not isinstance(index, BM25F) or not index.evidence_fields:
            raise RuntimeError(
                "this store cannot remember incrementally — build it with feedback=Feedback()"
            )
        id_index = self._id_index()
        for did in doc_ids:
            i = id_index.get(did)
            if i is None:
                continue
            c = self.chunks[i]
            content = {"title": tokenize(c.title), "body": tokenize(c.text)}
            before = dict(content, memory=tokenize(self.feedback.text(did)))
            self.feedback.remember(query, [did])
            after = dict(content, memory=tokenize(self.feedback.text(did)))
            index.update_evidence(i, before, after)

    def forget(self, query: str, doc_ids: list[str]) -> None:
        """Withdraw a remembered pair from the live index — the exact inverse of
        ``remember``. The updated index is bit-identical to one rebuilt without the pair;
        forgetting a pair that was never remembered is a no-op."""
        index = self._lexical_index() if self.feedback is not None else None
        if not isinstance(index, BM25F) or not index.evidence_fields:
            raise RuntimeError(
                "this store cannot forget incrementally — build it with feedback=Feedback()"
            )
        id_index = self._id_index()
        for did in doc_ids:
            i = id_index.get(did)
            if i is None:
                continue
            c = self.chunks[i]
            content = {"title": tokenize(c.title), "body": tokenize(c.text)}
            before = dict(content, memory=tokenize(self.feedback.text(did)))
            self.feedback.forget(query, [did])
            after = dict(content, memory=tokenize(self.feedback.text(did)))
            if before != after:
                index.update_evidence(i, before, after)

    def fetch(self, ids: list[str]) -> list[Chunk]:
        id_index = self._id_index()
        return [
            self.chunks[index]
            for chunk_id in ids
            if (index := id_index.get(chunk_id)) is not None
        ]

    def attach_embedder(self, embedder: Optional[Callable[[str], list[float]]]) -> None:
        """(Re)bind the query embedder — e.g. after loading a persisted index, where the
        callable could not be serialized. Re-derives whether dense retrieval is usable."""
        self.embedder = embedder
        self._dense = (
            embedder is not None
            and bool(self.chunks)
            and all(c.embedding for c in self.chunks)
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state["embedder"] = None  # a model/closure is not portable; re-attach on load
        state["_dense"] = False
        index = state.get("_bm25")
        if isinstance(index, CompactPostingsSnapshot):
            state["_bm25"] = _pack_static_lexical(index)
        return state

    def __setstate__(self, state: dict) -> None:
        restored = dict(state)
        restored.pop("_by_id", None)
        restored.setdefault("_ix", None)
        restored["_bm25"] = _unpack_static_lexical(restored.get("_bm25"))
        restored.setdefault("_bm25", None)
        restored.setdefault("_title_weight", _TITLE_WEIGHT)
        restored.setdefault("_idf_pow", _IDF_POW)
        if "_lexical_source" not in restored:
            restored["_lexical_source"] = (
                tuple((chunk.title, chunk.text) for chunk in restored["chunks"])
                if restored.get("_lexical") and restored["_bm25"] is None
                else None
            )
        self.__dict__.update(restored)
        self._lock = RLock()


class _EmptyMeta:
    __slots__ = ()

    def __reduce__(self):
        return _restore_empty_meta, ()


def _restore_empty_meta():
    return _EMPTY_META


_EMPTY_META = _EmptyMeta()


class _ValidatedChunk(NamedTuple):
    id: str
    text: str
    entities: tuple[str, ...]
    embedding: tuple[float, ...] | None
    meta: object
    title: str


class _PackedChunk(NamedTuple):
    id: str
    text: str
    entities: tuple[str, ...]
    embedding: tuple[float, ...] | None
    meta: object
    title: str

    @staticmethod
    def _validate_scalar_fields(
        chunk_id: object, text: object, title: object, meta: object
    ) -> None:
        if type(chunk_id) is not str or not chunk_id:
            raise ValueError("mutable chunks require an exact non-empty string id")
        if type(text) is not str or type(title) is not str:
            raise TypeError("chunk text and title must be exact strings")
        if not isinstance(meta, _EmptyMeta) and type(meta) is not dict:
            raise TypeError("chunk metadata must be a dict")

    @staticmethod
    def _validated_entity(entity: object) -> str:
        if type(entity) is not str:
            raise TypeError("chunk entities must contain exact strings")
        return entity

    @staticmethod
    def _validated_embedding_value(value: object) -> int | float:
        value_type = type(value)
        if value_type is not int and value_type is not float:
            raise TypeError("chunk embedding must contain exact int or float values")
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError):
            finite = False
        if not finite:
            raise ValueError("chunk embedding values must be finite")
        return value

    @classmethod
    def _prepare_entities(cls, entities: object) -> tuple[str, ...]:
        if type(entities) is tuple:
            for entity in entities:
                cls._validated_entity(entity)
            return entities
        if type(entities) is list and not entities:
            return ()
        try:
            iterator = iter(entities)
        except TypeError as exc:
            raise TypeError("chunk entities must be iterable") from exc
        return tuple(map(cls._validated_entity, iterator))

    @classmethod
    def _prepare_embedding(cls, embedding: object) -> tuple[float, ...]:
        if type(embedding) is tuple:
            for value in embedding:
                cls._validated_embedding_value(value)
            return embedding
        if type(embedding) is list and not embedding:
            return ()
        try:
            iterator = iter(embedding)
        except TypeError as exc:
            raise TypeError("chunk embedding must be iterable or None") from exc
        return tuple(map(cls._validated_embedding_value, iterator))

    @classmethod
    def prepare_ingress(cls, chunk: Chunk) -> _ValidatedChunk:
        chunk_id = chunk.id
        text = chunk.text
        entities_source = chunk.entities
        embedding_source = chunk.embedding
        meta = chunk.meta
        title = chunk.title
        cls._validate_scalar_fields(chunk_id, text, title, meta)
        entities = cls._prepare_entities(entities_source)
        if embedding_source is None:
            embedding = None
        else:
            embedding = cls._prepare_embedding(embedding_source)
        return _ValidatedChunk(
            id=chunk_id,
            text=text,
            entities=entities,
            embedding=embedding,
            meta=meta,
            title=title,
        )

    @classmethod
    def prepare_batch_ingress(cls, chunk: Chunk) -> "_PackedChunk | _ValidatedChunk":
        chunk_id = chunk.id
        text = chunk.text
        entities_source = chunk.entities
        embedding_source = chunk.embedding
        meta = chunk.meta
        title = chunk.title
        if (
            type(chunk_id) is str
            and chunk_id
            and type(text) is str
            and type(title) is str
            and type(entities_source) is list
            and not entities_source
            and embedding_source is None
            and type(meta) is dict
            and not meta
        ):
            return cls(chunk_id, text, (), None, _EMPTY_META, title)

        cls._validate_scalar_fields(chunk_id, text, title, meta)
        entities = cls._prepare_entities(entities_source)
        if embedding_source is None:
            embedding = None
        else:
            embedding = cls._prepare_embedding(embedding_source)
        if isinstance(meta, _EmptyMeta) or (type(meta) is dict and not meta):
            return cls(chunk_id, text, entities, embedding, _EMPTY_META, title)
        return _ValidatedChunk(
            id=chunk_id,
            text=text,
            entities=entities,
            embedding=embedding,
            meta=meta,
            title=title,
        )

    @classmethod
    def from_ingress(cls, chunk: _ValidatedChunk) -> "_PackedChunk":
        meta = (
            _EMPTY_META
            if isinstance(chunk.meta, _EmptyMeta)
            or (type(chunk.meta) is dict and not chunk.meta)
            else deepcopy(chunk.meta)
        )
        return cls(
            id=chunk.id,
            text=chunk.text,
            entities=chunk.entities,
            embedding=chunk.embedding,
            meta=meta,
            title=chunk.title,
        )

    @classmethod
    def from_batch_ingress(cls, chunk: _PreparedChunk) -> "_PackedChunk":
        if type(chunk) is cls:
            return chunk
        return cls.from_ingress(chunk)

    @classmethod
    def freeze(cls, chunk: Chunk) -> "_PackedChunk":
        chunk_id = chunk.id
        text = chunk.text
        entities_source = chunk.entities
        embedding_source = chunk.embedding
        meta_source = chunk.meta
        title = chunk.title
        if (
            type(chunk_id) is str
            and chunk_id
            and type(text) is str
            and type(title) is str
            and type(entities_source) is list
            and not entities_source
            and embedding_source is None
            and type(meta_source) is dict
            and not meta_source
        ):
            return cls(chunk_id, text, (), None, _EMPTY_META, title)
        cls._validate_scalar_fields(chunk_id, text, title, meta_source)
        entities = cls._prepare_entities(entities_source)
        if embedding_source is None:
            embedding = None
        else:
            embedding = cls._prepare_embedding(embedding_source)
        meta = (
            _EMPTY_META
            if isinstance(meta_source, _EmptyMeta)
            or (type(meta_source) is dict and not meta_source)
            else deepcopy(meta_source)
        )
        return cls(
            id=chunk_id,
            text=text,
            entities=entities,
            embedding=embedding,
            meta=meta,
            title=title,
        )

    def thaw(self) -> Chunk:
        return Chunk(
            id=self.id,
            text=self.text,
            entities=list(self.entities),
            embedding=list(self.embedding) if self.embedding is not None else None,
            meta={} if isinstance(self.meta, _EmptyMeta) else deepcopy(self.meta),
            title=self.title,
        )

    def _matches_ingress(self, chunk: _PreparedChunk) -> bool:
        stored_meta = {} if isinstance(self.meta, _EmptyMeta) else self.meta
        candidate_meta = {} if isinstance(chunk.meta, _EmptyMeta) else chunk.meta
        return (
            self.id == chunk.id
            and self.text == chunk.text
            and self.entities == chunk.entities
            and self.embedding == chunk.embedding
            and stored_meta == candidate_meta
            and self.title == chunk.title
        )

    def compare_for_upsert(
        self, chunk: _PreparedChunk
    ) -> tuple[bool, "_PackedChunk | None"]:
        if self._matches_ingress(chunk):
            return True, None
        return False, type(self).from_batch_ingress(chunk)


_PreparedChunk = _PackedChunk | _ValidatedChunk


def _token_counts(text: str) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    length = 0
    for term in tokenize(text):
        length += 1
        counts[term] = counts.get(term, 0) + 1
    return length, counts


class _VectorBaseRecordResolver:
    """Resolve trusted base records without retaining the owning vector."""

    __slots__ = ("chunks", "feedback", "fields")

    def __init__(
        self,
        chunks: dict[int, _PackedChunk],
        feedback: Feedback | None,
        fields: tuple[str, ...],
    ) -> None:
        self.chunks = chunks
        self.feedback = feedback
        self.fields = fields

    def plain(self, slot: int):
        chunk = self.chunks.get(slot)
        return None if chunk is None else _token_counts(chunk.text)

    def fielded(self, slot: int):
        chunk = self.chunks.get(slot)
        if chunk is None:
            return None
        values = {
            "title": chunk.title,
            "body": chunk.text,
            "memory": self.feedback.text(chunk.id) if self.feedback is not None else "",
        }
        records = tuple(_token_counts(values[field]) for field in self.fields)
        return tuple(record[0] for record in records), tuple(
            record[1] for record in records
        )


class MutableInMemoryVector(InMemoryVector):
    """Opt-in passage store with exact incremental corpus mutation.

    The default :class:`InMemoryVector` keeps its precomputed-weight fast path unchanged.
    This backend instead retains raw lexical sufficient statistics and assigns every inserted
    chunk a monotonic slot. Existing ids keep their slot; deleted slots are never reused, so
    score ties retain full-rebuild insertion order without renumbering corpus postings.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        embedder: Optional[Callable[[str], list[float]]] = None,
        title_weight: float = _TITLE_WEIGHT,
        lexical_weight: float = 0.8,
        dense_weight: float = 1.0,
        pool: int = 40,
        idf_pow: float = _IDF_POW,
        feedback: Optional[Feedback] = None,
    ):
        _validate_vector_scoring_config(
            title_weight,
            lexical_weight,
            dense_weight,
            pool,
            idf_pow,
        )
        packed_chunks: dict[int, _PackedChunk] = {}
        slot_by_id: dict[str, int] = {}
        title_count = lexical_count = embedding_count = 0
        freeze_chunk = _PackedChunk.freeze
        for slot, candidate in enumerate(chunks):
            if not isinstance(candidate, Chunk):
                raise TypeError(f"expected Chunk, got {type(candidate).__name__}")
            chunk = freeze_chunk(candidate)
            if chunk.id in slot_by_id:
                raise ValueError(f"duplicate chunk id {chunk.id!r}")
            packed_chunks[slot] = chunk
            slot_by_id[chunk.id] = slot
            title_count += bool(chunk.title)
            lexical_count += bool(chunk.title or chunk.text)
            embedding_count += bool(chunk.embedding)
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self._pool = pool
        self._title_weight = title_weight
        self._idf_pow = idf_pow
        # A mutable index owns its feedback snapshot. Otherwise direct mutation of a
        # caller-held Feedback could silently desynchronize the evidence postings.
        self._feedback = feedback.copy() if feedback is not None else None
        self._lock = RLock()
        self._revision = 0
        self._next_slot = len(packed_chunks)
        self._chunks = packed_chunks
        self._slot_by_id = slot_by_id
        self._title_count = title_count
        self._lexical_count = lexical_count
        self._embedding_count = embedding_count
        self._fielded = self._feedback is not None or self._title_count > 0
        self._bm25 = None
        self._persistence_in_progress = False
        self._refresh_modes()

    @staticmethod
    def _prepare_chunks(chunks) -> list[_PreparedChunk]:
        prepared: list[_PreparedChunk] = []
        seen: set[str] = set()
        prepare_ingress = _PackedChunk.prepare_batch_ingress
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise TypeError(f"expected Chunk, got {type(chunk).__name__}")
            candidate = prepare_ingress(chunk)
            if candidate.id in seen:
                raise ValueError(f"duplicate chunk id {candidate.id!r}")
            seen.add(candidate.id)
            prepared.append(candidate)
        return prepared

    @staticmethod
    def _prepare_ids(ids) -> list[str]:
        prepared = list(ids)
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in prepared):
            raise ValueError("chunk ids must be non-empty strings")
        if len(set(prepared)) != len(prepared):
            raise ValueError("a mutation batch cannot contain duplicate chunk ids")
        return prepared

    @staticmethod
    def _counts(chunks) -> tuple[int, int, int]:
        title = lexical = embedding = 0
        for chunk in chunks:
            title += bool(chunk.title)
            lexical += bool(chunk.title or chunk.text)
            embedding += chunk.embedding is not None and bool(chunk.embedding)
        return title, lexical, embedding

    def _refresh_modes(self) -> None:
        active = len(self._chunks)
        self._lexical = self._lexical_count > 0
        self._dense = (
            self.embedder is not None and active > 0 and self._embedding_count == active
        )

    def _field_tokens(
        self, chunk: _PackedChunk, *, memory_text: str | None = None
    ) -> dict[str, list[str]]:
        fields = {"title": tokenize(chunk.title), "body": tokenize(chunk.text)}
        if self._feedback is not None:
            text = self._feedback.text(chunk.id) if memory_text is None else memory_text
            fields["memory"] = tokenize(text)
        return fields

    def _index_document(self, chunk: _PackedChunk, fielded: bool):
        return self._field_tokens(chunk) if fielded else tokenize(chunk.text)

    def _build_lexical(self, fielded: bool, chunks: dict[int, _PackedChunk]):
        if fielded:
            weights = {"title": self._title_weight, "body": 1.0}
            evidence_fields: set[str] = set()
            if self._feedback is not None:
                weights["memory"] = 1.0
                evidence_fields.add("memory")
            docs = ((slot, self._field_tokens(chunk)) for slot, chunk in chunks.items())
            resolver = _VectorBaseRecordResolver(chunks, self._feedback, tuple(weights))
            return CompactMutableBM25F._from_vector(
                docs,
                weights,
                resolver.fielded,
                k1=_LEXICAL_K1,
                b=_LEXICAL_B,
                idf_pow=self._idf_pow,
                evidence_fields=evidence_fields,
            )
        docs = ((slot, tokenize(chunk.text)) for slot, chunk in chunks.items())
        resolver = _VectorBaseRecordResolver(chunks, self._feedback, ("body",))
        return CompactMutableBM25._from_vector(
            docs,
            resolver.plain,
            k1=_LEXICAL_K1,
            b=_LEXICAL_B,
            idf_pow=self._idf_pow,
        )

    def _ensure_lexical_index(self):
        """Materialize the current corpus once, without changing its revision."""
        with self._lock:
            self._reject_persistence_reentry()
            if self._bm25 is None and self._lexical:
                self._bm25 = self._build_lexical(self._fielded, self._chunks)
            return self._bm25

    def _prepare_for_persistence(self) -> None:
        """Materialize deferred lexical state before writing a warm snapshot."""
        if self._lexical:
            self._ensure_lexical_index()

    def _reject_persistence_reentry(self) -> None:
        if self._persistence_in_progress:
            raise RuntimeError(
                "mutable vector cannot change while it is being persisted"
            )

    def _publish_full_state(
        self,
        *,
        chunks: dict[int, _PackedChunk],
        next_slot: int,
        index,
        fielded: bool,
        counts: tuple[int, int, int],
    ) -> None:
        self._chunks = chunks
        self._next_slot = next_slot
        self._slot_by_id = {chunk.id: slot for slot, chunk in chunks.items()}
        self._bm25 = index
        self._fielded = fielded
        self._title_count, self._lexical_count, self._embedding_count = counts
        self._refresh_modes()

    @property
    def chunks(self) -> list[Chunk]:
        with self._lock:
            return [chunk.thaw() for chunk in self._chunks.values()]

    @property
    def feedback(self) -> Feedback | None:
        """A detached snapshot; mutate memory through ``remember``/``forget``."""
        with self._lock:
            return self._feedback.copy() if self._feedback is not None else None

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def is_mutable(self) -> bool:
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    @contextmanager
    def read_view(self):
        """Hold one corpus revision across a multi-call retrieval."""
        with self._lock:
            yield self._revision

    def _dense_ranked(self, query: str, limit: int) -> list[tuple[int, float]]:
        vector = self.embedder(query)  # type: ignore[misc]
        scored = [
            (slot, _cosine(vector, chunk.embedding))
            for slot, chunk in self._chunks.items()
            if chunk.embedding
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    def search(self, query: str, *, limit: int = 20) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._dense and self._lexical:
                index = self._ensure_lexical_index()
                pool = max(limit, self._pool)
                dense = _minmax(self._dense_ranked(query, pool))
                lexical = _minmax(
                    rerank_lexical_candidates(
                        query,
                        index.search(query, limit=pool, recover_partial_outlier=True),
                        lambda slot: self._chunks[slot].thaw(),
                    )
                )
                fused = {
                    slot: self.dense_weight * dense.get(slot, 0.0)
                    + self.lexical_weight * lexical.get(slot, 0.0)
                    for slot in set(dense) | set(lexical)
                }
                ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[
                    :limit
                ]
            elif self._dense:
                ranked = self._dense_ranked(query, limit)
            elif self._lexical:
                index = self._ensure_lexical_index()
                pool = max(limit, self._pool)
                candidates = index.search(
                    query, limit=pool, recover_partial_outlier=True
                )
                if not candidates:
                    candidates = rank_korean_character_fallback(
                        query, self._chunks, limit=pool
                    )
                ranked = rerank_lexical_candidates(
                    query,
                    candidates,
                    lambda slot: self._chunks[slot].thaw(),
                )[:limit]
            else:
                ranked = []
            return [(self._chunks[slot].thaw(), score) for slot, score in ranked]

    def fetch(self, ids: list[str]) -> list[Chunk]:
        with self._lock:
            return [
                self._chunks[self._slot_by_id[chunk_id]].thaw()
                for chunk_id in ids
                if chunk_id in self._slot_by_id
            ]

    def upsert_chunks(self, chunks: list[Chunk]) -> ChunkMutationResult:
        prepared = self._prepare_chunks(chunks)
        with self._lock:
            self._reject_persistence_reentry()
            inserted = updated = unchanged = 0
            changes: list[tuple[int, _PackedChunk, _PackedChunk | None]] = []
            next_slot = self._next_slot
            title_count = self._title_count
            lexical_count = self._lexical_count
            embedding_count = self._embedding_count
            for candidate in prepared:
                slot = self._slot_by_id.get(candidate.id)
                before = self._chunks.get(slot) if slot is not None else None
                chunk = None
                if before is not None:
                    matches, chunk = before.compare_for_upsert(candidate)
                    if matches:
                        unchanged += 1
                        continue
                if chunk is None:
                    chunk = _PackedChunk.from_batch_ingress(candidate)
                if before is None:
                    slot = next_slot
                    next_slot += 1
                    inserted += 1
                else:
                    updated += 1
                    title_count -= bool(before.title)
                    lexical_count -= bool(before.title or before.text)
                    embedding_count -= bool(before.embedding)
                title_count += bool(chunk.title)
                lexical_count += bool(chunk.title or chunk.text)
                embedding_count += bool(chunk.embedding)
                changes.append((slot, chunk, before))

            if not changes:
                return ChunkMutationResult(revision=self._revision, unchanged=unchanged)

            fielded = self._feedback is not None or title_count > 0
            counts = title_count, lexical_count, embedding_count
            if self._bm25 is None:
                for slot, chunk, before in changes:
                    self._chunks[slot] = chunk
                    self._slot_by_id[chunk.id] = slot
                self._next_slot = next_slot
                self._fielded = fielded
                self._title_count, self._lexical_count, self._embedding_count = counts
                self._refresh_modes()
                reindexed = 0
                rebuilt = False
            elif fielded != self._fielded:
                proposed_chunks = self._chunks.copy()
                for slot, chunk, _before in changes:
                    proposed_chunks[slot] = chunk
                proposed_index = self._build_lexical(fielded, proposed_chunks)
                self._publish_full_state(
                    chunks=proposed_chunks,
                    next_slot=next_slot,
                    index=proposed_index,
                    fielded=fielded,
                    counts=counts,
                )
                reindexed = len(proposed_chunks)
                rebuilt = True
            else:
                indexed = [
                    (slot, self._index_document(chunk, fielded))
                    for slot, chunk, _before in changes
                ]
                self._bm25.upsert_many(indexed)
                for slot, chunk, before in changes:
                    self._chunks[slot] = chunk
                    self._slot_by_id[chunk.id] = slot
                self._next_slot = next_slot
                self._title_count, self._lexical_count, self._embedding_count = counts
                self._refresh_modes()
                reindexed = 0
                rebuilt = False

            self._revision += 1
            return ChunkMutationResult(
                revision=self._revision,
                inserted=inserted,
                updated=updated,
                unchanged=unchanged,
                reindexed=reindexed,
                rebuilt=rebuilt,
            )

    def delete_chunks(self, ids: list[str]) -> ChunkMutationResult:
        prepared = self._prepare_ids(ids)
        with self._lock:
            self._reject_persistence_reentry()
            slots: list[int] = []
            removed_ids: list[str] = []
            missing = 0
            title_count = self._title_count
            lexical_count = self._lexical_count
            embedding_count = self._embedding_count
            for chunk_id in prepared:
                slot = self._slot_by_id.get(chunk_id)
                if slot is None:
                    missing += 1
                    continue
                chunk = self._chunks[slot]
                slots.append(slot)
                removed_ids.append(chunk.id)
                title_count -= bool(chunk.title)
                lexical_count -= bool(chunk.title or chunk.text)
                embedding_count -= bool(chunk.embedding)
            if not slots:
                return ChunkMutationResult(revision=self._revision, missing=missing)

            fielded = self._feedback is not None or title_count > 0
            counts = title_count, lexical_count, embedding_count

            if self._bm25 is None:
                for slot, chunk_id in zip(slots, removed_ids):
                    del self._chunks[slot]
                    del self._slot_by_id[chunk_id]
                self._fielded = fielded
                self._title_count, self._lexical_count, self._embedding_count = counts
                self._refresh_modes()
                reindexed = 0
                rebuilt = False
            elif fielded != self._fielded:
                deleted = set(slots)
                proposed_chunks = {
                    slot: chunk
                    for slot, chunk in self._chunks.items()
                    if slot not in deleted
                }
                proposed_index = self._build_lexical(fielded, proposed_chunks)
                self._publish_full_state(
                    chunks=proposed_chunks,
                    next_slot=self._next_slot,
                    index=proposed_index,
                    fielded=fielded,
                    counts=counts,
                )
                reindexed = len(proposed_chunks)
                rebuilt = True
            else:
                self._bm25.delete_many(slots)
                for slot, chunk_id in zip(slots, removed_ids):
                    del self._chunks[slot]
                    del self._slot_by_id[chunk_id]
                self._title_count, self._lexical_count, self._embedding_count = counts
                self._refresh_modes()
                reindexed = 0
                rebuilt = False

            if self._feedback is not None:
                self._feedback.drop(removed_ids)
            self._revision += 1
            return ChunkMutationResult(
                revision=self._revision,
                deleted=len(slots),
                missing=missing,
                reindexed=reindexed,
                rebuilt=rebuilt,
            )

    def remember(self, query: str, doc_ids: list[str]) -> None:
        self._change_feedback(query, doc_ids, forget=False)

    def forget(self, query: str, doc_ids: list[str]) -> None:
        self._change_feedback(query, doc_ids, forget=True)

    def _change_feedback(self, query: str, doc_ids: list[str], *, forget: bool) -> None:
        if self._feedback is None:
            raise RuntimeError(
                "this store cannot change memory incrementally — build it with feedback=Feedback()"
            )
        with self._lock:
            self._reject_persistence_reentry()
            normalized_query = (query or "").strip()
            changed: list[tuple[int, _PackedChunk, list[str]]] = []
            for chunk_id in dict.fromkeys(doc_ids):
                slot = self._slot_by_id.get(chunk_id)
                if slot is None:
                    continue
                before = self._feedback.queries(chunk_id)
                after = list(before)
                if forget:
                    if normalized_query in after:
                        after.remove(normalized_query)
                elif normalized_query and normalized_query not in after:
                    after.append(normalized_query)
                if after != before:
                    changed.append((slot, self._chunks[slot], after))
            if self._bm25 is not None:
                if not isinstance(self._bm25, (_MutableBM25F, CompactMutableBM25F)):
                    raise RuntimeError("feedback requires a fielded mutable index")
                self._bm25.upsert_many(
                    (
                        slot,
                        self._field_tokens(chunk, memory_text=" ".join(after)),
                    )
                    for slot, chunk, after in changed
                )
            if changed:
                for _slot, chunk, after in changed:
                    self._feedback._replace_queries(chunk.id, after)
                self._revision += 1

    def attach_embedder(self, embedder: Optional[Callable[[str], list[float]]]) -> None:
        with self._lock:
            self._reject_persistence_reentry()
            self.embedder = embedder
            self._refresh_modes()

    @staticmethod
    def _feedback_rows(feedback: Feedback | None) -> tuple | None:
        if feedback is None:
            return None
        if type(feedback) is not Feedback or type(feedback._mem) is not dict:
            raise TypeError("mutable vector feedback is not persistable")
        rows: list[tuple[str, tuple[str, ...]]] = []
        for chunk_id, queries in feedback._mem.items():
            if (
                type(chunk_id) is not str
                or type(queries) is not list
                or any(type(query) is not str for query in queries)
                or len(set(queries)) != len(queries)
            ):
                raise TypeError("mutable vector feedback is not persistable")
            rows.append((chunk_id, tuple(queries)))
        return tuple(rows)

    @staticmethod
    def _feedback_from_rows(value: object) -> Feedback | None:
        if value is None:
            return None
        if type(value) is not tuple:
            raise ValueError("mutable vector feedback rows are invalid")
        feedback = Feedback()
        seen: set[str] = set()
        for row in value:
            if type(row) is not tuple or len(row) != 2:
                raise ValueError("mutable vector feedback row is invalid")
            chunk_id, queries = row
            if (
                type(chunk_id) is not str
                or chunk_id in seen
                or type(queries) is not tuple
                or any(type(query) is not str for query in queries)
                or len(set(queries)) != len(queries)
            ):
                raise ValueError("mutable vector feedback row is invalid")
            feedback._mem[chunk_id] = list(queries)
            seen.add(chunk_id)
        return feedback

    def __getstate__(self) -> dict:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return type(self)._serialize_legacy_instance(self.__dict__)
        with lock:
            self._reject_persistence_reentry()
            self._persistence_in_progress = True
            try:
                _validate_vector_scoring_config(
                    self._title_weight,
                    self.lexical_weight,
                    self.dense_weight,
                    self._pool,
                    self._idf_pow,
                )
                lexical_state = None
                if self._bm25 is not None:
                    if isinstance(self._bm25, CompactMutableBM25F):
                        lexical_state = {
                            "kind": "bm25f",
                            "state": self._bm25._vector_packed_state(),
                        }
                    elif isinstance(self._bm25, CompactMutableBM25):
                        lexical_state = {
                            "kind": "bm25",
                            "state": self._bm25._vector_packed_state(),
                        }
                    else:
                        raise TypeError(
                            "mutable vector lexical index is not persistable"
                        )
                payload = {
                    "payload_version": _VECTOR_PAYLOAD_VERSION,
                    "chunks": self._chunks,
                    "next_slot": self._next_slot,
                    "title_weight": self._title_weight,
                    "lexical_weight": self.lexical_weight,
                    "dense_weight": self.dense_weight,
                    "pool": self._pool,
                    "idf_pow": self._idf_pow,
                    "feedback_rows": self._feedback_rows(self._feedback),
                    "revision": self._revision,
                    "materialized": self._bm25 is not None,
                    "lexical_state": lexical_state,
                }
                try:
                    payload_pickle = pickle.dumps(
                        payload, protocol=_VECTOR_PAYLOAD_PROTOCOL
                    )
                except (
                    AttributeError,
                    pickle.PickleError,
                    RecursionError,
                    TypeError,
                ) as exc:
                    raise TypeError("mutable vector state is not persistable") from exc
                return {
                    "state_version": _MUTABLE_VECTOR_STATE_VERSION,
                    "payload_pickle": payload_pickle,
                    "binding_sha256": _vector_payload_sha256(payload_pickle),
                }
            finally:
                self._persistence_in_progress = False

    def __setstate__(self, state: dict) -> None:
        if type(state) is dict and "state_version" in state:
            if (
                type(state.get("state_version")) is not int
                or state.get("state_version") != _MUTABLE_VECTOR_STATE_VERSION
            ):
                raise ValueError("unsupported mutable vector persistence state")
            restored = type(self)._restore_persistent_state(state)
            self.__dict__.clear()
            self.__dict__.update(restored)
            return
        restored = type(self)._restore_legacy_state(state)
        self.__dict__.clear()
        self.__dict__.update(restored)

    @classmethod
    def _restore_persistent_state(cls, state: dict) -> dict:
        expected = {"state_version", "payload_pickle", "binding_sha256"}
        if (
            type(state) is not dict
            or type(state.get("state_version")) is not int
            or state.get("state_version") != _MUTABLE_VECTOR_STATE_VERSION
            or set(state) != expected
        ):
            raise ValueError("invalid mutable vector persistence state")
        payload_pickle = state["payload_pickle"]
        binding_sha256 = state["binding_sha256"]
        if (
            type(payload_pickle) is not bytes
            or not payload_pickle.startswith(_VECTOR_PAYLOAD_PREFIX)
            or len(payload_pickle) <= len(_VECTOR_PAYLOAD_PREFIX)
            or type(binding_sha256) is not bytes
            or len(binding_sha256) != 32
        ):
            raise ValueError("mutable vector bound payload is invalid")
        expected_binding = _vector_payload_sha256(payload_pickle)
        if not hmac.compare_digest(binding_sha256, expected_binding):
            raise ValueError("mutable vector binding digest does not match its payload")
        # Pickle is a trusted-input format. The sealed bytes are decoded only after
        # their integrity binding has been verified.
        payload = _decode_vector_payload(payload_pickle)
        return cls._restore_payload_state(payload)

    @classmethod
    def _restore_payload_state(cls, state: object) -> dict:
        expected = {
            "payload_version",
            "chunks",
            "next_slot",
            "title_weight",
            "lexical_weight",
            "dense_weight",
            "pool",
            "idf_pow",
            "feedback_rows",
            "revision",
            "materialized",
            "lexical_state",
        }
        if (
            type(state) is not dict
            or type(state.get("payload_version")) is not int
            or state.get("payload_version") != _VECTOR_PAYLOAD_VERSION
            or set(state) != expected
        ):
            raise ValueError("invalid mutable vector payload state")
        raw_chunks = state["chunks"]
        if type(raw_chunks) is not dict:
            raise ValueError("mutable vector chunks are not canonical")
        seen_ids: set[str] = set()
        previous_slot = -1
        for slot, chunk in raw_chunks.items():
            if type(slot) is not int or slot < 0 or slot <= previous_slot:
                raise ValueError("mutable vector slots must be strictly increasing")
            if type(chunk) is not _PackedChunk:
                raise ValueError("mutable vector chunk payload is invalid")
            if (
                type(chunk.id) is not str
                or not chunk.id
                or type(chunk.text) is not str
                or type(chunk.title) is not str
                or type(chunk.entities) is not tuple
                or any(type(entity) is not str for entity in chunk.entities)
                or (
                    chunk.embedding is not None
                    and (
                        type(chunk.embedding) is not tuple
                        or any(
                            not _is_finite_builtin_number(value)
                            for value in chunk.embedding
                        )
                    )
                )
                or not (
                    type(chunk.meta) is _EmptyMeta
                    or (type(chunk.meta) is dict and bool(chunk.meta))
                )
                or chunk.id in seen_ids
            ):
                raise ValueError("mutable vector chunk payload is invalid")
            seen_ids.add(chunk.id)
            previous_slot = slot
        next_slot = state["next_slot"]
        if (
            type(next_slot) is not int
            or next_slot < 0
            or (raw_chunks and next_slot <= previous_slot)
        ):
            raise ValueError("mutable vector next slot is invalid")
        revision = state["revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("mutable vector revision is invalid")
        materialized = state["materialized"]
        if type(materialized) is not bool:
            raise ValueError("mutable vector materialization flag is invalid")
        title_weight = state["title_weight"]
        lexical_weight = state["lexical_weight"]
        dense_weight = state["dense_weight"]
        pool = state["pool"]
        idf_pow = state["idf_pow"]
        _validate_vector_scoring_config(
            title_weight,
            lexical_weight,
            dense_weight,
            pool,
            idf_pow,
        )
        feedback = cls._feedback_from_rows(state["feedback_rows"])
        chunks = raw_chunks

        title_count, lexical_count, embedding_count = cls._counts(chunks.values())
        fielded = feedback is not None or title_count > 0
        lexical = lexical_count > 0
        resolver_fields = (
            ("title", "body", "memory")
            if feedback is not None
            else (("title", "body") if fielded else ("body",))
        )
        resolver = _VectorBaseRecordResolver(chunks, feedback, resolver_fields)
        raw_lexical = state["lexical_state"]
        index = None
        if raw_lexical is not None:
            if type(raw_lexical) is not dict or set(raw_lexical) != {"kind", "state"}:
                raise ValueError("mutable vector lexical state is invalid")
            kind = raw_lexical["kind"]
            if type(kind) is not str:
                raise ValueError("mutable vector lexical kind is invalid")
            if kind == "bm25f" and fielded:
                index = CompactMutableBM25F._from_vector_packed_state(
                    raw_lexical["state"],
                    resolver.fielded,
                    expected_doc_ids=chunks,
                )
            elif kind == "bm25" and not fielded:
                index = CompactMutableBM25._from_vector_packed_state(
                    raw_lexical["state"],
                    resolver.plain,
                    expected_doc_ids=chunks,
                )
            else:
                raise ValueError("mutable vector lexical kind differs from its chunks")
        if index is not None and (
            (index.k1, index.b, index._idf_pow) != (_LEXICAL_K1, _LEXICAL_B, idf_pow)
        ):
            raise ValueError("mutable vector lexical configuration is invalid")
        if isinstance(index, CompactMutableBM25F):
            expected_weights = {"title": title_weight, "body": 1.0}
            expected_evidence: frozenset[str] = frozenset()
            if feedback is not None:
                expected_weights["memory"] = 1.0
                expected_evidence = frozenset({"memory"})
            if (
                index.w != expected_weights
                or tuple(index.w) != tuple(expected_weights)
                or index.evidence_fields != expected_evidence
            ):
                raise ValueError("mutable vector field configuration is invalid")
        if materialized != (index is not None):
            raise ValueError("mutable vector materialization state is inconsistent")
        if index is not None and index._max_doc_id >= next_slot:
            raise ValueError("mutable vector lexical high-water mark exceeds its slots")
        return {
            "embedder": None,
            "lexical_weight": lexical_weight,
            "dense_weight": dense_weight,
            "_pool": pool,
            "_title_weight": title_weight,
            "_idf_pow": idf_pow,
            "_feedback": feedback,
            "_lock": RLock(),
            "_revision": revision,
            "_next_slot": next_slot,
            "_chunks": chunks,
            "_slot_by_id": {chunk.id: slot for slot, chunk in chunks.items()},
            "_title_count": title_count,
            "_lexical_count": lexical_count,
            "_embedding_count": embedding_count,
            "_fielded": fielded,
            "_bm25": index,
            "_persistence_in_progress": False,
            "_lexical": lexical,
            "_dense": False,
        }

    @classmethod
    def _normalize_legacy_chunks(
        cls, raw_chunks: object, raw_order: object = None
    ) -> dict[int, _PackedChunk]:
        if type(raw_chunks) is not dict:
            raise ValueError("mutable vector legacy chunks are invalid")
        raw_slots = list(raw_chunks)
        if any(type(slot) is not int or slot < 0 for slot in raw_slots):
            raise ValueError("mutable vector legacy slots are invalid")
        if raw_order is None:
            ordered_slots = raw_slots
        elif type(raw_order) is list:
            ordered_slots = list(raw_order)
        elif type(raw_order) is dict:
            ordered_slots = list(raw_order)
        else:
            raise ValueError("mutable vector legacy order is invalid")
        if (
            any(type(slot) is not int or slot < 0 for slot in ordered_slots)
            or len(set(ordered_slots)) != len(ordered_slots)
            or set(ordered_slots) != set(raw_slots)
            or any(
                right <= left for left, right in zip(ordered_slots, ordered_slots[1:])
            )
        ):
            raise ValueError("mutable vector legacy order is invalid")

        chunks: dict[int, _PackedChunk] = {}
        seen_ids: set[str] = set()
        for slot in ordered_slots:
            raw_chunk = raw_chunks[slot]
            if type(raw_chunk) is _PackedChunk:
                chunk = raw_chunk
            elif type(raw_chunk) is Chunk:
                try:
                    chunk = _PackedChunk.freeze(raw_chunk)
                except (TypeError, ValueError) as exc:
                    raise ValueError("mutable vector legacy chunk is invalid") from exc
            else:
                raise ValueError("mutable vector legacy chunk is invalid")
            if (
                type(chunk.id) is not str
                or not chunk.id
                or type(chunk.text) is not str
                or type(chunk.title) is not str
                or type(chunk.entities) is not tuple
                or any(type(entity) is not str for entity in chunk.entities)
                or (
                    chunk.embedding is not None
                    and (
                        type(chunk.embedding) is not tuple
                        or any(
                            not _is_finite_builtin_number(value)
                            for value in chunk.embedding
                        )
                    )
                )
                or chunk.id in seen_ids
            ):
                raise ValueError("mutable vector legacy chunk is invalid")
            chunks[slot] = chunk
            seen_ids.add(chunk.id)
        return chunks

    @staticmethod
    def _copy_valid_feedback(value: object) -> Feedback | None:
        if value is None:
            return None
        if type(value) is not Feedback or type(value._mem) is not dict:
            raise ValueError("mutable vector feedback is invalid")
        for chunk_id, queries in value._mem.items():
            if (
                type(chunk_id) is not str
                or type(queries) is not list
                or any(type(query) is not str for query in queries)
                or len(set(queries)) != len(queries)
            ):
                raise ValueError("mutable vector feedback is invalid")
        return value.copy()

    @staticmethod
    def _validate_old_mutable_plain(
        index: _MutableBM25,
        chunks: dict[int, _PackedChunk],
        *,
        idf_pow: float,
        next_slot: int,
    ) -> None:
        expected_keys = {
            "k1",
            "b",
            "_idf_pow",
            "N",
            "_total_len",
            "_max_doc_id",
            "_docs",
            "_terms",
            "_df",
            "_postings",
            "_weight_cache",
            "_mutation_version",
        }
        if set(index.__dict__) != expected_keys:
            raise ValueError("legacy mutable BM25 schema is invalid")
        reference = _MutableBM25(
            ((slot, tokenize(chunk.text)) for slot, chunk in chunks.items()),
            k1=_LEXICAL_K1,
            b=_LEXICAL_B,
            idf_pow=idf_pow,
        )
        current_max = max(chunks, default=-1)
        if (
            (index.k1, index.b, index._idf_pow) != (_LEXICAL_K1, _LEXICAL_B, idf_pow)
            or type(index._max_doc_id) is not int
            or index._max_doc_id < current_max
            or index._max_doc_id >= next_slot
            or type(index._mutation_version) is not int
            or index._mutation_version < 0
            or index.N != reference.N
            or index._total_len != reference._total_len
            or index._docs != reference._docs
            or index._terms != reference._terms
            or index._df != reference._df
            or index._postings != reference._postings
            or type(index._weight_cache) is not dict
        ):
            raise ValueError("legacy mutable BM25 state differs from its chunks")
        for term, cached in index._weight_cache.items():
            if (
                type(term) is not str
                or term not in reference._terms
                or cached != reference._term_weights(term)
            ):
                raise ValueError("legacy mutable BM25 cache is invalid")

    @staticmethod
    def _validate_old_mutable_fielded(
        index: _MutableBM25F,
        chunks: dict[int, _PackedChunk],
        feedback: Feedback | None,
        *,
        title_weight: float,
        idf_pow: float,
        next_slot: int,
    ) -> None:
        expected_keys = {
            "k1",
            "b",
            "_idf_pow",
            "fields",
            "w",
            "_fw",
            "evidence_fields",
            "_is_ev",
            "N",
            "_totals",
            "_max_doc_id",
            "_docs",
            "_terms",
            "_df",
            "_dfe",
            "_postings",
            "_weight_cache",
            "_mutation_version",
        }
        if set(index.__dict__) != expected_keys:
            raise ValueError("legacy mutable BM25F schema is invalid")
        weights = {"title": title_weight, "body": 1.0}
        evidence_fields: frozenset[str] = frozenset()
        if feedback is not None:
            weights["memory"] = 1.0
            evidence_fields = frozenset({"memory"})

        def documents():
            for slot, chunk in chunks.items():
                fields = {
                    "title": tokenize(chunk.title),
                    "body": tokenize(chunk.text),
                }
                if feedback is not None:
                    fields["memory"] = tokenize(feedback.text(chunk.id))
                yield slot, fields

        reference = _MutableBM25F(
            documents(),
            weights,
            k1=_LEXICAL_K1,
            b=_LEXICAL_B,
            idf_pow=idf_pow,
            evidence_fields=evidence_fields,
        )
        current_max = max(chunks, default=-1)
        if (
            (index.k1, index.b, index._idf_pow) != (_LEXICAL_K1, _LEXICAL_B, idf_pow)
            or type(index._max_doc_id) is not int
            or index._max_doc_id < current_max
            or index._max_doc_id >= next_slot
            or type(index._mutation_version) is not int
            or index._mutation_version < 0
            or type(index.fields) is not list
            or index.fields != list(weights)
            or type(index.w) is not dict
            or index.w != weights
            or tuple(index.w) != tuple(weights)
            or index._fw != list(weights.values())
            or index.evidence_fields != evidence_fields
            or index._is_ev != [field in evidence_fields for field in weights]
            or index.N != reference.N
            or index._totals != reference._totals
            or index._docs != reference._docs
            or index._terms != reference._terms
            or index._df != reference._df
            or index._dfe != reference._dfe
            or index._postings != reference._postings
            or type(index._weight_cache) is not dict
        ):
            raise ValueError("legacy mutable BM25F state differs from its chunks")
        for term, cached in index._weight_cache.items():
            if (
                type(term) is not str
                or term not in reference._terms
                or cached != reference._term_weights(term)
            ):
                raise ValueError("legacy mutable BM25F cache is invalid")

    @classmethod
    def _serialize_legacy_instance(cls, raw: dict) -> dict:
        if frozenset(raw) not in {
            _LEGACY_MUTABLE_VECTOR_KEYS,
            _LEGACY_MUTABLE_VECTOR_KEYS | {"_order"},
        }:
            raise ValueError("mutable vector legacy instance is invalid")
        state = raw.copy()
        state["embedder"] = None
        state["_dense"] = False
        return state

    @classmethod
    def _restore_legacy_state(cls, state: dict) -> dict:
        if type(state) is not dict or frozenset(state) not in {
            _LEGACY_MUTABLE_VECTOR_KEYS,
            _LEGACY_MUTABLE_VECTOR_KEYS | {"_order"},
        }:
            raise ValueError("invalid legacy mutable vector state")
        if state["embedder"] is not None or state["_dense"] is not False:
            raise ValueError("legacy mutable vector runtime state is not portable")
        if "_order" in state and type(state["_order"]) not in {list, dict}:
            raise ValueError("mutable vector legacy order is invalid")
        title_weight = state["_title_weight"]
        lexical_weight = state["lexical_weight"]
        dense_weight = state["dense_weight"]
        pool = state["_pool"]
        idf_pow = state["_idf_pow"]
        _validate_vector_scoring_config(
            title_weight,
            lexical_weight,
            dense_weight,
            pool,
            idf_pow,
        )
        revision = state["_revision"]
        next_slot = state["_next_slot"]
        if (
            type(revision) is not int
            or revision < 0
            or type(next_slot) is not int
            or next_slot < 0
        ):
            raise ValueError("legacy mutable vector versions are invalid")

        chunks = cls._normalize_legacy_chunks(state["_chunks"], state.get("_order"))
        if chunks and next_slot <= max(chunks):
            raise ValueError("legacy mutable vector next slot is invalid")
        feedback = cls._copy_valid_feedback(state["_feedback"])
        title_count, lexical_count, embedding_count = cls._counts(chunks.values())
        fielded = feedback is not None or title_count > 0
        lexical = lexical_count > 0
        slot_by_id = {chunk.id: slot for slot, chunk in chunks.items()}
        derived = (
            ("_slot_by_id", slot_by_id),
            ("_title_count", title_count),
            ("_lexical_count", lexical_count),
            ("_embedding_count", embedding_count),
            ("_fielded", fielded),
            ("_lexical", lexical),
        )
        for key, expected in derived:
            value = state[key]
            if key == "_slot_by_id":
                if type(value) is not dict or value != expected:
                    raise ValueError("legacy mutable vector derived state is invalid")
            elif type(value) is not type(expected) or value != expected:
                raise ValueError("legacy mutable vector derived state is invalid")

        fields = (
            ("title", "body", "memory")
            if feedback is not None
            else ("title", "body")
            if fielded
            else ("body",)
        )
        resolver = _VectorBaseRecordResolver(chunks, feedback, fields)
        index = state["_bm25"]
        lexical_state = None
        if type(index) is CompactMutableBM25:
            if fielded:
                raise ValueError("legacy mutable BM25 kind differs from its chunks")
            try:
                index._base._validate_storage()
                index._validate_vector_source(resolver.plain, chunks)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError("legacy mutable BM25 state is invalid") from exc
            if (index.k1, index.b, index._idf_pow) != (
                _LEXICAL_K1,
                _LEXICAL_B,
                idf_pow,
            ) or index._max_doc_id >= next_slot:
                raise ValueError("legacy mutable BM25 configuration is invalid")
            lexical_state = {
                "kind": "bm25",
                "state": index._vector_packed_state(),
            }
        elif type(index) is CompactMutableBM25F:
            if not fielded:
                raise ValueError("legacy mutable BM25F kind differs from its chunks")
            try:
                index._base._validate_storage()
                index._validate_vector_source(resolver.fielded, chunks)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise ValueError("legacy mutable BM25F state is invalid") from exc
            weights = {"title": title_weight, "body": 1.0}
            evidence_fields: frozenset[str] = frozenset()
            if feedback is not None:
                weights["memory"] = 1.0
                evidence_fields = frozenset({"memory"})
            if (
                (index.k1, index.b, index._idf_pow)
                != (_LEXICAL_K1, _LEXICAL_B, idf_pow)
                or type(index.fields) is not list
                or index.fields != list(weights)
                or index.w != weights
                or tuple(index.w) != tuple(weights)
                or index._fw != list(weights.values())
                or index.evidence_fields != evidence_fields
                or index._is_ev != [field in evidence_fields for field in weights]
                or index._max_doc_id >= next_slot
            ):
                raise ValueError("legacy mutable BM25F configuration is invalid")
            lexical_state = {
                "kind": "bm25f",
                "state": index._vector_packed_state(),
            }
        elif type(index) is _MutableBM25:
            if fielded:
                raise ValueError("legacy mutable BM25 kind differs from its chunks")
            cls._validate_old_mutable_plain(
                index, chunks, idf_pow=idf_pow, next_slot=next_slot
            )
        elif type(index) is _MutableBM25F:
            if not fielded:
                raise ValueError("legacy mutable BM25F kind differs from its chunks")
            cls._validate_old_mutable_fielded(
                index,
                chunks,
                feedback,
                title_weight=title_weight,
                idf_pow=idf_pow,
                next_slot=next_slot,
            )
        elif index is not None:
            raise ValueError("legacy mutable vector lexical index type is unsupported")

        materialized = lexical_state is not None
        payload = {
            "payload_version": _VECTOR_PAYLOAD_VERSION,
            "chunks": chunks,
            "next_slot": next_slot,
            "title_weight": title_weight,
            "lexical_weight": lexical_weight,
            "dense_weight": dense_weight,
            "pool": pool,
            "idf_pow": idf_pow,
            "feedback_rows": cls._feedback_rows(feedback),
            "revision": revision,
            "materialized": materialized,
            "lexical_state": lexical_state,
        }
        return cls._restore_payload_state(payload)
