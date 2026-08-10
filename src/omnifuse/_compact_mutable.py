"""Bounded mutable overlays for compact lexical snapshots."""

from __future__ import annotations

from bisect import bisect_left
import math
from numbers import Real
from typing import Callable, cast, Final, Iterable, Sequence

from ._compact_metadata import _LayeredMetadata
from ._compact_postings import (
    _DEFAULT_IDF_POW,
    _UINT64_MAX,
    CompactPostingsSnapshot,
    _doc_id,
    _merge_sorted_postings,
    _top_k_scores,
    _WeightCacheBuilder,
)
from .text import _analyze_query, _coordinate_query_scores


_STATE_VERSION: Final = 1
_VECTOR_STATE_VERSION: Final = 1
_VECTOR_PACKED_STATE_VERSION: Final = 2
_UNRESOLVED_TERM_ID = object()

# A plain record owns its field length and canonical term frequencies.
_PlainRecord = tuple[int, dict[str, int]]
_BaseRecordResolver = Callable[[int], _PlainRecord | None]


def _record_tokens(record: _PlainRecord):
    _length, counts = record
    for term in sorted(counts):
        for _occurrence in range(counts[term]):
            yield term


class CompactMutableBM25:
    """Exact BM25 with an immutable compact base and one bounded latest overlay.

    Base postings are never rewritten by a logical mutation.  A base document that is
    updated or deleted is suppressed by ``_overrides``; only the latest live replacement
    contributes to ``_delta_postings``.  Repeated churn therefore remains proportional to
    the currently overridden documents rather than the mutation history.
    """

    def __init__(
        self,
        docs: Iterable[tuple[int, Iterable[str]]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
    ):
        self._validate_config(k1, b, idf_pow)
        base = CompactPostingsSnapshot.from_bm25(docs, k1=k1, b=b, idf_pow=idf_pow)
        self._initialize(base, k1, b, idf_pow, None)

    @classmethod
    def _from_vector(
        cls,
        docs: Iterable[tuple[int, Iterable[str]]],
        resolver: _BaseRecordResolver,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
    ) -> "CompactMutableBM25":
        cls._validate_config(k1, b, idf_pow)
        if not callable(resolver):
            raise TypeError("base record resolver must be callable")
        base = CompactPostingsSnapshot._from_bm25_for_vector(
            docs, k1=k1, b=b, idf_pow=idf_pow
        )
        candidate = cls.__new__(cls)
        candidate._initialize(base, k1, b, idf_pow, resolver)
        return candidate

    def _initialize(
        self,
        base: CompactPostingsSnapshot,
        k1: float,
        b: float,
        idf_pow: float,
        resolver: _BaseRecordResolver | None,
    ) -> None:
        self.k1, self.b, self._idf_pow = k1, b, idf_pow
        self._base = base
        self._base_record_resolver = resolver
        self._base_record_cache: dict[int, _PlainRecord] = {}
        self._overrides: dict[int, _PlainRecord | None] = {}
        self._delta_postings: dict[str, dict[int, int]] = {}
        self._use_base_metadata()
        self.N = base.N
        self._total_len = int(base._totals[0])
        self._max_doc_id = base.max_doc_id
        self._weight_cache: dict[str, tuple[Sequence[int], Sequence[float]]] = {}
        self._mutation_version = 0
        self._layout_epoch = 0

    @staticmethod
    def _validate_config(k1: float, b: float, idf_pow: float) -> None:
        if any(not isinstance(value, Real) for value in (k1, b, idf_pow)):
            raise TypeError("compact scoring parameters must be real numbers")
        try:
            finite = all(math.isfinite(value) for value in (k1, b, idf_pow))
        except (OverflowError, TypeError) as exc:
            raise ValueError("compact scoring parameters must be finite") from exc
        if not finite:
            raise ValueError("compact scoring parameters must be finite")

    def _use_base_metadata(self) -> None:
        self._metadata = _LayeredMetadata(self._base, self._base._df)
        self._terms = self._metadata.terms
        self._df = self._metadata.df

    def _release_metadata_if_pristine(self) -> None:
        if self._overrides or self._delta_postings:
            return
        self.N = self._base.N
        self._total_len = int(self._base._totals[0])
        self._use_base_metadata()

    @property
    def avgdl(self) -> float:
        return self._total_len / self.N if self.N else 0.0

    @property
    def idf(self) -> dict[str, float]:
        return {term: self._idf(term) for term in self._terms if term in self._df}

    def _idf(self, term: str) -> float:
        resolved = self._metadata.resolve_live(term)
        if resolved is None or not resolved[2]:
            raise KeyError(term)
        n = resolved[2]
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** self._idf_pow

    def _base_record(self, doc_id: int) -> _PlainRecord | None:
        if self._base_record_resolver is not None:
            cached = self._base_record_cache.get(doc_id)
            if cached is not None:
                return cached
            if doc_id not in self._base._doc_positions:
                return None
            record = self._base_record_resolver(doc_id)
            if record is None:
                if doc_id in self._overrides and self._overrides[doc_id] is None:
                    return None
                raise RuntimeError(
                    "vector source lost a live forward-only base document"
                )
            return record
        record = self._base._document_record(doc_id)
        if record is None:
            return None
        lengths, counts_by_field = record
        return (
            lengths[0],
            {
                self._base.terms[term_id]: tf
                for term_id, tf in counts_by_field[0].items()
            },
        )

    def _current_record(self, doc_id: int) -> _PlainRecord | None:
        if doc_id in self._overrides:
            return self._overrides[doc_id]
        return self._base_record(doc_id)

    def _freeze(
        self, tokens: Iterable[str], staged_terms: dict[str, str]
    ) -> _PlainRecord:
        counts: dict[str, int] = {}
        length = 0
        for term in tokens:
            if type(term) is not str:
                raise TypeError("tokens must contain only str values")
            canonical = self._metadata.canonicalize(term, staged_terms)
            length += 1
            counts[canonical] = counts.get(canonical, 0) + 1
        return length, counts

    def _remove_stats(self, record: _PlainRecord) -> None:
        length, _counts = record
        self._total_len -= length

    def _add_stats(self, record: _PlainRecord) -> None:
        length, _counts = record
        self._total_len += length

    @staticmethod
    def _stage_stats(
        changes: dict[str, int], record: _PlainRecord, amount: int
    ) -> None:
        for term in record[1]:
            changes[term] = changes.get(term, 0) + amount
            if not changes[term]:
                del changes[term]

    def _remove_delta(self, doc_id: int, record: _PlainRecord) -> None:
        for term in record[1]:
            posting = self._delta_postings[term]
            del posting[doc_id]
            if not posting:
                del self._delta_postings[term]

    def _add_delta(self, doc_id: int, record: _PlainRecord) -> None:
        for term, tf in record[1].items():
            self._delta_postings.setdefault(term, {})[doc_id] = tf

    def _restores_pristine_base(
        self,
        changes: Sequence[
            tuple[int, _PlainRecord, _PlainRecord | None, _PlainRecord | None]
        ],
    ) -> bool:
        return len(changes) == len(self._overrides) and all(
            doc_id in self._overrides
            and base_record is not None
            and frozen == base_record
            for doc_id, frozen, _before, base_record in changes
        )

    def _commit_pristine_restore(self, next_max: int, count: int) -> int:
        self._overrides.clear()
        self._delta_postings.clear()
        self._base_record_cache.clear()
        self._max_doc_id = next_max
        self._release_metadata_if_pristine()
        self._weight_cache.clear()
        self._mutation_version += 1
        return count

    def upsert_many(self, docs) -> int:
        prepared: list[
            tuple[int, _PlainRecord, _PlainRecord | None, _PlainRecord | None]
        ] = []
        staged_terms: dict[str, str] = {}
        seen: set[int] = set()
        next_max = self._max_doc_id
        for raw_doc_id, tokens in docs:
            doc_id = _doc_id(raw_doc_id)
            if doc_id in seen:
                raise ValueError(f"duplicate batch doc_id {doc_id}")
            seen.add(doc_id)
            base_record = self._base_record(doc_id)
            before = (
                self._overrides[doc_id] if doc_id in self._overrides else base_record
            )
            if before is None:
                reserved_tombstone = (
                    doc_id in self._overrides and self._overrides[doc_id] is None
                )
                if doc_id <= next_max and not reserved_tombstone:
                    raise ValueError(
                        "new doc_id must be greater than every previously assigned id"
                    )
                next_max = max(next_max, doc_id)
            frozen = self._freeze(tokens, staged_terms)
            prepared.append((doc_id, frozen, before, base_record))

        changes = [item for item in prepared if item[1] != item[2]]
        if not changes:
            return 0

        projected_total = self._total_len + sum(
            frozen[0] - (before[0] if before is not None else 0)
            for _, frozen, before, _ in changes
        )
        if projected_total > _UINT64_MAX:
            raise ValueError("compact mutable total length exceeds its storage domain")
        if self._restores_pristine_base(changes):
            return self._commit_pristine_restore(next_max, len(changes))

        metadata_changes: dict[str, int] = {}
        for _changed_doc_id, frozen, before, _base_record in changes:
            if before is not None:
                self._stage_stats(metadata_changes, before, -1)
            self._stage_stats(metadata_changes, frozen, 1)
        metadata_patch = self._metadata.prepare_patch(
            metadata_changes, new_terms=staged_terms
        )
        for doc_id, frozen, before, base_record in changes:
            previous_override = self._overrides.get(doc_id)
            if doc_id in self._overrides and previous_override is not None:
                self._remove_delta(doc_id, previous_override)
            if before is None:
                self.N += 1
            else:
                self._remove_stats(before)
            self._add_stats(frozen)

            if base_record is not None and frozen == base_record:
                self._overrides.pop(doc_id, None)
            else:
                self._overrides[doc_id] = frozen
                self._add_delta(doc_id, frozen)

        self._max_doc_id = next_max
        self._metadata.commit_patch(metadata_patch)
        if self._base_record_resolver is not None:
            for doc_id, _frozen, _before, base_record in changes:
                if base_record is None:
                    continue
                if self._overrides.get(doc_id) is None:
                    self._base_record_cache.pop(doc_id, None)
                else:
                    self._base_record_cache.setdefault(doc_id, base_record)
        self._release_metadata_if_pristine()
        self._weight_cache.clear()
        self._mutation_version += 1
        return len(changes)

    def upsert(self, doc_id: int, tokens) -> bool:
        return bool(self.upsert_many(((doc_id, tokens),)))

    def delete_many(self, doc_ids) -> int:
        prepared: list[tuple[int, _PlainRecord, _PlainRecord | None]] = []
        seen: set[int] = set()
        for raw_doc_id in doc_ids:
            doc_id = _doc_id(raw_doc_id)
            if doc_id in seen:
                raise ValueError(f"duplicate batch doc_id {doc_id}")
            seen.add(doc_id)
            base_record = self._base_record(doc_id)
            before = (
                self._overrides[doc_id] if doc_id in self._overrides else base_record
            )
            if before is not None:
                prepared.append((doc_id, before, base_record))
        if not prepared:
            return 0

        metadata_changes: dict[str, int] = {}
        for _deleted_doc_id, before, _base_record in prepared:
            self._stage_stats(metadata_changes, before, -1)
        metadata_patch = self._metadata.prepare_patch(metadata_changes)
        for doc_id, before, _base_record in prepared:
            previous_override = self._overrides.get(doc_id)
            if doc_id in self._overrides and previous_override is not None:
                self._remove_delta(doc_id, previous_override)
            self._remove_stats(before)
            self.N -= 1
            # A tombstone is retained even for a delta-only document: its id remains
            # permanently reserved until and after physical compaction.
            self._overrides[doc_id] = None

        self._metadata.commit_patch(metadata_patch)
        if self._base_record_resolver is not None:
            for doc_id, _before, base_record in prepared:
                if base_record is not None:
                    self._base_record_cache.setdefault(doc_id, base_record)
        self._release_metadata_if_pristine()
        self._weight_cache.clear()
        self._mutation_version += 1
        return len(prepared)

    def delete(self, doc_id: int) -> bool:
        return bool(self.delete_many((doc_id,)))

    def _term_frequencies(
        self,
        term: str,
        base_term_id: int | None | object = _UNRESOLVED_TERM_ID,
    ) -> Iterable[tuple[int, int]]:
        if base_term_id is _UNRESOLVED_TERM_ID:
            base_term_id = self._base._term_ids.get(term)
        base_term_id = cast(int | None, base_term_id)
        base = (
            (
                (doc_id, values[0])
                for doc_id, values in self._base._decode_posting(base_term_id)
                if doc_id not in self._overrides
            )
            if base_term_id is not None
            else ()
        )
        return _merge_sorted_postings(base, self._delta_postings.get(term))

    def _record_length(self, doc_id: int) -> int:
        if doc_id in self._overrides:
            record = self._overrides[doc_id]
            if record is None:
                raise RuntimeError("tombstoned document reached live scoring")
            return record[0]
        position = self._base._doc_positions[doc_id]
        return self._base._lengths[0][position]

    def _term_weights(self, term: str) -> tuple[Sequence[int], Sequence[float]]:
        cached = self._weight_cache.get(term)
        if cached is not None:
            return cached
        resolved = self._metadata.resolve_live(term)
        if resolved is None or not resolved[2]:
            return (), ()
        canonical, base_term_id, df, _dfe = resolved

        avg = self.avgdl or 1.0
        idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5)) ** self._idf_pow
        k1p1 = self.k1 + 1
        builder = _WeightCacheBuilder()
        for doc_id, tf in self._term_frequencies(canonical, base_term_id):
            doc_len = self._record_length(doc_id)
            norm = self.k1 * (1 - self.b + self.b * (doc_len or 1) / avg)
            weight = idf * k1p1 * tf / (tf + norm)
            builder.append(doc_id, weight)
        result = builder.finish()
        self._weight_cache[canonical] = result
        return result

    def score_tokens(self, tokens: Iterable[str], doc_id: int) -> float:
        doc_id = _doc_id(doc_id)
        score = 0.0
        for term in tokens:
            ids, weights = self._term_weights(term)
            position = bisect_left(ids, doc_id)
            if position < len(ids) and ids[position] == doc_id:
                score += weights[position]
        return score

    def score(self, tokens: list[str], doc_id: int) -> float:
        return self.score_tokens(tokens, doc_id)

    def search_tokens(
        self,
        tokens: Iterable[str],
        *,
        limit: int = 20,
        anchors: frozenset[str] = frozenset(),
        restricted: bool = False,
        recover_partial_outlier: bool = False,
    ) -> list[tuple[int, float]]:
        query_counts: dict[str, int] = {}
        for term in tokens:
            query_counts[term] = query_counts.get(term, 0) + 1
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        for term, query_count in query_counts.items():
            ids, weights = self._term_weights(term)
            if restricted and term in anchors:
                candidates.update(ids)
            if term in anchors and term.startswith("#"):
                complete_candidates.update(ids)
            if query_count == 1:
                for doc_id, weight in zip(ids, weights):
                    scores[doc_id] = scores.get(doc_id, 0.0) + weight
            else:
                for doc_id, weight in zip(ids, weights):
                    scores[doc_id] = scores.get(doc_id, 0.0) + query_count * weight
        scores = _coordinate_query_scores(
            scores,
            candidates,
            complete_candidates,
            recover_partial_outlier=recover_partial_outlier,
        )
        return _top_k_scores(scores, limit)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        recover_partial_outlier: bool = False,
    ) -> list[tuple[int, float]]:
        analysis = _analyze_query(query)
        return self.search_tokens(
            analysis.terms,
            limit=limit,
            anchors=analysis.anchors,
            restricted=analysis.restricted,
            recover_partial_outlier=recover_partial_outlier,
        )

    def _live_records(self):
        doc_ids = set(self._base.doc_ids)
        doc_ids.update(self._overrides)
        for doc_id in sorted(doc_ids):
            record = self._current_record(doc_id)
            if record is not None:
                yield doc_id, record

    def compact(self) -> bool:
        """Absorb the latest overlay without changing logical state or warm caches."""
        if not self._overrides:
            return False

        def documents():
            for doc_id, record in self._live_records():
                yield doc_id, _record_tokens(record)

        factory = (
            CompactPostingsSnapshot.from_bm25
            if self._base_record_resolver is None
            else CompactPostingsSnapshot._from_bm25_for_vector
        )
        candidate = factory(
            documents(),
            k1=self.k1,
            b=self.b,
            idf_pow=self._idf_pow,
            max_doc_id=self._max_doc_id,
        )
        self._base = candidate
        self._overrides = {}
        self._delta_postings = {}
        self._base_record_cache.clear()
        self._use_base_metadata()
        self.N = candidate.N
        self._total_len = int(candidate._totals[0])
        self._layout_epoch += 1
        return True

    def storage_stats(self) -> dict[str, int]:
        return {
            "base_bytes": self._base.storage_nbytes(),
            "base_docs": self._base.N,
            "override_docs": len(self._overrides),
            "live_overrides": sum(
                record is not None for record in self._overrides.values()
            ),
            "tombstones": sum(record is None for record in self._overrides.values()),
            "delta_terms": len(self._delta_postings),
            "delta_postings": sum(
                len(posting) for posting in self._delta_postings.values()
            ),
            "layout_epoch": self._layout_epoch,
            **self._metadata.sparse_counts(),
        }

    def _vector_state(self) -> dict:
        return {
            "state_version": _VECTOR_STATE_VERSION,
            "k1": self.k1,
            "b": self.b,
            "idf_pow": self._idf_pow,
            "base_state": self._base._export_forward_state(),
            "overrides": {
                doc_id: (None if record is None else (record[0], dict(record[1])))
                for doc_id, record in self._overrides.items()
            },
            "max_doc_id": self._max_doc_id,
            "mutation_version": self._mutation_version,
            "layout_epoch": self._layout_epoch,
        }

    def _vector_packed_state(self) -> dict:
        return {
            "state_version": _VECTOR_PACKED_STATE_VERSION,
            "k1": self.k1,
            "b": self.b,
            "idf_pow": self._idf_pow,
            "base_state": self._base._export_packed_forward_state(),
            "overrides": self._overrides,
            "max_doc_id": self._max_doc_id,
            "mutation_version": self._mutation_version,
            "layout_epoch": self._layout_epoch,
        }

    @classmethod
    def _from_vector_packed_state(
        cls,
        state: dict,
        resolver: _BaseRecordResolver,
        *,
        expected_doc_ids: Iterable[int],
    ) -> "CompactMutableBM25":
        expected = {
            "state_version",
            "k1",
            "b",
            "idf_pow",
            "base_state",
            "overrides",
            "max_doc_id",
            "mutation_version",
            "layout_epoch",
        }
        if (
            type(state) is not dict
            or type(state.get("state_version")) is not int
            or state.get("state_version") != _VECTOR_PACKED_STATE_VERSION
            or set(state) != expected
            or not callable(resolver)
        ):
            raise ValueError("unsupported packed vector mutable BM25 state")
        raw_overrides = state.get("overrides")
        if type(raw_overrides) is not dict:
            raise ValueError("packed vector mutable BM25 overrides must be a dict")
        expected_ids = tuple(expected_doc_ids)
        base, captured = CompactPostingsSnapshot._from_packed_forward_state(
            state.get("base_state"),
            capture_doc_ids=raw_overrides,
        )
        base_records = {
            doc_id: (record[0][0], record[1][0]) for doc_id, record in captured.items()
        }
        restored = {
            "state_version": _STATE_VERSION,
            "k1": state.get("k1"),
            "b": state.get("b"),
            "idf_pow": state.get("idf_pow"),
            "base": base,
            "overrides": raw_overrides,
            "max_doc_id": state.get("max_doc_id"),
            "mutation_version": state.get("mutation_version"),
            "layout_epoch": state.get("layout_epoch"),
        }
        candidate = cls._from_state(
            restored,
            _vector_resolver=resolver,
            _captured_base_records=base_records,
        )
        candidate._validate_vector_source(resolver, expected_ids, base_records)
        return candidate

    @classmethod
    def _from_vector_state(
        cls,
        state: dict,
        resolver: _BaseRecordResolver,
        *,
        expected_doc_ids: Iterable[int],
    ) -> "CompactMutableBM25":
        expected = {
            "state_version",
            "k1",
            "b",
            "idf_pow",
            "base_state",
            "overrides",
            "max_doc_id",
            "mutation_version",
            "layout_epoch",
        }
        if (
            type(state) is not dict
            or type(state.get("state_version")) is not int
            or state.get("state_version") != _VECTOR_STATE_VERSION
            or set(state) != expected
            or not callable(resolver)
        ):
            raise ValueError("unsupported vector mutable BM25 state")
        raw_overrides = state.get("overrides")
        if type(raw_overrides) is not dict:
            raise ValueError("vector mutable BM25 overrides must be a dict")
        expected_ids = tuple(expected_doc_ids)
        base, captured = CompactPostingsSnapshot._from_forward_state(
            state.get("base_state"),
            capture_doc_ids=raw_overrides,
        )
        base_records = {
            doc_id: (record[0][0], record[1][0]) for doc_id, record in captured.items()
        }
        restored = {
            "state_version": _STATE_VERSION,
            "k1": state.get("k1"),
            "b": state.get("b"),
            "idf_pow": state.get("idf_pow"),
            "base": base,
            "overrides": raw_overrides,
            "max_doc_id": state.get("max_doc_id"),
            "mutation_version": state.get("mutation_version"),
            "layout_epoch": state.get("layout_epoch"),
        }
        candidate = cls._from_state(
            restored,
            _vector_resolver=resolver,
            _captured_base_records=base_records,
        )
        candidate._validate_vector_source(resolver, expected_ids, base_records)
        return candidate

    def _validate_vector_source(
        self,
        resolver: _BaseRecordResolver,
        expected_doc_ids: Iterable[int],
        base_records: dict[int, _PlainRecord] | None = None,
    ) -> None:
        expected = {_doc_id(doc_id) for doc_id in expected_doc_ids}
        live = set(self._base.doc_ids)
        for doc_id, record in self._overrides.items():
            if record is None:
                live.discard(doc_id)
            else:
                live.add(doc_id)
        if live != expected:
            raise ValueError("vector mutable BM25 membership differs from its chunks")
        if base_records is not None:
            self._base._validate_vector_source_records(resolver, base_records)
        for doc_id in expected:
            if doc_id in self._overrides:
                record = self._overrides[doc_id]
            elif base_records is None:
                record = self._current_record(doc_id)
            else:
                continue
            if record != resolver(doc_id):
                raise ValueError("vector mutable BM25 record differs from its chunk")

    def __getstate__(self) -> dict:
        if self._base_record_resolver is not None:
            raise TypeError("vector-owned mutable indexes are derived state")
        return {
            "state_version": _STATE_VERSION,
            "k1": self.k1,
            "b": self.b,
            "idf_pow": self._idf_pow,
            "base": self._base,
            "overrides": self._overrides,
            "max_doc_id": self._max_doc_id,
            "mutation_version": self._mutation_version,
            "layout_epoch": self._layout_epoch,
        }

    def __setstate__(self, state: dict) -> None:
        candidate = self._from_state(state)
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    @classmethod
    def _from_state(
        cls,
        state: dict,
        *,
        _vector_resolver: _BaseRecordResolver | None = None,
        _captured_base_records: dict[int, _PlainRecord] | None = None,
    ) -> "CompactMutableBM25":
        if type(state) is not dict or state.get("state_version") != _STATE_VERSION:
            raise ValueError("unsupported compact mutable BM25 state")
        try:
            k1 = state["k1"]
            b = state["b"]
            idf_pow = state["idf_pow"]
            base = state["base"]
            raw_overrides = state["overrides"]
            max_doc_id = state["max_doc_id"]
            mutation_version = state["mutation_version"]
            layout_epoch = state["layout_epoch"]
        except KeyError as exc:
            raise ValueError("invalid compact mutable BM25 state") from exc
        cls._validate_config(k1, b, idf_pow)
        if not isinstance(base, CompactPostingsSnapshot) or base.mode != "bm25":
            raise ValueError("compact mutable BM25 base is invalid")
        if (base.k1, base.b, base.idf_pow) != (k1, b, idf_pow):
            raise ValueError("compact mutable BM25 configuration differs from its base")
        if _vector_resolver is None:
            base._validate_storage()
        elif not callable(_vector_resolver):
            raise ValueError("vector mutable BM25 resolver is invalid")
        if type(raw_overrides) is not dict:
            raise ValueError("compact mutable BM25 overrides must be a dict")
        if type(max_doc_id) is not int or max_doc_id < base.max_doc_id:
            raise ValueError("compact mutable BM25 high-water mark is invalid")
        if (
            type(mutation_version) is not int
            or mutation_version < 0
            or type(layout_epoch) is not int
            or layout_epoch < 0
        ):
            raise ValueError("compact mutable BM25 versions are invalid")

        parsed: dict[int, _PlainRecord | None] = {}
        for raw_doc_id, raw_record in raw_overrides.items():
            doc_id = _doc_id(raw_doc_id)
            if doc_id > max_doc_id:
                raise ValueError("compact mutable override exceeds its high-water mark")
            if doc_id not in base._doc_positions and doc_id <= base.max_doc_id:
                raise ValueError(
                    "compact mutable override id is not a base or delta id"
                )
            if raw_record is None:
                parsed[doc_id] = None
                continue
            if (
                type(raw_record) is not tuple
                or len(raw_record) != 2
                or type(raw_record[0]) is not int
                or raw_record[0] < 0
                or raw_record[0] > _UINT64_MAX
                or type(raw_record[1]) is not dict
            ):
                raise ValueError("compact mutable override record is invalid")
            counts: dict[str, int] = {}
            for term, tf in raw_record[1].items():
                if (
                    type(term) is not str
                    or type(tf) is not int
                    or tf <= 0
                    or tf > _UINT64_MAX
                ):
                    raise ValueError("compact mutable override frequencies are invalid")
                counts[term] = tf
            if sum(counts.values()) != raw_record[0]:
                raise ValueError("compact mutable override length is invalid")
            parsed[doc_id] = raw_record[0], counts

        candidate = cls.__new__(cls)
        candidate.k1, candidate.b, candidate._idf_pow = k1, b, idf_pow
        candidate._base = base
        candidate._base_record_resolver = _vector_resolver
        candidate._base_record_cache = dict(_captured_base_records or {})
        candidate._overrides = parsed
        candidate._delta_postings = {}
        if not parsed:
            candidate._use_base_metadata()
            candidate.N = base.N
            candidate._total_len = int(base._totals[0])
            candidate._max_doc_id = max_doc_id
            candidate._weight_cache = {}
            candidate._mutation_version = mutation_version
            candidate._layout_epoch = layout_epoch
            candidate._base_record_cache.clear()
            return candidate

        candidate._use_base_metadata()
        staged_terms: dict[str, str] = {}
        for record in parsed.values():
            if record is not None:
                for term in record[1]:
                    candidate._metadata.canonicalize(term, staged_terms)
        # Canonicalize override records before rebuilding their derived postings.
        candidate._overrides = {
            doc_id: (
                None
                if record is None
                else (
                    record[0],
                    {
                        candidate._metadata.canonicalize(term, staged_terms): tf
                        for term, tf in record[1].items()
                    },
                )
            )
            for doc_id, record in parsed.items()
        }
        candidate.N = base.N
        candidate._total_len = int(base._totals[0])
        metadata_changes: dict[str, int] = {}
        for doc_id, record in candidate._overrides.items():
            base_record = candidate._base_record(doc_id)
            if base_record is not None:
                candidate.N -= 1
                candidate._remove_stats(base_record)
                candidate._stage_stats(metadata_changes, base_record, -1)
            if record is None:
                continue
            candidate.N += 1
            candidate._add_stats(record)
            candidate._stage_stats(metadata_changes, record, 1)
            candidate._add_delta(doc_id, record)
        if candidate._total_len > _UINT64_MAX:
            raise ValueError("compact mutable total length exceeds its storage domain")
        metadata_patch = candidate._metadata.prepare_patch(
            metadata_changes, new_terms=staged_terms
        )
        candidate._metadata.commit_patch(metadata_patch)
        candidate._max_doc_id = max_doc_id
        candidate._weight_cache = {}
        candidate._mutation_version = mutation_version
        candidate._layout_epoch = layout_epoch
        if _vector_resolver is not None:
            candidate._base_record_cache = {
                doc_id: record
                for doc_id, record in candidate._base_record_cache.items()
                if doc_id in candidate._overrides
            }
        return candidate
