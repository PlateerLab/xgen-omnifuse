from __future__ import annotations

from bisect import bisect_left
import math
from numbers import Real
from typing import Callable, Final, Iterable, Mapping, Sequence

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
_FieldRecord = tuple[tuple[int, ...], tuple[dict[str, int], ...]]
_BaseRecordResolver = Callable[[int], _FieldRecord | None]


def _record_fields(
    record: _FieldRecord, fields: tuple[str, ...]
) -> dict[str, Iterable[str]]:
    def tokens(counts: dict[str, int]):
        for term in sorted(counts):
            for _occurrence in range(counts[term]):
                yield term

    return {
        field: tokens(counts) for field, counts in zip(fields, record[1], strict=True)
    }


class CompactMutableBM25F:
    """Exact BM25F backed by a compact base and one bounded latest overlay.

    Updating or deleting a base document suppresses that document's immutable posting.
    The delta contains only each document's latest live replacement, so repeated churn is
    bounded by current logical state instead of growing with mutation history.
    """

    def __init__(
        self,
        docs: Iterable[tuple[int, Mapping[str, Iterable[str]]]],
        weights: Mapping[str, float],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
        evidence_fields: Iterable[str] | None = None,
    ):
        weight_map = dict(weights)
        evidence_names = frozenset(evidence_fields or ())
        self._validate_config(weight_map, evidence_names, k1, b, idf_pow)
        base = CompactPostingsSnapshot.from_bm25f(
            docs,
            weight_map,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            evidence_fields=evidence_names,
        )
        self._initialize(base, weight_map, evidence_names, k1, b, idf_pow, None)

    @classmethod
    def _from_vector(
        cls,
        docs: Iterable[tuple[int, Mapping[str, Iterable[str]]]],
        weights: Mapping[str, float],
        resolver: _BaseRecordResolver,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
        evidence_fields: Iterable[str] | None = None,
    ) -> "CompactMutableBM25F":
        weight_map = dict(weights)
        evidence_names = frozenset(evidence_fields or ())
        cls._validate_config(weight_map, evidence_names, k1, b, idf_pow)
        if not callable(resolver):
            raise TypeError("base record resolver must be callable")
        base = CompactPostingsSnapshot._from_bm25f_for_vector(
            docs,
            weight_map,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            evidence_fields=evidence_names,
        )
        candidate = cls.__new__(cls)
        candidate._initialize(
            base, weight_map, evidence_names, k1, b, idf_pow, resolver
        )
        return candidate

    def _initialize(
        self,
        base: CompactPostingsSnapshot,
        weight_map: dict[str, float],
        evidence_names: frozenset[str],
        k1: float,
        b: float,
        idf_pow: float,
        resolver: _BaseRecordResolver | None,
    ) -> None:
        self.k1, self.b, self._idf_pow = k1, b, idf_pow
        self.fields = list(base.fields)
        self.w = weight_map
        self._fw = list(base.weights)
        self.evidence_fields = evidence_names
        self._is_ev = list(base.evidence)
        self._base = base
        self._base_record_resolver = resolver
        self._base_record_cache: dict[int, _FieldRecord] = {}
        self._overrides: dict[int, _FieldRecord | None] = {}
        self._delta_postings: dict[str, dict[int, tuple[int, ...]]] = {}
        self._use_base_metadata()
        self.N = base.N
        self._totals = [int(total) for total in base._totals]
        self._max_doc_id = base.max_doc_id
        self._weight_cache: dict[str, tuple[Sequence[int], Sequence[float]]] = {}
        self._mutation_version = 0
        self._layout_epoch = 0

    @staticmethod
    def _validate_config(
        weights: Mapping[str, float],
        evidence_fields: frozenset[str],
        k1: float,
        b: float,
        idf_pow: float,
    ) -> None:
        if any(type(field) is not str for field in weights):
            raise ValueError("compact fields must be strings")
        if any(type(field) is not str for field in evidence_fields):
            raise ValueError("compact evidence fields must be strings")
        if any(not isinstance(weight, Real) for weight in weights.values()):
            raise TypeError("compact field weights must be real numbers")
        if any(not isinstance(value, Real) for value in (k1, b, idf_pow)):
            raise TypeError("compact scoring parameters must be real numbers")
        try:
            finite = all(math.isfinite(weight) for weight in weights.values()) and all(
                math.isfinite(value) for value in (k1, b, idf_pow)
            )
        except (OverflowError, TypeError) as exc:
            raise ValueError("compact scoring values must be finite") from exc
        if not finite:
            raise ValueError("compact scoring values must be finite")

    def _use_base_metadata(self) -> None:
        self._metadata = _LayeredMetadata(self._base, self._base._df, self._base._dfe)
        self._terms = self._metadata.terms
        self._df = self._metadata.df
        assert self._metadata.dfe is not None
        self._dfe = self._metadata.dfe

    def _release_metadata_if_pristine(self) -> None:
        if self._overrides or self._delta_postings:
            return
        self.N = self._base.N
        self._totals = [int(total) for total in self._base._totals]
        self._use_base_metadata()

    @property
    def avglen(self) -> dict[str, float]:
        return {
            field: (self._totals[index] / self.N if self.N else 0.0)
            for index, field in enumerate(self.fields)
        }

    @property
    def idf(self) -> dict[str, float]:
        return {
            term: self._idf(term)
            for term in self._terms
            if term in self._df or term in self._dfe
        }

    def _idf(self, term: str) -> float:
        resolved = self._metadata.resolve_live(term)
        if resolved is None or not (resolved[2] or resolved[3]):
            raise KeyError(term)
        n = resolved[2] or resolved[3]
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** self._idf_pow

    def _base_record(self, doc_id: int) -> _FieldRecord | None:
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
        lengths, raw_counts = record
        return lengths, tuple(
            {self._base.terms[term_id]: tf for term_id, tf in counts.items()}
            for counts in raw_counts
        )

    def _current_record(self, doc_id: int) -> _FieldRecord | None:
        if doc_id in self._overrides:
            return self._overrides[doc_id]
        return self._base_record(doc_id)

    def _freeze(
        self,
        raw_fields: Mapping[str, Iterable[str]],
        staged_terms: dict[str, str],
    ) -> _FieldRecord:
        if not isinstance(raw_fields, Mapping):
            raise TypeError("BM25F documents must be field mappings")
        lengths: list[int] = []
        counts_by_field: list[dict[str, int]] = []
        for field in self.fields:
            counts: dict[str, int] = {}
            length = 0
            for term in raw_fields.get(field, ()):
                if type(term) is not str:
                    raise TypeError("tokens must contain only str values")
                canonical = self._metadata.canonicalize(term, staged_terms)
                length += 1
                counts[canonical] = counts.get(canonical, 0) + 1
            lengths.append(length)
            counts_by_field.append(counts)
        return tuple(lengths), tuple(counts_by_field)

    def _presence(self, record: _FieldRecord) -> tuple[set[str], set[str]]:
        content: set[str] = set()
        evidence: set[str] = set()
        for index, counts in enumerate(record[1]):
            (evidence if self._is_ev[index] else content).update(counts)
        return content, evidence

    def _record_terms(self, record: _FieldRecord) -> set[str]:
        content, evidence = self._presence(record)
        return content | evidence

    def _remove_stats(self, record: _FieldRecord) -> None:
        lengths, _counts_by_field = record
        for index, length in enumerate(lengths):
            self._totals[index] -= length

    def _add_stats(self, record: _FieldRecord) -> None:
        lengths, _counts_by_field = record
        for index, length in enumerate(lengths):
            self._totals[index] += length

    def _stage_stats(
        self,
        df_changes: dict[str, int],
        dfe_changes: dict[str, int],
        record: _FieldRecord,
        amount: int,
    ) -> None:
        content, evidence = self._presence(record)
        for term in content:
            df_changes[term] = df_changes.get(term, 0) + amount
            if not df_changes[term]:
                del df_changes[term]
        for term in evidence:
            dfe_changes[term] = dfe_changes.get(term, 0) + amount
            if not dfe_changes[term]:
                del dfe_changes[term]

    def _remove_delta(self, doc_id: int, record: _FieldRecord) -> None:
        for term in self._record_terms(record):
            posting = self._delta_postings[term]
            del posting[doc_id]
            if not posting:
                del self._delta_postings[term]

    def _add_delta(self, doc_id: int, record: _FieldRecord) -> None:
        frequencies: dict[str, list[int]] = {}
        for index, counts in enumerate(record[1]):
            for term, tf in counts.items():
                frequencies.setdefault(term, [0] * len(self.fields))[index] = tf
        for term, values in frequencies.items():
            self._delta_postings.setdefault(term, {})[doc_id] = tuple(values)

    def _restores_pristine_base(
        self,
        changes: Sequence[
            tuple[int, _FieldRecord, _FieldRecord | None, _FieldRecord | None]
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

    def _validate_projected_totals(
        self,
        changes: list[
            tuple[int, _FieldRecord, _FieldRecord | None, _FieldRecord | None]
        ],
    ) -> None:
        projected = list(self._totals)
        for _document_id, frozen, before, _base_record in changes:
            for index, length in enumerate(frozen[0]):
                projected[index] += length - (before[0][index] if before else 0)
        if any(total > _UINT64_MAX for total in projected):
            raise ValueError("compact mutable field total exceeds its storage domain")

    def upsert_many(self, docs) -> int:
        prepared: list[
            tuple[int, _FieldRecord, _FieldRecord | None, _FieldRecord | None]
        ] = []
        staged_terms: dict[str, str] = {}
        seen: set[int] = set()
        next_max = self._max_doc_id
        for raw_doc_id, fields in docs:
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
            frozen = self._freeze(fields, staged_terms)
            prepared.append((doc_id, frozen, before, base_record))

        changes = [item for item in prepared if item[1] != item[2]]
        if not changes:
            return 0
        self._validate_projected_totals(changes)
        if self._restores_pristine_base(changes):
            return self._commit_pristine_restore(next_max, len(changes))

        df_changes: dict[str, int] = {}
        dfe_changes: dict[str, int] = {}
        for _changed_doc_id, frozen, before, _base_record in changes:
            if before is not None:
                self._stage_stats(df_changes, dfe_changes, before, -1)
            self._stage_stats(df_changes, dfe_changes, frozen, 1)
        metadata_patch = self._metadata.prepare_patch(
            df_changes, dfe_changes, new_terms=staged_terms
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

    def upsert(self, doc_id: int, fields: Mapping[str, Iterable[str]]) -> bool:
        return bool(self.upsert_many(((doc_id, fields),)))

    def delete_many(self, doc_ids) -> int:
        prepared: list[tuple[int, _FieldRecord, _FieldRecord | None]] = []
        seen: set[int] = set()
        for raw_doc_id in doc_ids:
            doc_id = _doc_id(raw_doc_id)
            if doc_id in seen:
                raise ValueError(f"duplicate batch doc_id {doc_id}")
            seen.add(doc_id)
            base_record = self._base_record(doc_id)
            before = (
                self._overrides[doc_id]
                if doc_id in self._overrides
                else base_record
            )
            if before is not None:
                prepared.append((doc_id, before, base_record))
        if not prepared:
            return 0

        df_changes: dict[str, int] = {}
        dfe_changes: dict[str, int] = {}
        for _deleted_doc_id, before, _base_record in prepared:
            self._stage_stats(df_changes, dfe_changes, before, -1)
        metadata_patch = self._metadata.prepare_patch(df_changes, dfe_changes)
        for doc_id, before, _base_record in prepared:
            previous_override = self._overrides.get(doc_id)
            if doc_id in self._overrides and previous_override is not None:
                self._remove_delta(doc_id, previous_override)
            self._remove_stats(before)
            self.N -= 1
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
        self, term: str, base_term_id: int | None
    ) -> Iterable[tuple[int, tuple[int, ...]]]:
        base = (
            (
                (doc_id, values)
                for doc_id, values in self._base._decode_posting(base_term_id)
                if doc_id not in self._overrides
            )
            if base_term_id is not None
            else ()
        )
        return _merge_sorted_postings(base, self._delta_postings.get(term))

    def _term_weights(self, term: str) -> tuple[Sequence[int], Sequence[float]]:
        cached = self._weight_cache.get(term)
        if cached is not None:
            return cached
        resolved = self._metadata.resolve_live(term)
        if resolved is None or not (resolved[2] or resolved[3]):
            return (), ()
        canonical, base_term_id, df, dfe = resolved

        avgl = [
            (self._totals[index] / self.N if self.N else 0.0) or 1
            for index in range(len(self.fields))
        ]
        n = df or dfe
        idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** self._idf_pow
        k1p1 = self.k1 + 1
        builder = _WeightCacheBuilder()
        overrides = self._overrides
        base_lengths = self._base._lengths
        doc_positions = self._base._doc_positions
        for doc_id, term_frequencies in self._term_frequencies(canonical, base_term_id):
            if doc_id in overrides:
                record = overrides[doc_id]
                if record is None:
                    raise RuntimeError("tombstoned document reached live scoring")
                lengths = record[0]
                doc_position = None
            else:
                lengths = None
                doc_position = doc_positions[doc_id]

            tfw = 0.0
            for index, tf in enumerate(term_frequencies):
                if not tf:
                    continue
                if self._is_ev[index]:
                    norm = 1.0
                else:
                    doc_length = (
                        lengths[index]
                        if lengths is not None
                        else base_lengths[index][doc_position]
                    )
                    norm = 1 - self.b + self.b * (doc_length or 1) / avgl[index]
                tfw = tfw + self._fw[index] * tf / norm
            if not tfw:
                continue
            weight = idf * tfw * k1p1 / (self.k1 + tfw)
            builder.append(doc_id, weight)
        result = builder.finish()
        self._weight_cache[canonical] = result
        return result

    def score_tokens(self, tokens: Iterable[str], doc_id: int) -> float:
        doc_id = _doc_id(doc_id)
        score = 0.0
        for term in dict.fromkeys(tokens):
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
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        for term in dict.fromkeys(tokens):
            ids, weights = self._term_weights(term)
            if restricted and term in anchors:
                candidates.update(ids)
            if term in anchors and term.startswith("#"):
                complete_candidates.update(ids)
            for doc_id, weight in zip(ids, weights):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight
        scores = _coordinate_query_scores(scores, candidates, complete_candidates)
        return _top_k_scores(scores, limit)

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        analysis = _analyze_query(query)
        return self.search_tokens(
            analysis.terms,
            limit=limit,
            anchors=analysis.anchors,
            restricted=analysis.restricted,
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

        fields = tuple(self.fields)

        def documents():
            for doc_id, record in self._live_records():
                yield doc_id, _record_fields(record, fields)

        factory = (
            CompactPostingsSnapshot.from_bm25f
            if self._base_record_resolver is None
            else CompactPostingsSnapshot._from_bm25f_for_vector
        )
        candidate = factory(
            documents(),
            self.w,
            k1=self.k1,
            b=self.b,
            idf_pow=self._idf_pow,
            evidence_fields=self.evidence_fields,
            max_doc_id=self._max_doc_id,
        )
        self._base = candidate
        self._overrides = {}
        self._delta_postings = {}
        self._base_record_cache.clear()
        self._use_base_metadata()
        self.N = candidate.N
        self._totals = [int(total) for total in candidate._totals]
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
            "weights": dict(self.w),
            "evidence_fields": self.evidence_fields,
            "base_state": self._base._export_forward_state(),
            "overrides": {
                doc_id: (
                    None
                    if record is None
                    else (
                        tuple(record[0]),
                        tuple(dict(counts) for counts in record[1]),
                    )
                )
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
            "weights": self.w,
            "evidence_fields": self.evidence_fields,
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
    ) -> "CompactMutableBM25F":
        expected = {
            "state_version",
            "k1",
            "b",
            "idf_pow",
            "weights",
            "evidence_fields",
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
            raise ValueError("unsupported packed vector mutable BM25F state")
        raw_overrides = state.get("overrides")
        if type(raw_overrides) is not dict:
            raise ValueError("packed vector mutable BM25F overrides must be a dict")
        expected_ids = tuple(expected_doc_ids)
        base, captured = CompactPostingsSnapshot._from_packed_forward_state(
            state.get("base_state"),
            capture_doc_ids=raw_overrides,
        )
        restored = {
            "state_version": _STATE_VERSION,
            "k1": state.get("k1"),
            "b": state.get("b"),
            "idf_pow": state.get("idf_pow"),
            "weights": state.get("weights"),
            "evidence_fields": state.get("evidence_fields"),
            "base": base,
            "overrides": raw_overrides,
            "max_doc_id": state.get("max_doc_id"),
            "mutation_version": state.get("mutation_version"),
            "layout_epoch": state.get("layout_epoch"),
        }
        candidate = cls._from_state(
            restored,
            _vector_resolver=resolver,
            _captured_base_records={
                doc_id: captured[doc_id]
                for doc_id in raw_overrides
                if doc_id in captured
            },
        )
        candidate._validate_vector_source(resolver, expected_ids, captured)
        return candidate

    @classmethod
    def _from_vector_state(
        cls,
        state: dict,
        resolver: _BaseRecordResolver,
        *,
        expected_doc_ids: Iterable[int],
    ) -> "CompactMutableBM25F":
        expected = {
            "state_version",
            "k1",
            "b",
            "idf_pow",
            "weights",
            "evidence_fields",
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
            raise ValueError("unsupported vector mutable BM25F state")
        raw_overrides = state.get("overrides")
        if type(raw_overrides) is not dict:
            raise ValueError("vector mutable BM25F overrides must be a dict")
        expected_ids = tuple(expected_doc_ids)
        base, captured = CompactPostingsSnapshot._from_forward_state(
            state.get("base_state"),
            capture_doc_ids=raw_overrides,
        )
        restored = {
            "state_version": _STATE_VERSION,
            "k1": state.get("k1"),
            "b": state.get("b"),
            "idf_pow": state.get("idf_pow"),
            "weights": state.get("weights"),
            "evidence_fields": state.get("evidence_fields"),
            "base": base,
            "overrides": raw_overrides,
            "max_doc_id": state.get("max_doc_id"),
            "mutation_version": state.get("mutation_version"),
            "layout_epoch": state.get("layout_epoch"),
        }
        candidate = cls._from_state(
            restored,
            _vector_resolver=resolver,
            _captured_base_records={
                doc_id: captured[doc_id]
                for doc_id in raw_overrides
                if doc_id in captured
            },
        )
        candidate._validate_vector_source(resolver, expected_ids, captured)
        return candidate

    def _validate_vector_source(
        self,
        resolver: _BaseRecordResolver,
        expected_doc_ids: Iterable[int],
        base_records: dict[int, _FieldRecord] | None = None,
    ) -> None:
        expected = {_doc_id(doc_id) for doc_id in expected_doc_ids}
        live = set(self._base.doc_ids)
        for doc_id, record in self._overrides.items():
            if record is None:
                live.discard(doc_id)
            else:
                live.add(doc_id)
        if live != expected:
            raise ValueError("vector mutable BM25F membership differs from its chunks")
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
                raise ValueError("vector mutable BM25F record differs from its chunk")

    def __getstate__(self) -> dict:
        if self._base_record_resolver is not None:
            raise TypeError("vector-owned mutable indexes are derived state")
        return {
            "state_version": _STATE_VERSION,
            "k1": self.k1,
            "b": self.b,
            "idf_pow": self._idf_pow,
            "weights": self.w,
            "evidence_fields": self.evidence_fields,
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
        _captured_base_records: dict[int, _FieldRecord] | None = None,
    ) -> "CompactMutableBM25F":
        if type(state) is not dict or state.get("state_version") != _STATE_VERSION:
            raise ValueError("unsupported compact mutable BM25F state")
        try:
            k1 = state["k1"]
            b = state["b"]
            idf_pow = state["idf_pow"]
            raw_weights = state["weights"]
            evidence_fields = state["evidence_fields"]
            base = state["base"]
            raw_overrides = state["overrides"]
            max_doc_id = state["max_doc_id"]
            mutation_version = state["mutation_version"]
            layout_epoch = state["layout_epoch"]
        except KeyError as exc:
            raise ValueError("invalid compact mutable BM25F state") from exc
        if type(raw_weights) is not dict or type(evidence_fields) is not frozenset:
            raise ValueError("compact mutable BM25F field configuration is invalid")
        weights = dict(raw_weights)
        cls._validate_config(weights, evidence_fields, k1, b, idf_pow)
        expected_evidence = tuple(field in evidence_fields for field in weights)
        if not isinstance(base, CompactPostingsSnapshot) or base.mode != "bm25f":
            raise ValueError("compact mutable BM25F base is invalid")
        if (
            base.fields != tuple(weights)
            or base.weights != tuple(weights.values())
            or base.evidence != expected_evidence
            or (base.k1, base.b, base.idf_pow) != (k1, b, idf_pow)
        ):
            raise ValueError(
                "compact mutable BM25F configuration differs from its base"
            )
        if _vector_resolver is None:
            base._validate_storage()
        elif not callable(_vector_resolver):
            raise ValueError("vector mutable BM25F resolver is invalid")
        if type(raw_overrides) is not dict:
            raise ValueError("compact mutable BM25F overrides must be a dict")
        if type(max_doc_id) is not int or max_doc_id < base.max_doc_id:
            raise ValueError("compact mutable BM25F high-water mark is invalid")
        if (
            type(mutation_version) is not int
            or mutation_version < 0
            or type(layout_epoch) is not int
            or layout_epoch < 0
        ):
            raise ValueError("compact mutable BM25F versions are invalid")

        parsed: dict[int, _FieldRecord | None] = {}
        field_count = len(weights)
        for raw_doc_id, raw_record in raw_overrides.items():
            doc_id = _doc_id(raw_doc_id)
            if doc_id > max_doc_id:
                raise ValueError(
                    "compact mutable BM25F override exceeds its high-water mark"
                )
            if doc_id not in base._doc_positions and doc_id <= base.max_doc_id:
                raise ValueError(
                    "compact mutable BM25F override id is not a base or delta id"
                )
            if raw_record is None:
                parsed[doc_id] = None
                continue
            if (
                type(raw_record) is not tuple
                or len(raw_record) != 2
                or type(raw_record[0]) is not tuple
                or len(raw_record[0]) != field_count
                or type(raw_record[1]) is not tuple
                or len(raw_record[1]) != field_count
            ):
                raise ValueError("compact mutable BM25F override record is invalid")
            lengths: list[int] = []
            counts_by_field: list[dict[str, int]] = []
            for raw_length, raw_counts in zip(
                raw_record[0], raw_record[1], strict=True
            ):
                if (
                    type(raw_length) is not int
                    or raw_length < 0
                    or raw_length > _UINT64_MAX
                    or type(raw_counts) is not dict
                ):
                    raise ValueError("compact mutable BM25F field record is invalid")
                counts: dict[str, int] = {}
                for term, tf in raw_counts.items():
                    if (
                        type(term) is not str
                        or type(tf) is not int
                        or tf <= 0
                        or tf > _UINT64_MAX
                    ):
                        raise ValueError(
                            "compact mutable BM25F frequencies are invalid"
                        )
                    counts[term] = tf
                if sum(counts.values()) != raw_length:
                    raise ValueError("compact mutable BM25F field length is invalid")
                lengths.append(raw_length)
                counts_by_field.append(counts)
            parsed[doc_id] = tuple(lengths), tuple(counts_by_field)

        candidate = cls.__new__(cls)
        candidate.k1, candidate.b, candidate._idf_pow = k1, b, idf_pow
        candidate.fields = list(weights)
        candidate.w = weights
        candidate._fw = list(weights.values())
        candidate.evidence_fields = evidence_fields
        candidate._is_ev = list(expected_evidence)
        candidate._base = base
        candidate._base_record_resolver = _vector_resolver
        candidate._base_record_cache = dict(_captured_base_records or {})
        candidate._overrides = parsed
        candidate._delta_postings = {}
        if not parsed:
            candidate._use_base_metadata()
            candidate.N = base.N
            candidate._totals = [int(total) for total in base._totals]
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
                for counts in record[1]:
                    for term in counts:
                        candidate._metadata.canonicalize(term, staged_terms)
        candidate._overrides = {
            doc_id: (
                None
                if record is None
                else (
                    record[0],
                    tuple(
                        {
                            candidate._metadata.canonicalize(term, staged_terms): tf
                            for term, tf in counts.items()
                        }
                        for counts in record[1]
                    ),
                )
            )
            for doc_id, record in parsed.items()
        }
        candidate.N = base.N
        candidate._totals = [int(total) for total in base._totals]
        df_changes: dict[str, int] = {}
        dfe_changes: dict[str, int] = {}
        for doc_id, record in candidate._overrides.items():
            base_record = candidate._base_record(doc_id)
            if base_record is not None:
                candidate.N -= 1
                candidate._remove_stats(base_record)
                candidate._stage_stats(df_changes, dfe_changes, base_record, -1)
            if record is None:
                continue
            candidate.N += 1
            candidate._add_stats(record)
            candidate._stage_stats(df_changes, dfe_changes, record, 1)
            candidate._add_delta(doc_id, record)
        if any(total > _UINT64_MAX for total in candidate._totals):
            raise ValueError("compact mutable field total exceeds its storage domain")
        metadata_patch = candidate._metadata.prepare_patch(
            df_changes, dfe_changes, new_terms=staged_terms
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
