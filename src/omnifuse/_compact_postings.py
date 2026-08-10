"""Immutable compact raw-term-frequency snapshots for lexical retrieval.

The snapshot keeps the scoring inputs, not precomputed scores.  That makes the
representation suitable as the immutable base of a future mutable index: corpus
statistics can change while document and term identities remain stable.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import Counter
from heapq import heappush, heapreplace, merge
from io import BytesIO
import math
from numbers import Real
import operator
import sys
from typing import (
    Final,
    Iterable,
    Iterator,
    Mapping,
    overload,
    Sequence,
    SupportsIndex,
    TypeVar,
)

from .settings import DEFAULT_IDF_POW
from .text import _analyze_query, _coordinate_query_scores


_STATE_VERSION: Final = 1
_VECTOR_FORWARD_STATE_VERSION: Final = 1
_VECTOR_PACKED_FORWARD_STATE_VERSION: Final = 2
_UINT64_MAX: Final = (1 << 64) - 1
_PLAIN_FIELD: Final = "body"
_DEFAULT_IDF_POW: Final = DEFAULT_IDF_POW
_UNSIGNED_ARRAY_LAYOUTS: Final = (
    ("B", 1, (1 << 8) - 1),
    ("H", 2, (1 << 16) - 1),
    ("I", 4, (1 << 32) - 1),
    ("Q", 8, _UINT64_MAX),
)
_UNSIGNED_ARRAY_LIMITS: Final = {
    typecode: maximum for typecode, _itemsize, maximum in _UNSIGNED_ARRAY_LAYOUTS
}
_UNSIGNED_ARRAY_ITEMSIZES: Final = {
    typecode: itemsize for typecode, itemsize, _maximum in _UNSIGNED_ARRAY_LAYOUTS
}
for _typecode, _itemsize, _maximum in _UNSIGNED_ARRAY_LAYOUTS:
    if array(_typecode).itemsize != _itemsize:
        raise RuntimeError(f"unsupported {_typecode} array itemsize")
_CachedWeights = tuple[Sequence[int], Sequence[float]]
_PostingValue = TypeVar("_PostingValue")


def _is_finite_builtin_number(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _unsigned_typecode(value: int) -> str:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise ValueError("packed unsigned values must be uint64-compatible ints")
    for typecode, _itemsize, maximum in _UNSIGNED_ARRAY_LAYOUTS:
        if value <= maximum:
            return typecode
    raise AssertionError("uint64 value has no packed representation")


def _is_canonical_unsigned_array(values: object) -> bool:
    return (
        type(values) is array
        and values.typecode in _UNSIGNED_ARRAY_ITEMSIZES
        and values.itemsize == _UNSIGNED_ARRAY_ITEMSIZES[values.typecode]
    )


def _is_minimal_unsigned_array(values: object) -> bool:
    if not _is_canonical_unsigned_array(values):
        return False
    maximum = max(values, default=0)
    return values.typecode == _unsigned_typecode(maximum)


def _pack_unsigned_array(values: array) -> tuple[str, int, bytes]:
    if not _is_minimal_unsigned_array(values):
        raise TypeError("packed unsigned array is not canonical")
    if sys.byteorder == "little" or values.itemsize == 1:
        payload = values.tobytes()
    else:
        normalized = array(values.typecode, values)
        normalized.byteswap()
        payload = normalized.tobytes()
    return values.typecode, len(values), payload


def _unpack_unsigned_array(value: object) -> array:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("packed unsigned array wire state is invalid")
    typecode, count, payload = value
    if (
        type(typecode) is not str
        or typecode not in _UNSIGNED_ARRAY_ITEMSIZES
        or type(count) is not int
        or count < 0
        or type(payload) is not bytes
        or len(payload) != count * _UNSIGNED_ARRAY_ITEMSIZES[typecode]
    ):
        raise ValueError("packed unsigned array wire state is invalid")
    restored = array(typecode)
    restored.frombytes(payload)
    if sys.byteorder != "little" and restored.itemsize > 1:
        restored.byteswap()
    if len(restored) != count or not _is_minimal_unsigned_array(restored):
        raise ValueError("packed unsigned array wire state is not minimal")
    return restored


def _append_adaptive_uint(values: array, value: int) -> array:
    if value <= _UNSIGNED_ARRAY_LIMITS[values.typecode]:
        values.append(value)
        return values
    target_typecode = _unsigned_typecode(value)
    promoted = array(target_typecode, values)
    promoted.append(value)
    return promoted


def _minimal_uint_array(values: Sequence[int]) -> array:
    maximum = 0
    for value in values:
        if type(value) is not int or not 0 <= value <= _UINT64_MAX:
            raise ValueError("packed unsigned values must be uint64-compatible ints")
        if value > maximum:
            maximum = value
    return array(_unsigned_typecode(maximum), values)


def _packed_slot_capacity(count: int) -> int:
    if type(count) is not int or count < 0 or count > _UINT64_MAX:
        raise ValueError("packed vocabulary size must be a uint64-compatible int")
    required = max(8, (count * 4 + 2) // 3)
    return 1 << (required - 1).bit_length()


def _insert_packed_slot(slots: array, mask: int, term: str, term_id: int) -> None:
    digest = hash(term) & _UINT64_MAX
    slot = digest & mask
    step = ((digest >> 32) | 1) & mask
    for _probe in range(len(slots)):
        if not slots[slot]:
            slots[slot] = term_id + 1
            return
        slot = (slot + step) & mask
    raise ValueError("packed vocabulary lookup table is full")


class _VocabularyStaging:
    """Build final UTF-8 storage as new terms are first observed."""

    __slots__ = ("_blob", "_offsets")

    def __init__(self) -> None:
        self._blob = bytearray()
        self._offsets = array("B", [0])

    def append(self, term: str) -> None:
        self._blob.extend(term.encode("utf-8", "surrogatepass"))
        self._offsets = _append_adaptive_uint(self._offsets, len(self._blob))

    def finish(self, term_ids: dict[str, int]) -> "_PackedVocabulary":
        count = len(term_ids)
        if len(self._offsets) != count + 1:
            raise ValueError("packed vocabulary staging differs from its term lookup")
        capacity = _packed_slot_capacity(count)
        slots = array(_unsigned_typecode(count), [0]) * capacity
        mask = capacity - 1
        for expected_id, (term, term_id) in enumerate(term_ids.items()):
            if type(term) is not str or term_id != expected_id:
                raise ValueError("packed vocabulary term ids are not canonical")
            _insert_packed_slot(slots, mask, term, term_id)
        return _PackedVocabulary(bytes(self._blob), self._offsets, slots)


class _PackedVocabulary:
    """Lossless UTF-8 vocabulary plus a process-local salted lookup table."""

    __slots__ = ("_blob", "_offsets", "_slots", "_mask")

    def __init__(self, blob: bytes, offsets: array, slots: array) -> None:
        self._blob = blob
        self._offsets = offsets
        self._slots = slots
        self._mask = len(slots) - 1

    @classmethod
    def from_validated_terms(cls, terms: Sequence[str]) -> "_PackedVocabulary":
        count = len(terms)
        capacity = _packed_slot_capacity(count)
        slots = array(_unsigned_typecode(count), [0]) * capacity
        mask = capacity - 1
        staging = _VocabularyStaging()
        for term_id, term in enumerate(terms):
            if type(term) is not str:
                raise ValueError("packed vocabulary terms must be strings")
            staging.append(term)
            _insert_packed_slot(slots, mask, term, term_id)
        return cls(bytes(staging._blob), staging._offsets, slots)

    @classmethod
    def _from_packed_forward_state(
        cls, blob: object, offsets: object
    ) -> "_PackedVocabulary":
        if type(blob) is not bytes or not _is_minimal_unsigned_array(offsets):
            raise ValueError("packed vocabulary wire state is invalid")
        count = len(offsets) - 1
        if count < 0:
            raise ValueError("packed vocabulary offset table is empty")
        _validate_offsets(offsets, size=len(blob), count=count)
        capacity = _packed_slot_capacity(count)
        slots = array(_unsigned_typecode(count), [0]) * capacity
        mask = capacity - 1
        candidate = cls(blob, offsets, slots)
        for term_id in range(count):
            start = offsets[term_id]
            end = offsets[term_id + 1]
            try:
                term = blob[start:end].decode("utf-8", "surrogatepass")
            except UnicodeDecodeError as exc:
                raise ValueError("packed vocabulary contains invalid UTF-8") from exc
            if (
                term.encode("utf-8", "surrogatepass") != blob[start:end]
                or candidate.find(term) is not None
            ):
                raise ValueError("packed vocabulary terms are not canonical")
            _insert_packed_slot(slots, mask, term, term_id)
        return candidate

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def term(self, term_id: int) -> str:
        if term_id < 0:
            term_id += len(self)
        if not 0 <= term_id < len(self):
            raise IndexError(term_id)
        start = self._offsets[term_id]
        end = self._offsets[term_id + 1]
        return self._blob[start:end].decode("utf-8", "surrogatepass")

    def find(self, term: object) -> int | None:
        if not isinstance(term, str):
            return None
        encoded = term.encode("utf-8", "surrogatepass")
        digest = hash(term) & _UINT64_MAX
        slot = digest & self._mask
        step = ((digest >> 32) | 1) & self._mask
        for _probe in range(len(self._slots)):
            entry = self._slots[slot]
            if not entry:
                return None
            term_id = entry - 1
            if term_id >= len(self):
                raise ValueError("packed vocabulary lookup contains an invalid term id")
            start = self._offsets[term_id]
            end = self._offsets[term_id + 1]
            if end - start == len(encoded) and self._blob.startswith(
                encoded, start, end
            ):
                return term_id
            slot = (slot + step) & self._mask
        raise ValueError("packed vocabulary lookup has no terminating empty slot")

    def storage_nbytes(self) -> int:
        return (
            len(self._blob)
            + len(self._offsets) * self._offsets.itemsize
            + len(self._slots) * self._slots.itemsize
        )

    def _validate_storage(self) -> None:
        if type(self._blob) is not bytes:
            raise ValueError("packed vocabulary blob must be bytes")
        if not _is_canonical_unsigned_array(
            self._offsets
        ) or not _is_canonical_unsigned_array(self._slots):
            raise ValueError("packed vocabulary arrays are invalid")
        count = len(self._offsets) - 1
        if count < 0:
            raise ValueError("packed vocabulary offset table is empty")
        _validate_offsets(self._offsets, size=len(self._blob), count=count)
        capacity = _packed_slot_capacity(count)
        if (
            len(self._slots) != capacity
            or self._mask != capacity - 1
            or self._slots.typecode != _unsigned_typecode(count)
        ):
            raise ValueError("packed vocabulary lookup capacity is invalid")

        seen = bytearray(count)
        occupied = 0
        for entry in self._slots:
            if not entry:
                continue
            term_id = entry - 1
            if term_id >= count or seen[term_id]:
                raise ValueError("packed vocabulary lookup term ids are invalid")
            seen[term_id] = 1
            occupied += 1
        if occupied != count:
            raise ValueError("packed vocabulary lookup is incomplete")

        for term_id in range(count):
            try:
                term = self.term(term_id)
            except UnicodeDecodeError as exc:
                raise ValueError("packed vocabulary contains invalid UTF-8") from exc
            start = self._offsets[term_id]
            end = self._offsets[term_id + 1]
            if term.encode("utf-8", "surrogatepass") != self._blob[start:end]:
                raise ValueError("packed vocabulary term encoding is not canonical")
            if self.find(term) != term_id:
                raise ValueError("packed vocabulary lookup differs from term order")


class _PackedTerms(Sequence[str]):
    __slots__ = ("_vocabulary",)

    def __init__(self, vocabulary: _PackedVocabulary) -> None:
        self._vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self._vocabulary)

    @overload
    def __getitem__(self, index: int, /) -> str: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            return tuple(
                self._vocabulary.term(term_id)
                for term_id in range(*index.indices(len(self)))
            )
        return self._vocabulary.term(operator.index(index))

    def __iter__(self) -> Iterator[str]:
        return (
            self._vocabulary.term(term_id) for term_id in range(len(self._vocabulary))
        )

    def __reversed__(self) -> Iterator[str]:
        return (
            self._vocabulary.term(term_id)
            for term_id in range(len(self._vocabulary) - 1, -1, -1)
        )

    def __contains__(self, value: object) -> bool:
        return self._vocabulary.find(value) is not None

    def index(
        self,
        value: object,
        start: SupportsIndex = 0,
        stop: SupportsIndex = sys.maxsize,
    ) -> int:
        size = len(self)
        start_index = operator.index(start)
        stop_index = operator.index(stop)
        lower, upper, _step = slice(start_index, stop_index, 1).indices(size)
        term_id = self._vocabulary.find(value)
        if term_id is None or not lower <= term_id < upper:
            raise ValueError(f"{value!r} is not in vocabulary")
        return term_id

    def count(self, value: object) -> int:
        return int(self._vocabulary.find(value) is not None)

    def __eq__(self, other: object) -> bool:
        if type(other) is _PackedTerms:
            return self._vocabulary._blob == other._vocabulary._blob and tuple(
                self._vocabulary._offsets
            ) == tuple(other._vocabulary._offsets)
        if type(other) is tuple:
            return len(self) == len(other) and all(
                left == right for left, right in zip(self, other, strict=True)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return repr(tuple(self))


class _PackedTermIds(Mapping[str, int]):
    __slots__ = ("_vocabulary", "_resolved")

    def __init__(self, vocabulary: _PackedVocabulary) -> None:
        self._vocabulary = vocabulary
        self._resolved: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._vocabulary)

    def __iter__(self) -> Iterator[str]:
        return (
            self._vocabulary.term(term_id) for term_id in range(len(self._vocabulary))
        )

    def __getitem__(self, term: str) -> int:
        try:
            return self._resolved[term]
        except KeyError:
            pass
        term_id = self._vocabulary.find(term)
        if term_id is None:
            raise KeyError(term)
        return term_id

    def get(self, term: str, default=None):
        if isinstance(term, str):
            try:
                return self._resolved[term]
            except KeyError:
                pass
        term_id = self._vocabulary.find(term)
        if term_id is None:
            return default
        canonical = term if type(term) is str else self._vocabulary.term(term_id)
        self._resolved[canonical] = term_id
        return term_id

    def __contains__(self, term: object) -> bool:
        if isinstance(term, str) and term in self._resolved:
            return True
        return self._vocabulary.find(term) is not None

    def _validate_cache(self) -> None:
        if type(self._resolved) is not dict or any(
            type(term) is not str
            or type(term_id) is not int
            or self._vocabulary.find(term) != term_id
            for term, term_id in self._resolved.items()
        ):
            raise ValueError("compact term lookup cache is invalid")


class _IdentityDocPositions(Mapping[int, int]):
    __slots__ = ("_count",)

    def __init__(self, count: int) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("identity document position count is invalid")
        self._count = count

    def __setattr__(self, name: str, value: object) -> None:
        if name != "_count" or hasattr(self, "_count"):
            raise AttributeError("identity document positions are immutable")
        object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._count))

    def __getitem__(self, doc_id: int) -> int:
        if type(doc_id) is int and 0 <= doc_id < self._count:
            return doc_id
        raise KeyError(doc_id)

    def __contains__(self, doc_id: object) -> bool:
        return type(doc_id) is int and 0 <= doc_id < self._count


def _encode_uvarint(value: int) -> bytes:
    """Encode a non-negative integer as canonical unsigned LEB128."""
    if type(value) is not int or value < 0:
        raise ValueError("unsigned LEB128 values must be non-negative ints")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _append_uvarint(target: bytearray, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("unsigned LEB128 values must be non-negative ints")
    while value >= 0x80:
        target.append((value & 0x7F) | 0x80)
        value >>= 7
    target.append(value)


def _freeze_posting_streams(
    posting_streams: Sequence[bytearray],
) -> tuple[bytes, array]:
    """Join owned term streams while releasing each staged payload promptly."""
    output = BytesIO()
    offsets = array("B", [0])
    for posting in posting_streams:
        output.write(posting)
        offsets = _append_adaptive_uint(offsets, output.tell())
        posting.clear()
    return output.getvalue(), offsets


def _consume_staged_document(
    doc_id: int,
    raw_fields: Mapping[str, Iterable[str]],
    *,
    fields: tuple[str, ...],
    evidence: tuple[bool, ...],
    vocabulary: _VocabularyStaging | None,
    term_ids: dict[str, int],
    df: list[int],
    dfe: list[int],
    totals: list[int],
    lengths: list[array],
    posting_streams: list[bytearray],
    posting_last_doc: list[int],
    posting_counts: list[int],
    reverse_blob: bytearray | None,
) -> None:
    """Detach one input document while limiting transient state to this call."""
    lengths_by_field: list[int] | None = [] if reverse_blob is not None else None
    counts_by_field: list[dict[int, int]] | None = (
        [] if reverse_blob is not None else None
    )
    field_count = len(fields)
    has_evidence = any(evidence)
    content: set[int] | None = set() if has_evidence else None
    evidence_only: set[int] | None = set() if has_evidence else None
    term_frequencies: dict[int, list[int]] = {}
    for field_index, field in enumerate(fields):
        counts: dict[int, int] | None = {} if counts_by_field is not None else None
        raw_counts = Counter(raw_fields.get(field, ()))
        length = sum(raw_counts.values())
        present = (
            (evidence_only if evidence[field_index] else content)
            if has_evidence
            else None
        )
        for term, tf in raw_counts.items():
            if type(term) is not str:
                raise TypeError("tokens must contain only str values")
            term_id = term_ids.get(term)
            if term_id is None:
                term_id = len(term_ids)
                term_ids[term] = term_id
                if vocabulary is not None:
                    vocabulary.append(term)
                df.append(0)
                dfe.append(0)
                posting_streams.append(bytearray())
                posting_last_doc.append(0)
                posting_counts.append(0)
            if counts is not None:
                counts[term_id] = tf
            if present is not None:
                present.add(term_id)
            frequencies = term_frequencies.get(term_id)
            if frequencies is None:
                frequencies = [0] * field_count
                term_frequencies[term_id] = frequencies
            frequencies[field_index] = tf
        if lengths_by_field is not None:
            lengths_by_field.append(length)
        lengths[field_index] = _append_adaptive_uint(lengths[field_index], length)
        totals[field_index] += length
        if counts_by_field is not None:
            assert counts is not None
            counts_by_field.append(counts)
    if has_evidence:
        assert content is not None and evidence_only is not None
        for term_id in content:
            df[term_id] += 1
        for term_id in evidence_only:
            dfe[term_id] += 1
    else:
        for term_id in term_frequencies:
            df[term_id] += 1
    for term_id, frequencies in term_frequencies.items():
        posting_index = posting_counts[term_id]
        delta = doc_id if not posting_index else doc_id - posting_last_doc[term_id]
        posting = posting_streams[term_id]
        if delta < 0x80:
            posting.append(delta)
        else:
            _append_uvarint(posting, delta)
        for tf in frequencies:
            if tf < 0x80:
                posting.append(tf)
            else:
                _append_uvarint(posting, tf)
        posting_last_doc[term_id] = doc_id
        posting_counts[term_id] += 1

    if reverse_blob is not None:
        assert lengths_by_field is not None and counts_by_field is not None
        for length in lengths_by_field:
            _append_uvarint(reverse_blob, length)
        for counts in counts_by_field:
            entries = sorted(counts.items())
            _append_uvarint(reverse_blob, len(entries))
            previous = 0
            for entry_index, (term_id, tf) in enumerate(entries):
                delta = term_id if entry_index == 0 else term_id - previous
                _append_uvarint(reverse_blob, delta)
                _append_uvarint(reverse_blob, tf)
                previous = term_id


def _consume_raw_document(
    doc_id: int,
    raw_fields: Mapping[str, Iterable[str]],
    *,
    fields: tuple[str, ...],
    term_ids: dict[str, int],
    df: list[int],
    totals: list[int],
    lengths: list[array],
    posting_streams: list[bytearray],
    posting_last_doc: list[int],
) -> None:
    """Build the disk-only one/two-field stream without generic reverse state."""
    if len(fields) not in {1, 2}:
        raise ValueError("raw document fast path supports one or two fields")
    counts = [Counter(raw_fields.get(field, ())) for field in fields]
    for field_index, field_counts in enumerate(counts):
        length = sum(field_counts.values())
        lengths[field_index] = _append_adaptive_uint(lengths[field_index], length)
        totals[field_index] += length

    if len(fields) == 1:
        for term, frequency in counts[0].items():
            if type(term) is not str:
                raise TypeError("tokens must contain only str values")
            term_id = term_ids.get(term)
            if term_id is None:
                term_id = len(term_ids)
                term_ids[term] = term_id
                df.append(0)
                posting_streams.append(bytearray())
                posting_last_doc.append(0)
            document_frequency = df[term_id]
            delta = (
                doc_id if not document_frequency else doc_id - posting_last_doc[term_id]
            )
            posting = posting_streams[term_id]
            if delta < 0x80:
                posting.append(delta)
            else:
                _append_uvarint(posting, delta)
            if frequency < 0x80:
                posting.append(frequency)
            else:
                _append_uvarint(posting, frequency)
            posting_last_doc[term_id] = doc_id
            df[term_id] = document_frequency + 1
        return

    left, right = counts
    for term, left_frequency in left.items():
        if type(term) is not str:
            raise TypeError("tokens must contain only str values")
        term_id = term_ids.get(term)
        if term_id is None:
            term_id = len(term_ids)
            term_ids[term] = term_id
            df.append(0)
            posting_streams.append(bytearray())
            posting_last_doc.append(0)
        document_frequency = df[term_id]
        delta = doc_id if not document_frequency else doc_id - posting_last_doc[term_id]
        posting = posting_streams[term_id]
        if delta < 0x80:
            posting.append(delta)
        else:
            _append_uvarint(posting, delta)
        right_frequency = right.pop(term, 0)
        if left_frequency < 0x80:
            posting.append(left_frequency)
        else:
            _append_uvarint(posting, left_frequency)
        if right_frequency < 0x80:
            posting.append(right_frequency)
        else:
            _append_uvarint(posting, right_frequency)
        posting_last_doc[term_id] = doc_id
        df[term_id] = document_frequency + 1

    for term, right_frequency in right.items():
        if type(term) is not str:
            raise TypeError("tokens must contain only str values")
        term_id = term_ids.get(term)
        if term_id is None:
            term_id = len(term_ids)
            term_ids[term] = term_id
            df.append(0)
            posting_streams.append(bytearray())
            posting_last_doc.append(0)
        document_frequency = df[term_id]
        delta = doc_id if not document_frequency else doc_id - posting_last_doc[term_id]
        posting = posting_streams[term_id]
        if delta < 0x80:
            posting.append(delta)
        else:
            _append_uvarint(posting, delta)
        posting.append(0)
        if right_frequency < 0x80:
            posting.append(right_frequency)
        else:
            _append_uvarint(posting, right_frequency)
        posting_last_doc[term_id] = doc_id
        df[term_id] = document_frequency + 1


def _decode_uvarint(data: bytes, position: int, end: int) -> tuple[int, int]:
    """Decode one canonical unsigned LEB128 value inside ``[position, end)``."""
    if type(data) is not bytes:
        raise TypeError("unsigned LEB128 data must be bytes")
    if (
        type(position) is not int
        or type(end) is not int
        or position < 0
        or end < position
        or end > len(data)
    ):
        raise ValueError("invalid unsigned LEB128 bounds")
    value = 0
    shift = 0
    start = position
    while position < end:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if position - start > 1 and byte == 0:
                raise ValueError("non-canonical unsigned LEB128")
            return value, position
        shift += 7
    raise ValueError("truncated unsigned LEB128")


def _doc_id(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("doc_id must be a non-negative int")
    return value


def _high_water(value: int, highest_active: int) -> int:
    if type(value) is not int or value < -1:
        raise ValueError("max_doc_id must be an int greater than or equal to -1")
    if value < highest_active:
        raise ValueError("max_doc_id cannot be below the highest active doc_id")
    return value


def _validate_uint_values(values: Sequence[int], *, upper: int, label: str) -> None:
    if any(type(value) is not int or not 0 <= value <= upper for value in values):
        raise ValueError(f"{label} are invalid")


def _validate_offsets(values: Sequence[int], *, size: int, count: int) -> None:
    if len(values) != count + 1:
        raise ValueError("compact offset table has the wrong length")
    previous = -1
    for value in values:
        if type(value) is not int or value < 0 or value > _UINT64_MAX:
            raise ValueError("compact offsets must be unsigned 64-bit ints")
        if value < previous:
            raise ValueError("compact offsets must be monotonic")
        previous = value
    if not values or values[0] != 0 or values[-1] != size:
        raise ValueError("compact offsets do not span their byte stream")


def _top_k_scores(scores: dict[int, float], limit: int) -> list[tuple[int, float]]:
    """Match ``text._top_k_scores`` without introducing an import cycle."""
    if limit < 0:
        positive = ((doc_id, score) for doc_id, score in scores.items() if score > 0)
        return sorted(positive, key=lambda item: (-item[1], item[0]))[:limit]
    if not limit:
        return []

    frontier: list[tuple[float, int, int]] = []
    for doc_id, score in scores.items():
        if not score > 0:
            continue
        candidate = (score, -doc_id, doc_id)
        if len(frontier) < limit:
            heappush(frontier, candidate)
        elif candidate > frontier[0]:
            heapreplace(frontier, candidate)
    frontier.sort(key=lambda item: (-item[0], item[2]))
    return [(doc_id, score) for score, _negated_id, doc_id in frontier]


class _WeightCacheBuilder:
    """Build a derived-score cache without a full-size Python staging list."""

    __slots__ = ("_id_limit", "_ids", "_weights")

    def __init__(self) -> None:
        self._ids: array[int] | list[int] = array("B")
        self._id_limit = _UNSIGNED_ARRAY_LIMITS["B"]
        self._weights = array("d")

    def append(self, doc_id: int, weight: float) -> None:
        ids = self._ids
        if isinstance(ids, array):
            if doc_id <= self._id_limit:
                ids.append(doc_id)
            elif doc_id > _UINT64_MAX:
                self._ids = [*ids, doc_id]
            else:
                typecode = _unsigned_typecode(doc_id)
                promoted = array(typecode, ids)
                promoted.append(doc_id)
                self._ids = promoted
                self._id_limit = _UNSIGNED_ARRAY_LIMITS[typecode]
        else:
            ids.append(doc_id)
        self._weights.append(weight)

    def finish(self) -> _CachedWeights:
        ids = self._ids
        packed_ids: Sequence[int] = ids if isinstance(ids, array) else tuple(ids)
        return packed_ids, self._weights


def _merge_sorted_postings(
    base: Iterable[tuple[int, _PostingValue]],
    delta: Mapping[int, _PostingValue] | None,
) -> Iterator[tuple[int, _PostingValue]]:
    """Merge a large sorted base with a bounded, independently ordered delta."""
    if not delta:
        return iter(base)
    delta_stream = ((doc_id, delta[doc_id]) for doc_id in sorted(delta))
    return merge(base, delta_stream, key=lambda posting: posting[0])


class _RawPostingsSnapshot:
    """Minimal forward-only state consumed by the durable SQLite builder."""

    __slots__ = (
        "mode",
        "fields",
        "weights",
        "evidence",
        "k1",
        "b",
        "idf_pow",
        "N",
        "terms",
        "_totals",
        "_lengths",
        "_df",
        "_posting_streams",
    )

    def __init__(
        self,
        *,
        mode: str,
        fields: tuple[str, ...],
        weights: tuple[float, ...],
        evidence: tuple[bool, ...],
        k1: float,
        b: float,
        idf_pow: float,
        document_count: int,
        terms: tuple[str, ...],
        totals: array,
        lengths: tuple[array, ...],
        df: array,
        posting_streams: list[bytearray],
    ) -> None:
        self.mode = mode
        self.fields = fields
        self.weights = weights
        self.evidence = evidence
        self.k1 = k1
        self.b = b
        self.idf_pow = idf_pow
        self.N = document_count
        self.terms = terms
        self._totals = totals
        self._lengths = lengths
        self._df = df
        self._posting_streams = posting_streams


class CompactPostingsSnapshot:
    """Read-only compact BM25/BM25F scoring state.

    Use :meth:`from_bm25` for one-field BM25 query multiplicity or
    :meth:`from_bm25f` for unique-term BM25F semantics.  Derived lookup and
    weight caches are intentionally absent from pickle state.  Document ids
    must increase strictly, matching the mutable index's permanent-slot
    contract and allowing each streamed document to be detached before the
    producer advances or reuses its input buffer.
    """

    @classmethod
    def from_bm25(
        cls,
        docs: Iterable[tuple[int, Iterable[str]]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
        max_doc_id: int | None = None,
    ) -> "CompactPostingsSnapshot":
        def prepared():
            for doc_id, tokens in docs:
                yield _doc_id(doc_id), {_PLAIN_FIELD: tokens}

        return cls._build(
            prepared(),
            mode="bm25",
            fields=(_PLAIN_FIELD,),
            weights=(1.0,),
            evidence=(False,),
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=max_doc_id,
        )

    @classmethod
    def _from_bm25_for_vector(
        cls,
        docs: Iterable[tuple[int, Iterable[str]]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
        max_doc_id: int | None = None,
    ) -> "CompactPostingsSnapshot":
        def prepared():
            for doc_id, tokens in docs:
                yield _doc_id(doc_id), {_PLAIN_FIELD: tokens}

        return cls._build(
            prepared(),
            mode="bm25",
            fields=(_PLAIN_FIELD,),
            weights=(1.0,),
            evidence=(False,),
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=max_doc_id,
            retain_reverse=False,
        )

    @classmethod
    def from_bm25f(
        cls,
        docs: Iterable[tuple[int, Mapping[str, Iterable[str]]]],
        weights: Mapping[str, float],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
        evidence_fields: Iterable[str] | None = None,
        max_doc_id: int | None = None,
    ) -> "CompactPostingsSnapshot":
        fields = tuple(weights)
        evidence_names = frozenset(evidence_fields or ())

        def prepared():
            for doc_id, raw_fields in docs:
                if not isinstance(raw_fields, Mapping):
                    raise TypeError("BM25F documents must be field mappings")
                yield _doc_id(doc_id), raw_fields

        return cls._build(
            prepared(),
            mode="bm25f",
            fields=fields,
            weights=tuple(weights[field] for field in fields),
            evidence=tuple(field in evidence_names for field in fields),
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=max_doc_id,
        )

    @classmethod
    def _from_bm25f_for_vector(
        cls,
        docs: Iterable[tuple[int, Mapping[str, Iterable[str]]]],
        weights: Mapping[str, float],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
        evidence_fields: Iterable[str] | None = None,
        max_doc_id: int | None = None,
    ) -> "CompactPostingsSnapshot":
        fields = tuple(weights)
        evidence_names = frozenset(evidence_fields or ())

        def prepared():
            for doc_id, raw_fields in docs:
                if not isinstance(raw_fields, Mapping):
                    raise TypeError("BM25F documents must be field mappings")
                yield _doc_id(doc_id), raw_fields

        return cls._build(
            prepared(),
            mode="bm25f",
            fields=fields,
            weights=tuple(weights[field] for field in fields),
            evidence=tuple(field in evidence_names for field in fields),
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=max_doc_id,
            retain_reverse=False,
        )

    @classmethod
    def _raw_bm25_for_disk(
        cls,
        docs: Iterable[tuple[int, Iterable[str]]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
    ) -> "_RawPostingsSnapshot":
        def prepared():
            for doc_id, tokens in docs:
                yield _doc_id(doc_id), {_PLAIN_FIELD: tokens}

        return cls._build(
            prepared(),
            mode="bm25",
            fields=(_PLAIN_FIELD,),
            weights=(1.0,),
            evidence=(False,),
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=None,
            retain_reverse=False,
            raw_export=True,
        )

    @classmethod
    def _raw_bm25f_for_disk(
        cls,
        docs: Iterable[tuple[int, Mapping[str, Iterable[str]]]],
        weights: Mapping[str, float],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _DEFAULT_IDF_POW,
    ) -> "_RawPostingsSnapshot":
        fields = tuple(weights)

        def prepared():
            for doc_id, raw_fields in docs:
                if not isinstance(raw_fields, Mapping):
                    raise TypeError("BM25F documents must be field mappings")
                yield _doc_id(doc_id), raw_fields

        return cls._build(
            prepared(),
            mode="bm25f",
            fields=fields,
            weights=tuple(weights[field] for field in fields),
            evidence=(False,) * len(fields),
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=None,
            retain_reverse=False,
            raw_export=True,
        )

    @classmethod
    def _build(
        cls,
        prepared: Iterable[tuple[int, Mapping[str, Iterable[str]]]],
        *,
        mode: str,
        fields: tuple[str, ...],
        weights: tuple[float, ...],
        evidence: tuple[bool, ...],
        k1: float,
        b: float,
        idf_pow: float,
        max_doc_id: int | None,
        retain_reverse: bool = True,
        raw_export: bool = False,
    ) -> "CompactPostingsSnapshot | _RawPostingsSnapshot":
        if type(retain_reverse) is not bool:
            raise TypeError("retain_reverse must be bool")
        if type(raw_export) is not bool or (raw_export and retain_reverse):
            raise TypeError("raw_export requires a non-reverse bool build")
        cls._validate_config(mode, fields, weights, evidence, k1, b, idf_pow)
        forward_fast_path = (
            not retain_reverse and not any(evidence) and len(fields) <= 2
        )
        doc_ids_list: list[int] | None = None if raw_export else []
        document_count = 0
        vocabulary_staging = None if forward_fast_path else _VocabularyStaging()
        term_ids: dict[str, int] = {}
        df: list[int] = []
        dfe: list[int] | None = None if forward_fast_path else []
        totals = [0] * len(fields)
        lengths = [array("B") for _field in fields]
        posting_streams: list[bytearray] = []
        posting_last_doc: list[int] = []
        posting_counts: list[int] = []
        reverse_blob = bytearray() if retain_reverse else None
        reverse_offsets = array("B", [0])

        for doc_id, raw_fields in prepared:
            doc_id = _doc_id(doc_id)
            if raw_export:
                if doc_id != document_count:
                    raise ValueError(
                        "raw disk postings require contiguous document ids"
                    )
                document_count += 1
            else:
                assert doc_ids_list is not None
                if doc_ids_list and doc_id <= doc_ids_list[-1]:
                    raise ValueError(
                        "compact snapshot doc_ids must be strictly increasing"
                    )
                doc_ids_list.append(doc_id)
            if forward_fast_path:
                _consume_raw_document(
                    doc_id,
                    raw_fields,
                    fields=fields,
                    term_ids=term_ids,
                    df=df,
                    totals=totals,
                    lengths=lengths,
                    posting_streams=posting_streams,
                    posting_last_doc=posting_last_doc,
                )
            else:
                assert dfe is not None
                _consume_staged_document(
                    doc_id,
                    raw_fields,
                    fields=fields,
                    evidence=evidence,
                    vocabulary=vocabulary_staging,
                    term_ids=term_ids,
                    df=df,
                    dfe=dfe,
                    totals=totals,
                    lengths=lengths,
                    posting_streams=posting_streams,
                    posting_last_doc=posting_last_doc,
                    posting_counts=posting_counts,
                    reverse_blob=reverse_blob,
                )
            if reverse_blob is not None:
                reverse_offsets = _append_adaptive_uint(
                    reverse_offsets, len(reverse_blob)
                )
            del raw_fields

        if raw_export:
            del posting_last_doc, posting_counts, reverse_blob, dfe
            raw = _RawPostingsSnapshot(
                mode=mode,
                fields=fields,
                weights=weights,
                evidence=evidence,
                k1=k1,
                b=b,
                idf_pow=idf_pow,
                document_count=document_count,
                terms=tuple(term_ids),
                totals=_minimal_uint_array(totals),
                lengths=tuple(lengths),
                df=_minimal_uint_array(df),
                posting_streams=posting_streams,
            )
            return raw

        assert doc_ids_list is not None
        doc_ids = tuple(doc_ids_list)
        highest = doc_ids[-1] if doc_ids else -1
        high_water = highest if max_doc_id is None else _high_water(max_doc_id, highest)
        posting_blob, posting_offsets = _freeze_posting_streams(posting_streams)
        del posting_streams, posting_last_doc, posting_counts
        reverse_bytes = bytes(reverse_blob) if reverse_blob is not None else b""
        del reverse_blob

        vocabulary = (
            _PackedVocabulary.from_validated_terms(tuple(term_ids))
            if vocabulary_staging is None
            else vocabulary_staging.finish(term_ids)
        )
        totals_array = _minimal_uint_array(totals)
        df_array = _minimal_uint_array(df)
        dfe_array = (
            array("B", [0]) * len(df) if dfe is None else _minimal_uint_array(dfe)
        )
        del vocabulary_staging, term_ids, totals, df, dfe, doc_ids_list

        candidate = cls.__new__(cls)
        candidate._install(
            mode=mode,
            fields=fields,
            weights=weights,
            evidence=evidence,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=high_water,
            doc_ids=doc_ids,
            vocabulary=vocabulary,
            totals=totals_array,
            lengths=tuple(lengths),
            df=df_array,
            dfe=dfe_array,
            posting_blob=posting_blob,
            posting_offsets=posting_offsets,
            reverse_blob=reverse_bytes,
            reverse_offsets=reverse_offsets,
            retains_reverse=retain_reverse,
        )
        # Every value above was derived from already validated local records.  Re-reading
        # both streams here would double build work and peak memory; untrusted pickle state
        # still takes the full cross-stream validation path in ``__setstate__``.
        return candidate

    @staticmethod
    def _validate_config(
        mode: str,
        fields: tuple[str, ...],
        weights: tuple[float, ...],
        evidence: tuple[bool, ...],
        k1: float,
        b: float,
        idf_pow: float,
    ) -> None:
        if type(mode) is not str or mode not in {"bm25", "bm25f"}:
            raise ValueError("unsupported compact scoring mode")
        if any(type(field) is not str for field in fields) or len(set(fields)) != len(
            fields
        ):
            raise ValueError("compact fields must be unique strings")
        if len(weights) != len(fields) or len(evidence) != len(fields):
            raise ValueError("compact field configuration lengths differ")
        if any(type(flag) is not bool for flag in evidence):
            raise ValueError("compact evidence flags must be bool values")
        if any(not isinstance(weight, Real) for weight in weights):
            raise TypeError("compact field weights must be real numbers")
        if any(not isinstance(value, Real) for value in (k1, b, idf_pow)):
            raise TypeError("compact scoring parameters must be real numbers")
        try:
            finite_weights = all(math.isfinite(weight) for weight in weights)
            finite_parameters = all(math.isfinite(value) for value in (k1, b, idf_pow))
        except (OverflowError, TypeError) as exc:
            raise ValueError("compact scoring values must be finite") from exc
        if not finite_weights or not finite_parameters:
            raise ValueError("compact scoring values must be finite")
        if mode == "bm25" and (
            fields != (_PLAIN_FIELD,) or weights != (1.0,) or evidence != (False,)
        ):
            raise ValueError(
                "plain BM25 requires its canonical one-field configuration"
            )

    def _install(
        self,
        *,
        mode: str,
        fields: tuple[str, ...],
        weights: tuple[float, ...],
        evidence: tuple[bool, ...],
        k1: float,
        b: float,
        idf_pow: float,
        max_doc_id: int,
        doc_ids: tuple[int, ...],
        vocabulary: _PackedVocabulary,
        totals: array,
        lengths: tuple[array, ...],
        df: array,
        dfe: array,
        posting_blob: bytes,
        posting_offsets: array,
        reverse_blob: bytes,
        reverse_offsets: array,
        retains_reverse: bool,
    ) -> None:
        self.mode = mode
        self.fields = fields
        self.weights = weights
        self.evidence = evidence
        self.k1 = k1
        self.b = b
        self.idf_pow = idf_pow
        self.max_doc_id = max_doc_id
        self.doc_ids = doc_ids
        self._vocabulary = vocabulary
        self.terms = _PackedTerms(vocabulary)
        self._totals = totals
        self._lengths = lengths
        self._df = df
        self._dfe = dfe
        self._posting_blob = posting_blob
        self._posting_offsets = posting_offsets
        self._reverse_blob = reverse_blob
        self._reverse_offsets = reverse_offsets
        self._retains_reverse = retains_reverse
        self._term_ids = _PackedTermIds(vocabulary)
        self._doc_positions: Mapping[int, int]
        if not doc_ids or (doc_ids[0] == 0 and doc_ids[-1] == len(doc_ids) - 1):
            self._doc_positions = _IdentityDocPositions(len(doc_ids))
        else:
            self._doc_positions = {
                doc_id: position for position, doc_id in enumerate(doc_ids)
            }
        self._weight_cache: dict[int, _CachedWeights] = {}

    @property
    def N(self) -> int:
        return len(self.doc_ids)

    @property
    def avgdl(self) -> float:
        if self.mode != "bm25":
            raise AttributeError("avgdl is available only for BM25 snapshots")
        return self._totals[0] / self.N if self.N else 0.0

    @property
    def avglen(self) -> dict[str, float]:
        if self.mode != "bm25f":
            raise AttributeError("avglen is available only for BM25F snapshots")
        return {
            field: (self._totals[index] / self.N if self.N else 0.0)
            for index, field in enumerate(self.fields)
        }

    @property
    def idf(self) -> dict[str, float]:
        return {term: self._idf(term_id) for term_id, term in enumerate(self.terms)}

    def _idf(self, term_id: int) -> float:
        n = self._df[term_id] or self._dfe[term_id]
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** self.idf_pow

    def _decode_posting(self, term_id: int):
        position = self._posting_offsets[term_id]
        end = self._posting_offsets[term_id + 1]
        previous = 0
        posting_index = 0
        while position < end:
            delta, position = _decode_uvarint(self._posting_blob, position, end)
            if posting_index and delta == 0:
                raise ValueError(
                    "posting doc_id deltas after the first must be positive"
                )
            doc_id = delta if not posting_index else previous + delta
            frequencies: list[int] = []
            for _field in self.fields:
                tf, position = _decode_uvarint(self._posting_blob, position, end)
                frequencies.append(tf)
            if not any(frequencies):
                raise ValueError(
                    "compact postings cannot contain an all-zero frequency"
                )
            yield doc_id, tuple(frequencies)
            previous = doc_id
            posting_index += 1
        if position != end:
            raise ValueError("compact posting did not consume its byte range")

    def _term_weights(self, term: str) -> _CachedWeights:
        term_id = self._term_ids.get(term)
        if term_id is None:
            return (), ()
        cached = self._weight_cache.get(term_id)
        if cached is not None:
            return cached

        builder = _WeightCacheBuilder()
        idf = self._idf(term_id)
        k1p1 = self.k1 + 1
        doc_positions = self._doc_positions
        identity_positions = type(doc_positions) is _IdentityDocPositions
        if self.mode == "bm25":
            avg = self.avgdl or 1.0
            for doc_id, frequencies in self._decode_posting(term_id):
                tf = frequencies[0]
                doc_position = doc_id if identity_positions else doc_positions[doc_id]
                doc_len = self._lengths[0][doc_position]
                norm = self.k1 * (1 - self.b + self.b * (doc_len or 1) / avg)
                weight = idf * k1p1 * tf / (tf + norm)
                builder.append(doc_id, weight)
        else:
            avgl = [
                (self._totals[index] / self.N if self.N else 0.0) or 1
                for index in range(len(self.fields))
            ]
            for doc_id, frequencies in self._decode_posting(term_id):
                doc_position = doc_id if identity_positions else doc_positions[doc_id]
                tfw = 0.0
                for field_index, tf in enumerate(frequencies):
                    if not tf:
                        continue
                    norm = (
                        1.0
                        if self.evidence[field_index]
                        else 1
                        - self.b
                        + self.b
                        * (self._lengths[field_index][doc_position] or 1)
                        / avgl[field_index]
                    )
                    tfw = tfw + self.weights[field_index] * tf / norm
                if not tfw:
                    continue
                weight = idf * tfw * k1p1 / (self.k1 + tfw)
                builder.append(doc_id, weight)
        result = builder.finish()
        self._weight_cache[term_id] = result
        return result

    def score_tokens(self, tokens: Iterable[str], doc_id: int) -> float:
        doc_id = _doc_id(doc_id)
        score = 0.0
        query_terms = tokens if self.mode == "bm25" else dict.fromkeys(tokens)
        for term in query_terms:
            ids, weights = self._term_weights(term)
            position = bisect_left(ids, doc_id)
            if position < len(ids) and ids[position] == doc_id:
                score += weights[position]
        return score

    def search_tokens(
        self,
        tokens: Iterable[str],
        *,
        limit: int = 20,
        anchors: frozenset[str] = frozenset(),
        restricted: bool = False,
        recover_partial_outlier: bool = False,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        if self.mode == "bm25":
            query_counts: dict[str, int] = {}
            for term in tokens:
                query_counts[term] = query_counts.get(term, 0) + 1
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
        else:
            for term in dict.fromkeys(tokens):
                ids, weights = self._term_weights(term)
                if restricted and term in anchors:
                    candidates.update(ids)
                if term in anchors and term.startswith("#"):
                    complete_candidates.update(ids)
                for doc_id, weight in zip(ids, weights):
                    scores[doc_id] = scores.get(doc_id, 0.0) + weight
        scores = _coordinate_query_scores(
            scores,
            candidates,
            complete_candidates,
            recover_partial_outlier=recover_partial_outlier,
        )
        return _top_k_scores(scores, limit)

    def score(self, tokens: list[str], doc_id: int) -> float:
        return self.score_tokens(tokens, doc_id)

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

    def storage_nbytes(self) -> int:
        """Return bytes owned by compact streams (excluding object headers/caches)."""
        arrays = (
            self._totals,
            self._df,
            self._dfe,
            self._posting_offsets,
            self._reverse_offsets,
            *self._lengths,
        )
        return (
            len(self._posting_blob)
            + len(self._reverse_blob)
            + sum(len(values) * values.itemsize for values in arrays)
            + self._vocabulary.storage_nbytes()
        )

    def _export_forward_state(self) -> dict:
        """Return the vector-private canonical state without reverse postings."""
        return {
            "state_version": _VECTOR_FORWARD_STATE_VERSION,
            "mode": self.mode,
            "fields": self.fields,
            "weights": self.weights,
            "evidence": self.evidence,
            "k1": self.k1,
            "b": self.b,
            "idf_pow": self.idf_pow,
            "max_doc_id": self.max_doc_id,
            "doc_ids": self.doc_ids,
            "terms": tuple(self.terms),
            "totals": tuple(self._totals),
            "lengths": tuple(tuple(values) for values in self._lengths),
            "df": tuple(self._df),
            "dfe": tuple(self._dfe),
            "posting_blob": self._posting_blob,
            "posting_offsets": tuple(self._posting_offsets),
        }

    def _export_packed_forward_state(self) -> dict:
        """Return the vector-private forward state without unpacking its columns."""
        return {
            "state_version": _VECTOR_PACKED_FORWARD_STATE_VERSION,
            "mode": self.mode,
            "fields": self.fields,
            "weights": self.weights,
            "evidence": self.evidence,
            "k1": self.k1,
            "b": self.b,
            "idf_pow": self.idf_pow,
            "max_doc_id": self.max_doc_id,
            "doc_ids": self.doc_ids,
            "vocabulary_blob": self._vocabulary._blob,
            "vocabulary_offsets": _pack_unsigned_array(self._vocabulary._offsets),
            "totals": _pack_unsigned_array(self._totals),
            "lengths": tuple(_pack_unsigned_array(values) for values in self._lengths),
            "df": _pack_unsigned_array(self._df),
            "dfe": _pack_unsigned_array(self._dfe),
            "posting_blob": self._posting_blob,
            "posting_offsets": _pack_unsigned_array(self._posting_offsets),
        }

    @classmethod
    def _from_packed_forward_state(
        cls, state: dict, *, capture_doc_ids: Iterable[int] = ()
    ) -> tuple[
        "CompactPostingsSnapshot",
        dict[int, tuple[tuple[int, ...], tuple[dict[str, int], ...]]],
    ]:
        expected = {
            "state_version",
            "mode",
            "fields",
            "weights",
            "evidence",
            "k1",
            "b",
            "idf_pow",
            "max_doc_id",
            "doc_ids",
            "vocabulary_blob",
            "vocabulary_offsets",
            "totals",
            "lengths",
            "df",
            "dfe",
            "posting_blob",
            "posting_offsets",
        }
        if (
            type(state) is not dict
            or type(state.get("state_version")) is not int
            or state.get("state_version") != _VECTOR_PACKED_FORWARD_STATE_VERSION
            or set(state) != expected
        ):
            raise ValueError("unsupported packed vector forward snapshot state")
        try:
            mode = state["mode"]
            fields = state["fields"]
            weights = state["weights"]
            evidence = state["evidence"]
            k1 = state["k1"]
            b = state["b"]
            idf_pow = state["idf_pow"]
            max_doc_id = state["max_doc_id"]
            doc_ids = state["doc_ids"]
            vocabulary_blob = state["vocabulary_blob"]
            vocabulary_offsets = _unpack_unsigned_array(state["vocabulary_offsets"])
            totals = _unpack_unsigned_array(state["totals"])
            raw_lengths = state["lengths"]
            df = _unpack_unsigned_array(state["df"])
            dfe = _unpack_unsigned_array(state["dfe"])
            posting_blob = state["posting_blob"]
            posting_offsets = _unpack_unsigned_array(state["posting_offsets"])
        except KeyError as exc:
            raise ValueError("invalid packed vector forward snapshot state") from exc
        tuple_values = (fields, weights, evidence, doc_ids, raw_lengths)
        if any(type(value) is not tuple for value in tuple_values):
            raise ValueError("packed vector forward containers are not canonical")
        if any(
            not _is_finite_builtin_number(value) for value in (*weights, k1, b, idf_pow)
        ):
            raise ValueError("packed vector forward scoring values are invalid")
        lengths = tuple(_unpack_unsigned_array(value) for value in raw_lengths)
        vocabulary = _PackedVocabulary._from_packed_forward_state(
            vocabulary_blob, vocabulary_offsets
        )
        high_water = cls._validate_metadata_values(
            mode=mode,
            fields=fields,
            weights=weights,
            evidence=evidence,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=max_doc_id,
            doc_ids=doc_ids,
            terms=_PackedTerms(vocabulary),
            totals=totals,
            lengths=lengths,
            df=df,
            dfe=dfe,
            posting_blob=posting_blob,
            posting_offsets=posting_offsets,
            reverse_blob=b"",
            reverse_offsets=(0,) * (len(doc_ids) + 1),
            validate_terms=False,
        )
        candidate = cls.__new__(cls)
        candidate._install(
            mode=mode,
            fields=fields,
            weights=weights,
            evidence=evidence,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=high_water,
            doc_ids=doc_ids,
            vocabulary=vocabulary,
            totals=totals,
            lengths=lengths,
            df=df,
            dfe=dfe,
            posting_blob=posting_blob,
            posting_offsets=posting_offsets,
            reverse_blob=b"",
            reverse_offsets=array("B", [0]),
            retains_reverse=False,
        )
        return candidate, candidate._validate_forward_storage(capture_doc_ids)

    @classmethod
    def _from_forward_state(
        cls, state: dict, *, capture_doc_ids: Iterable[int] = ()
    ) -> tuple[
        "CompactPostingsSnapshot",
        dict[int, tuple[tuple[int, ...], tuple[dict[str, int], ...]]],
    ]:
        expected = {
            "state_version",
            "mode",
            "fields",
            "weights",
            "evidence",
            "k1",
            "b",
            "idf_pow",
            "max_doc_id",
            "doc_ids",
            "terms",
            "totals",
            "lengths",
            "df",
            "dfe",
            "posting_blob",
            "posting_offsets",
        }
        if (
            type(state) is not dict
            or type(state.get("state_version")) is not int
            or state.get("state_version") != _VECTOR_FORWARD_STATE_VERSION
            or set(state) != expected
        ):
            raise ValueError("unsupported vector forward snapshot state")
        try:
            mode = state["mode"]
            fields = state["fields"]
            weights = state["weights"]
            evidence = state["evidence"]
            k1 = state["k1"]
            b = state["b"]
            idf_pow = state["idf_pow"]
            max_doc_id = state["max_doc_id"]
            doc_ids = state["doc_ids"]
            terms = state["terms"]
            totals = state["totals"]
            lengths = state["lengths"]
            df = state["df"]
            dfe = state["dfe"]
            posting_blob = state["posting_blob"]
            posting_offsets = state["posting_offsets"]
        except KeyError as exc:
            raise ValueError("invalid vector forward snapshot state") from exc
        tuple_values = (
            fields,
            weights,
            evidence,
            doc_ids,
            terms,
            totals,
            lengths,
            df,
            dfe,
            posting_offsets,
        )
        if any(type(value) is not tuple for value in tuple_values) or any(
            type(column) is not tuple for column in lengths
        ):
            raise ValueError("vector forward snapshot containers are not canonical")
        scoring_values = (*weights, k1, b, idf_pow)
        if any(not _is_finite_builtin_number(value) for value in scoring_values):
            raise ValueError("vector forward scoring values are invalid")

        high_water = cls._validate_metadata_values(
            mode=mode,
            fields=fields,
            weights=weights,
            evidence=evidence,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=max_doc_id,
            doc_ids=doc_ids,
            terms=terms,
            totals=totals,
            lengths=lengths,
            df=df,
            dfe=dfe,
            posting_blob=posting_blob,
            posting_offsets=posting_offsets,
            reverse_blob=b"",
            reverse_offsets=(0,) * (len(doc_ids) + 1),
        )
        candidate = cls.__new__(cls)
        candidate._install(
            mode=mode,
            fields=fields,
            weights=weights,
            evidence=evidence,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=high_water,
            doc_ids=doc_ids,
            vocabulary=_PackedVocabulary.from_validated_terms(terms),
            totals=_minimal_uint_array(totals),
            lengths=tuple(_minimal_uint_array(values) for values in lengths),
            df=_minimal_uint_array(df),
            dfe=_minimal_uint_array(dfe),
            posting_blob=posting_blob,
            posting_offsets=_minimal_uint_array(posting_offsets),
            reverse_blob=b"",
            reverse_offsets=array("B", [0]),
            retains_reverse=False,
        )
        return candidate, candidate._validate_forward_storage(capture_doc_ids)

    def _validate_forward_storage(
        self, capture_doc_ids: Iterable[int] = ()
    ) -> dict[int, tuple[tuple[int, ...], tuple[dict[str, int], ...]]]:
        """Validate a forward-only base and recover selected original records."""
        capture = {_doc_id(doc_id) for doc_id in capture_doc_ids}
        capture.intersection_update(self.doc_ids)
        self._validate_config(
            self.mode,
            self.fields,
            self.weights,
            self.evidence,
            self.k1,
            self.b,
            self.idf_pow,
        )
        if (
            type(self.fields) is not tuple
            or type(self.weights) is not tuple
            or type(self.evidence) is not tuple
            or type(self.doc_ids) is not tuple
            or type(self._lengths) is not tuple
            or type(self._posting_blob) is not bytes
            or type(self._vocabulary) is not _PackedVocabulary
            or type(self.terms) is not _PackedTerms
            or self.terms._vocabulary is not self._vocabulary
            or type(self._term_ids) is not _PackedTermIds
            or self._term_ids._vocabulary is not self._vocabulary
        ):
            raise ValueError("vector forward snapshot metadata is invalid")
        self._vocabulary._validate_storage()
        self._term_ids._validate_cache()
        arrays = (
            self._totals,
            *self._lengths,
            self._df,
            self._dfe,
            self._posting_offsets,
        )
        if any(not _is_canonical_unsigned_array(values) for values in arrays):
            raise ValueError("vector forward snapshot numeric arrays are invalid")
        self._validate_metadata_values(
            mode=self.mode,
            fields=self.fields,
            weights=self.weights,
            evidence=self.evidence,
            k1=self.k1,
            b=self.b,
            idf_pow=self.idf_pow,
            max_doc_id=self.max_doc_id,
            doc_ids=self.doc_ids,
            terms=self.terms,
            totals=self._totals,
            lengths=self._lengths,
            df=self._df,
            dfe=self._dfe,
            posting_blob=self._posting_blob,
            posting_offsets=self._posting_offsets,
            reverse_blob=b"",
            reverse_offsets=(0,) * (self.N + 1),
            validate_terms=False,
        )
        doc_positions = self._doc_positions
        dense_doc_ids = not self.doc_ids or (
            self.doc_ids[0] == 0 and self.doc_ids[-1] == self.N - 1
        )
        if type(doc_positions) is _IdentityDocPositions:
            if not dense_doc_ids or len(doc_positions) != self.N:
                raise ValueError("vector forward document lookup is invalid")
        elif (
            dense_doc_ids
            or type(doc_positions) is not dict
            or doc_positions
            != {doc_id: position for position, doc_id in enumerate(self.doc_ids)}
        ):
            raise ValueError("vector forward document lookup is invalid")

        field_count = len(self.fields)
        recomputed_lengths = [array("Q", [0]) * self.N for _field in self.fields]
        recomputed_totals = [0] * field_count
        recomputed_df = array("Q", [0]) * len(self.terms)
        recomputed_dfe = array("Q", [0]) * len(self.terms)
        captured_counts = {
            doc_id: tuple({} for _field in self.fields) for doc_id in capture
        }
        for term_id in range(len(self.terms)):
            posting_count = 0
            for doc_id, frequencies in self._decode_posting(term_id):
                doc_position = doc_positions.get(doc_id)
                if doc_position is None:
                    raise ValueError(
                        "vector forward posting references a missing document"
                    )
                posting_count += 1
                has_content = False
                has_evidence = False
                for field_index, tf in enumerate(frequencies):
                    if tf > _UINT64_MAX:
                        raise ValueError("vector forward frequency exceeds uint64")
                    current_length = recomputed_lengths[field_index][doc_position]
                    if current_length > _UINT64_MAX - tf:
                        raise ValueError("vector forward field length exceeds uint64")
                    recomputed_lengths[field_index][doc_position] = current_length + tf
                    if recomputed_totals[field_index] > _UINT64_MAX - tf:
                        raise ValueError("vector forward field total exceeds uint64")
                    recomputed_totals[field_index] += tf
                    if tf:
                        if self.evidence[field_index]:
                            has_evidence = True
                        else:
                            has_content = True
                        counts = captured_counts.get(doc_id)
                        if counts is not None:
                            counts[field_index][self.terms[term_id]] = tf
                recomputed_df[term_id] += int(has_content)
                recomputed_dfe[term_id] += int(has_evidence)
            if not posting_count:
                raise ValueError("vector forward vocabulary terms require a posting")
        if tuple(recomputed_totals) != tuple(self._totals):
            raise ValueError("vector forward totals do not match postings")
        if tuple(recomputed_df) != tuple(self._df):
            raise ValueError("vector forward content frequencies do not match postings")
        if tuple(recomputed_dfe) != tuple(self._dfe):
            raise ValueError(
                "vector forward evidence frequencies do not match postings"
            )
        if any(
            tuple(recomputed) != tuple(stored)
            for recomputed, stored in zip(
                recomputed_lengths, self._lengths, strict=True
            )
        ):
            raise ValueError("vector forward field lengths do not match postings")
        return {
            doc_id: (
                tuple(
                    recomputed_lengths[field_index][doc_positions[doc_id]]
                    for field_index in range(field_count)
                ),
                counts,
            )
            for doc_id, counts in captured_counts.items()
        }

    def _validate_vector_source_records(
        self,
        resolver,
        captured_base_records: dict[int, object],
    ) -> None:
        """Compare source records exactly with compact per-term posting cursors."""
        if not callable(resolver) or type(captured_base_records) is not dict:
            raise ValueError("vector forward source validator is invalid")
        if any(doc_id not in self._doc_positions for doc_id in captured_base_records):
            raise ValueError("captured vector base record is not in the snapshot")

        term_count = len(self.terms)
        cursors = self._posting_offsets[:-1]
        previous_doc_positions = array(_unsigned_typecode(self.N), [0]) * term_count
        field_count = len(self.fields)

        for doc_position, doc_id in enumerate(self.doc_ids):
            record = (
                captured_base_records[doc_id]
                if doc_id in captured_base_records
                else resolver(doc_id)
            )
            if self.mode == "bm25":
                if (
                    type(record) is not tuple
                    or len(record) != 2
                    or type(record[0]) is not int
                    or type(record[1]) is not dict
                ):
                    raise ValueError("vector source BM25 record is invalid")
                lengths = (record[0],)
                counts_by_field = (record[1],)
            else:
                if (
                    type(record) is not tuple
                    or len(record) != 2
                    or type(record[0]) is not tuple
                    or len(record[0]) != field_count
                    or type(record[1]) is not tuple
                    or len(record[1]) != field_count
                    or any(type(counts) is not dict for counts in record[1])
                ):
                    raise ValueError("vector source BM25F record is invalid")
                lengths = record[0]
                counts_by_field = record[1]

            expected_frequencies: dict[int, list[int]] = {}
            for field_index, (length, counts) in enumerate(
                zip(lengths, counts_by_field, strict=True)
            ):
                if (
                    type(length) is not int
                    or not 0 <= length <= _UINT64_MAX
                    or type(counts) is not dict
                    or length != self._lengths[field_index][doc_position]
                ):
                    raise ValueError("vector source field length is invalid")
                total = 0
                for term, tf in counts.items():
                    if (
                        type(term) is not str
                        or type(tf) is not int
                        or not 0 < tf <= _UINT64_MAX
                        or total > _UINT64_MAX - tf
                    ):
                        raise ValueError("vector source frequencies are invalid")
                    term_id = self._vocabulary.find(term)
                    if term_id is None:
                        raise ValueError(
                            "vector source term is absent from its snapshot"
                        )
                    total += tf
                    frequencies = expected_frequencies.setdefault(
                        term_id, [0] * field_count
                    )
                    frequencies[field_index] = tf
                if total != length:
                    raise ValueError(
                        "vector source field length differs from its frequencies"
                    )

            for term_id, expected in expected_frequencies.items():
                cursor = cursors[term_id]
                end = self._posting_offsets[term_id + 1]
                if cursor >= end:
                    raise ValueError("vector source posting is missing")
                delta, cursor = _decode_uvarint(self._posting_blob, cursor, end)
                previous_position = previous_doc_positions[term_id]
                if previous_position:
                    if delta == 0:
                        raise ValueError("vector source posting delta is not positive")
                    posting_doc_id = self.doc_ids[previous_position - 1] + delta
                else:
                    posting_doc_id = delta
                frequencies: list[int] = []
                for _field in self.fields:
                    tf, cursor = _decode_uvarint(self._posting_blob, cursor, end)
                    frequencies.append(tf)
                if posting_doc_id != doc_id or frequencies != expected:
                    raise ValueError("vector source posting differs from its snapshot")
                cursors[term_id] = cursor
                previous_doc_positions[term_id] = doc_position + 1

        if any(
            cursor != self._posting_offsets[term_id + 1]
            for term_id, cursor in enumerate(cursors)
        ):
            raise ValueError("vector source snapshot has unexpected postings")

    @classmethod
    def _validate_metadata_values(
        cls,
        *,
        mode: str,
        fields: tuple[str, ...],
        weights: tuple[float, ...],
        evidence: tuple[bool, ...],
        k1: float,
        b: float,
        idf_pow: float,
        max_doc_id: int,
        doc_ids: Sequence[int],
        terms: Sequence[str],
        totals: Sequence[int],
        lengths: Sequence[Sequence[int]],
        df: Sequence[int],
        dfe: Sequence[int],
        posting_blob: bytes,
        posting_offsets: Sequence[int],
        reverse_blob: bytes,
        reverse_offsets: Sequence[int],
        validate_terms: bool = True,
    ) -> int:
        cls._validate_config(mode, fields, weights, evidence, k1, b, idf_pow)
        if type(posting_blob) is not bytes or type(reverse_blob) is not bytes:
            raise ValueError("compact snapshot streams must be bytes")
        if any(type(doc_id) is not int or doc_id < 0 for doc_id in doc_ids):
            raise ValueError("compact snapshot doc_ids are invalid")
        if any(left >= right for left, right in zip(doc_ids, doc_ids[1:])):
            raise ValueError("compact snapshot doc_ids must be strictly increasing")
        highest = doc_ids[-1] if doc_ids else -1
        high_water = _high_water(max_doc_id, highest)
        if validate_terms and (
            any(type(term) is not str for term in terms)
            or len(set(terms)) != len(terms)
        ):
            raise ValueError("compact snapshot terms must be unique strings")
        if len(totals) != len(fields):
            raise ValueError("compact snapshot totals are invalid")
        _validate_uint_values(
            totals, upper=_UINT64_MAX, label="compact snapshot totals"
        )
        if len(lengths) != len(fields) or any(
            len(values) != len(doc_ids) for values in lengths
        ):
            raise ValueError("compact snapshot length columns are invalid")
        for values in lengths:
            _validate_uint_values(
                values,
                upper=_UINT64_MAX,
                label="compact snapshot document lengths",
            )
        if len(df) != len(terms) or len(dfe) != len(terms):
            raise ValueError("compact snapshot frequency arrays are invalid")
        _validate_uint_values(
            df, upper=len(doc_ids), label="compact snapshot document frequencies"
        )
        _validate_uint_values(
            dfe, upper=len(doc_ids), label="compact snapshot document frequencies"
        )
        _validate_offsets(posting_offsets, size=len(posting_blob), count=len(terms))
        _validate_offsets(reverse_offsets, size=len(reverse_blob), count=len(doc_ids))
        return high_water

    def _validate_installed_metadata(self) -> None:
        if (
            type(self.fields) is not tuple
            or type(self.weights) is not tuple
            or type(self.evidence) is not tuple
            or type(self.doc_ids) is not tuple
            or type(self._lengths) is not tuple
        ):
            raise ValueError("compact snapshot metadata containers are invalid")
        if (
            type(self._vocabulary) is not _PackedVocabulary
            or type(self.terms) is not _PackedTerms
            or self.terms._vocabulary is not self._vocabulary
            or type(self._term_ids) is not _PackedTermIds
            or self._term_ids._vocabulary is not self._vocabulary
        ):
            raise ValueError("compact term lookup differs from its vocabulary")
        self._vocabulary._validate_storage()
        self._term_ids._validate_cache()
        arrays = (
            self._totals,
            *self._lengths,
            self._df,
            self._dfe,
            self._posting_offsets,
            self._reverse_offsets,
        )
        if any(not _is_canonical_unsigned_array(values) for values in arrays):
            raise ValueError("compact snapshot numeric arrays are invalid")
        self._validate_metadata_values(
            mode=self.mode,
            fields=self.fields,
            weights=self.weights,
            evidence=self.evidence,
            k1=self.k1,
            b=self.b,
            idf_pow=self.idf_pow,
            max_doc_id=self.max_doc_id,
            doc_ids=self.doc_ids,
            terms=self.terms,
            totals=self._totals,
            lengths=self._lengths,
            df=self._df,
            dfe=self._dfe,
            posting_blob=self._posting_blob,
            posting_offsets=self._posting_offsets,
            reverse_blob=self._reverse_blob,
            reverse_offsets=self._reverse_offsets,
            validate_terms=False,
        )
        doc_positions = self._doc_positions
        identity_positions = type(doc_positions) is _IdentityDocPositions
        dense_doc_ids = not self.doc_ids or (
            self.doc_ids[0] == 0 and self.doc_ids[-1] == len(self.doc_ids) - 1
        )
        if (
            identity_positions
            and (not dense_doc_ids or len(doc_positions) != len(self.doc_ids))
        ) or (
            not identity_positions
            and (
                dense_doc_ids
                or type(doc_positions) is not dict
                or doc_positions
                != {doc_id: position for position, doc_id in enumerate(self.doc_ids)}
            )
        ):
            raise ValueError("compact document lookup differs from its ordered ids")
        if type(self._weight_cache) is not dict:
            raise ValueError("compact weight cache is invalid")

    def __getstate__(self) -> dict:
        if not self._retains_reverse:
            raise TypeError("vector-owned forward-only snapshots are derived state")
        return {
            "state_version": _STATE_VERSION,
            "mode": self.mode,
            "fields": self.fields,
            "weights": self.weights,
            "evidence": self.evidence,
            "k1": self.k1,
            "b": self.b,
            "idf_pow": self.idf_pow,
            "max_doc_id": self.max_doc_id,
            "doc_ids": self.doc_ids,
            "terms": tuple(self.terms),
            "totals": tuple(self._totals),
            "lengths": tuple(tuple(values) for values in self._lengths),
            "df": tuple(self._df),
            "dfe": tuple(self._dfe),
            "posting_blob": self._posting_blob,
            "posting_offsets": tuple(self._posting_offsets),
            "reverse_blob": self._reverse_blob,
            "reverse_offsets": tuple(self._reverse_offsets),
        }

    def __setstate__(self, state: dict) -> None:
        if type(state) is not dict or state.get("state_version") != _STATE_VERSION:
            raise ValueError("unsupported compact snapshot state")
        try:
            mode = state["mode"]
            fields = tuple(state["fields"])
            weights = tuple(state["weights"])
            evidence = tuple(state["evidence"])
            k1 = state["k1"]
            b = state["b"]
            idf_pow = state["idf_pow"]
            max_doc_id = state["max_doc_id"]
            doc_ids = tuple(state["doc_ids"])
            terms = tuple(state["terms"])
            totals_raw = tuple(state["totals"])
            lengths_raw = tuple(tuple(values) for values in state["lengths"])
            df_raw = tuple(state["df"])
            dfe_raw = tuple(state["dfe"])
            posting_blob = state["posting_blob"]
            posting_offsets_raw = tuple(state["posting_offsets"])
            reverse_blob = state["reverse_blob"]
            reverse_offsets_raw = tuple(state["reverse_offsets"])
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid compact snapshot state") from exc

        high_water = self._validate_metadata_values(
            mode=mode,
            fields=fields,
            weights=weights,
            evidence=evidence,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=max_doc_id,
            doc_ids=doc_ids,
            terms=terms,
            totals=totals_raw,
            lengths=lengths_raw,
            df=df_raw,
            dfe=dfe_raw,
            posting_blob=posting_blob,
            posting_offsets=posting_offsets_raw,
            reverse_blob=reverse_blob,
            reverse_offsets=reverse_offsets_raw,
        )
        candidate = type(self).__new__(type(self))
        vocabulary = _PackedVocabulary.from_validated_terms(terms)
        candidate._install(
            mode=mode,
            fields=fields,
            weights=weights,
            evidence=evidence,
            k1=k1,
            b=b,
            idf_pow=idf_pow,
            max_doc_id=high_water,
            doc_ids=doc_ids,
            vocabulary=vocabulary,
            totals=_minimal_uint_array(totals_raw),
            lengths=tuple(_minimal_uint_array(values) for values in lengths_raw),
            df=_minimal_uint_array(df_raw),
            dfe=_minimal_uint_array(dfe_raw),
            posting_blob=posting_blob,
            posting_offsets=_minimal_uint_array(posting_offsets_raw),
            reverse_blob=reverse_blob,
            reverse_offsets=_minimal_uint_array(reverse_offsets_raw),
            retains_reverse=True,
        )
        candidate._validate_storage()
        self.__dict__.update(candidate.__dict__)

    def _decode_document(
        self, doc_position: int
    ) -> tuple[tuple[int, ...], tuple[dict[int, int], ...]]:
        """Decode one base document into field lengths and term-id frequencies."""
        if not self._retains_reverse:
            raise RuntimeError("forward-only snapshot has no reverse document records")
        if type(doc_position) is not int or not 0 <= doc_position < self.N:
            raise ValueError("compact document position is out of range")
        position = self._reverse_offsets[doc_position]
        end = self._reverse_offsets[doc_position + 1]
        decoded_lengths: list[int] = []
        for _field in self.fields:
            length, position = _decode_uvarint(self._reverse_blob, position, end)
            decoded_lengths.append(length)

        counts_by_field: list[dict[int, int]] = []
        for field_index, _field in enumerate(self.fields):
            entry_count, position = _decode_uvarint(self._reverse_blob, position, end)
            previous = 0
            counts: dict[int, int] = {}
            for entry_index in range(entry_count):
                delta, position = _decode_uvarint(self._reverse_blob, position, end)
                if entry_index and delta == 0:
                    raise ValueError(
                        "reverse term-id deltas after the first must be positive"
                    )
                term_id = delta if not entry_index else previous + delta
                tf, position = _decode_uvarint(self._reverse_blob, position, end)
                if term_id >= len(self.terms) or tf <= 0 or term_id in counts:
                    raise ValueError("compact reverse term record is invalid")
                counts[term_id] = tf
                previous = term_id
            if sum(counts.values()) != decoded_lengths[field_index]:
                raise ValueError("compact reverse length does not match frequencies")
            if decoded_lengths[field_index] != self._lengths[field_index][doc_position]:
                raise ValueError("compact reverse and length column differ")
            counts_by_field.append(counts)
        if position != end:
            raise ValueError("compact reverse record has trailing bytes")
        return tuple(decoded_lengths), tuple(counts_by_field)

    def _document_record(
        self, doc_id: int
    ) -> tuple[tuple[int, ...], tuple[dict[int, int], ...]] | None:
        position = self._doc_positions.get(_doc_id(doc_id))
        return None if position is None else self._decode_document(position)

    def _iter_document_records(self):
        for position, doc_id in enumerate(self.doc_ids):
            yield doc_id, self._decode_document(position)

    def _validate_storage(self) -> None:
        if not self._retains_reverse:
            raise RuntimeError("forward-only snapshots are trusted derived state")
        self._validate_installed_metadata()

        for term_id in range(len(self.terms)):
            previous = -1
            posting_count = 0
            for doc_id, frequencies in self._decode_posting(term_id):
                if doc_id <= previous or doc_id not in self._doc_positions:
                    raise ValueError("compact posting doc_ids are invalid")
                previous = doc_id
                posting_count += 1
            if not posting_count:
                raise ValueError("compact vocabulary terms require a posting")

        recomputed_totals = [0] * len(self.fields)
        recomputed_df = [0] * len(self.terms)
        recomputed_dfe = [0] * len(self.terms)
        expected_postings = [bytearray() for _term in self.terms]
        previous_posting_doc = [0] * len(self.terms)
        posting_counts = [0] * len(self.terms)
        for doc_position, doc_id in enumerate(self.doc_ids):
            decoded_lengths, counts_by_field = self._decode_document(doc_position)
            content: set[int] = set()
            evidence_only: set[int] = set()
            doc_frequencies: dict[int, list[int]] = {}
            for field_index, counts in enumerate(counts_by_field):
                present = evidence_only if self.evidence[field_index] else content
                present.update(counts)
                for term_id, tf in counts.items():
                    frequencies = doc_frequencies.setdefault(
                        term_id, [0] * len(self.fields)
                    )
                    frequencies[field_index] = tf
            for term_id, frequencies in doc_frequencies.items():
                posting_index = posting_counts[term_id]
                delta = (
                    doc_id
                    if not posting_index
                    else doc_id - previous_posting_doc[term_id]
                )
                _append_uvarint(expected_postings[term_id], delta)
                for tf in frequencies:
                    _append_uvarint(expected_postings[term_id], tf)
                previous_posting_doc[term_id] = doc_id
                posting_counts[term_id] += 1
            for field_index, length in enumerate(decoded_lengths):
                recomputed_totals[field_index] += length
            for term_id in content:
                recomputed_df[term_id] += 1
            for term_id in evidence_only:
                recomputed_dfe[term_id] += 1

        expected_blob = bytearray()
        expected_offsets = [0]
        for posting in expected_postings:
            expected_blob.extend(posting)
            expected_offsets.append(len(expected_blob))
        if bytes(expected_blob) != self._posting_blob or tuple(
            expected_offsets
        ) != tuple(self._posting_offsets):
            raise ValueError("compact forward and reverse postings differ")
        if tuple(recomputed_totals) != tuple(self._totals):
            raise ValueError("compact totals do not match reverse records")
        if tuple(recomputed_df) != tuple(self._df) or tuple(recomputed_dfe) != tuple(
            self._dfe
        ):
            raise ValueError(
                "compact document frequencies do not match reverse records"
            )
