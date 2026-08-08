"""Sparse mutable metadata layered over an immutable compact vocabulary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._compact_postings import CompactPostingsSnapshot


class _ExtraTerm:
    __slots__ = ("term", "df", "dfe")

    def __init__(self, term: str, df: int, dfe: int) -> None:
        self.term = term
        self.df = df
        self.dfe = dfe


class _TermView(Mapping[str, str]):
    __slots__ = ("_metadata", "_base")

    def __init__(self, metadata: _LayeredMetadata) -> None:
        self._metadata = metadata
        self._base = metadata._base

    def __getitem__(self, term: str) -> str:
        resolved = self._metadata.resolve_live(term)
        if resolved is None:
            raise KeyError(term)
        return resolved[0]

    def __iter__(self):
        yield from self._metadata.iter_terms()

    def __len__(self) -> int:
        return sum(1 for _term in self)

    def __contains__(self, term: object) -> bool:
        return self._metadata.resolve_live(term) is not None


class _TermValueView(Mapping[str, int]):
    __slots__ = ("_metadata", "_base", "_values", "_evidence")

    def __init__(
        self,
        metadata: _LayeredMetadata,
        values: Sequence[int],
        *,
        evidence: bool,
    ) -> None:
        self._metadata = metadata
        self._base = metadata._base
        self._values = values
        self._evidence = evidence

    def __getitem__(self, term: str) -> int:
        resolved = self._metadata.resolve_live(term)
        if resolved is None:
            raise KeyError(term)
        value = resolved[3] if self._evidence else resolved[2]
        if not value:
            raise KeyError(term)
        return value

    def __iter__(self):
        yield from self._metadata.iter_values(evidence=self._evidence)

    def __len__(self) -> int:
        return sum(1 for _term in self)

    def __contains__(self, term: object) -> bool:
        resolved = self._metadata.resolve_live(term)
        return resolved is not None and bool(
            resolved[3] if self._evidence else resolved[2]
        )


class _LayeredMetadata:
    """Current term statistics without copying the immutable base vocabulary.

    Base changes are signed deltas keyed by compact term id. Extra terms own one
    canonical string and absolute statistics. The public-facing mappings are views;
    only explicit iteration or ``len`` walks the base vocabulary.
    """

    __slots__ = (
        "_base",
        "_base_df",
        "_base_dfe",
        "_base_df_delta",
        "_base_dfe_delta",
        "_extras",
        "_term_tail",
        "_df_tail",
        "_dfe_tail",
        "terms",
        "df",
        "dfe",
    )

    def __init__(
        self,
        base: CompactPostingsSnapshot,
        df: Sequence[int],
        dfe: Sequence[int] | None = None,
    ) -> None:
        self._base = base
        self._base_df = df
        self._base_dfe = dfe
        self._base_df_delta: dict[int, int] = {}
        self._base_dfe_delta: dict[int, int] = {}
        self._extras: dict[str, _ExtraTerm] = {}
        self._term_tail: dict[int | str, None] = {}
        self._df_tail: dict[int | str, None] = {}
        self._dfe_tail: dict[int | str, None] = {}
        self.terms = _TermView(self)
        self.df = _TermValueView(self, df, evidence=False)
        self.dfe = None if dfe is None else _TermValueView(self, dfe, evidence=True)

    def _term_id(self, term: object) -> int | None:
        return self._base._term_ids.get(cast(str, term))

    def _base_stats(self, term_id: int) -> tuple[int, int]:
        df = int(self._base_df[term_id]) + self._base_df_delta.get(term_id, 0)
        dfe = (
            0
            if self._base_dfe is None
            else int(self._base_dfe[term_id]) + self._base_dfe_delta.get(term_id, 0)
        )
        return df, dfe

    def resolve_live(self, term: object) -> tuple[str, int | None, int, int] | None:
        term_id = self._term_id(term)
        if term_id is not None:
            df, dfe = self._base_stats(term_id)
            if df or dfe:
                return self._base.terms[term_id], term_id, df, dfe
            return None
        extra = self._extras.get(cast(str, term))
        if extra is None:
            return None
        return extra.term, None, extra.df, extra.dfe

    def canonicalize(self, term: str, staged: dict[str, str]) -> str:
        term_id = self._term_id(term)
        if term_id is not None:
            canonical = self._base.terms[term_id]
            df, dfe = self._base_stats(term_id)
            if not (df or dfe):
                staged.setdefault(canonical, canonical)
            return canonical
        extra = self._extras.get(term)
        if extra is not None:
            return extra.term
        return staged.setdefault(term, term)

    def iter_terms(self):
        for term_id, term in enumerate(self._base.terms):
            df, dfe = self._base_stats(term_id)
            if (df or dfe) and term_id not in self._term_tail:
                yield term
        for key in self._term_tail:
            if type(key) is int:
                df, dfe = self._base_stats(key)
                if df or dfe:
                    yield self._base.terms[key]
                continue
            extra = self._extras.get(cast(str, key))
            if extra is not None and (extra.df or extra.dfe):
                yield extra.term

    def iter_values(self, *, evidence: bool):
        tail = self._dfe_tail if evidence else self._df_tail
        for term_id, term in enumerate(self._base.terms):
            df, dfe = self._base_stats(term_id)
            value = dfe if evidence else df
            if value and term_id not in tail:
                yield term
        for key in tail:
            if type(key) is int:
                df, dfe = self._base_stats(key)
                value = dfe if evidence else df
                if value:
                    yield self._base.terms[key]
                continue
            extra = self._extras.get(cast(str, key))
            if extra is None:
                continue
            value = extra.dfe if evidence else extra.df
            if value:
                yield extra.term

    @staticmethod
    def _transition(old: int, new: int) -> int:
        if not old and new:
            return 1
        if old and not new:
            return -1
        return 0

    def prepare_patch(
        self,
        df_changes: Mapping[str, int],
        dfe_changes: Mapping[str, int] | None = None,
        *,
        new_terms: Mapping[str, str] | None = None,
    ) -> tuple[tuple[Any, ...], ...]:
        evidence_changes = dfe_changes or {}
        if self._base_dfe is None and any(evidence_changes.values()):
            raise ValueError(
                "plain compact metadata cannot contain evidence statistics"
            )
        staged_terms = new_terms or {}
        terms = dict.fromkeys((*staged_terms, *df_changes, *evidence_changes))
        prepared: list[tuple[Any, ...]] = []
        for term in terms:
            df_change = df_changes.get(term, 0)
            dfe_change = evidence_changes.get(term, 0)
            term_id = self._term_id(term)
            if term_id is not None:
                old_df, old_dfe = self._base_stats(term_id)
                new_df = old_df + df_change
                new_dfe = old_dfe + dfe_change
                if new_df < 0 or new_dfe < 0:
                    raise ValueError("compact metadata statistics cannot be negative")
                base_df = int(self._base_df[term_id])
                base_dfe = 0 if self._base_dfe is None else int(self._base_dfe[term_id])
                prepared.append(
                    (
                        0,
                        term_id,
                        new_df - base_df,
                        new_dfe - base_dfe,
                        self._transition(old_df or old_dfe, new_df or new_dfe),
                        self._transition(old_df, new_df),
                        self._transition(old_dfe, new_dfe),
                    )
                )
                continue
            extra = self._extras.get(term)
            old_df = 0 if extra is None else extra.df
            old_dfe = 0 if extra is None else extra.dfe
            new_df = old_df + df_change
            new_dfe = old_dfe + dfe_change
            if new_df < 0 or new_dfe < 0:
                raise ValueError("compact metadata statistics cannot be negative")
            canonical = staged_terms.get(term, term) if extra is None else extra.term
            prepared.append(
                (
                    1,
                    canonical,
                    new_df,
                    new_dfe,
                    self._transition(old_df or old_dfe, new_df or new_dfe),
                    self._transition(old_df, new_df),
                    self._transition(old_dfe, new_dfe),
                )
            )
        return tuple(prepared)

    @staticmethod
    def _apply_order(
        tail: dict[int | str, None], key: int | str, transition: int
    ) -> None:
        if transition > 0:
            tail[key] = None
        elif transition < 0:
            tail.pop(key, None)

    def commit_patch(self, patch: Sequence[tuple[Any, ...]]) -> None:
        for kind, key, df, dfe, term_order, df_order, dfe_order in patch:
            if kind == 0:
                if df:
                    self._base_df_delta[key] = df
                else:
                    self._base_df_delta.pop(key, None)
                if self._base_dfe is not None:
                    if dfe:
                        self._base_dfe_delta[key] = dfe
                    else:
                        self._base_dfe_delta.pop(key, None)
            elif df or dfe:
                extra = self._extras.get(key)
                if extra is None:
                    self._extras[key] = _ExtraTerm(key, df, dfe)
                else:
                    extra.df = df
                    extra.dfe = dfe
            else:
                self._extras.pop(key, None)
            self._apply_order(self._term_tail, key, term_order)
            self._apply_order(self._df_tail, key, df_order)
            self._apply_order(self._dfe_tail, key, dfe_order)

    def sparse_counts(self) -> dict[str, int]:
        return {
            "base_df_deltas": len(self._base_df_delta),
            "base_dfe_deltas": len(self._base_dfe_delta),
            "extra_terms": len(self._extras),
        }
