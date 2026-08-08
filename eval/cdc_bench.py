"""Claim-grade incremental corpus benchmark against synaptic-memory v0.27.0.

The controller reuses the official external-dataset preprocessing and provenance
contracts from :mod:`perf_bench`.  Each system runs in a fresh isolated process.
Within a worker, a deterministic, qrels-blind change trace is applied to the
system's native mutable path and compared bit-for-bit with a fresh full rebuild
of the same system and final corpus.

Example::

    python -X utf8 eval/cdc_bench.py \
        --data-dir <synaptic>/tests/benchmark/data \
        --dataset nfcorpus.json --synaptic-repo <synaptic> \
        --doctor-manifest <immutable-doctor.json> --out <new-result.json>
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(EVAL_DIR))

import perf_bench as perf  # noqa: E402 - sibling benchmark contracts
from metrics import BenchmarkResult  # noqa: E402 - byte-identical official scorer
from provenance import (  # noqa: E402 - sibling benchmark helper
    ProvenanceError,
    assert_artifact_unchanged,
    assert_unchanged,
    canonical_json_sha256,
    capture_worker_identity,
    default_worker_directory,
    ensure_output_absent,
    file_fingerprint,
    new_worker_run_id,
    read_bytes_artifact,
    read_json_artifact,
    run_with_launcher_pid,
    validate_worker_identity,
    worker_process_summary,
    write_json_once,
)

PROTOCOL = "official-external-memory-cdc-v2"
SCHEMA = "omnifuse.eval.cdc"
SCHEMA_VERSION = 3
WORKER_SCHEMA = "omnifuse.eval.cdc.worker"
WORKER_SCHEMA_VERSION = 3
PROVENANCE_LEVEL = "official-tag-cdc-isolated-write-once-v3"
WORKER_INPUT_DISPLAY_PATH = "worker-input/cdc.json"
DEFAULT_SEED = 42
DEFAULT_GROUP_FRACTION = 0.01
DEFAULT_TRIALS = 2
DEFAULT_STEADY_REPEATS = 5
K = 10
CANDIDATE_LIMIT = 20
SYSTEMS = ("omnifuse", "synaptic")
SYSTEM_NAMES = {"omnifuse": "OmniFuse", "synaptic": "synaptic"}
METRIC_NAMES = (
    "mrr_at_20",
    "mrr_at_10",
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "ndcg_at_10",
)
MEMORY_SCOPE = (
    "entire fresh worker process before the separate phase-checkpoint "
    "verification lane"
)
SYNAPTIC_NOOP_ADAPTER_SEMANTICS = (
    "get_node field equality check; skip MemoryBackend.update_node when unchanged"
)
WORKER_INPUT_KEYS = {
    "path",
    "sha256",
    "bytes",
    "documents",
    "source_scored_queries",
    "active_scored_queries",
    "active_query_ids_ordered_sha256",
    "active_relevance_judgments",
}
RESULT_KEYS = {
    "system",
    "trace_sha256",
    "mutation",
    "timing",
    "memory",
    "metrics",
    "active_queries",
    "exactness",
    "phase_checkpoints",
    "runtime",
}
OMNIFUSE_BINDINGS = frozenset(
    {
        "package",
        "build_inmemory",
        "retrieve",
        "upsert_chunks",
        "delete_chunks",
        "compact_postings",
        "compact_mutable_bm25",
        "compact_mutable_bm25f",
        "tokenize",
    }
)
SYNAPTIC_BINDINGS = frozenset(
    {
        "package",
        "memory_backend",
        "graph",
        "save_node",
        "update_node",
        "delete_node",
        "official_external_driver",
    }
)

CorpusRow = perf.CorpusRow
Query = perf.Query
ScoredHit = tuple[str, str]
Rankings = dict[str, list[ScoredHit]]


@dataclass(frozen=True)
class MutationTrace:
    """Disjoint deterministic operations selected without consulting qrels."""

    seed: int
    group_fraction: float
    insert: tuple[str, ...]
    update: tuple[str, ...]
    delete: tuple[str, ...]
    noop: tuple[str, ...]

    @property
    def group_size(self) -> int:
        return len(self.insert)

    def payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "group_fraction": self.group_fraction,
            "insert": list(self.insert),
            "update": list(self.update),
            "delete": list(self.delete),
            "noop": list(self.noop),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.payload())


def _trace_key(seed: int, document_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{document_id}".encode("utf-8")).digest()


def build_mutation_trace(
    corpus: Sequence[CorpusRow], *, seed: int, group_fraction: float
) -> MutationTrace:
    """Choose four exact-size groups using only ``seed`` and document IDs."""
    perf._require_unique_nonempty_ids(corpus, label="corpus", error_type=ValueError)
    if not 0.0 < group_fraction <= 0.25:
        raise ValueError("group_fraction must be greater than 0 and at most 0.25")
    if len(corpus) < 4:
        raise ValueError("CDC benchmark requires at least four corpus documents")
    group_size = max(1, math.floor(len(corpus) * group_fraction))
    if group_size * 4 > len(corpus):
        raise ValueError("mutation groups exceed the corpus")
    ordered = sorted(
        (document_id for document_id, _title, _text in corpus),
        key=lambda document_id: (_trace_key(seed, document_id), document_id),
    )
    groups = [
        tuple(ordered[offset : offset + group_size])
        for offset in range(0, group_size * 4, group_size)
    ]
    trace = MutationTrace(seed, group_fraction, *groups)
    selected = trace.insert + trace.update + trace.delete + trace.noop
    if len(set(selected)) != len(selected):
        raise RuntimeError("mutation trace groups are not disjoint")
    return trace


def _half_text(text: str) -> str:
    if not text:
        raise ValueError("an update target must have non-empty text")
    shortened = text[: len(text) // 2]
    if shortened == text:
        raise RuntimeError("update pre-image unexpectedly equals its final text")
    return shortened


def _official_synaptic_fields(title: str, text: str) -> tuple[str, str]:
    """Match upstream ``_build_graph`` title fallback and graph NFC ingest."""
    return (
        unicodedata.normalize("NFC", title or text[:80]),
        unicodedata.normalize("NFC", text),
    )


def _corpus_states(
    corpus: Sequence[CorpusRow], trace: MutationTrace
) -> tuple[list[CorpusRow], list[CorpusRow]]:
    """Return candidate pre-image and final-oracle corpus in identical tie order."""
    rows = {row[0]: row for row in corpus}
    if set(trace.insert + trace.update + trace.delete + trace.noop) - set(rows):
        raise ValueError("mutation trace references a document outside the corpus")
    inserts = set(trace.insert)
    updates = set(trace.update)
    deletes = set(trace.delete)
    initial: list[CorpusRow] = []
    for document_id, title, text in corpus:
        if document_id in inserts:
            continue
        initial.append(
            (document_id, title, _half_text(text) if document_id in updates else text)
        )
    final = [row for row in corpus if row[0] not in inserts and row[0] not in deletes]
    final.extend(rows[document_id] for document_id in trace.insert)
    return initial, final


def _phase_corpora(
    corpus: Sequence[CorpusRow], trace: MutationTrace
) -> list[tuple[str, list[CorpusRow]]]:
    """Materialize the expected corpus after each ordered CDC phase."""
    initial, _final = _corpus_states(corpus, trace)
    source = {row[0]: row for row in corpus}
    current = list(initial)
    checkpoints: list[tuple[str, list[CorpusRow]]] = []

    current.extend(source[document_id] for document_id in trace.insert)
    checkpoints.append(("insert", list(current)))

    updates = set(trace.update)
    current = [source[row[0]] if row[0] in updates else row for row in current]
    checkpoints.append(("update", list(current)))

    deletes = set(trace.delete)
    current = [row for row in current if row[0] not in deletes]
    checkpoints.append(("delete", list(current)))

    checkpoints.append(("noop", list(current)))
    return checkpoints


def _active_queries(
    queries: Sequence[Query], corpus: Sequence[CorpusRow]
) -> list[Query]:
    """Filter qrels to live documents and drop queries with no live judgment."""
    active_ids = {row[0] for row in corpus}
    active: list[Query] = []
    for query_id, text, relevant in queries:
        filtered = relevant & active_ids
        if filtered:
            active.append((query_id, text, filtered))
    if not active:
        raise ValueError("CDC corpus has no actively scored queries")
    return active


def _active_query_contract(queries: Sequence[Query]) -> dict[str, Any]:
    return {
        "count": len(queries),
        "query_ids_ordered_sha256": canonical_json_sha256(
            [query_id for query_id, _text, _relevant in queries]
        ),
        "relevance_judgments": sum(
            len(relevant) for _query_id, _text, relevant in queries
        ),
    }


def _checkpoint_contracts(
    corpus: Sequence[CorpusRow], queries: Sequence[Query], trace: MutationTrace
) -> list[dict[str, Any]]:
    return [
        {"phase": phase, **_active_query_contract(_active_queries(queries, rows))}
        for phase, rows in _phase_corpora(corpus, trace)
    ]


def _trace_summary(trace: MutationTrace) -> dict[str, Any]:
    payload = trace.payload()
    return {
        "algorithm": "sort by SHA256(utf8(seed + NUL + external_document_id))",
        "qrels_used_for_selection": False,
        "sha256": trace.sha256,
        "group_size": trace.group_size,
        "groups": payload,
    }


def _score_hex(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("retrieval produced a non-finite score")
    return number.hex()


def _normalize_hits(hits: Iterable[tuple[Any, Any]]) -> list[ScoredHit]:
    normalized: list[ScoredHit] = []
    seen: set[str] = set()
    for raw_id, raw_score in hits:
        document_id = str(raw_id)
        if not document_id or document_id in seen:
            raise RuntimeError("retrieval returned an empty or duplicate document ID")
        normalized.append((document_id, _score_hex(float(raw_score))))
        seen.add(document_id)
        if len(normalized) == CANDIDATE_LIMIT:
            break
    return normalized


def _ranking_rows(queries: Sequence[Query], rankings: Rankings) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query_id,
            "hits": [
                {"document_id": document_id, "score_hex": score_hex}
                for document_id, score_hex in rankings[query_id]
            ],
        }
        for query_id, _text, _relevant in queries
    ]


def _ranking_sha256(queries: Sequence[Query], rankings: Rankings) -> str:
    return canonical_json_sha256(_ranking_rows(queries, rankings))


def _assert_exact_rankings(expected: Rankings, actual: Rankings, *, label: str) -> None:
    if actual != expected:
        expected_hash = canonical_json_sha256(expected)
        actual_hash = canonical_json_sha256(actual)
        raise RuntimeError(
            f"{label} ranking/score mismatch: expected={expected_hash}, actual={actual_hash}"
        )


def _latency(samples_ms: Sequence[float]) -> dict[str, float | int]:
    if not samples_ms:
        raise RuntimeError("query timing produced no samples")
    return {
        "count": len(samples_ms),
        "min_ms": min(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": perf._percentile(samples_ms, 0.95),
        "mean_ms": statistics.fmean(samples_ms),
        "max_ms": max(samples_ms),
    }


def _metrics(queries: Sequence[Query], rankings: Rankings) -> dict[str, float]:
    benchmark = BenchmarkResult()
    mrr_at_10: list[float] = []
    for query_id, text, relevant in queries:
        retrieved = [document_id for document_id, _score in rankings[query_id]]
        benchmark.add(
            query_id=query_id,
            query=text,
            retrieved=retrieved,
            relevant=relevant,
            k=K,
        )
        mrr_at_10.append(
            next(
                (
                    1.0 / rank
                    for rank, document_id in enumerate(retrieved[:K], 1)
                    if document_id in relevant
                ),
                0.0,
            )
        )
    summary = benchmark.summary()
    if not summary:
        raise RuntimeError("official scorer produced no rows")
    return {
        "mrr_at_20": float(summary["mrr"]),
        "mrr_at_10": statistics.fmean(mrr_at_10),
        "precision_at_10": float(summary["mean_precision@k"]),
        "recall_at_10": float(summary["mean_recall@k"]),
        "f1_at_10": float(summary["mean_f1@k"]),
        "ndcg_at_10": float(summary["mean_ndcg@k"]),
    }


def _memory_snapshot() -> dict[str, Any]:
    current, peak, kind = perf._process_memory_bytes()
    return {
        "kind": kind,
        "current_rss_mb": current / 1_000_000 if current is not None else None,
        "peak_rss_mb": peak / 1_000_000 if peak is not None else None,
    }


def _stabilized_memory_snapshot() -> dict[str, Any]:
    gc.collect()
    return _memory_snapshot()


def _prepare_omnifuse_documents(corpus: Sequence[CorpusRow]) -> list[dict[str, str]]:
    return [
        {"id": document_id, "title": title, "text": text}
        for document_id, title, text in corpus
    ]


def _prepare_synaptic_documents(
    corpus: Sequence[CorpusRow],
) -> dict[str, dict[str, str]]:
    return {
        document_id: {"title": title, "text": text}
        for document_id, title, text in corpus
    }


def _timed_sync_native_build(
    corpus: Sequence[CorpusRow],
    *,
    prepare: Callable[[Sequence[CorpusRow]], Any],
    build: Callable[[Any], Any],
) -> tuple[Any, float]:
    native_input = prepare(corpus)
    started = time.perf_counter_ns()
    result = build(native_input)
    elapsed = time.perf_counter_ns() - started
    del native_input
    gc.collect()
    if elapsed < 0:
        raise RuntimeError("perf_counter_ns moved backwards")
    return result, elapsed / 1_000_000_000.0


async def _timed_async_native_build(
    corpus: Sequence[CorpusRow],
    *,
    prepare: Callable[[Sequence[CorpusRow]], Any],
    build: Callable[[Any], Awaitable[Any]],
    finalize: Callable[[Any, Any], Any] | None = None,
) -> tuple[Any, float]:
    native_input = prepare(corpus)
    started = time.perf_counter_ns()
    built = await build(native_input)
    elapsed = time.perf_counter_ns() - started
    result = built if finalize is None else finalize(native_input, built)
    del native_input
    del built
    gc.collect()
    if elapsed < 0:
        raise RuntimeError("perf_counter_ns moved backwards")
    return result, elapsed / 1_000_000_000.0


def _memory_delta(
    row: Mapping[str, Any], phase: str, *, field: str = "current_rss_mb"
) -> float | None:
    before = row["memory"]["before_initial_ingest"][field]
    after = row["memory"][phase][field]
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _measure_sync_round(
    search: Callable[[str], Iterable[tuple[Any, Any]]], queries: Sequence[Query]
) -> tuple[Rankings, list[float], float]:
    rankings: Rankings = {}
    samples: list[float] = []
    round_started = time.perf_counter_ns()
    for query_id, text, _relevant in queries:
        started = time.perf_counter_ns()
        rankings[query_id] = _normalize_hits(search(text))
        elapsed = time.perf_counter_ns() - started
        if elapsed < 0:
            raise RuntimeError("perf_counter_ns moved backwards")
        samples.append(elapsed / 1_000_000.0)
    round_elapsed = time.perf_counter_ns() - round_started
    return rankings, samples, round_elapsed / 1_000_000_000.0


async def _measure_async_round(
    search: Callable[[str], Awaitable[Iterable[tuple[Any, Any]]]],
    queries: Sequence[Query],
) -> tuple[Rankings, list[float], float]:
    rankings: Rankings = {}
    samples: list[float] = []
    round_started = time.perf_counter_ns()
    for query_id, text, _relevant in queries:
        started = time.perf_counter_ns()
        rankings[query_id] = _normalize_hits(await search(text))
        elapsed = time.perf_counter_ns() - started
        if elapsed < 0:
            raise RuntimeError("perf_counter_ns moved backwards")
        samples.append(elapsed / 1_000_000.0)
    round_elapsed = time.perf_counter_ns() - round_started
    return rankings, samples, round_elapsed / 1_000_000_000.0


def _measurement_result(
    *,
    system: str,
    trace: MutationTrace,
    queries: Sequence[Query],
    initial_ingest_s: float,
    mutation_s: float,
    oracle_rebuild_s: float,
    cold_rankings: Rankings,
    cold_samples: Sequence[float],
    cold_round_s: float,
    steady_samples: Sequence[float],
    steady_rounds_s: Sequence[float],
    memory_before_initial: Mapping[str, Any],
    memory_after_initial: Mapping[str, Any],
    memory_after_mutation: Mapping[str, Any],
    memory_after_queries: Mapping[str, Any],
    mutation_detail: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    ranking_rows = _ranking_rows(queries, cold_rankings)
    steady_mean = statistics.fmean(steady_rounds_s)
    return {
        "system": SYSTEM_NAMES[system],
        "trace_sha256": trace.sha256,
        "mutation": {
            "seconds": mutation_s,
            **dict(mutation_detail),
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "initial_ingest_seconds": initial_ingest_s,
            "mutation_seconds": mutation_s,
            "cold_first": {
                "round_seconds": cold_round_s,
                "latency": _latency(cold_samples),
            },
            "steady": {
                "rounds": len(steady_rounds_s),
                "round_seconds": list(steady_rounds_s),
                "mean_round_seconds": steady_mean,
                "latency": _latency(steady_samples),
            },
            "end_to_end": {
                "incremental_mutation_plus_cold_seconds": mutation_s + cold_round_s,
                "initial_ingest_plus_mutation_plus_cold_seconds": initial_ingest_s
                + mutation_s
                + cold_round_s,
            },
            "oracle_full_rebuild_seconds_verification_only": oracle_rebuild_s,
        },
        "memory": {
            "scope": MEMORY_SCOPE,
            "before_initial_ingest": dict(memory_before_initial),
            "after_initial_ingest": dict(memory_after_initial),
            "after_mutation": dict(memory_after_mutation),
            "after_measured_queries": dict(memory_after_queries),
        },
        "metrics": _metrics(queries, cold_rankings),
        "active_queries": _active_query_contract(queries),
        "exactness": {
            "ordered_external_ids_and_float_hex": True,
            "candidate_equals_full_rebuild_oracle": True,
            "query_count": len(ranking_rows),
            "rankings_sha256": canonical_json_sha256(ranking_rows),
            "rankings": ranking_rows,
        },
        "phase_checkpoints": list(checkpoints),
        "runtime": dict(runtime),
    }


def run_omnifuse_cdc(
    corpus: Sequence[CorpusRow],
    queries: Sequence[Query],
    trace: MutationTrace,
    *,
    steady_repeats: int,
) -> dict[str, Any]:
    """Apply the trace via ``MutableInMemoryVector`` and verify a static oracle."""
    source_root = (ROOT / "src").resolve()
    sys.path.insert(0, str(source_root))
    import omnifuse as package
    from omnifuse import Chunk, build_inmemory
    from omnifuse._compact_mutable import CompactMutableBM25
    from omnifuse._compact_mutable_fielded import CompactMutableBM25F
    from omnifuse._compact_postings import CompactPostingsSnapshot
    from omnifuse.text import tokenize

    initial, final = _corpus_states(corpus, trace)
    active_queries = _active_queries(queries, final)
    memory_before_initial = _stabilized_memory_snapshot()
    candidate, initial_ingest_s = _timed_sync_native_build(
        initial,
        prepare=_prepare_omnifuse_documents,
        build=lambda documents: build_inmemory(
            [], [], documents, mutable=True, vector_k=CANDIDATE_LIMIT
        ),
    )
    memory_after_initial = _stabilized_memory_snapshot()
    vector_type = type(candidate.vector)
    bindings = {
        "package": perf._module_binding(
            package,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse",
        ),
        "build_inmemory": perf._module_binding(
            build_inmemory,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse.build_inmemory",
        ),
        "retrieve": perf._module_binding(
            type(candidate).retrieve,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse.OmniFuse.retrieve",
        ),
        "upsert_chunks": perf._module_binding(
            vector_type.upsert_chunks,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse.MutableInMemoryVector.upsert_chunks",
        ),
        "delete_chunks": perf._module_binding(
            vector_type.delete_chunks,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse.MutableInMemoryVector.delete_chunks",
        ),
        "compact_postings": perf._module_binding(
            CompactPostingsSnapshot,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse._compact_postings.CompactPostingsSnapshot",
        ),
        "compact_mutable_bm25": perf._module_binding(
            CompactMutableBM25,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse._compact_mutable.CompactMutableBM25",
        ),
        "compact_mutable_bm25f": perf._module_binding(
            CompactMutableBM25F,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse._compact_mutable_fielded.CompactMutableBM25F",
        ),
        "tokenize": perf._module_binding(
            tokenize,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse.text.tokenize",
        ),
    }
    rows = {document_id: (title, text) for document_id, title, text in corpus}

    def chunks(document_ids: Sequence[str]) -> list[Any]:
        return [
            Chunk(
                id=document_id,
                title=rows[document_id][0],
                text=rows[document_id][1],
            )
            for document_id in document_ids
        ]

    def apply_phase(target: Any, phase: str) -> Any:
        if phase == "insert":
            return target.upsert_chunks(chunks(trace.insert))
        if phase == "update":
            return target.upsert_chunks(chunks(trace.update))
        if phase == "delete":
            return target.delete_chunks(list(trace.delete))
        if phase == "noop":
            return target.upsert_chunks(chunks(trace.noop))
        raise RuntimeError(f"unknown CDC phase {phase!r}")

    phase_results: dict[str, Any] = {}
    phase_seconds: dict[str, float] = {}
    for phase in ("insert", "update", "delete", "noop"):
        started = time.perf_counter_ns()
        phase_results[phase] = apply_phase(candidate, phase)
        phase_seconds[phase] = (time.perf_counter_ns() - started) / 1_000_000_000.0
    mutation_s = sum(phase_seconds.values())
    if any(not result.incremental for result in phase_results.values()):
        raise RuntimeError("OmniFuse CDC phase performed a corpus-wide reindex")
    expected_writes = len(trace.insert) + len(trace.update) + len(trace.delete)
    changed = sum(result.changed for result in phase_results.values())
    if changed != expected_writes:
        raise RuntimeError("OmniFuse native changed-write count is invalid")
    if phase_results["noop"].unchanged != len(trace.noop):
        raise RuntimeError("OmniFuse no-op upserts were not recognized as unchanged")
    memory_after_mutation = _stabilized_memory_snapshot()

    def candidate_search(
        text: str, _candidate: Any = candidate
    ) -> Iterable[tuple[str, float]]:
        return (
            (chunk.id, score)
            for chunk, score in _candidate.retrieve(text, limit=CANDIDATE_LIMIT)
        )

    cold, cold_samples, cold_s = _measure_sync_round(candidate_search, active_queries)
    steady_samples: list[float] = []
    steady_rounds: list[float] = []
    for repeat in range(steady_repeats):
        rankings, samples, round_s = _measure_sync_round(
            candidate_search, active_queries
        )
        _assert_exact_rankings(
            cold, rankings, label=f"OmniFuse steady round {repeat + 1}"
        )
        steady_samples.extend(samples)
        steady_rounds.append(round_s)

    memory_after_queries = _stabilized_memory_snapshot()
    del candidate_search, candidate
    gc.collect()
    verification, _verification_build_s = _timed_sync_native_build(
        initial,
        prepare=_prepare_omnifuse_documents,
        build=lambda documents: build_inmemory(
            [], [], documents, mutable=True, vector_k=CANDIDATE_LIMIT
        ),
    )

    def verification_search(text: str) -> Iterable[tuple[str, float]]:
        return (
            (chunk.id, score)
            for chunk, score in verification.retrieve(text, limit=CANDIDATE_LIMIT)
        )

    checkpoints: list[dict[str, Any]] = []
    oracle_rebuild_s = 0.0
    for phase, phase_corpus in _phase_corpora(corpus, trace):
        verification_result = apply_phase(verification, phase)
        if not verification_result.incremental:
            raise RuntimeError(f"OmniFuse {phase} verification phase reindexed")
        checkpoint_queries = _active_queries(queries, phase_corpus)
        candidate_rankings, _samples, candidate_query_s = _measure_sync_round(
            verification_search, checkpoint_queries
        )
        oracle, oracle_build_s = _timed_sync_native_build(
            phase_corpus,
            prepare=_prepare_omnifuse_documents,
            build=lambda documents: build_inmemory(
                [], [], documents, vector_k=CANDIDATE_LIMIT
            ),
        )

        def oracle_search(
            text: str, _oracle: Any = oracle
        ) -> Iterable[tuple[str, float]]:
            return (
                (chunk.id, score)
                for chunk, score in _oracle.retrieve(text, limit=CANDIDATE_LIMIT)
            )

        oracle_rankings, _oracle_samples, oracle_query_s = _measure_sync_round(
            oracle_search, checkpoint_queries
        )
        _assert_exact_rankings(
            oracle_rankings,
            candidate_rankings,
            label=f"OmniFuse {phase} CDC checkpoint",
        )
        if phase == "noop":
            _assert_exact_rankings(
                cold,
                candidate_rankings,
                label="OmniFuse measured vs verification candidate",
            )
            oracle_rebuild_s = oracle_build_s
        ranking_hash = _ranking_sha256(checkpoint_queries, candidate_rankings)
        checkpoints.append(
            {
                "phase": phase,
                **_active_query_contract(checkpoint_queries),
                "ordered_external_ids_and_float_hex": True,
                "candidate_equals_full_rebuild_oracle": True,
                "difference_count": 0,
                "candidate_rankings_sha256": ranking_hash,
                "oracle_rankings_sha256": _ranking_sha256(
                    checkpoint_queries, oracle_rankings
                ),
                "verification_only": {
                    "candidate_query_seconds": candidate_query_s,
                    "oracle_rebuild_seconds": oracle_build_s,
                    "oracle_query_seconds": oracle_query_s,
                },
            }
        )
        del oracle_search, oracle
        gc.collect()
    return _measurement_result(
        system="omnifuse",
        trace=trace,
        queries=active_queries,
        initial_ingest_s=initial_ingest_s,
        mutation_s=mutation_s,
        oracle_rebuild_s=oracle_rebuild_s,
        cold_rankings=cold,
        cold_samples=cold_samples,
        cold_round_s=cold_s,
        steady_samples=steady_samples,
        steady_rounds_s=steady_rounds,
        memory_before_initial=memory_before_initial,
        memory_after_initial=memory_after_initial,
        memory_after_mutation=memory_after_mutation,
        memory_after_queries=memory_after_queries,
        mutation_detail={
            "phase_seconds": phase_seconds,
            "native_document_writes": expected_writes,
            "inserted": phase_results["insert"].inserted,
            "updated": phase_results["update"].updated,
            "deleted": phase_results["delete"].deleted,
            "unchanged": phase_results["noop"].unchanged,
            "missing": phase_results["delete"].missing,
            "incremental": True,
            "reindexed_documents": sum(
                result.reindexed for result in phase_results.values()
            ),
            "final_revision": phase_results["noop"].revision,
        },
        checkpoints=checkpoints,
        runtime={
            "package_version": getattr(package, "__version__", None)
            or perf._package_version("omnifuse"),
            "source_bindings": bindings,
            "tokenizer": None,
        },
    )


async def _build_synaptic_graph(
    prepared: Mapping[str, Mapping[str, str]], *, driver: Any
) -> tuple[Any, Mapping[Any, Any]]:
    return await driver._build_graph(prepared, no_embedding=True)


def _finalize_synaptic_graph(
    prepared: Mapping[str, Mapping[str, str]], built: tuple[Any, Mapping[Any, Any]]
) -> tuple[Any, Any, dict[str, str]]:
    graph, id_map = built
    if set(id_map) != set(prepared):
        missing = sorted(set(prepared) - set(id_map))[:5]
        raise RuntimeError(
            f"official Synaptic builder skipped CDC pre-image documents: {missing}"
        )
    return graph, graph.backend, {str(key): str(value) for key, value in id_map.items()}


async def run_synaptic_cdc(
    repo: Path,
    corpus: Sequence[CorpusRow],
    queries: Sequence[Query],
    trace: MutationTrace,
    *,
    steady_repeats: int,
) -> dict[str, Any]:
    """Apply native MemoryBackend CRUD and verify a fresh MemoryBackend oracle."""
    import direct_external_bench as direct

    direct._validate_tag_checkout(repo)
    driver, _scorer, official_runtime = direct._load_upstream_driver(repo)
    source_root = (repo / "src").resolve()
    sys.path.insert(0, str(source_root))
    import synaptic as package
    from synaptic.backends import sqlite as sqlite_module
    from synaptic.backends.memory import MemoryBackend
    from synaptic.graph import SynapticGraph
    from synaptic.models import NodeKind

    initial, final = _corpus_states(corpus, trace)
    active_queries = _active_queries(queries, final)
    memory_before_initial = _stabilized_memory_snapshot()
    (
        (candidate, backend, runtime_ids),
        initial_ingest_s,
    ) = await _timed_async_native_build(
        initial,
        prepare=_prepare_synaptic_documents,
        build=lambda documents: _build_synaptic_graph(documents, driver=driver),
        finalize=_finalize_synaptic_graph,
    )
    memory_after_initial = _stabilized_memory_snapshot()
    bindings = {
        "package": perf._module_binding(
            package,
            source_root=source_root,
            repository_root=repo,
            name="synaptic",
        ),
        "memory_backend": perf._module_binding(
            MemoryBackend,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.memory.MemoryBackend",
        ),
        "graph": perf._module_binding(
            SynapticGraph,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.graph.SynapticGraph",
        ),
        "save_node": perf._module_binding(
            MemoryBackend.save_node,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.memory.MemoryBackend.save_node",
        ),
        "update_node": perf._module_binding(
            MemoryBackend.update_node,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.memory.MemoryBackend.update_node",
        ),
        "delete_node": perf._module_binding(
            MemoryBackend.delete_node,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.memory.MemoryBackend.delete_node",
        ),
        "official_external_driver": perf._module_binding(
            driver,
            source_root=(repo / "tests").resolve(),
            repository_root=repo,
            name="tests.benchmark.test_external_datasets",
        ),
    }
    rows = {document_id: (title, text) for document_id, title, text in corpus}

    async def apply_phase(
        graph: Any,
        active_backend: Any,
        active_runtime_ids: dict[str, str],
        phase: str,
    ) -> dict[str, int]:
        counts = {"inserted": 0, "updated": 0, "deleted": 0, "unchanged": 0}
        if phase == "insert":
            for document_id in trace.insert:
                title, text = rows[document_id]
                title, text = _official_synaptic_fields(title, text)
                node = await graph.add(
                    title=title,
                    content=text,
                    kind=NodeKind.CONCEPT,
                    source="benchmark",
                )
                runtime_id = str(node.id)
                if runtime_id in active_runtime_ids.values():
                    raise RuntimeError("synaptic insert reused a runtime node ID")
                active_runtime_ids[document_id] = runtime_id
                counts["inserted"] += 1
            return counts
        if phase == "update":
            for document_id in trace.update:
                title, text = rows[document_id]
                title, text = _official_synaptic_fields(title, text)
                runtime_id = active_runtime_ids.get(document_id)
                if runtime_id is None:
                    raise RuntimeError("synaptic update external ID is unmapped")
                node = await graph.update(runtime_id, title=title, content=text)
                if node is None:
                    raise RuntimeError("synaptic update target is missing")
                counts["updated"] += 1
            return counts
        if phase == "delete":
            for document_id in trace.delete:
                runtime_id = active_runtime_ids.get(document_id)
                if runtime_id is None or not await graph.remove(runtime_id):
                    raise RuntimeError("synaptic delete target is missing")
                del active_runtime_ids[document_id]
                counts["deleted"] += 1
            return counts
        if phase == "noop":
            for document_id in trace.noop:
                title, text = rows[document_id]
                title, text = _official_synaptic_fields(title, text)
                runtime_id = active_runtime_ids.get(document_id)
                node = (
                    await active_backend.get_node(runtime_id)
                    if runtime_id is not None
                    else None
                )
                if node is None:
                    raise RuntimeError("synaptic no-op target is missing")
                if node.title != title or node.content != text:
                    raise RuntimeError(
                        "synaptic no-op target differs from its final document"
                    )
                counts["unchanged"] += 1
            return counts
        raise RuntimeError(f"unknown CDC phase {phase!r}")

    phase_counts: dict[str, dict[str, int]] = {}
    phase_seconds: dict[str, float] = {}
    for phase in ("insert", "update", "delete", "noop"):
        started = time.perf_counter_ns()
        phase_counts[phase] = await apply_phase(candidate, backend, runtime_ids, phase)
        phase_seconds[phase] = (time.perf_counter_ns() - started) / 1_000_000_000.0
    mutation_s = sum(phase_seconds.values())
    inserted = phase_counts["insert"]["inserted"]
    updated = phase_counts["update"]["updated"]
    deleted = phase_counts["delete"]["deleted"]
    unchanged = phase_counts["noop"]["unchanged"]
    native_writes = inserted + updated + deleted
    memory_after_mutation = _stabilized_memory_snapshot()
    candidate_reverse_ids = {
        runtime_id: external_id for external_id, runtime_id in runtime_ids.items()
    }

    async def candidate_search(
        text: str,
        _candidate: Any = candidate,
        _reverse_ids: Mapping[str, str] = candidate_reverse_ids,
    ) -> Iterable[tuple[str, float]]:
        result = await _candidate.search(text, limit=CANDIDATE_LIMIT)
        hits: list[tuple[str, float]] = []
        for activated in result.nodes:
            try:
                external_id = _reverse_ids[str(activated.node.id)]
            except KeyError as exc:
                raise RuntimeError(
                    f"synaptic returned unknown runtime node {exc.args[0]!r}"
                ) from exc
            hits.append((external_id, activated.resonance))
        return hits

    cold, cold_samples, cold_s = await _measure_async_round(
        candidate_search, active_queries
    )
    steady_samples: list[float] = []
    steady_rounds: list[float] = []
    for repeat in range(steady_repeats):
        rankings, samples, round_s = await _measure_async_round(
            candidate_search, active_queries
        )
        _assert_exact_rankings(
            cold, rankings, label=f"synaptic steady round {repeat + 1}"
        )
        steady_samples.extend(samples)
        steady_rounds.append(round_s)

    memory_after_queries = _stabilized_memory_snapshot()
    await candidate.backend.close()
    del candidate_search, candidate, backend, runtime_ids, candidate_reverse_ids
    gc.collect()

    (
        (
            verification,
            verification_backend,
            verification_ids,
        ),
        _verification_build_s,
    ) = await _timed_async_native_build(
        initial,
        prepare=_prepare_synaptic_documents,
        build=lambda documents: _build_synaptic_graph(documents, driver=driver),
        finalize=_finalize_synaptic_graph,
    )
    verification_reverse_ids: dict[str, str] = {}

    async def verification_search(text: str) -> Iterable[tuple[str, float]]:
        result = await verification.search(text, limit=CANDIDATE_LIMIT)
        hits: list[tuple[str, float]] = []
        for activated in result.nodes:
            if _score_hex(activated.activation) != _score_hex(activated.resonance):
                raise RuntimeError(
                    "synaptic evidence activation and resonance scores differ"
                )
            try:
                external_id = verification_reverse_ids[str(activated.node.id)]
            except KeyError as exc:
                raise RuntimeError(
                    f"synaptic returned unknown runtime node {exc.args[0]!r}"
                ) from exc
            hits.append((external_id, activated.resonance))
        return hits

    checkpoints: list[dict[str, Any]] = []
    oracle_rebuild_s = 0.0
    for phase, phase_corpus in _phase_corpora(corpus, trace):
        await apply_phase(verification, verification_backend, verification_ids, phase)
        verification_reverse_ids.clear()
        verification_reverse_ids.update(
            {
                runtime_id: external_id
                for external_id, runtime_id in verification_ids.items()
            }
        )
        checkpoint_queries = _active_queries(queries, phase_corpus)
        candidate_rankings, _samples, candidate_query_s = await _measure_async_round(
            verification_search, checkpoint_queries
        )
        (
            (
                oracle,
                _oracle_backend,
                oracle_ids,
            ),
            oracle_build_s,
        ) = await _timed_async_native_build(
            phase_corpus,
            prepare=_prepare_synaptic_documents,
            build=lambda documents: _build_synaptic_graph(documents, driver=driver),
            finalize=_finalize_synaptic_graph,
        )
        oracle_reverse_ids = {
            runtime_id: external_id for external_id, runtime_id in oracle_ids.items()
        }

        async def oracle_search(
            text: str,
            _oracle: Any = oracle,
            _reverse_ids: Mapping[str, str] = oracle_reverse_ids,
        ) -> Iterable[tuple[str, float]]:
            result = await _oracle.search(text, limit=CANDIDATE_LIMIT)
            hits: list[tuple[str, float]] = []
            for activated in result.nodes:
                if _score_hex(activated.activation) != _score_hex(activated.resonance):
                    raise RuntimeError(
                        "synaptic oracle activation and resonance scores differ"
                    )
                try:
                    external_id = _reverse_ids[str(activated.node.id)]
                except KeyError as exc:
                    raise RuntimeError(
                        f"synaptic oracle returned unknown runtime node {exc.args[0]!r}"
                    ) from exc
                hits.append((external_id, activated.resonance))
            return hits

        oracle_rankings, _oracle_samples, oracle_query_s = await _measure_async_round(
            oracle_search, checkpoint_queries
        )
        _assert_exact_rankings(
            oracle_rankings,
            candidate_rankings,
            label=f"synaptic {phase} CDC checkpoint",
        )
        if phase == "noop":
            _assert_exact_rankings(
                cold,
                candidate_rankings,
                label="synaptic measured vs verification candidate",
            )
            oracle_rebuild_s = oracle_build_s
        ranking_hash = _ranking_sha256(checkpoint_queries, candidate_rankings)
        checkpoints.append(
            {
                "phase": phase,
                **_active_query_contract(checkpoint_queries),
                "ordered_external_ids_and_float_hex": True,
                "candidate_equals_full_rebuild_oracle": True,
                "difference_count": 0,
                "candidate_rankings_sha256": ranking_hash,
                "oracle_rankings_sha256": _ranking_sha256(
                    checkpoint_queries, oracle_rankings
                ),
                "verification_only": {
                    "candidate_query_seconds": candidate_query_s,
                    "oracle_rebuild_seconds": oracle_build_s,
                    "oracle_query_seconds": oracle_query_s,
                },
            }
        )
        await oracle.backend.close()
        del oracle_search, oracle, oracle_ids, oracle_reverse_ids
        gc.collect()
    result = _measurement_result(
        system="synaptic",
        trace=trace,
        queries=active_queries,
        initial_ingest_s=initial_ingest_s,
        mutation_s=mutation_s,
        oracle_rebuild_s=oracle_rebuild_s,
        cold_rankings=cold,
        cold_samples=cold_samples,
        cold_round_s=cold_s,
        steady_samples=steady_samples,
        steady_rounds_s=steady_rounds,
        memory_before_initial=memory_before_initial,
        memory_after_initial=memory_after_initial,
        memory_after_mutation=memory_after_mutation,
        memory_after_queries=memory_after_queries,
        mutation_detail={
            "phase_seconds": phase_seconds,
            "native_document_writes": native_writes,
            "inserted": inserted,
            "updated": updated,
            "deleted": deleted,
            "unchanged": unchanged,
            "missing": 0,
            "incremental": True,
            "reindexed_documents": 0,
            "noop_adapter_semantics": SYNAPTIC_NOOP_ADAPTER_SEMANTICS,
        },
        checkpoints=checkpoints,
        runtime={
            "package_version": getattr(package, "__version__", None)
            or perf._package_version("synaptic-memory"),
            "source_bindings": bindings,
            "tokenizer": perf._synaptic_tokenizer_evidence(sqlite_module),
            "official_external_runtime": official_runtime,
        },
    )
    await verification.backend.close()
    return result


def _worker_environment_contract(environment: Mapping[str, Any]) -> None:
    if set(environment) != perf.WORKER_ENVIRONMENT_KEYS:
        raise ProvenanceError("CDC worker environment fields are incomplete")
    perf._require_protocol_utf8_mode(
        perf.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        context="CDC worker",
        environment=environment,
    )
    required_true = ("isolated", "ignore_environment", "no_user_site", "safe_path")
    if any(environment.get(key) is not True for key in required_true):
        raise ProvenanceError("CDC worker is not isolated")
    if environment.get("user_site_enabled") is not False:
        raise ProvenanceError("CDC worker user site must be disabled")
    if environment.get("python_no_user_site_env") != "1":
        raise ProvenanceError("CDC worker did not enforce PYTHONNOUSERSITE")
    if any(
        environment.get(key) is not None
        for key in ("pythonpath", "pythonhome", "pythonusersite")
    ):
        raise ProvenanceError("CDC worker inherited a Python path override")
    if not all(
        isinstance(environment.get(key), str) and environment.get(key)
        for key in ("python", "python_executable")
    ):
        raise ProvenanceError("CDC worker Python identity is incomplete")


def _run_worker(args: argparse.Namespace) -> None:
    if (
        args.input_file is None
        or args.result_file is None
        or args.worker is None
        or args.worker_run_id is None
    ):
        raise ValueError(
            "worker mode requires input, result, system, and worker run ID"
        )
    if args.synaptic_repo is None:
        raise ValueError("worker mode requires the official synaptic checkout")
    ensure_output_absent(args.result_file)
    payload, input_fingerprint = read_bytes_artifact(
        args.input_file, display_path=WORKER_INPUT_DISPLAY_PATH
    )
    corpus, queries = perf._parse_frozen_input(payload)
    assert_artifact_unchanged(
        "CDC frozen input after read",
        args.input_file,
        input_fingerprint,
    )
    trace = build_mutation_trace(
        corpus, seed=args.seed, group_fraction=args.mutation_group_fraction
    )
    _initial, final = _corpus_states(corpus, trace)
    active_queries = _active_queries(queries, final)
    active_contract = _active_query_contract(active_queries)
    environment_before = perf._official_environment_lock_evidence(args.synaptic_repo)
    if args.worker == "omnifuse":
        result = run_omnifuse_cdc(
            corpus, queries, trace, steady_repeats=args.steady_repeats
        )
    else:
        result = asyncio.run(
            run_synaptic_cdc(
                args.synaptic_repo,
                corpus,
                queries,
                trace,
                steady_repeats=args.steady_repeats,
            )
        )
    environment_after = perf._official_environment_lock_evidence(args.synaptic_repo)
    assert_unchanged(
        "official CDC worker environment lock",
        environment_before,
        environment_after,
    )
    assert_artifact_unchanged(
        "CDC frozen input after measurement",
        args.input_file,
        input_fingerprint,
    )
    worker_environment = perf._worker_environment_snapshot()
    _worker_environment_contract(worker_environment)
    worker_identity = capture_worker_identity(args.worker_run_id)
    write_json_once(
        args.result_file,
        {
            "schema": WORKER_SCHEMA,
            "schema_version": WORKER_SCHEMA_VERSION,
            "status": "ok",
            "system": args.worker,
            "protocol": PROTOCOL,
            "contract": {
                "k": K,
                "candidate_limit": CANDIDATE_LIMIT,
                "steady_repeats": args.steady_repeats,
                "seed": args.seed,
                "mutation_group_fraction": args.mutation_group_fraction,
            },
            "input": {
                **input_fingerprint,
                "documents": len(corpus),
                "source_scored_queries": len(queries),
                "active_scored_queries": active_contract["count"],
                "active_query_ids_ordered_sha256": active_contract[
                    "query_ids_ordered_sha256"
                ],
                "active_relevance_judgments": active_contract["relevance_judgments"],
            },
            "trace": _trace_summary(trace),
            "worker_identity": worker_identity,
            "environment": worker_environment,
            "official_environment_lock": {
                "before": environment_before,
                "after": environment_after,
            },
            "result": result,
        },
    )


def _worker_command(
    args: argparse.Namespace,
    *,
    system: str,
    input_file: Path,
    result_file: Path,
    worker_run_id: str,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(SCRIPT_PATH),
        "--data-dir",
        str(args.data_dir),
        "--dataset",
        args.dataset,
        "--synaptic-repo",
        str(args.synaptic_repo),
        "--steady-repeats",
        str(args.steady_repeats),
        "--seed",
        str(args.seed),
        "--mutation-group-fraction",
        str(args.mutation_group_fraction),
        "--worker",
        system,
        "--input-file",
        str(input_file),
        "--result-file",
        str(result_file),
        "--worker-run-id",
        worker_run_id,
    ]


def _validate_bindings(
    raw: object, *, system: str, synaptic_repo: Path
) -> dict[str, Any]:
    expected = OMNIFUSE_BINDINGS if system == "omnifuse" else SYNAPTIC_BINDINGS
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ProvenanceError(f"{system} CDC source bindings are incomplete")
    source_root = (
        (ROOT / "src").resolve()
        if system == "omnifuse"
        else (synaptic_repo / "src").resolve()
    )
    validated: dict[str, Any] = {}
    for name, binding in raw.items():
        if not isinstance(binding, dict):
            raise ProvenanceError(f"{system} binding {name!r} is invalid")
        resolved = binding.get("resolved_path")
        if not isinstance(resolved, str):
            raise ProvenanceError(f"{system} binding {name!r} has no path")
        path = Path(resolved).resolve()
        expected_root = (
            (synaptic_repo / "tests").resolve()
            if system == "synaptic" and name == "official_external_driver"
            else source_root
        )
        if not perf._is_below(path, expected_root):
            raise ProvenanceError(f"{system} binding {name!r} escaped {expected_root}")
        if (
            system == "synaptic"
            and name == "official_external_driver"
            and path
            != (synaptic_repo / perf.SYNAPTIC_EXTERNAL_DRIVER_RELATIVE).resolve()
        ):
            raise ProvenanceError("synaptic official driver binding path is invalid")
        live = file_fingerprint(path, display_path=str(binding.get("path")))
        if live["sha256"] != binding.get("sha256") or live["bytes"] != binding.get(
            "bytes"
        ):
            raise ProvenanceError(f"{system} binding {name!r} changed after execution")
        validated[name] = dict(binding)
    return validated


def _require_number(value: object, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ProvenanceError(f"{label} is outside its valid range")
    return number


def _validate_exactness_rows(
    rows: object, *, expected_queries: Sequence[Query]
) -> Rankings:
    if not isinstance(rows, list) or len(rows) != len(expected_queries):
        raise ProvenanceError("CDC exactness rows have the wrong query count")
    rankings: Rankings = {}
    for row, (expected_query_id, _text, _relevant) in zip(
        rows, expected_queries, strict=True
    ):
        if (
            not isinstance(row, dict)
            or set(row) != {"query_id", "hits"}
            or row.get("query_id") != expected_query_id
        ):
            raise ProvenanceError("CDC exactness query ID/order is invalid")
        hits = row["hits"]
        if not isinstance(hits, list) or len(hits) > CANDIDATE_LIMIT:
            raise ProvenanceError("CDC exactness hit list is invalid")
        parsed: list[ScoredHit] = []
        document_ids: set[str] = set()
        for hit in hits:
            if not isinstance(hit, dict) or set(hit) != {
                "document_id",
                "score_hex",
            }:
                raise ProvenanceError("CDC exactness hit is invalid")
            document_id = hit["document_id"]
            score_hex = hit["score_hex"]
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id in document_ids
                or not isinstance(score_hex, str)
            ):
                raise ProvenanceError("CDC exactness hit fields are invalid")
            try:
                score = float.fromhex(score_hex)
            except ValueError as exc:
                raise ProvenanceError("CDC exactness score hex is invalid") from exc
            if not math.isfinite(score) or score.hex() != score_hex:
                raise ProvenanceError("CDC exactness score hex is not canonical")
            document_ids.add(document_id)
            parsed.append((document_id, score_hex))
        rankings[expected_query_id] = parsed
    return rankings


def _validate_phase_checkpoints(
    raw: object,
    *,
    expected: Sequence[Mapping[str, Any]],
    final_rankings_sha256: str,
    system: str,
) -> None:
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise ProvenanceError(f"{system} CDC phase checkpoints are incomplete")
    for checkpoint, contract in zip(raw, expected, strict=True):
        expected_keys = set(contract) | {
            "ordered_external_ids_and_float_hex",
            "candidate_equals_full_rebuild_oracle",
            "difference_count",
            "candidate_rankings_sha256",
            "oracle_rankings_sha256",
            "verification_only",
        }
        if not isinstance(checkpoint, dict) or set(checkpoint) != expected_keys:
            raise ProvenanceError(f"{system} CDC phase checkpoint is invalid")
        if any(checkpoint.get(key) != value for key, value in contract.items()):
            raise ProvenanceError(f"{system} CDC phase query contract changed")
        candidate_hash = checkpoint.get("candidate_rankings_sha256")
        oracle_hash = checkpoint.get("oracle_rankings_sha256")
        if (
            checkpoint.get("ordered_external_ids_and_float_hex") is not True
            or checkpoint.get("candidate_equals_full_rebuild_oracle") is not True
            or isinstance(checkpoint.get("difference_count"), bool)
            or checkpoint.get("difference_count") != 0
            or not isinstance(candidate_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", candidate_hash) is None
            or not isinstance(oracle_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", oracle_hash) is None
            or candidate_hash != oracle_hash
        ):
            raise ProvenanceError(f"{system} CDC phase exactness is invalid")
        verification = checkpoint.get("verification_only")
        if not isinstance(verification, dict) or set(verification) != {
            "candidate_query_seconds",
            "oracle_rebuild_seconds",
            "oracle_query_seconds",
        }:
            raise ProvenanceError(f"{system} CDC checkpoint timing is invalid")
        for name, value in verification.items():
            _require_number(value, label=f"{system}.{contract['phase']}.{name}")
    if raw[-1].get("candidate_rankings_sha256") != final_rankings_sha256:
        raise ProvenanceError(
            f"{system} final checkpoint differs from measured candidate rankings"
        )


def _validate_latency_summary(raw: object, *, expected_count: int, label: str) -> None:
    expected_keys = {"count", "min_ms", "p50_ms", "p95_ms", "mean_ms", "max_ms"}
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ProvenanceError(f"{label} latency summary is invalid")
    if isinstance(raw["count"], bool) or raw["count"] != expected_count:
        raise ProvenanceError(f"{label} latency sample count is invalid")
    values = {
        key: _require_number(raw[key], label=f"{label}.{key}")
        for key in expected_keys - {"count"}
    }
    if not (
        values["min_ms"] <= values["p50_ms"] <= values["p95_ms"] <= values["max_ms"]
        and values["min_ms"] <= values["mean_ms"] <= values["max_ms"]
    ):
        raise ProvenanceError(f"{label} latency distribution is inconsistent")


def _validate_memory(raw: object, *, system: str) -> None:
    phases = (
        "before_initial_ingest",
        "after_initial_ingest",
        "after_mutation",
        "after_measured_queries",
    )
    if (
        not isinstance(raw, dict)
        or set(raw) != {"scope", *phases}
        or raw.get("scope") != MEMORY_SCOPE
    ):
        raise ProvenanceError(f"{system} CDC memory evidence is invalid")
    kinds: list[str | None] = []
    peaks: list[float] = []
    for phase in phases:
        snapshot = raw.get(phase)
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "kind",
            "current_rss_mb",
            "peak_rss_mb",
        }:
            raise ProvenanceError(f"{system} CDC {phase} memory is invalid")
        kind = snapshot["kind"]
        if kind is not None and (not isinstance(kind, str) or not kind):
            raise ProvenanceError(f"{system} CDC memory kind is invalid")
        kinds.append(kind)
        values: dict[str, float | None] = {}
        for name in ("current_rss_mb", "peak_rss_mb"):
            value = snapshot[name]
            values[name] = (
                None
                if value is None
                else _require_number(value, label=f"{system}.{phase}.{name}")
            )
        if kind is None and any(value is not None for value in values.values()):
            raise ProvenanceError(f"{system} CDC memory kind is missing")
        current = values["current_rss_mb"]
        peak = values["peak_rss_mb"]
        if current is not None and peak is not None and current > peak:
            raise ProvenanceError(f"{system} CDC current RSS exceeds peak RSS")
        if peak is not None:
            peaks.append(peak)
    if len(set(kinds)) != 1:
        raise ProvenanceError(f"{system} CDC memory kind changed across phases")
    if any(after < before for before, after in zip(peaks, peaks[1:])):
        raise ProvenanceError(f"{system} CDC peak RSS decreased across phases")


def _validate_worker_result(
    raw: object,
    *,
    system: str,
    expected_input: Mapping[str, Any],
    expected_trace: MutationTrace,
    expected_queries: Sequence[Query],
    expected_checkpoints: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    trial_number: int,
    order_position: int,
    expected_worker_run_id: str,
    official_environment_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ProvenanceError("CDC worker output is not an object")
    if set(raw) != {
        "schema",
        "schema_version",
        "status",
        "system",
        "protocol",
        "contract",
        "input",
        "trace",
        "worker_identity",
        "environment",
        "official_environment_lock",
        "result",
    }:
        raise ProvenanceError(f"{system} CDC worker envelope is not strict")
    if (
        raw.get("schema") != WORKER_SCHEMA
        or raw.get("schema_version") != WORKER_SCHEMA_VERSION
        or raw.get("status") != "ok"
        or raw.get("system") != system
        or raw.get("protocol") != PROTOCOL
    ):
        raise ProvenanceError(f"{system} CDC worker envelope is invalid")
    contract = raw.get("contract")
    expected_contract = {
        "k": K,
        "candidate_limit": CANDIDATE_LIMIT,
        "steady_repeats": args.steady_repeats,
        "seed": args.seed,
        "mutation_group_fraction": args.mutation_group_fraction,
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != set(expected_contract)
        or contract != expected_contract
        or any(
            isinstance(contract[key], bool) or not isinstance(contract[key], int)
            for key in ("k", "candidate_limit", "steady_repeats", "seed")
        )
        or isinstance(contract["mutation_group_fraction"], bool)
        or not isinstance(contract["mutation_group_fraction"], (int, float))
    ):
        raise ProvenanceError(f"{system} CDC worker contract changed")
    worker_input = raw.get("input")
    if (
        not isinstance(worker_input, dict)
        or set(worker_input) != WORKER_INPUT_KEYS
        or not isinstance(worker_input["path"], str)
        or not worker_input["path"]
        or not isinstance(worker_input["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", worker_input["sha256"]) is None
        or not isinstance(worker_input["active_query_ids_ordered_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}", worker_input["active_query_ids_ordered_sha256"]
        )
        is None
        or any(
            isinstance(worker_input[key], bool)
            or not isinstance(worker_input[key], int)
            or worker_input[key] < 0
            for key in (
                "bytes",
                "documents",
                "source_scored_queries",
                "active_scored_queries",
                "active_relevance_judgments",
            )
        )
        or dict(worker_input) != {
            key: expected_input[key] for key in WORKER_INPUT_KEYS
        }
    ):
        raise ProvenanceError(f"{system} CDC worker input binding is invalid")
    trace = raw.get("trace")
    if trace != _trace_summary(expected_trace):
        raise ProvenanceError(f"{system} CDC mutation trace changed")
    worker_identity = validate_worker_identity(
        raw.get("worker_identity"),
        expected_run_id=expected_worker_run_id,
        label=f"{system} CDC worker identity",
    )
    environment = raw.get("environment")
    if not isinstance(environment, dict):
        raise ProvenanceError(f"{system} CDC worker environment is invalid")
    _worker_environment_contract(environment)
    locks = raw.get("official_environment_lock")
    if not isinstance(locks, dict) or set(locks) != {"before", "after"}:
        raise ProvenanceError(f"{system} worker lock evidence is invalid")
    for phase in ("before", "after"):
        lock = locks[phase]
        if lock != official_environment_lock:
            raise ProvenanceError(f"{system} worker environment lock differs")
        perf._validate_official_environment_lock(lock, args.synaptic_repo)
    result = raw.get("result")
    if (
        not isinstance(result, dict)
        or set(result) != RESULT_KEYS
        or result.get("system") != SYSTEM_NAMES[system]
        or result.get("trace_sha256") != expected_trace.sha256
    ):
        raise ProvenanceError(f"{system} CDC result is invalid")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
        raise ProvenanceError(f"{system} CDC metrics are incomplete")
    for name in METRIC_NAMES:
        metric = _require_number(metrics[name], label=f"{system}.{name}")
        if metric > 1.0:
            raise ProvenanceError(f"{system}.{name} exceeds 1")
    active_contract = _active_query_contract(expected_queries)
    if result.get("active_queries") != active_contract:
        raise ProvenanceError(f"{system} active query contract changed")
    exactness = result.get("exactness")
    if (
        not isinstance(exactness, dict)
        or set(exactness)
        != {
            "ordered_external_ids_and_float_hex",
            "candidate_equals_full_rebuild_oracle",
            "query_count",
            "rankings_sha256",
            "rankings",
        }
        or exactness.get("ordered_external_ids_and_float_hex") is not True
        or exactness.get("candidate_equals_full_rebuild_oracle") is not True
        or exactness.get("query_count") != len(expected_queries)
        or isinstance(exactness.get("query_count"), bool)
        or not isinstance(exactness.get("rankings"), list)
        or not isinstance(exactness.get("rankings_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", exactness["rankings_sha256"]) is None
        or canonical_json_sha256(exactness["rankings"])
        != exactness.get("rankings_sha256")
    ):
        raise ProvenanceError(f"{system} CDC exactness evidence is invalid")
    parsed_rankings = _validate_exactness_rows(
        exactness["rankings"], expected_queries=expected_queries
    )
    recomputed_metrics = _metrics(expected_queries, parsed_rankings)
    if recomputed_metrics != metrics:
        raise ProvenanceError(f"{system} CDC metrics do not match ranking evidence")
    _validate_phase_checkpoints(
        result.get("phase_checkpoints"),
        expected=expected_checkpoints,
        final_rankings_sha256=exactness["rankings_sha256"],
        system=system,
    )
    mutation = result.get("mutation")
    expected_writes = expected_trace.group_size * 3
    mutation_keys = {
        "seconds",
        "phase_seconds",
        "native_document_writes",
        "inserted",
        "updated",
        "deleted",
        "unchanged",
        "missing",
        "incremental",
        "reindexed_documents",
        "final_revision"
        if system == "omnifuse"
        else "noop_adapter_semantics",
    }
    if (
        not isinstance(mutation, dict)
        or set(mutation) != mutation_keys
        or any(
            isinstance(mutation.get(key), bool)
            or not isinstance(mutation.get(key), int)
            or mutation[key] < 0
            for key in (
                "native_document_writes",
                "inserted",
                "updated",
                "deleted",
                "unchanged",
                "missing",
                "reindexed_documents",
            )
        )
        or mutation.get("native_document_writes") != expected_writes
        or mutation.get("inserted") != expected_trace.group_size
        or mutation.get("updated") != expected_trace.group_size
        or mutation.get("deleted") != expected_trace.group_size
        or mutation.get("unchanged") != expected_trace.group_size
        or mutation.get("missing") != 0
        or mutation.get("incremental") is not True
        or mutation.get("reindexed_documents") != 0
    ):
        raise ProvenanceError(f"{system} CDC mutation accounting is invalid")
    if system == "omnifuse":
        final_revision = mutation["final_revision"]
        if (
            isinstance(final_revision, bool)
            or not isinstance(final_revision, int)
            or final_revision < 0
        ):
            raise ProvenanceError("omnifuse CDC final revision is invalid")
    elif mutation["noop_adapter_semantics"] != SYNAPTIC_NOOP_ADAPTER_SEMANTICS:
        raise ProvenanceError("synaptic CDC no-op adapter semantics changed")
    timing = result.get("timing")
    if not isinstance(timing, dict) or set(timing) != {
        "clock",
        "initial_ingest_seconds",
        "mutation_seconds",
        "cold_first",
        "steady",
        "end_to_end",
        "oracle_full_rebuild_seconds_verification_only",
    }:
        raise ProvenanceError(f"{system} CDC timing is invalid")
    if timing["clock"] != "time.perf_counter_ns":
        raise ProvenanceError(f"{system} CDC clock is invalid")
    for name in ("initial_ingest_seconds", "mutation_seconds"):
        _require_number(timing.get(name), label=f"{system}.{name}")
    if mutation.get("seconds") != timing.get("mutation_seconds"):
        raise ProvenanceError(f"{system} CDC mutation timing is inconsistent")
    phase_seconds = mutation.get("phase_seconds")
    if not isinstance(phase_seconds, dict) or set(phase_seconds) != {
        "insert",
        "update",
        "delete",
        "noop",
    }:
        raise ProvenanceError(f"{system} CDC phase mutation timing is incomplete")
    phase_total = sum(
        _require_number(value, label=f"{system}.{phase}_seconds")
        for phase, value in phase_seconds.items()
    )
    if not math.isclose(
        phase_total, timing["mutation_seconds"], rel_tol=0.0, abs_tol=1e-15
    ):
        raise ProvenanceError(f"{system} CDC phase mutation timing is inconsistent")
    cold = timing.get("cold_first")
    steady = timing.get("steady")
    end_to_end = timing.get("end_to_end")
    if (
        not isinstance(cold, dict)
        or set(cold) != {"round_seconds", "latency"}
        or not isinstance(steady, dict)
        or set(steady) != {"rounds", "round_seconds", "mean_round_seconds", "latency"}
        or isinstance(steady.get("rounds"), bool)
        or steady.get("rounds") != args.steady_repeats
        or not isinstance(steady.get("round_seconds"), list)
        or len(steady["round_seconds"]) != args.steady_repeats
        or not isinstance(end_to_end, dict)
        or set(end_to_end)
        != {
            "incremental_mutation_plus_cold_seconds",
            "initial_ingest_plus_mutation_plus_cold_seconds",
        }
    ):
        raise ProvenanceError(f"{system} CDC timing detail is incomplete")
    _validate_latency_summary(
        cold["latency"], expected_count=len(expected_queries), label=f"{system}.cold"
    )
    _validate_latency_summary(
        steady["latency"],
        expected_count=len(expected_queries) * args.steady_repeats,
        label=f"{system}.steady",
    )
    steady_rounds = [
        _require_number(value, label=f"{system}.steady_round_seconds")
        for value in steady["round_seconds"]
    ]
    steady_mean = _require_number(
        steady.get("mean_round_seconds"), label=f"{system}.steady_mean_seconds"
    )
    if not math.isclose(
        steady_mean,
        statistics.fmean(steady_rounds),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ProvenanceError(f"{system} CDC steady timing is inconsistent")
    mutation_seconds = _require_number(
        timing["mutation_seconds"], label=f"{system}.mutation_seconds"
    )
    cold_seconds = _require_number(
        cold.get("round_seconds"), label=f"{system}.cold_first_round_seconds"
    )
    actual_e2e = _require_number(
        end_to_end.get("incremental_mutation_plus_cold_seconds"),
        label=f"{system}.incremental_end_to_end_seconds",
    )
    if not math.isclose(
        actual_e2e,
        mutation_seconds + cold_seconds,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ProvenanceError(f"{system} CDC end-to-end timing is inconsistent")
    total_e2e = _require_number(
        end_to_end.get("initial_ingest_plus_mutation_plus_cold_seconds"),
        label=f"{system}.total_end_to_end_seconds",
    )
    if not math.isclose(
        total_e2e,
        timing["initial_ingest_seconds"] + mutation_seconds + cold_seconds,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ProvenanceError(f"{system} CDC total timing is inconsistent")
    final_oracle_seconds = result["phase_checkpoints"][-1]["verification_only"][
        "oracle_rebuild_seconds"
    ]
    if (
        timing.get("oracle_full_rebuild_seconds_verification_only")
        != final_oracle_seconds
    ):
        raise ProvenanceError(f"{system} CDC oracle timing is inconsistent")
    _validate_memory(result.get("memory"), system=system)
    runtime = result.get("runtime")
    runtime_keys = {"package_version", "source_bindings", "tokenizer"}
    if system == "synaptic":
        runtime_keys.add("official_external_runtime")
    if not isinstance(runtime, dict) or set(runtime) != runtime_keys:
        raise ProvenanceError(f"{system} CDC runtime is invalid")
    package_version = runtime["package_version"]
    if package_version is not None and (
        not isinstance(package_version, str) or not package_version
    ):
        raise ProvenanceError(f"{system} CDC package version is invalid")
    bindings = _validate_bindings(
        runtime.get("source_bindings"),
        system=system,
        synaptic_repo=args.synaptic_repo,
    )
    perf._validate_tokenizer_evidence(
        runtime.get("tokenizer"), system=system, require_kiwi=True
    )
    if system == "synaptic":
        official_runtime = runtime["official_external_runtime"]
        if not isinstance(official_runtime, dict) or set(official_runtime) != {
            "python_executable",
            "synaptic_package",
            "synaptic_version",
            "upstream_driver",
            "upstream_scorer",
        }:
            raise ProvenanceError(
                "synaptic official external runtime evidence is invalid"
            )
        expected_runtime_paths = {
            "python_executable": environment["python_executable"],
            "synaptic_package": bindings["package"]["resolved_path"],
            "upstream_driver": bindings["official_external_driver"][
                "resolved_path"
            ],
            "upstream_scorer": str(
                (args.synaptic_repo / perf.SYNAPTIC_SCORER_RELATIVE).resolve()
            ),
        }
        if any(
            official_runtime[key] != expected
            for key, expected in expected_runtime_paths.items()
        ):
            raise ProvenanceError(
                "synaptic official external runtime path evidence is invalid"
            )
        if official_runtime["synaptic_version"] != package_version:
            raise ProvenanceError(
                "synaptic official external runtime version evidence is invalid"
            )
    return {
        **result,
        "trial": {
            "number": trial_number,
            "order_position": order_position,
        },
        "worker_identity": worker_identity,
        "worker_environment": environment,
    }


def _aggregate(trials: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for system in SYSTEMS:
        rows = list(trials[system])
        if not rows:
            raise ProvenanceError(f"{system} has no CDC trials")
        ranking_hashes = {row["exactness"]["rankings_sha256"] for row in rows}
        metric_payloads = {canonical_json_sha256(row["metrics"]) for row in rows}
        checkpoint_payloads = {
            canonical_json_sha256(
                [
                    {
                        "phase": checkpoint["phase"],
                        "rankings_sha256": checkpoint["candidate_rankings_sha256"],
                    }
                    for checkpoint in row["phase_checkpoints"]
                ]
            )
            for row in rows
        }
        if (
            len(ranking_hashes) != 1
            or len(metric_payloads) != 1
            or len(checkpoint_payloads) != 1
        ):
            raise ProvenanceError(f"{system} CDC accuracy changed across trials")
        distributions = {
            "initial_ingest_seconds": perf._distribution(
                [row["timing"]["initial_ingest_seconds"] for row in rows]
            ),
            "mutation_seconds": perf._distribution(
                [row["timing"]["mutation_seconds"] for row in rows]
            ),
            "cold_first_round_seconds": perf._distribution(
                [row["timing"]["cold_first"]["round_seconds"] for row in rows]
            ),
            "steady_mean_round_seconds": perf._distribution(
                [row["timing"]["steady"]["mean_round_seconds"] for row in rows]
            ),
            "incremental_end_to_end_seconds": perf._distribution(
                [
                    row["timing"]["end_to_end"][
                        "incremental_mutation_plus_cold_seconds"
                    ]
                    for row in rows
                ]
            ),
            "rss_before_initial_ingest_mb": perf._optional_distribution(
                [
                    row["memory"]["before_initial_ingest"]["current_rss_mb"]
                    for row in rows
                ]
            ),
            "peak_rss_before_initial_ingest_mb": perf._optional_distribution(
                [row["memory"]["before_initial_ingest"]["peak_rss_mb"] for row in rows]
            ),
            "rss_after_initial_ingest_mb": perf._optional_distribution(
                [
                    row["memory"]["after_initial_ingest"]["current_rss_mb"]
                    for row in rows
                ]
            ),
            "peak_rss_after_initial_ingest_mb": perf._optional_distribution(
                [row["memory"]["after_initial_ingest"]["peak_rss_mb"] for row in rows]
            ),
            "rss_initial_ingest_delta_mb": perf._optional_distribution(
                [_memory_delta(row, "after_initial_ingest") for row in rows]
            ),
            "rss_after_mutation_delta_mb": perf._optional_distribution(
                [_memory_delta(row, "after_mutation") for row in rows]
            ),
            "rss_after_measured_queries_delta_mb": perf._optional_distribution(
                [_memory_delta(row, "after_measured_queries") for row in rows]
            ),
            "rss_after_mutation_mb": perf._optional_distribution(
                [row["memory"]["after_mutation"]["current_rss_mb"] for row in rows]
            ),
            "peak_rss_after_mutation_mb": perf._optional_distribution(
                [row["memory"]["after_mutation"]["peak_rss_mb"] for row in rows]
            ),
            "rss_after_measured_queries_mb": perf._optional_distribution(
                [
                    row["memory"]["after_measured_queries"]["current_rss_mb"]
                    for row in rows
                ]
            ),
            "peak_rss_after_measured_queries_mb": perf._optional_distribution(
                [row["memory"]["after_measured_queries"]["peak_rss_mb"] for row in rows]
            ),
        }
        results.append(
            {
                "system": SYSTEM_NAMES[system],
                "trial_count": len(rows),
                "order_positions": [row["trial"]["order_position"] for row in rows],
                "accuracy_consistent": True,
                "exact_rebuild_equivalence": True,
                "rankings_sha256": next(iter(ranking_hashes)),
                "phase_checkpoint_rankings_consistent": True,
                "phase_checkpoint_sha256": next(iter(checkpoint_payloads)),
                "metrics": rows[0]["metrics"],
                "distributions": distributions,
                "trials": rows,
            }
        )
    return results


def _worker_directory(output: Path, configured: Path | None) -> Path:
    if configured is not None:
        return configured.resolve()
    return default_worker_directory(ROOT, output, kind="cdc")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="nfcorpus.json")
    parser.add_argument("--synaptic-repo", required=True, type=Path)
    parser.add_argument("--doctor-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--workers-dir",
        type=Path,
        help="new durable directory for frozen input and raw worker JSON artifacts",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--steady-repeats", type=int, default=DEFAULT_STEADY_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mutation-group-fraction", type=float, default=DEFAULT_GROUP_FRACTION
    )
    parser.add_argument("--worker", choices=SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--input-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.steady_repeats < 2:
        parser.error("--steady-repeats must be at least 2")
    if args.trials < 2 or args.trials % 2:
        parser.error("--trials must be an even value of at least 2 for AB/BA order")
    if not 0.0 < args.mutation_group_fraction <= 0.25:
        parser.error("--mutation-group-fraction must be in (0, 0.25]")
    if args.worker is None and (args.out is None or args.doctor_manifest is None):
        parser.error("controller mode requires --out and --doctor-manifest")
    if args.worker is not None and (
        args.input_file is None
        or args.result_file is None
        or args.worker_run_id is None
    ):
        parser.error(
            "worker mode requires --input-file, --result-file, and --worker-run-id"
        )


def _report(
    *,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    postflight: Mapping[str, Any],
    cdc_source: Mapping[str, Any],
    frozen_input: Mapping[str, Any],
    trace: MutationTrace,
    active_queries: Sequence[Query],
    checkpoint_contracts: Sequence[Mapping[str, Any]],
    schedule: Sequence[Sequence[str]],
    results: Sequence[Mapping[str, Any]],
    workers_directory: Path,
    frozen_input_artifact: Mapping[str, Any],
    worker_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    process_summary = worker_process_summary(
        worker_records, expected_count=sum(len(order) for order in schedule)
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance_level": PROVENANCE_LEVEL,
        "dataset": {**state["dataset"], "doctor": state["doctor_link"]},
        "repositories": state["repositories"],
        "doctor_manifest": state["doctor_manifest"],
        "provenance": {
            "benchmark_sources": {**state["sources"], "cdc_runner": cdc_source},
            "frozen_worker_input": dict(frozen_input),
            "frozen_input_artifact": dict(frozen_input_artifact),
            "controller_before": {
                "repositories": state["repositories"],
                "dataset": state["dataset_fingerprint"],
                "cdc_runner": cdc_source,
                "official_environment_probe": state["official_environment_probe"],
            },
            "controller_after": postflight["after"],
        },
        "integrity": {
            **postflight["checks"],
            "cdc_runner_unchanged": True,
            "qrels_not_used_for_mutation_selection": True,
            "workers_exact_against_same_system_full_rebuild": True,
            "frozen_input_artifact_unchanged_before_publication": True,
            "worker_artifacts_unchanged_before_publication": True,
            "worker_run_ids_unique": True,
        },
        "workers_directory": str(workers_directory.resolve()),
        "worker_artifacts": [
            {
                key: record[key]
                for key in (
                    "trial_number",
                    "order_position",
                    "system",
                    "worker_run_id",
                    "artifact",
                )
            }
            for record in worker_records
        ],
        "worker_processes": [
            {
                key: record[key]
                for key in (
                    "trial_number",
                    "order_position",
                    "system",
                    "worker_run_id",
                    "launcher_pid",
                    "worker_pid",
                    "same_process_id",
                )
            }
            for record in worker_records
        ],
        "worker_process_summary": process_summary,
        "contract": {
            "protocol": PROTOCOL,
            "k": K,
            "candidate_limit": CANDIDATE_LIMIT,
            "seed": args.seed,
            "mutation_group_fraction": args.mutation_group_fraction,
            "trace": _trace_summary(trace),
            "final_active_queries": _active_query_contract(active_queries),
            "phase_checkpoint_query_contracts": list(checkpoint_contracts),
            "steady_repeats": args.steady_repeats,
            "independent_trials_per_system": args.trials,
            "trial_order": [list(order) for order in schedule],
            "counterbalanced_ab_ba": True,
            "accuracy_scorer": "byte-identical official tests/benchmark/metrics.py",
            "exactness": "ordered external IDs plus float.hex scores at top 20",
            "timing": "time.perf_counter_ns; mutation, cold-first, steady, and end-to-end are separate",
            "initial_ingest_boundary": (
                "canonical rows are converted to each system's native input before the "
                "clock; only the native build call is timed; ID-map validation and "
                "normalization are outside the clock"
            ),
            "memory_scope": (
                "whole fresh worker process sampled after full garbage collection, "
                "before native-input adaptation and after native-input release; "
                "current-RSS deltas use the same worker's before-initial-ingest "
                "snapshot; all samples precede the separate checkpoint verification lane"
            ),
            "worker_model": "one fresh isolated process per system and trial",
            "timed_result_adapter": {
                "omnifuse": "materialize native chunk ID and score pairs",
                "synaptic": (
                    "materialize native hits and map runtime node IDs to official "
                    "external document IDs; activation/resonance audit runs only in "
                    "the untimed checkpoint lane"
                ),
            },
            "noop_adapter_semantics": {
                "omnifuse": (
                    "native upsert_chunks call; unchanged result must report zero "
                    "changed writes"
                ),
                "synaptic": (
                    "MemoryBackend exposes save/update/delete but no semantic upsert; "
                    "the adapter reads and compares official title/content fields, then "
                    "skips update_node when unchanged"
                ),
            },
            "mutation_call_scope": {
                "omnifuse": (
                    "one native batch call per phase; document adaptation and result "
                    "accounting are included in the phase timer"
                ),
                "synaptic": (
                    "one native graph/backend call per document; field adaptation and "
                    "runtime-ID bookkeeping are included in the phase timer"
                ),
            },
        },
        "results": list(results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.data_dir = args.data_dir.resolve()
    args.synaptic_repo = args.synaptic_repo.resolve()
    if args.doctor_manifest is not None:
        args.doctor_manifest = args.doctor_manifest.resolve()
    if args.out is not None:
        args.out = args.out.resolve()
    if args.workers_dir is not None:
        args.workers_dir = args.workers_dir.resolve()
    if args.input_file is not None:
        args.input_file = args.input_file.resolve()
    if args.result_file is not None:
        args.result_file = args.result_file.resolve()
    try:
        perf._require_protocol_utf8_mode(
            perf.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
            context="CDC controller" if args.worker is None else "CDC worker",
        )
        if args.worker is not None:
            _run_worker(args)
            return 0
        dataset_path = (args.data_dir / args.dataset).resolve()
        worker_root = _worker_directory(args.out, args.workers_dir)
        perf._validate_worker_directory(
            worker_root, output=args.out, synaptic_repo=args.synaptic_repo
        )
        state, corpus, queries = perf._machine_preflight(
            output=args.out,
            doctor_manifest=args.doctor_manifest,
            synaptic_repo=args.synaptic_repo,
            dataset_path=dataset_path,
            protocol=perf.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        )
        cdc_source = file_fingerprint(SCRIPT_PATH, display_path="eval/cdc_bench.py")
        trace = build_mutation_trace(
            corpus, seed=args.seed, group_fraction=args.mutation_group_fraction
        )
        _initial, final = _corpus_states(corpus, trace)
        active_queries = _active_queries(queries, final)
        active_contract = _active_query_contract(active_queries)
        checkpoint_contracts = _checkpoint_contracts(corpus, queries, trace)
        frozen_payload = perf._frozen_input_payload(corpus, queries)
        frozen_input = {
            **perf._bytes_fingerprint(frozen_payload, path=WORKER_INPUT_DISPLAY_PATH),
            "documents": len(corpus),
            "source_scored_queries": len(queries),
            "active_scored_queries": active_contract["count"],
            "active_query_ids_ordered_sha256": active_contract[
                "query_ids_ordered_sha256"
            ],
            "active_relevance_judgments": active_contract["relevance_judgments"],
        }
        schedule = perf._trial_schedule(SYSTEMS, args.trials)
        expected_lock = state["official_environment_probe"]["environment_lock"]
        trial_results: dict[str, list[dict[str, Any]]] = {
            system: [] for system in SYSTEMS
        }
        worker_root.mkdir(parents=True, exist_ok=False)
        input_file = worker_root / "cdc-input.json"
        perf._write_bytes_once(input_file, frozen_payload)
        _frozen_bytes, frozen_input_artifact = read_bytes_artifact(input_file)
        worker_records: list[dict[str, Any]] = []
        for trial_number, order in enumerate(schedule, 1):
            for order_position, system in enumerate(order, 1):
                assert_unchanged(
                    "CDC frozen input before worker",
                    {key: frozen_input[key] for key in ("path", "sha256", "bytes")},
                    file_fingerprint(
                        input_file, display_path=WORKER_INPUT_DISPLAY_PATH
                    ),
                )
                worker_run_id = new_worker_run_id()
                if any(
                    record["worker_run_id"] == worker_run_id
                    for record in worker_records
                ):
                    raise ProvenanceError(
                        "controller generated a duplicate worker run ID"
                    )
                result_file = worker_root / (
                    f"trial-{trial_number:02d}-position-{order_position}-{system}.json"
                )
                _completed, launcher_pid = run_with_launcher_pid(
                    _worker_command(
                        args,
                        system=system,
                        input_file=input_file,
                        result_file=result_file,
                        worker_run_id=worker_run_id,
                    ),
                    cwd=ROOT,
                    check=True,
                    env=perf._isolated_worker_environment(),
                )
                raw, worker_artifact = read_json_artifact(result_file)
                validated = _validate_worker_result(
                    raw,
                    system=system,
                    expected_input=frozen_input,
                    expected_trace=trace,
                    expected_queries=active_queries,
                    expected_checkpoints=checkpoint_contracts,
                    args=args,
                    trial_number=trial_number,
                    order_position=order_position,
                    expected_worker_run_id=worker_run_id,
                    official_environment_lock=expected_lock,
                )
                trial_results[system].append(validated)
                identity = validated["worker_identity"]
                worker_records.append(
                    {
                        "trial_number": trial_number,
                        "order_position": order_position,
                        "system": system,
                        "worker_run_id": identity["worker_run_id"],
                        "launcher_pid": launcher_pid,
                        "worker_pid": identity["worker_pid"],
                        "same_process_id": launcher_pid == identity["worker_pid"],
                        "artifact": worker_artifact,
                    }
                )
        assert_unchanged(
            "CDC frozen input after workers",
            {key: frozen_input[key] for key in ("path", "sha256", "bytes")},
            file_fingerprint(input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
        )
        worker_process_summary(
            worker_records, expected_count=sum(len(order) for order in schedule)
        )
        results = _aggregate(trial_results)
        postflight = perf._verify_machine_postflight(
            state,
            synaptic_repo=args.synaptic_repo,
            dataset_path=dataset_path,
            protocol=perf.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        )
        assert_unchanged(
            "CDC benchmark source",
            cdc_source,
            file_fingerprint(SCRIPT_PATH, display_path="eval/cdc_bench.py"),
        )
        report = _report(
            args=args,
            state=state,
            postflight=postflight,
            cdc_source=cdc_source,
            frozen_input=frozen_input,
            trace=trace,
            active_queries=active_queries,
            checkpoint_contracts=checkpoint_contracts,
            schedule=schedule,
            results=results,
            workers_directory=worker_root,
            frozen_input_artifact=frozen_input_artifact,
            worker_records=worker_records,
        )
        assert_artifact_unchanged(
            "CDC frozen input artifact before report publication",
            Path(str(frozen_input_artifact["path"])),
            frozen_input_artifact,
        )
        for record in worker_records:
            artifact = record["artifact"]
            assert_artifact_unchanged(
                f"CDC worker artifact {record['worker_run_id']} before publication",
                Path(str(artifact["path"])),
                artifact,
            )
        write_json_once(args.out, report)
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
