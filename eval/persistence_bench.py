"""Claim-grade native SQLite persistence comparison.

The controller freezes one official synaptic-memory benchmark input, launches one
fresh process per system and trial in AB/BA order, and records durable creation,
clean open, first-query, steady-query, artifact, RSS, and six retrieval metrics.

OmniFuse uses :func:`save_sqlite_index` / :func:`open_sqlite_index`. Synaptic uses
its native ``SqliteGraphBackend.save_nodes_batch`` and ``SynapticGraph.search``.
The optional ``--omnifuse-idf-pow`` is a global retrieval profile, never a
per-query or per-document switch.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import math
import os
import platform
import site
import statistics
import sys
from threading import Event, Thread
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = (ROOT / "src").resolve()
sys.path.insert(0, str(EVAL_DIR))

import perf_bench as perf  # noqa: E402
from metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from provenance import (  # noqa: E402
    ProvenanceError,
    assert_artifact_unchanged,
    assert_unchanged,
    canonical_json_sha256,
    capture_worker_identity,
    ensure_output_absent,
    file_fingerprint,
    new_worker_run_id,
    read_json_artifact,
    run_with_launcher_pid,
    sha256_file,
    validate_worker_identity,
    worker_process_summary,
    write_json_once,
)

SCHEMA = "omnifuse.eval.sqlite_persistence_comparison"
SCHEMA_VERSION = 2
WORKER_SCHEMA = "omnifuse.eval.sqlite_persistence_worker"
WORKER_SCHEMA_VERSION = 2
PROVENANCE_LEVEL = "strict-preflight-postflight-isolated-write-once-v2"
SYSTEMS = ("omnifuse", "synaptic")
SYSTEM_LABELS = {"omnifuse": "OmniFuse", "synaptic": "synaptic"}
K = 10
CANDIDATE_LIMIT = 20
WORKER_INPUT_DISPLAY_PATH = "worker-input/persistence.json"
RSS_SAMPLE_INTERVAL_SECONDS = 0.01
WORKER_ENVIRONMENT_KEYS = {
    "python",
    "python_executable",
    "isolated",
    "ignore_environment",
    "no_user_site",
    "safe_path",
    "utf8_mode",
    "user_site_enabled",
    "pythonpath",
    "pythonhome",
    "pythonusersite",
    "python_no_user_site_env",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptic-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="nfcorpus.json")
    parser.add_argument("--doctor-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers-dir", type=Path)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--omnifuse-idf-pow", type=float, default=1.2)
    parser.add_argument("--worker", choices=SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--input-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    return parser


def _input_preflight_protocol(filename: str) -> str:
    import direct_external_bench

    if any(case.filename == filename for case in direct_external_bench.CASES):
        return perf.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
    return perf.PROTOCOL_SQLITE_NATIVE


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class _WorkloadRssSampler:
    """Sample current RSS after system import, excluding frozen-input parse overlap."""

    def __init__(self) -> None:
        self._stop = Event()
        self._peak: int | None = None
        self._thread = Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        current, _lifetime_peak, _kind = perf._process_memory_bytes()
        if current is not None and (self._peak is None or current > self._peak):
            self._peak = current

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(RSS_SAMPLE_INTERVAL_SECONDS):
            self._sample()
        self._sample()

    def __enter__(self) -> "_WorkloadRssSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak(self) -> int | None:
        return self._peak


def _is_below(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _module_binding(
    value: Any, *, source_root: Path, repository_root: Path, name: str
) -> dict[str, Any]:
    module = value if inspect.ismodule(value) else inspect.getmodule(value)
    raw_path = getattr(module, "__file__", None) if module is not None else None
    if not isinstance(raw_path, str):
        raise RuntimeError(f"cannot resolve source module for {name}")
    path = Path(raw_path).resolve()
    if not _is_below(path, source_root):
        raise RuntimeError(f"loaded {name} from {path}, expected below {source_root}")
    return {
        **file_fingerprint(
            path, display_path=path.relative_to(repository_root.resolve()).as_posix()
        ),
        "resolved_path": str(path),
    }


def _prepend_import_path(path: Path) -> None:
    value = str(path.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)
    importlib.invalidate_caches()


def _worker_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "isolated": bool(sys.flags.isolated),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "no_user_site": bool(sys.flags.no_user_site),
        "safe_path": bool(getattr(sys.flags, "safe_path", False)),
        "utf8_mode": bool(sys.flags.utf8_mode),
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "pythonhome": os.environ.get("PYTHONHOME"),
        "pythonusersite": os.environ.get("PYTHONUSERBASE"),
        "python_no_user_site_env": os.environ.get("PYTHONNOUSERSITE"),
    }


def _isolated_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _artifact(directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "bytes": sum(row["bytes"] for row in files),
        "files": files,
        "manifest_sha256": canonical_json_sha256(files),
    }


def _current_rss_mb() -> float | None:
    current, _peak, _kind = perf._process_memory_bytes()
    return None if current is None else current / 1_000_000.0


def _unique(values: Iterable[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
            if len(out) == limit:
                break
    return out


def _reciprocal_rank(values: Sequence[str], relevant: set[str], limit: int) -> float:
    return next(
        (
            1.0 / rank
            for rank, value in enumerate(values[:limit], 1)
            if value in relevant
        ),
        0.0,
    )


def _accuracy(
    queries: Sequence[perf.Query], rankings: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    rows: list[tuple[float, ...]] = []
    for query_id, _text, relevant in queries:
        retrieved = list(rankings[query_id])
        rows.append(
            (
                _reciprocal_rank(retrieved, relevant, CANDIDATE_LIMIT),
                _reciprocal_rank(retrieved, relevant, K),
                precision_at_k(retrieved, relevant, K),
                recall_at_k(retrieved, relevant, K),
                f1_at_k(retrieved, relevant, K),
                ndcg_at_k(retrieved, relevant, K),
            )
        )
    names = (
        "mrr_at_20",
        "mrr_at_10",
        "precision_at_10",
        "recall_at_10",
        "f1_at_10",
        "ndcg_at_10",
    )
    return {
        name: statistics.fmean(row[index] for row in rows)
        for index, name in enumerate(names)
    }


def _timing_summary(samples_ns: Sequence[int]) -> dict[str, float | int]:
    samples_ms = [value / 1_000_000.0 for value in samples_ns]
    return {
        "samples": len(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": perf._percentile(samples_ms, 0.95),
        "mean_ms": statistics.fmean(samples_ms),
    }


def _rankings_hash(
    queries: Sequence[perf.Query], rankings: Mapping[str, Sequence[str]]
) -> str:
    return canonical_json_sha256(
        [
            {"query_id": query_id, "retrieved_top_20": list(rankings[query_id])}
            for query_id, _text, _relevant in queries
        ]
    )


def _measure_sync(
    search: Callable[[str], Iterable[str]],
    queries: Sequence[perf.Query],
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    canonical: dict[str, list[str]] = {}
    first_text = queries[0][1]
    started = time.perf_counter_ns()
    first = _unique(search(first_text), CANDIDATE_LIMIT)
    first_ns = time.perf_counter_ns() - started
    for _ in range(warmup):
        for query_id, text, _relevant in queries:
            ranking = _unique(search(text), CANDIDATE_LIMIT)
            expected = canonical.setdefault(query_id, ranking)
            if ranking != expected:
                raise RuntimeError(f"ranking changed for query {query_id!r}")
    samples: list[int] = []
    rounds: list[float] = []
    for _ in range(repeats):
        round_ns = 0
        for query_id, text, _relevant in queries:
            started = time.perf_counter_ns()
            ranking = _unique(search(text), CANDIDATE_LIMIT)
            elapsed = time.perf_counter_ns() - started
            round_ns += elapsed
            samples.append(elapsed)
            if ranking != canonical[query_id]:
                raise RuntimeError(f"ranking changed for query {query_id!r}")
        rounds.append(round_ns / 1_000_000_000.0)
    if first != canonical[queries[0][0]]:
        raise RuntimeError("first-query ranking differs from the steady ranking")
    return canonical, {
        "first_query_ms": first_ns / 1_000_000.0,
        "steady": _timing_summary(samples),
        "round_seconds": rounds,
    }


async def _measure_async(
    search: Callable[[str], Awaitable[Iterable[str]]],
    queries: Sequence[perf.Query],
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    canonical: dict[str, list[str]] = {}
    started = time.perf_counter_ns()
    first = _unique(await search(queries[0][1]), CANDIDATE_LIMIT)
    first_ns = time.perf_counter_ns() - started
    for _ in range(warmup):
        for query_id, text, _relevant in queries:
            ranking = _unique(await search(text), CANDIDATE_LIMIT)
            expected = canonical.setdefault(query_id, ranking)
            if ranking != expected:
                raise RuntimeError(f"ranking changed for query {query_id!r}")
    samples: list[int] = []
    rounds: list[float] = []
    for _ in range(repeats):
        round_ns = 0
        for query_id, text, _relevant in queries:
            started = time.perf_counter_ns()
            ranking = _unique(await search(text), CANDIDATE_LIMIT)
            elapsed = time.perf_counter_ns() - started
            round_ns += elapsed
            samples.append(elapsed)
            if ranking != canonical[query_id]:
                raise RuntimeError(f"ranking changed for query {query_id!r}")
        rounds.append(round_ns / 1_000_000_000.0)
    if first != canonical[queries[0][0]]:
        raise RuntimeError("first-query ranking differs from the steady ranking")
    return canonical, {
        "first_query_ms": first_ns / 1_000_000.0,
        "steady": _timing_summary(samples),
        "round_seconds": rounds,
    }


def _omnifuse_worker(
    corpus: Sequence[perf.CorpusRow],
    queries: Sequence[perf.Query],
    *,
    idf_pow: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    _prepend_import_path(SOURCE_ROOT)
    import omnifuse
    from omnifuse import build_sqlite_index, open_sqlite_index

    bindings = {
        "package": _module_binding(
            omnifuse, source_root=SOURCE_ROOT, repository_root=ROOT, name="omnifuse"
        ),
        "build_sqlite_index": _module_binding(
            build_sqlite_index,
            source_root=SOURCE_ROOT,
            repository_root=ROOT,
            name="build_sqlite_index",
        ),
        "open_sqlite_index": _module_binding(
            open_sqlite_index,
            source_root=SOURCE_ROOT,
            repository_root=ROOT,
            name="open_sqlite_index",
        ),
    }
    with (
        _WorkloadRssSampler() as rss,
        TemporaryDirectory(prefix="omnifuse-persistence-") as directory,
    ):
        root = Path(directory)
        path = root / "index.sqlite"
        started = time.perf_counter_ns()
        build_sqlite_index(
            [],
            [],
            (
                {"id": doc_id, "title": title, "text": text}
                for doc_id, title, text in corpus
            ),
            path,
            vector_kwargs={"idf_pow": idf_pow},
        )
        create_ns = time.perf_counter_ns() - started
        created_artifact = _artifact(root)
        gc.collect()
        post_create_rss_mb = _current_rss_mb()
        started = time.perf_counter_ns()
        opened = open_sqlite_index(path, vector_k=CANDIDATE_LIMIT)
        open_ns = time.perf_counter_ns() - started
        clean_open_rss_mb = _current_rss_mb()
        try:
            rankings, query_timing = _measure_sync(
                lambda text: (
                    chunk.id
                    for chunk, _score in opened.retrieve(text, limit=CANDIDATE_LIMIT)
                ),
                queries,
                warmup=warmup,
                repeats=repeats,
            )
            post_query_rss_mb = _current_rss_mb()
        finally:
            opened.close()
        final_artifact = _artifact(root)
    current, peak, kind = perf._process_memory_bytes()
    return {
        "system": "OmniFuse",
        "configuration": {"idf_pow": idf_pow, "title_weight": 4.0},
        "durable_create_seconds": create_ns / 1_000_000_000.0,
        "clean_open_ms": open_ns / 1_000_000.0,
        "query": query_timing,
        "accuracy": _accuracy(queries, rankings),
        "rankings_sha256": _rankings_hash(queries, rankings),
        "artifact_after_create": created_artifact,
        "artifact_after_queries": final_artifact,
        "process_memory": {
            "kind": kind,
            "current_rss_mb": None if current is None else current / 1_000_000.0,
            "post_create_rss_mb": post_create_rss_mb,
            "clean_open_rss_mb": clean_open_rss_mb,
            "post_query_rss_mb": post_query_rss_mb,
            "peak_rss_mb": None if peak is None else peak / 1_000_000.0,
            "workload_peak_rss_mb": (
                None if rss.peak is None else rss.peak / 1_000_000.0
            ),
        },
        "runtime": {
            "package_path": str(Path(omnifuse.__file__).resolve()),
            "package_version": getattr(omnifuse, "__version__", None)
            or _package_version("omnifuse"),
            "source_bindings": bindings,
            "tokenizer": None,
        },
    }


async def _synaptic_worker(
    repo: Path,
    corpus: Sequence[perf.CorpusRow],
    queries: Sequence[perf.Query],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    source_root = (repo / "src").resolve()
    _prepend_import_path(source_root)
    import synaptic
    from synaptic.backends import sqlite as sqlite_module
    from synaptic.backends.sqlite import SQLiteBackend
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph
    from synaptic.models import Node

    bindings = {
        "package": _module_binding(
            synaptic, source_root=source_root, repository_root=repo, name="synaptic"
        ),
        "sqlite_backend": _module_binding(
            SqliteGraphBackend,
            source_root=source_root,
            repository_root=repo,
            name="SqliteGraphBackend",
        ),
        "sqlite_backend_base": _module_binding(
            SQLiteBackend,
            source_root=source_root,
            repository_root=repo,
            name="SQLiteBackend",
        ),
        "graph": _module_binding(
            SynapticGraph,
            source_root=source_root,
            repository_root=repo,
            name="SynapticGraph",
        ),
        "node": _module_binding(
            Node, source_root=source_root, repository_root=repo, name="Node"
        ),
    }
    with (
        _WorkloadRssSampler() as rss,
        TemporaryDirectory(prefix="synaptic-persistence-") as directory,
    ):
        root = Path(directory)
        path = root / "graph.sqlite"
        started = time.perf_counter_ns()
        backend = SqliteGraphBackend(str(path))
        await backend.connect()
        nodes = [
            Node(
                id=doc_id,
                title=title or doc_id,
                content=text,
                properties={"doc_id": doc_id},
            )
            for doc_id, title, text in corpus
            if title or text
        ]
        await backend.save_nodes_batch(nodes)
        await backend.close()
        create_ns = time.perf_counter_ns() - started
        created_artifact = _artifact(root)
        del nodes
        gc.collect()
        post_create_rss_mb = _current_rss_mb()

        started = time.perf_counter_ns()
        opened_backend = SqliteGraphBackend(str(path))
        opened = SynapticGraph(opened_backend, embedder=None, reranker=None)
        await opened.connect()
        open_ns = time.perf_counter_ns() - started
        clean_open_rss_mb = _current_rss_mb()

        async def search(text: str) -> Iterable[str]:
            result = await opened.search(text, limit=CANDIDATE_LIMIT)
            return (
                str((hit.node.properties or {}).get("doc_id", ""))
                for hit in result.nodes
            )

        try:
            rankings, query_timing = await _measure_async(
                search, queries, warmup=warmup, repeats=repeats
            )
            post_query_rss_mb = _current_rss_mb()
        finally:
            await opened.close()
        final_artifact = _artifact(root)
    current, peak, kind = perf._process_memory_bytes()
    return {
        "system": "synaptic",
        "configuration": {
            "backend": "SqliteGraphBackend",
            "ingest": "save_nodes_batch",
        },
        "durable_create_seconds": create_ns / 1_000_000_000.0,
        "clean_open_ms": open_ns / 1_000_000.0,
        "query": query_timing,
        "accuracy": _accuracy(queries, rankings),
        "rankings_sha256": _rankings_hash(queries, rankings),
        "artifact_after_create": created_artifact,
        "artifact_after_queries": final_artifact,
        "process_memory": {
            "kind": kind,
            "current_rss_mb": None if current is None else current / 1_000_000.0,
            "post_create_rss_mb": post_create_rss_mb,
            "clean_open_rss_mb": clean_open_rss_mb,
            "post_query_rss_mb": post_query_rss_mb,
            "peak_rss_mb": None if peak is None else peak / 1_000_000.0,
            "workload_peak_rss_mb": (
                None if rss.peak is None else rss.peak / 1_000_000.0
            ),
        },
        "runtime": {
            "package_path": str(Path(synaptic.__file__).resolve()),
            "package_version": getattr(synaptic, "__version__", None)
            or _package_version("synaptic-memory"),
            "source_bindings": bindings,
            "tokenizer": perf._synaptic_tokenizer_evidence(sqlite_module),
        },
    }


def _run_worker(args: argparse.Namespace) -> int:
    if (
        args.input_file is None
        or args.result_file is None
        or args.worker_run_id is None
    ):
        raise ValueError("worker mode requires input, result, and run ID")
    ensure_output_absent(args.result_file)
    fingerprint, corpus, queries = perf._load_frozen_input_file(
        args.input_file,
        display_path=WORKER_INPUT_DISPLAY_PATH,
    )
    if args.worker == "omnifuse":
        result = _omnifuse_worker(
            corpus,
            queries,
            idf_pow=args.omnifuse_idf_pow,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    else:
        result = asyncio.run(
            _synaptic_worker(
                args.synaptic_repo.resolve(),
                corpus,
                queries,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    assert_unchanged(
        "worker input after measurement",
        fingerprint,
        file_fingerprint(args.input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
    )
    write_json_once(
        args.result_file,
        {
            "schema": WORKER_SCHEMA,
            "schema_version": WORKER_SCHEMA_VERSION,
            "status": "ok",
            "system": args.worker,
            "input": {
                **fingerprint,
                "documents": len(corpus),
                "scored_queries": len(queries),
            },
            "contract": {
                "k": K,
                "candidate_limit": CANDIDATE_LIMIT,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "omnifuse_idf_pow": args.omnifuse_idf_pow,
            },
            "worker_identity": capture_worker_identity(args.worker_run_id),
            "environment": _worker_environment(),
            "result": result,
        },
    )
    return 0


def _validate_worker(
    payload: Any,
    *,
    system: str,
    input_fingerprint: Mapping[str, Any],
    documents: int,
    queries: int,
    run_id: str,
    trial: int,
    position: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != WORKER_SCHEMA:
        raise ProvenanceError(f"invalid {system} worker schema")
    if (
        payload.get("schema_version") != WORKER_SCHEMA_VERSION
        or payload.get("status") != "ok"
        or payload.get("system") != system
    ):
        raise ProvenanceError(f"invalid {system} worker contract")
    raw_input = payload.get("input")
    expected_input = {
        **input_fingerprint,
        "documents": documents,
        "scored_queries": queries,
    }
    assert_unchanged(f"{system} worker input", expected_input, raw_input)
    identity = validate_worker_identity(
        payload.get("worker_identity"), expected_run_id=run_id, label=f"{system} worker"
    )
    environment = payload.get("environment")
    if not isinstance(environment, dict) or set(environment) != WORKER_ENVIRONMENT_KEYS:
        raise ProvenanceError(f"invalid {system} worker environment")
    for flag in (
        "isolated",
        "ignore_environment",
        "no_user_site",
        "safe_path",
        "utf8_mode",
    ):
        if environment[flag] is not True:
            raise ProvenanceError(f"{system} worker did not enable {flag}")
    if environment["user_site_enabled"] is not False:
        raise ProvenanceError(f"{system} worker enabled user site")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("system") != SYSTEM_LABELS[system]:
        raise ProvenanceError(f"invalid {system} metrics")
    numeric_paths = [
        ("durable_create_seconds", result.get("durable_create_seconds")),
        ("clean_open_ms", result.get("clean_open_ms")),
        ("first_query_ms", result.get("query", {}).get("first_query_ms")),
        (
            "workload_peak_rss_mb",
            result.get("process_memory", {}).get("workload_peak_rss_mb"),
        ),
        (
            "post_create_rss_mb",
            result.get("process_memory", {}).get("post_create_rss_mb"),
        ),
        (
            "clean_open_rss_mb",
            result.get("process_memory", {}).get("clean_open_rss_mb"),
        ),
        (
            "post_query_rss_mb",
            result.get("process_memory", {}).get("post_query_rss_mb"),
        ),
    ]
    for label, value in numeric_paths:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ProvenanceError(f"invalid {system} {label}")
    accuracy = result.get("accuracy")
    if not isinstance(accuracy, dict) or set(accuracy) != {
        "mrr_at_20",
        "mrr_at_10",
        "precision_at_10",
        "recall_at_10",
        "f1_at_10",
        "ndcg_at_10",
    }:
        raise ProvenanceError(f"invalid {system} accuracy")
    if any(
        not isinstance(value, (int, float)) or not 0 <= value <= 1
        for value in accuracy.values()
    ):
        raise ProvenanceError(f"invalid {system} accuracy value")
    ranking_hash = result.get("rankings_sha256")
    if not isinstance(ranking_hash, str) or len(ranking_hash) != 64:
        raise ProvenanceError(f"invalid {system} ranking hash")
    return {
        **result,
        "trial": {"number": trial, "order_position": position},
        "worker_identity": identity,
        "worker_environment": environment,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "p50": statistics.median(values),
        "p95": perf._percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _aggregate(rows: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system, trials in rows.items():
        hashes = {trial["rankings_sha256"] for trial in trials}
        accuracies = {canonical_json_sha256(trial["accuracy"]) for trial in trials}
        if len(hashes) != 1 or len(accuracies) != 1:
            raise ProvenanceError(f"{system} rankings changed across trials")
        out[system] = {
            "trial_count": len(trials),
            "durable_create_seconds": _distribution(
                [trial["durable_create_seconds"] for trial in trials]
            ),
            "clean_open_ms": _distribution(
                [trial["clean_open_ms"] for trial in trials]
            ),
            "first_query_ms": _distribution(
                [trial["query"]["first_query_ms"] for trial in trials]
            ),
            "steady_p50_ms": _distribution(
                [trial["query"]["steady"]["p50_ms"] for trial in trials]
            ),
            "steady_p95_ms": _distribution(
                [trial["query"]["steady"]["p95_ms"] for trial in trials]
            ),
            "query_round_seconds": _distribution(
                [statistics.fmean(trial["query"]["round_seconds"]) for trial in trials]
            ),
            "artifact_bytes": _distribution(
                [float(trial["artifact_after_queries"]["bytes"]) for trial in trials]
            ),
            "current_rss_mb": _distribution(
                [trial["process_memory"]["current_rss_mb"] for trial in trials]
            ),
            "post_create_rss_mb": _distribution(
                [trial["process_memory"]["post_create_rss_mb"] for trial in trials]
            ),
            "clean_open_rss_mb": _distribution(
                [trial["process_memory"]["clean_open_rss_mb"] for trial in trials]
            ),
            "post_query_rss_mb": _distribution(
                [trial["process_memory"]["post_query_rss_mb"] for trial in trials]
            ),
            "peak_rss_mb": _distribution(
                [trial["process_memory"]["peak_rss_mb"] for trial in trials]
            ),
            "workload_peak_rss_mb": _distribution(
                [trial["process_memory"]["workload_peak_rss_mb"] for trial in trials]
            ),
            "accuracy": trials[0]["accuracy"],
            "rankings_sha256": trials[0]["rankings_sha256"],
            "trials": list(trials),
        }
    return out


def _worker_command(
    args: argparse.Namespace,
    *,
    system: str,
    input_file: Path,
    result_file: Path,
    run_id: str,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(SCRIPT_PATH),
        "--synaptic-repo",
        str(args.synaptic_repo.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--dataset",
        args.dataset,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--omnifuse-idf-pow",
        str(args.omnifuse_idf_pow),
        "--worker",
        system,
        "--input-file",
        str(input_file),
        "--result-file",
        str(result_file),
        "--worker-run-id",
        run_id,
    ]


def _controller(args: argparse.Namespace) -> int:
    if args.out is None or args.doctor_manifest is None:
        raise ValueError("claim-grade mode requires --out and --doctor-manifest")
    if args.trials < 2 or args.trials % 2:
        raise ValueError("--trials must be an even value of at least 2")
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("claim-grade mode requires warmup >= 1 and repeats >= 2")
    if not math.isfinite(args.omnifuse_idf_pow) or args.omnifuse_idf_pow <= 0:
        raise ValueError("--omnifuse-idf-pow must be positive and finite")
    output = args.out.resolve()
    doctor = args.doctor_manifest.resolve()
    repo = args.synaptic_repo.resolve()
    dataset = (args.data_dir / args.dataset).resolve()
    input_protocol = _input_preflight_protocol(args.dataset)
    ensure_output_absent(output)
    harness_before = file_fingerprint(
        SCRIPT_PATH, display_path="eval/persistence_bench.py"
    )
    state, corpus, queries = perf._machine_preflight(
        output=output,
        doctor_manifest=doctor,
        synaptic_repo=repo,
        dataset_path=dataset,
        protocol=input_protocol,
    )
    frozen = perf._frozen_input_payload(corpus, queries)
    workers = (
        args.workers_dir.resolve()
        if args.workers_dir is not None
        else output.parent / f".{output.stem}.workers"
    )
    if workers.exists():
        raise ProvenanceError(f"refusing to reuse worker directory {workers}")
    workers.mkdir(parents=True)
    input_file = workers / "input.json"
    input_file.write_bytes(frozen)
    input_fingerprint = {
        "path": WORKER_INPUT_DISPLAY_PATH,
        "sha256": hashlib.sha256(frozen).hexdigest(),
        "bytes": len(frozen),
    }
    schedule = [
        list(SYSTEMS) if trial % 2 == 0 else list(reversed(SYSTEMS))
        for trial in range(args.trials)
    ]
    rows: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    processes: list[dict[str, Any]] = []
    for trial_number, order in enumerate(schedule, 1):
        for position, system in enumerate(order, 1):
            assert_unchanged(
                "frozen input before worker",
                input_fingerprint,
                file_fingerprint(input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
            )
            result_file = workers / f"trial-{trial_number:02d}-{position}-{system}.json"
            run_id = new_worker_run_id()
            completed, launched = run_with_launcher_pid(
                _worker_command(
                    args,
                    system=system,
                    input_file=input_file,
                    result_file=result_file,
                    run_id=run_id,
                ),
                cwd=ROOT,
                env=_isolated_environment(),
                check=True,
            )
            payload, result_artifact = read_json_artifact(result_file)
            validated = _validate_worker(
                payload,
                system=system,
                input_fingerprint=input_fingerprint,
                documents=len(corpus),
                queries=len(queries),
                run_id=run_id,
                trial=trial_number,
                position=position,
            )
            rows[system].append(validated)
            identity = validated["worker_identity"]
            processes.append(
                {
                    "trial_number": trial_number,
                    "order_position": position,
                    "system": system,
                    "worker_run_id": identity["worker_run_id"],
                    "launcher_pid": launched,
                    "worker_pid": identity["worker_pid"],
                    "same_process_id": launched == identity["worker_pid"],
                    "returncode": completed.returncode,
                    "result_artifact": result_artifact,
                }
            )
    assert_artifact_unchanged(
        "frozen input after workers",
        input_file,
        input_fingerprint,
    )
    postflight = perf._verify_machine_postflight(
        state,
        synaptic_repo=repo,
        dataset_path=dataset,
        protocol=input_protocol,
    )
    assert_unchanged(
        "persistence harness source",
        harness_before,
        file_fingerprint(SCRIPT_PATH, display_path="eval/persistence_bench.py"),
    )
    aggregated = _aggregate(rows)
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance_level": PROVENANCE_LEVEL,
        "contract": {
            "dataset": args.dataset,
            "input_preflight_protocol": input_protocol,
            "k": K,
            "candidate_limit": CANDIDATE_LIMIT,
            "trials_per_system": args.trials,
            "trial_order": schedule,
            "warmup_rounds": args.warmup,
            "measurement_rounds": args.repeats,
            "omnifuse_idf_pow": args.omnifuse_idf_pow,
            "synaptic_ingest": "SqliteGraphBackend.save_nodes_batch",
            "artifact_scope": "native SQLite files after clean close",
        },
        "dataset": state["dataset"],
        "repositories": state["repositories"],
        "sources": {**state["sources"], "persistence_harness": harness_before},
        "doctor_manifest": state["doctor_manifest"],
        "worker_input": input_fingerprint,
        "worker_processes": processes,
        "worker_process_summary": worker_process_summary(
            processes, expected_count=sum(len(order) for order in schedule)
        ),
        "results": aggregated,
        "postflight": postflight,
    }
    write_json_once(output, report)
    for system in SYSTEMS:
        row = aggregated[system]
        print(
            f"{system:9} create={row['durable_create_seconds']['p50']:.6f}s "
            f"open={row['clean_open_ms']['p50']:.4f}ms "
            f"first={row['first_query_ms']['p50']:.4f}ms "
            f"steady={row['steady_p50_ms']['p50']:.4f}ms "
            f"bytes={row['artifact_bytes']['p50']:.0f} "
            f"mrr10={row['accuracy']['mrr_at_10']:.6f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.worker is not None:
            return _run_worker(args)
        return _controller(args)
    except (OSError, ValueError, RuntimeError, ProvenanceError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
