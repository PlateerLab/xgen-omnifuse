"""Isolated comparison of Synaptic's official QA performance contract.

The benchmark reproduces ``tests/qa/conftest.py::combined_graph`` and the exact
16-query sequence from ``tests/qa/test_performance.py``.  Each system runs in a
fresh isolated process in counterbalanced AB/BA order.  The report separates the
official first pass (which includes lazy tokenizer/index initialization) from
steady repeated passes so test-order warming cannot hide cold-start cost.

Batch throughput is recorded with an explicit capability boundary.  Synaptic's
upstream test times ``MemoryBackend.save_nodes_batch``.  OmniFuse records both
raw lazy-store construction and full lexical materialization; those operations
are not declared capability-equivalent.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import site
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = (ROOT / "src").resolve()
sys.path.insert(0, str(EVAL_DIR))

import perf_bench as perf  # noqa: E402
from provenance import (  # noqa: E402
    ProvenanceError,
    assert_unchanged,
    canonical_json_sha256,
    capture_worker_identity,
    ensure_output_absent,
    file_fingerprint,
    load_doctor_manifest,
    new_worker_run_id,
    read_json_artifact,
    repository_fingerprint,
    run_with_launcher_pid,
    validate_worker_identity,
    verify_doctor_manifest,
    verify_doctor_runtime,
    worker_process_summary,
    write_json_once,
)

SCHEMA = "omnifuse.eval.qa_memory_performance"
SCHEMA_VERSION = 1
WORKER_SCHEMA = "omnifuse.eval.qa_memory_performance_worker"
WORKER_SCHEMA_VERSION = 1
PROVENANCE_LEVEL = "strict-doctor-isolated-ab-ba-preflight-postflight-write-once-v1"
SYSTEMS = ("omnifuse", "synaptic")
SYSTEM_LABELS = {"omnifuse": "OmniFuse", "synaptic": "synaptic-memory"}
TARGET_ID = "qa_combined"
K = 10
OFFICIAL_P95_LIMIT_MS = 100.0
OFFICIAL_AVERAGE_LIMIT_MS = 50.0
OFFICIAL_BATCH_MIN_NODES_PER_SECOND = 50.0
BATCH_SIZE = 100
QA_SOURCES = (
    ("wikipedia", "tests/qa/data/wikipedia_ko_tech.json"),
    ("commits", "tests/qa/data/github_commits.json"),
    ("issues", "tests/qa/data/github_issues.json"),
)
QUERIES = (
    "데이터베이스",
    "프로그래밍",
    "네트워크",
    "보안",
    "알고리즘",
    "웹",
    "클라우드",
    "인공지능",
    "fix bug",
    "deploy",
    "performance",
    "refactor",
    "API",
    "테스트",
    "Python",
    "설계",
)
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
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--doctor-manifest", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers-dir", type=Path)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list) or not value:
        raise ValueError(f"QA source must be a non-empty array: {path}")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"QA source rows must be objects: {path}")
    return value


def _source_paths(repo: Path, data_dir: Path | None = None) -> dict[str, Path]:
    base = (
        data_dir.resolve()
        if data_dir is not None
        else repo.resolve() / "tests" / "qa" / "data"
    )
    return {
        "wikipedia": base / "wikipedia_ko_tech.json",
        "commits": base / "github_commits.json",
        "issues": base / "github_issues.json",
    }


def _prepare_documents(
    repo: Path, data_dir: Path | None = None
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    paths = _source_paths(repo, data_dir)
    raw = {role: _load_list(path) for role, path in paths.items()}
    documents: list[dict[str, str]] = []
    selected = {"wikipedia": 0, "commits": 0, "issues": 0}

    for source_index, article in enumerate(raw["wikipedia"][:50]):
        title = str(article.get("title", ""))
        content = str(article.get("content", ""))[:1500]
        if not title or not content:
            continue
        documents.append(
            {
                "id": f"wikipedia-{source_index:04d}",
                "title": title,
                "content": content,
                "kind": "concept",
                "source": "wikipedia:ko",
            }
        )
        selected["wikipedia"] += 1

    for source_index, commit in enumerate(raw["commits"][:50]):
        message = str(commit.get("message", ""))
        if not message or len(message) < 10:
            continue
        documents.append(
            {
                "id": f"commit-{source_index:04d}",
                "title": message.split("\n", maxsplit=1)[0][:100],
                "content": message,
                "kind": "artifact",
                "source": "github:commit",
            }
        )
        selected["commits"] += 1

    for source_index, issue in enumerate(raw["issues"][:50]):
        title = str(issue.get("title", ""))
        body = str(issue.get("body", "") or "")[:1500]
        if not title:
            continue
        documents.append(
            {
                "id": f"issue-{source_index:04d}",
                "title": title[:100],
                "content": body or title,
                "kind": "entity",
                "source": "github:issue",
            }
        )
        selected["issues"] += 1

    if len(documents) <= 10:
        raise ValueError("combined QA fixture produced 10 or fewer documents")
    files = []
    relative_by_role = dict(QA_SOURCES)
    for role, path in paths.items():
        files.append(
            file_fingerprint(
                path,
                display_path=relative_by_role[role],
            )
            | {"role": role}
        )
    state = {
        "files": files,
        "raw_records": {role: len(rows) for role, rows in raw.items()},
        "selected_records": selected,
        "documents": len(documents),
        "queries": len(QUERIES),
        "selected_payload_sha256": canonical_json_sha256(documents),
    }
    return documents, state


def _module_binding(
    value: Any, *, source_root: Path, repository_root: Path, name: str
) -> dict[str, Any]:
    module = value if inspect.ismodule(value) else inspect.getmodule(value)
    raw_path = getattr(module, "__file__", None) if module is not None else None
    if not isinstance(raw_path, str):
        raise RuntimeError(f"cannot resolve source module for {name}")
    path = Path(raw_path).resolve()
    try:
        display = path.relative_to(repository_root.resolve()).as_posix()
        path.relative_to(source_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"loaded {name} from {path}, outside {source_root}") from exc
    return {
        **file_fingerprint(path, display_path=display),
        "resolved_path": str(path),
    }


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


def _process_memory() -> dict[str, Any]:
    current, lifetime_peak, kind = perf._process_memory_bytes()
    return {
        "current_rss_mb": None if current is None else current / 1_000_000.0,
        "lifetime_peak_rss_mb": (
            None if lifetime_peak is None else lifetime_peak / 1_000_000.0
        ),
        "measurement_kind": kind,
    }


def _official_p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("latency percentile requires samples")
    ordered = sorted(values)
    return ordered[int(len(ordered) * 0.95)]


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency summary requires samples")
    return {
        "samples": len(values),
        "minimum_ms": min(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _official_p95(values),
        "average_ms": statistics.fmean(values),
        "maximum_ms": max(values),
    }


def _measurement_result(
    first_latencies: list[float],
    first_rankings: list[list[str]],
    steady_latencies: list[float],
    steady_rounds_ms: list[float],
    ranking_hashes: list[str],
) -> dict[str, Any]:
    official = _latency_summary(first_latencies)
    official["first_query_ms"] = first_latencies[0]
    official["passes"] = {
        "p95_under_100_ms": official["p95_ms"] < OFFICIAL_P95_LIMIT_MS,
        "average_under_50_ms": official["average_ms"] < OFFICIAL_AVERAGE_LIMIT_MS,
    }
    return {
        "official_first_pass": official,
        "steady": {
            **_latency_summary(steady_latencies),
            "round_ms": {
                "per_round": steady_rounds_ms,
                "median": statistics.median(steady_rounds_ms),
                "minimum": min(steady_rounds_ms),
                "maximum": max(steady_rounds_ms),
            },
        },
        "official_rankings_sha256": canonical_json_sha256(first_rankings),
        "all_ranking_hashes": ranking_hashes,
        "rankings_deterministic": len(set(ranking_hashes)) == 1,
    }


def _measure_sync(
    search: Callable[[str], list[str]], *, repeats: int
) -> dict[str, Any]:
    first_latencies: list[float] = []
    first_rankings: list[list[str]] = []
    for query in QUERIES:
        started = time.perf_counter_ns()
        ranking = list(search(query))
        first_latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        first_rankings.append(ranking)

    hashes = [canonical_json_sha256(first_rankings)]
    steady_latencies: list[float] = []
    round_ms: list[float] = []
    for _ in range(repeats):
        rankings: list[list[str]] = []
        round_started = time.perf_counter_ns()
        for query in QUERIES:
            started = time.perf_counter_ns()
            ranking = list(search(query))
            steady_latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
            rankings.append(ranking)
        round_ms.append((time.perf_counter_ns() - round_started) / 1_000_000.0)
        hashes.append(canonical_json_sha256(rankings))
    return _measurement_result(
        first_latencies, first_rankings, steady_latencies, round_ms, hashes
    )


async def _measure_async(
    search: Callable[[str], Awaitable[list[str]]], *, repeats: int
) -> dict[str, Any]:
    first_latencies: list[float] = []
    first_rankings: list[list[str]] = []
    for query in QUERIES:
        started = time.perf_counter_ns()
        ranking = list(await search(query))
        first_latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        first_rankings.append(ranking)

    hashes = [canonical_json_sha256(first_rankings)]
    steady_latencies: list[float] = []
    round_ms: list[float] = []
    for _ in range(repeats):
        rankings: list[list[str]] = []
        round_started = time.perf_counter_ns()
        for query in QUERIES:
            started = time.perf_counter_ns()
            ranking = list(await search(query))
            steady_latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
            rankings.append(ranking)
        round_ms.append((time.perf_counter_ns() - round_started) / 1_000_000.0)
        hashes.append(canonical_json_sha256(rankings))
    return _measurement_result(
        first_latencies, first_rankings, steady_latencies, round_ms, hashes
    )


def _batch_result(elapsed_ns: int, *, operation: str) -> dict[str, Any]:
    elapsed_seconds = elapsed_ns / 1_000_000_000.0
    throughput = BATCH_SIZE / elapsed_seconds if elapsed_seconds else math.inf
    return {
        "operation": operation,
        "nodes": BATCH_SIZE,
        "elapsed_ms": elapsed_ns / 1_000_000.0,
        "nodes_per_second": throughput,
        "passes_upstream_50_nodes_per_second": (
            throughput > OFFICIAL_BATCH_MIN_NODES_PER_SECOND
        ),
    }


def _run_omnifuse(documents: list[dict[str, str]], *, repeats: int) -> dict[str, Any]:
    sys.path.insert(0, str(SOURCE_ROOT))
    import omnifuse as package
    from omnifuse import Chunk, build_inmemory
    from omnifuse.oneshot import OmniFuse

    bindings = {
        "package": _module_binding(
            package,
            source_root=SOURCE_ROOT,
            repository_root=ROOT,
            name="omnifuse",
        ),
        "build_inmemory": _module_binding(
            build_inmemory,
            source_root=SOURCE_ROOT,
            repository_root=ROOT,
            name="omnifuse.build_inmemory",
        ),
        "retrieve": _module_binding(
            OmniFuse.retrieve,
            source_root=SOURCE_ROOT,
            repository_root=ROOT,
            name="omnifuse.OmniFuse.retrieve",
        ),
    }
    chunks = [
        Chunk(
            id=row["id"],
            title=row["title"],
            text=row["content"],
            meta={"kind": row["kind"], "source": row["source"]},
        )
        for row in documents
    ]
    before = _process_memory()
    started = time.perf_counter_ns()
    graph = build_inmemory([], [], chunks, vector_k=K)
    build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    post_build = _process_memory()

    def search(query: str) -> list[str]:
        return [chunk.id for chunk, _score in graph.retrieve(query, limit=K)]

    query = _measure_sync(search, repeats=repeats)
    post_query = _process_memory()

    batch_chunks = [
        Chunk(
            id=f"batch-{index}",
            title=f"Batch node {index}",
            text=(
                f"Content for batch node {index} about software engineering "
                f"topic {index % 10}"
            ),
        )
        for index in range(BATCH_SIZE)
    ]
    started = time.perf_counter_ns()
    batch_graph = build_inmemory([], [], batch_chunks)
    raw_elapsed = time.perf_counter_ns() - started
    materialize_started = time.perf_counter_ns()
    batch_graph.vector._prepare_for_persistence()
    materialize_elapsed = time.perf_counter_ns() - materialize_started
    batch = {
        "raw_lazy_store": _batch_result(
            raw_elapsed,
            operation="build_inmemory over 100 chunks; lexical index remains lazy",
        ),
        "lexical_materialization": _batch_result(
            raw_elapsed + materialize_elapsed,
            operation="build_inmemory plus complete lexical index materialization",
        ),
    }
    graph.close()
    batch_graph.close()
    return {
        "system": SYSTEM_LABELS["omnifuse"],
        "build_ms": build_ms,
        "query": query,
        "batch": batch,
        "process_memory": {
            "before_build": before,
            "post_build": post_build,
            "post_query": post_query,
        },
        "runtime": {
            "package_version": _package_version("omnifuse"),
            "source_bindings": bindings,
        },
    }


async def _run_synaptic(
    repo: Path, documents: list[dict[str, str]], *, repeats: int
) -> dict[str, Any]:
    source_root = (repo / "src").resolve()
    sys.path.insert(0, str(source_root))
    import synaptic as package
    from synaptic.backends.memory import MemoryBackend
    from synaptic.extensions.tagger_regex import RegexTagExtractor
    from synaptic.graph import SynapticGraph
    from synaptic.models import Node, NodeKind

    bindings = {
        "package": _module_binding(
            package,
            source_root=source_root,
            repository_root=repo,
            name="synaptic",
        ),
        "memory_backend": _module_binding(
            MemoryBackend,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.MemoryBackend",
        ),
        "graph": _module_binding(
            SynapticGraph,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.SynapticGraph",
        ),
        "regex_tagger": _module_binding(
            RegexTagExtractor,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.RegexTagExtractor",
        ),
    }
    kind_map = {
        "concept": NodeKind.CONCEPT,
        "artifact": NodeKind.ARTIFACT,
        "entity": NodeKind.ENTITY,
    }
    before = _process_memory()
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend, tag_extractor=RegexTagExtractor())
    reverse_ids: dict[str, str] = {}
    started = time.perf_counter_ns()
    for row in documents:
        node = await graph.add(
            title=row["title"],
            content=row["content"],
            kind=kind_map[row["kind"]],
            source=row["source"],
        )
        reverse_ids[str(node.id)] = row["id"]
    build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    post_build = _process_memory()

    async def search(query: str) -> list[str]:
        result = await graph.search(query, limit=K)
        return [
            reverse_ids.get(str(hit.node.id), f"native:{hit.node.id}")
            for hit in result.nodes
        ]

    query = await _measure_async(search, repeats=repeats)
    post_query = _process_memory()
    batch_nodes = [
        Node(
            kind=NodeKind.CONCEPT,
            title=f"Batch node {index}",
            content=(
                f"Content for batch node {index} about software engineering "
                f"topic {index % 10}"
            ),
        )
        for index in range(BATCH_SIZE)
    ]
    started = time.perf_counter_ns()
    await backend.save_nodes_batch(batch_nodes)
    batch = {
        "upstream_backend_batch": _batch_result(
            time.perf_counter_ns() - started,
            operation="MemoryBackend.save_nodes_batch over 100 Node objects",
        )
    }
    await backend.close()
    return {
        "system": SYSTEM_LABELS["synaptic"],
        "build_ms": build_ms,
        "query": query,
        "batch": batch,
        "process_memory": {
            "before_build": before,
            "post_build": post_build,
            "post_query": post_query,
        },
        "runtime": {
            "package_version": _package_version("synaptic-memory"),
            "source_bindings": bindings,
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_worker(args: argparse.Namespace) -> int:
    if args.result_file is None or args.worker_run_id is None:
        raise ValueError("worker mode requires --result-file and --worker-run-id")
    ensure_output_absent(args.result_file)
    repo = args.synaptic_repo.resolve()
    documents, data_state = _prepare_documents(repo, args.data_dir)
    if args.worker == "omnifuse":
        result = _run_omnifuse(documents, repeats=args.repeats)
    else:
        result = asyncio.run(_run_synaptic(repo, documents, repeats=args.repeats))
    _documents_after, data_after = _prepare_documents(repo, args.data_dir)
    assert_unchanged("QA worker inputs", data_state, data_after)
    write_json_once(
        args.result_file,
        {
            "schema": WORKER_SCHEMA,
            "schema_version": WORKER_SCHEMA_VERSION,
            "status": "ok",
            "system": args.worker,
            "data": data_state,
            "configuration": {
                "k": K,
                "queries": list(QUERIES),
                "repeats": args.repeats,
            },
            "worker_identity": capture_worker_identity(args.worker_run_id),
            "environment": _worker_environment(),
            "result": result,
        },
    )
    return 0


def _doctor_inputs(data_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": f"QA combined {item['role']}",
            "target_id": TARGET_ID,
            "role": item["role"],
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
        }
        for item in data_state["files"]
    ]


def _source_state(repo: Path) -> dict[str, Any]:
    return {
        "harness": file_fingerprint(
            SCRIPT_PATH, display_path="eval/qa_performance_bench.py"
        ),
        "process_memory_support": file_fingerprint(
            EVAL_DIR / "perf_bench.py", display_path="eval/perf_bench.py"
        ),
        "upstream_fixture": file_fingerprint(
            repo / "tests" / "qa" / "conftest.py",
            display_path="tests/qa/conftest.py",
        ),
        "upstream_performance_test": file_fingerprint(
            repo / "tests" / "qa" / "test_performance.py",
            display_path="tests/qa/test_performance.py",
        ),
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.out is None or args.doctor_manifest is None:
        raise ValueError("claim-grade mode requires --out and --doctor-manifest")
    if args.trials < 2 or args.trials % 2:
        raise ValueError("--trials must be an even value of at least 2")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    output = args.out.resolve()
    ensure_output_absent(output)
    repo = args.synaptic_repo.resolve()
    if not (repo / "src" / "synaptic").is_dir():
        raise FileNotFoundError(f"Synaptic source package not found below {repo}")
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable not found: {python}")
    _documents, data = _prepare_documents(repo, args.data_dir)
    repositories = {
        "omnifuse": repository_fingerprint(ROOT),
        "synaptic_memory": repository_fingerprint(repo),
    }
    sources = _source_state(repo)
    scorer = {
        "omnifuse": file_fingerprint(
            ROOT / "eval" / "metrics.py", display_path="eval/metrics.py"
        ),
        "synaptic_memory": file_fingerprint(
            repo / "tests" / "benchmark" / "metrics.py",
            display_path="tests/benchmark/metrics.py",
        ),
    }
    doctor, links = load_doctor_manifest(
        args.doctor_manifest.resolve(), _doctor_inputs(data)
    )
    verify_doctor_runtime(
        doctor,
        omnifuse_repository=repositories["omnifuse"],
        synaptic_repository=repositories["synaptic_memory"],
        omnifuse_scorer=scorer["omnifuse"],
        synaptic_scorer=scorer["synaptic_memory"],
    )
    return {
        "output": output,
        "repo": repo,
        "python": file_fingerprint(python),
        "data": data,
        "repositories": repositories,
        "sources": sources,
        "scorer": scorer,
        "doctor_manifest": doctor,
        "doctor_links": links,
    }


def _worker_command(
    args: argparse.Namespace,
    *,
    system: str,
    result_file: Path,
    run_id: str,
) -> list[str]:
    command = [
        str(args.python.resolve()),
        "-I",
        "-X",
        "utf8",
        "-B",
        str(SCRIPT_PATH),
        "--synaptic-repo",
        str(args.synaptic_repo.resolve()),
        "--repeats",
        str(args.repeats),
        "--worker",
        system,
        "--result-file",
        str(result_file),
        "--worker-run-id",
        run_id,
    ]
    if args.data_dir is not None:
        command.extend(["--data-dir", str(args.data_dir.resolve())])
    return command


def _binding_root(binding: Mapping[str, Any], *, root: Path, label: str) -> None:
    resolved = binding.get("resolved_path")
    if not isinstance(resolved, str):
        raise ProvenanceError(f"{label} binding omitted resolved path")
    try:
        Path(resolved).resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ProvenanceError(f"{label} binding is outside {root}") from exc


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ProvenanceError(f"{label} must be finite and non-negative")
    return number


def _validate_worker(
    payload: Any,
    *,
    system: str,
    data: Mapping[str, Any],
    repeats: int,
    run_id: str,
    repo: Path,
    trial: int,
    position: int,
) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != WORKER_SCHEMA
        or payload.get("schema_version") != WORKER_SCHEMA_VERSION
        or payload.get("status") != "ok"
        or payload.get("system") != system
    ):
        raise ProvenanceError(f"invalid {system} QA worker contract")
    assert_unchanged(f"{system} QA worker data", data, payload.get("data"))
    configuration = payload.get("configuration")
    expected_configuration = {
        "k": K,
        "queries": list(QUERIES),
        "repeats": repeats,
    }
    assert_unchanged(
        f"{system} QA worker configuration", expected_configuration, configuration
    )
    identity = validate_worker_identity(
        payload.get("worker_identity"),
        expected_run_id=run_id,
        label=f"{system} QA worker",
    )
    environment = payload.get("environment")
    if not isinstance(environment, dict) or set(environment) != WORKER_ENVIRONMENT_KEYS:
        raise ProvenanceError(f"invalid {system} QA worker environment")
    for flag in (
        "isolated",
        "ignore_environment",
        "no_user_site",
        "safe_path",
        "utf8_mode",
    ):
        if environment[flag] is not True:
            raise ProvenanceError(f"{system} QA worker did not enable {flag}")
    if environment["user_site_enabled"] is not False:
        raise ProvenanceError(f"{system} QA worker enabled user site")

    result = payload.get("result")
    if not isinstance(result, dict) or result.get("system") != SYSTEM_LABELS[system]:
        raise ProvenanceError(f"invalid {system} QA metrics")
    query = result.get("query")
    if not isinstance(query, dict):
        raise ProvenanceError(f"{system} QA worker omitted query metrics")
    official = query.get("official_first_pass")
    steady = query.get("steady")
    if not isinstance(official, dict) or not isinstance(steady, dict):
        raise ProvenanceError(f"{system} QA latency metrics are invalid")
    p95 = _number(official.get("p95_ms"), label=f"{system} official p95")
    average = _number(official.get("average_ms"), label=f"{system} official average")
    _number(official.get("first_query_ms"), label=f"{system} first query")
    passes = official.get("passes")
    expected_passes = {
        "p95_under_100_ms": p95 < OFFICIAL_P95_LIMIT_MS,
        "average_under_50_ms": average < OFFICIAL_AVERAGE_LIMIT_MS,
    }
    assert_unchanged(f"{system} official threshold verdict", expected_passes, passes)
    for key in ("p50_ms", "p95_ms", "average_ms"):
        _number(steady.get(key), label=f"{system} steady {key}")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(
        runtime.get("source_bindings"), dict
    ):
        raise ProvenanceError(f"{system} QA worker omitted source bindings")
    expected_bindings = (
        {"package", "build_inmemory", "retrieve"}
        if system == "omnifuse"
        else {"package", "memory_backend", "graph", "regex_tagger"}
    )
    bindings = runtime["source_bindings"]
    if set(bindings) != expected_bindings:
        raise ProvenanceError(f"{system} QA source binding set is invalid")
    binding_root = SOURCE_ROOT if system == "omnifuse" else repo / "src"
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ProvenanceError(f"{system} QA binding {name} is invalid")
        _binding_root(binding, root=binding_root, label=f"{system} {name}")
    return {
        **result,
        "trial": {"number": trial, "order_position": position},
        "worker_identity": identity,
        "worker_environment": environment,
    }


def _nested(row: Mapping[str, Any], path: str) -> float | None:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    if value is None:
        return None
    return _number(value, label=path)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "median": None, "minimum": None, "maximum": None}
    return {
        "samples": len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "per_trial": list(values),
    }


AGGREGATE_PATHS = {
    "build_ms": "build_ms",
    "first_query_ms": "query.official_first_pass.first_query_ms",
    "official_p95_ms": "query.official_first_pass.p95_ms",
    "official_average_ms": "query.official_first_pass.average_ms",
    "steady_p50_ms": "query.steady.p50_ms",
    "steady_p95_ms": "query.steady.p95_ms",
    "steady_average_ms": "query.steady.average_ms",
    "steady_round_ms": "query.steady.round_ms.median",
    "post_query_rss_mb": "process_memory.post_query.current_rss_mb",
    "lifetime_peak_rss_mb": "process_memory.post_query.lifetime_peak_rss_mb",
}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for name, path in AGGREGATE_PATHS.items():
        values = [value for row in rows if (value := _nested(row, path)) is not None]
        metrics[name] = _distribution(values)
    pass_counts = {
        "p95_under_100_ms": sum(
            bool(row["query"]["official_first_pass"]["passes"]["p95_under_100_ms"])
            for row in rows
        ),
        "average_under_50_ms": sum(
            bool(row["query"]["official_first_pass"]["passes"]["average_under_50_ms"])
            for row in rows
        ),
    }
    return {
        "trials": len(rows),
        "metrics": metrics,
        "official_contract_pass_trials": pass_counts,
        "ranking_deterministic_trials": sum(
            bool(row["query"]["rankings_deterministic"]) for row in rows
        ),
    }


def _head_to_head(aggregates: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {}
    wins = {"omnifuse": 0, "synaptic": 0, "ties": 0}
    for name in AGGREGATE_PATHS:
        left = aggregates["omnifuse"]["metrics"][name]["median"]
        right = aggregates["synaptic"]["metrics"][name]["median"]
        if left is None or right is None:
            winner = None
        elif left < right:
            winner = "omnifuse"
            wins["omnifuse"] += 1
        elif right < left:
            winner = "synaptic"
            wins["synaptic"] += 1
        else:
            winner = "tie"
            wins["ties"] += 1
        metrics[name] = {
            "direction": "lower_is_better",
            "omnifuse_median": left,
            "synaptic_median": right,
            "winner": winner,
        }
    return {"metrics": metrics, "verdict": wins}


def _controller(args: argparse.Namespace) -> int:
    state = _preflight(args)
    output: Path = state["output"]
    workers = (
        args.workers_dir.resolve()
        if args.workers_dir is not None
        else output.parent / f".{output.stem}.workers"
    )
    if workers.exists():
        raise ProvenanceError(f"refusing to reuse worker directory {workers}")
    workers.mkdir(parents=True)
    schedule = [
        ("omnifuse", "synaptic") if trial % 2 else ("synaptic", "omnifuse")
        for trial in range(1, args.trials + 1)
    ]
    rows: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    processes: list[dict[str, Any]] = []
    for trial_number, order in enumerate(schedule, 1):
        for position, system in enumerate(order, 1):
            result_file = workers / f"trial-{trial_number:02d}-{position}-{system}.json"
            run_id = new_worker_run_id()
            completed, launcher_pid = run_with_launcher_pid(
                _worker_command(
                    args,
                    system=system,
                    result_file=result_file,
                    run_id=run_id,
                ),
                cwd=ROOT,
                env=_isolated_environment(),
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"{system} QA worker failed with {completed.returncode}: {detail}"
                )
            payload, artifact = read_json_artifact(result_file)
            validated = _validate_worker(
                payload,
                system=system,
                data=state["data"],
                repeats=args.repeats,
                run_id=run_id,
                repo=state["repo"],
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
                    "launcher_pid": launcher_pid,
                    "worker_pid": identity["worker_pid"],
                    "same_process_id": launcher_pid == identity["worker_pid"],
                    "returncode": completed.returncode,
                    "result_artifact": artifact,
                    "stdout_bytes": len((completed.stdout or "").encode("utf-8")),
                    "stderr_bytes": len((completed.stderr or "").encode("utf-8")),
                }
            )

    _documents_after, data_after = _prepare_documents(state["repo"], args.data_dir)
    assert_unchanged("QA data postflight", state["data"], data_after)
    assert_unchanged(
        "repository fingerprints postflight",
        state["repositories"],
        {
            "omnifuse": repository_fingerprint(ROOT),
            "synaptic_memory": repository_fingerprint(state["repo"]),
        },
    )
    assert_unchanged(
        "benchmark sources postflight", state["sources"], _source_state(state["repo"])
    )
    assert_unchanged(
        "Python executable postflight",
        state["python"],
        file_fingerprint(args.python.resolve()),
    )
    verify_doctor_manifest(state["doctor_manifest"])
    verify_doctor_runtime(
        state["doctor_manifest"],
        omnifuse_repository=state["repositories"]["omnifuse"],
        synaptic_repository=state["repositories"]["synaptic_memory"],
        omnifuse_scorer=state["scorer"]["omnifuse"],
        synaptic_scorer=state["scorer"]["synaptic_memory"],
    )
    aggregates = {system: _aggregate(rows[system]) for system in SYSTEMS}
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_level": PROVENANCE_LEVEL,
        "repositories": state["repositories"],
        "sources": state["sources"],
        "scorer": state["scorer"],
        "doctor_manifest": state["doctor_manifest"],
        "doctor_links": state["doctor_links"],
        "python": state["python"],
        "data": state["data"],
        "contract": {
            "upstream_fixture": "tests/qa/conftest.py::combined_graph",
            "upstream_test": (
                "tests/qa/test_performance.py::"
                "TestSearchPerformance.test_search_latency_p95"
            ),
            "queries": list(QUERIES),
            "k": K,
            "official_p95_limit_ms": OFFICIAL_P95_LIMIT_MS,
            "official_average_limit_ms": OFFICIAL_AVERAGE_LIMIT_MS,
            "official_percentile_rule": "sorted_latencies[int(n * 0.95)]",
            "fresh_process_per_system_trial": True,
            "counterbalanced_ab_ba": True,
            "trials": args.trials,
            "steady_repeats": args.repeats,
            "batch_capability_caveat": (
                "Synaptic times MemoryBackend.save_nodes_batch. OmniFuse records raw "
                "lazy-store construction and complete lexical materialization separately; "
                "batch rates are threshold evidence, not a head-to-head winner metric."
            ),
        },
        "worker_processes": processes,
        "worker_process_summary": worker_process_summary(
            processes, expected_count=args.trials * len(SYSTEMS)
        ),
        "results": {
            system: {"trials": rows[system], "aggregate": aggregates[system]}
            for system in SYSTEMS
        },
        "head_to_head": _head_to_head(aggregates),
        "postflight": {
            "data_unchanged": True,
            "repositories_unchanged": True,
            "sources_unchanged": True,
            "python_unchanged": True,
            "doctor_unchanged": True,
        },
    }
    write_json_once(output, report)
    print(
        "QA memory benchmark: "
        f"OmniFuse official p95 median={aggregates['omnifuse']['metrics']['official_p95_ms']['median']:.3f}ms; "
        f"Synaptic={aggregates['synaptic']['metrics']['official_p95_ms']['median']:.3f}ms"
    )
    print(f"wrote {output}")
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
