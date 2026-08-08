"""Compare OmniFuse with a specified synaptic-memory checkout on finreg.

OmniFuse is rebuilt from the tracked corpus in this process. The
synaptic-memory arm queries a caller-supplied, prebuilt SQLite graph; therefore
the timing columns are diagnostic and are not a controlled build/query speed
comparison. Both ``--synaptic-repo`` and ``--synaptic-graph`` are required.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.metadata
import json
import os
import site
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = (REPOSITORY_ROOT / "src").resolve()
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from _common import (  # noqa: E402
    K,
    build_reference_triples,
    load_corpus,
    load_queries,
    score_mrr,
    score_strict,
    to_chunks_nodes,
)
from finreg_bench import (  # noqa: E402
    CANDIDATE_LIMIT,
    COMMON_HARNESS_PATH,
    DEDUPE_POLICY,
    MULTIHOP_QUERY_PATH,
    PROVENANCE_PATH,
    SINGLE_QUERY_PATH,
    SCRIPT_PATH as FINREG_HARNESS_PATH,
    _atomic_write_json,
    _doctor_input_specs,
    _file_record,
    _group_doctor_links,
    _input_records,
    _omnifuse_import_sources,
    _runtime_environment,
    _scorer_records,
)
from omnifuse import build_inmemory  # noqa: E402
from provenance import (  # noqa: E402
    ProvenanceError,
    assert_unchanged,
    canonical_json_sha256,
    ensure_output_absent,
    load_doctor_manifest,
    repository_fingerprint,
    verify_doctor_manifest,
    verify_doctor_runtime,
)


INDEX_CONDITION_CAVEAT = (
    "OmniFuse is rebuilt in-process from the tracked corpus, while synaptic-memory "
    "queries the supplied prebuilt SQLite graph. This harness fingerprints that "
    "artifact but cannot infer or verify its ingestion provenance against the corpus. "
    "Treat the scores as an accuracy comparison under stated index conditions and do "
    "not compare the elapsed times as build or query-speed equivalents."
)
PROTOCOL_BOUNDARY = (
    "This comparison executes both systems in this invocation, but their index "
    "capabilities differ. Scores copied from standalone finreg_bench artifacts or "
    "earlier comparison artifacts must not be paired as if they came from this run."
)
PROVENANCE_LEVEL = (
    "strict-doctor-fresh-worker-runtime-sqlite-watch-preflight-postflight-write-once-v2"
)
WORKER_SCHEMA = "omnifuse.eval.finreg_synaptic_worker"
WORKER_SCHEMA_VERSION = 1
RUNTIME_DISTRIBUTIONS = {
    "aiosqlite": "aiosqlite",
    "kiwipiepy": "kiwipiepy",
    "kiwipiepy_model": "kiwipiepy-model",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synaptic-repo",
        type=Path,
        required=True,
        help="synaptic-memory checkout whose src/ package must be executed",
    )
    parser.add_argument(
        "--synaptic-graph",
        type=Path,
        required=True,
        help="prebuilt synaptic-memory SQLite graph for the finreg corpus",
    )
    parser.add_argument(
        "--synaptic-python",
        type=Path,
        help=(
            "Python executable containing the selected synaptic runtime; required "
            "for machine reports"
        ),
    )
    parser.add_argument("--out", type=Path, help="write the JSON report atomically")
    parser.add_argument(
        "--doctor-manifest",
        type=Path,
        help="strict eval/bench.py doctor JSON; required with --out",
    )
    parser.add_argument(
        "--_synaptic-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    return parser


def _is_below(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _verify_module_under(module: Any, expected_source: Path, name: str) -> Path:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise RuntimeError(f"{name} has no inspectable module path")
    actual_path = Path(raw_path).resolve()
    if not _is_below(actual_path, expected_source):
        raise RuntimeError(
            f"loaded {name} from {actual_path}; expected checkout source below "
            f"{expected_source.resolve()}"
        )
    return actual_path


def _prepend_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != resolved]
    sys.path.insert(0, resolved)
    importlib.invalidate_caches()


def _load_synaptic_api(repo: Path) -> tuple[Any, Any, dict[str, str]]:
    source = (repo.resolve() / "src").resolve()
    package_init = source / "synaptic" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            f"synaptic checkout source package not found: {package_init}"
        )

    _prepend_import_path(source)
    package = importlib.import_module("synaptic")
    package_path = _verify_module_under(package, source, "synaptic")
    backend_base_module = importlib.import_module("synaptic.backends.sqlite")
    backend_module = importlib.import_module("synaptic.backends.sqlite_graph")
    search_module = importlib.import_module("synaptic.extensions.evidence_search")
    paths = {
        "package": str(package_path),
        "sqlite_backend_base": str(
            _verify_module_under(
                backend_base_module, source, "synaptic.backends.sqlite"
            )
        ),
        "sqlite_backend": str(
            _verify_module_under(
                backend_module, source, "synaptic.backends.sqlite_graph"
            )
        ),
        "evidence_search": str(
            _verify_module_under(
                search_module, source, "synaptic.extensions.evidence_search"
            )
        ),
    }
    return backend_module.SqliteGraphBackend, search_module.EvidenceSearch, paths


def _query_map(queries: Sequence[dict[str, Any]]) -> dict[str, str]:
    by_text: dict[str, str] = {}
    for query in queries:
        text = str(query["query"])
        query_id = str(query.get("qid", ""))
        previous = by_text.setdefault(text, query_id)
        if previous != query_id:
            raise ValueError(
                "duplicate query text has multiple query ids; scoring would be ambiguous"
            )
    return by_text


def run_omnifuse(
    single_hop: Sequence[dict[str, Any]],
    multi_hop: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    docs = load_corpus()
    nodes, chunks = to_chunks_nodes(docs)
    triples = build_reference_triples(docs)
    _query_map([*single_hop, *multi_hop])

    total_started = time.perf_counter()
    build_started = time.perf_counter()
    omnifuse = build_inmemory(
        nodes,
        triples,
        chunks,
        graph_fusion=True,
        vector_k=CANDIDATE_LIMIT,
    )
    build_seconds = time.perf_counter() - build_started

    def retrieve(text: str) -> list[str]:
        return [
            chunk.id for chunk, _score in omnifuse.retrieve(text, limit=CANDIDATE_LIMIT)
        ]

    query_started = time.perf_counter()
    scores = {
        "single_hop": score_mrr(retrieve, single_hop, k=K),
        "multi_hop": score_strict(retrieve, multi_hop, k=K),
    }
    return {
        "scores": scores,
        "timing_seconds": {
            "rebuild": build_seconds,
            "query_and_score": time.perf_counter() - query_started,
            "total": time.perf_counter() - total_started,
        },
    }


async def run_synaptic(
    repo: Path,
    graph_path: Path,
    single_hop: Sequence[dict[str, Any]],
    multi_hop: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    backend_type, search_type, module_paths = _load_synaptic_api(repo)
    all_queries = [*single_hop, *multi_hop]
    _query_map(all_queries)
    backend = backend_type(str(graph_path.resolve()))
    cache: dict[str, list[str]] = {}
    total_started = time.perf_counter()
    try:
        await backend.connect()
        searcher = search_type(backend=backend, embedder=None, reranker=None)
        query_started = time.perf_counter()
        for query in all_queries:
            result = await searcher.search(
                str(query["query"]),
                k=CANDIDATE_LIMIT,
                fts_seed_limit=30,
            )
            ranked: list[str] = []
            seen: set[str] = set()
            for evidence in result.evidence:
                properties = evidence.node.properties or {}
                document_id = evidence.document_id or properties.get("doc_id", "")
                normalized = str(document_id)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    ranked.append(normalized)
            cache[str(query["query"])] = ranked
        query_seconds = time.perf_counter() - query_started

        def retrieve(text: str) -> list[str]:
            return cache[text]

        scores = {
            "single_hop": score_mrr(retrieve, single_hop, k=K),
            "multi_hop": score_strict(retrieve, multi_hop, k=K),
        }
        return {
            "scores": scores,
            "timing_seconds": {
                "connect_query_and_score": time.perf_counter() - total_started,
                "query_only": query_seconds,
                "prebuilt_index_rebuild": None,
            },
            "module_paths": module_paths,
        }
    finally:
        await backend.close()


def _artifact_manifest(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"synaptic SQLite graph not found: {resolved}")

    def group(candidates: Sequence[Path]) -> dict[str, object]:
        files = [
            _file_record(candidate) for candidate in candidates if candidate.is_file()
        ]
        fingerprint_payload = [
            {
                "name": Path(str(item["path"])).name,
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in files
        ]
        return {
            "files": files,
            "artifact_manifest_sha256": canonical_json_sha256(fingerprint_payload),
        }

    return {
        "database_path": str(resolved),
        "durable": group((resolved, Path(f"{resolved}-wal"))),
        "transient": group((Path(f"{resolved}-shm"),)),
    }


def _durable_artifact_identity(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "database_path": manifest["database_path"],
        "durable": manifest["durable"],
    }


def _durable_files_by_name(
    manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    durable = manifest.get("durable")
    if not isinstance(durable, dict):
        raise ProvenanceError("SQLite artifact has no durable manifest")
    files = durable.get("files")
    if not isinstance(files, list):
        raise ProvenanceError("SQLite artifact durable file list is invalid")
    return {Path(str(record["path"])).name: record for record in files}


def _open_graph_watcher(graph_path: Path) -> tuple[sqlite3.Connection, int]:
    resolved = graph_path.resolve()
    uri = f"file:{resolved.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        connection.execute("PRAGMA query_only=ON")
        row = connection.execute("PRAGMA data_version").fetchone()
        if row is None:
            raise ProvenanceError("SQLite watcher returned no PRAGMA data_version")
        return connection, int(row[0])
    except Exception:
        connection.close()
        raise


def _close_graph_watcher(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    watcher = state.pop("_graph_watcher", None)
    if watcher is not None:
        watcher.close()


def _graph_guard_preflight(graph_path: Path) -> dict[str, Any]:
    initial = _artifact_manifest(graph_path)
    watcher, data_version = _open_graph_watcher(graph_path)
    try:
        watcher_open = _artifact_manifest(graph_path)
        initial_files = _durable_files_by_name(initial)
        watcher_files = _durable_files_by_name(watcher_open)
        database_name = Path(str(initial["database_path"])).name
        assert_unchanged(
            "synaptic database while opening watcher",
            initial_files[database_name],
            watcher_files[database_name],
        )
        wal_name = f"{database_name}-wal"
        initial_wal = initial_files.get(wal_name)
        watcher_wal = watcher_files.get(wal_name)
        if initial_wal is not None:
            assert_unchanged(
                "synaptic WAL while opening watcher", initial_wal, watcher_wal
            )
        elif watcher_wal is not None and (
            watcher_wal.get("bytes") != 0
            or watcher_wal.get("sha256")
            != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ):
            raise ProvenanceError(
                "synaptic WAL gained durable content while opening watcher"
            )
    except Exception:
        watcher.close()
        raise
    return {
        "graph": initial,
        "graph_watcher_open": watcher_open,
        "graph_data_version_before": data_version,
        "_graph_watcher": watcher,
    }


def _verify_graph_guard(
    state: dict[str, Any], graph_path: Path
) -> tuple[dict[str, object], int]:
    watcher = state.get("_graph_watcher")
    if watcher is None:
        raise ProvenanceError("SQLite graph watcher is missing at postflight")
    row = watcher.execute("PRAGMA data_version").fetchone()
    if row is None:
        raise ProvenanceError("SQLite watcher returned no postflight data_version")
    data_version_after = int(row[0])
    if data_version_after != state["graph_data_version_before"]:
        raise ProvenanceError(
            "synaptic graph changed: SQLite PRAGMA data_version advanced during run"
        )
    after = _artifact_manifest(graph_path)
    assert_unchanged(
        "synaptic graph durable files",
        _durable_artifact_identity(state["graph_watcher_open"]),
        _durable_artifact_identity(after),
    )
    return after, data_version_after


def _synaptic_import_sources(
    repo: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    root = repo.resolve()
    source = root / "src"
    expected = {
        "package": source / "synaptic" / "__init__.py",
        "sqlite_backend_base": source / "synaptic" / "backends" / "sqlite.py",
        "sqlite_backend": source / "synaptic" / "backends" / "sqlite_graph.py",
        "evidence_search": source / "synaptic" / "extensions" / "evidence_search.py",
    }
    records: dict[str, object] = {}
    module_paths: dict[str, str] = {}
    for name, expected_path in expected.items():
        resolved = expected_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"selected synaptic source file not found for {name}: {resolved}"
            )
        display_path = resolved.relative_to(root).as_posix()
        records[name] = _file_record(resolved, display_path=display_path)
        module_paths[name] = str(resolved)
    return records, module_paths


def _comparison_sources(
    synaptic_repo: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    synaptic_sources, module_paths = _synaptic_import_sources(synaptic_repo)
    return (
        {
            "entrypoint": _file_record(
                SCRIPT_PATH, display_path="eval/compare_synaptic.py"
            ),
            "finreg_support": _file_record(
                FINREG_HARNESS_PATH, display_path="eval/finreg_bench.py"
            ),
            "provenance_helper": _file_record(
                PROVENANCE_PATH, display_path="eval/provenance.py"
            ),
            "shared_finreg_logic": _file_record(
                COMMON_HARNESS_PATH, display_path="eval/_common.py"
            ),
            "scorer": _scorer_records(synaptic_repo),
            "imported_omnifuse": _omnifuse_import_sources(),
            "imported_synaptic_memory": synaptic_sources,
        },
        module_paths,
    )


def _python_executable_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"synaptic Python executable not found: {resolved}")
    return _file_record(resolved)


def _distribution_record(module_name: str, distribution_name: str) -> dict[str, object]:
    module = importlib.import_module(module_name)
    module_path = getattr(module, "__file__", None)
    if not module_path:
        raise ProvenanceError(f"{module_name} has no inspectable module path")
    resolved_module = Path(module_path).resolve()
    if not resolved_module.is_file():
        raise ProvenanceError(
            f"{module_name} module file does not exist: {resolved_module}"
        )

    distribution = importlib.metadata.distribution(distribution_name)
    files: list[dict[str, object]] = []
    for entry in sorted(distribution.files or (), key=lambda item: str(item)):
        installed = Path(distribution.locate_file(entry)).resolve()
        if installed.is_file():
            files.append(
                _file_record(installed, display_path=str(entry).replace("\\", "/"))
            )
    if not files:
        raise ProvenanceError(
            f"installed distribution has no fingerprintable files: {distribution_name}"
        )
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "module": _file_record(resolved_module),
        "installed_files": files,
        "installed_files_manifest_sha256": canonical_json_sha256(files),
    }


def _normalized_sys_path() -> list[str]:
    return [str(Path(entry).resolve()) for entry in sys.path if entry]


def _worker_isolation_record(repo: Path) -> dict[str, object]:
    raw_user_sites = site.getusersitepackages()
    if isinstance(raw_user_sites, str):
        raw_user_sites = [raw_user_sites]
    user_sites = [str(Path(entry).resolve()) for entry in raw_user_sites]
    normalized = _normalized_sys_path()
    normalized_cases = {os.path.normcase(path) for path in normalized}
    return {
        "isolated": bool(sys.flags.isolated),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "no_user_site": bool(sys.flags.no_user_site),
        "safe_path": bool(getattr(sys.flags, "safe_path", False)),
        "utf8_mode": bool(sys.flags.utf8_mode),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "pythonhome": os.environ.get("PYTHONHOME"),
        "pythonuserbase": os.environ.get("PYTHONUSERBASE"),
        "python_no_user_site_env": os.environ.get("PYTHONNOUSERSITE"),
        "sys_path": list(sys.path),
        "normalized_sys_path": normalized,
        "required_source_roots": [
            str((repo.resolve() / "src").resolve()),
            str(EVAL_DIR.resolve()),
            str(SOURCE_ROOT.resolve()),
        ],
        "user_site_paths": user_sites,
        "user_site_present_on_sys_path": any(
            os.path.normcase(path) in normalized_cases for path in user_sites
        ),
    }


def _validate_worker_isolation_record(
    record: object, *, expected_synaptic_source: Path
) -> None:
    if not isinstance(record, dict):
        raise ProvenanceError("synaptic worker omitted isolation evidence")
    for flag in (
        "isolated",
        "ignore_environment",
        "no_user_site",
        "safe_path",
        "utf8_mode",
        "dont_write_bytecode",
    ):
        if record.get(flag) is not True:
            raise ProvenanceError(f"synaptic worker did not enable {flag}")
    if record.get("user_site_enabled") is not False:
        raise ProvenanceError("synaptic worker left user-site packages enabled")
    if record.get("user_site_present_on_sys_path") is not False:
        raise ProvenanceError("synaptic worker sys.path contains a user-site directory")
    for variable in ("pythonpath", "pythonhome", "pythonuserbase"):
        if record.get(variable) is not None:
            raise ProvenanceError(f"synaptic worker inherited {variable}")
    if record.get("python_no_user_site_env") != "1":
        raise ProvenanceError(
            "synaptic worker did not receive the no-user-site environment guard"
        )
    normalized = record.get("normalized_sys_path")
    required = record.get("required_source_roots")
    if not isinstance(normalized, list) or not isinstance(required, list):
        raise ProvenanceError("synaptic worker omitted normalized sys.path evidence")
    expected_roots = [
        str(expected_synaptic_source.resolve()),
        str(EVAL_DIR.resolve()),
        str(SOURCE_ROOT.resolve()),
    ]
    if required != expected_roots:
        raise ProvenanceError("synaptic worker reported unexpected source roots")
    normalized_cases = {os.path.normcase(str(path)) for path in normalized}
    if any(os.path.normcase(str(path)) not in normalized_cases for path in required):
        raise ProvenanceError("synaptic worker sys.path omitted a required source root")
    raw_sys_path = record.get("sys_path")
    if not isinstance(raw_sys_path, list) or any(not entry for entry in raw_sys_path):
        raise ProvenanceError("synaptic worker sys.path contains the current directory")


def _synaptic_runtime_binding(
    repo: Path,
) -> tuple[dict[str, object], Any, Any]:
    backend_type, search_type, module_paths = _load_synaptic_api(repo)
    root = repo.resolve()
    module_records: dict[str, object] = {}
    for name, raw_path in module_paths.items():
        resolved = Path(raw_path).resolve()
        if not _is_below(resolved, root / "src"):
            raise ProvenanceError(
                f"loaded synaptic module {name} from outside selected checkout: "
                f"{resolved}"
            )
        module_records[name] = _file_record(
            resolved, display_path=resolved.relative_to(root).as_posix()
        )

    backend_base_module = importlib.import_module("synaptic.backends.sqlite")
    get_kiwi = getattr(backend_base_module, "_get_kiwi", None)
    if not callable(get_kiwi):
        raise ProvenanceError("selected synaptic backend has no _get_kiwi runtime hook")
    kiwi = get_kiwi()
    kiwi_available = getattr(backend_base_module, "_kiwi_available", None)
    if kiwi is None or kiwi_available is not True:
        raise ProvenanceError(
            "canonical synaptic worker requires active kiwipiepy tokenization"
        )

    distributions = {
        module_name: _distribution_record(module_name, distribution_name)
        for module_name, distribution_name in RUNTIME_DISTRIBUTIONS.items()
    }
    isolation = _worker_isolation_record(repo)
    _validate_worker_isolation_record(
        isolation, expected_synaptic_source=repo.resolve() / "src"
    )
    return (
        {
            "python": {
                "version": sys.version,
                "executable": _python_executable_record(Path(sys.executable)),
            },
            "synaptic_modules": module_records,
            "distributions": distributions,
            "kiwi": {
                "active": True,
                "availability_flag": True,
                "implementation_type": (
                    f"{type(kiwi).__module__}.{type(kiwi).__qualname__}"
                ),
            },
            "isolation": isolation,
        },
        backend_type,
        search_type,
    )


def _worker_payload(
    *, synaptic_repo: Path, graph_path: Path, expected_python: Path
) -> dict[str, object]:
    expected_executable = expected_python.resolve()
    actual_executable = Path(sys.executable).resolve()
    if actual_executable != expected_executable:
        raise ProvenanceError(
            f"worker executed with {actual_executable}; expected {expected_executable}"
        )

    inputs_before = _input_records()
    graph_before = _artifact_manifest(graph_path)
    scorer_before = _scorer_records(synaptic_repo)
    runtime_before, _backend_type, _search_type = _synaptic_runtime_binding(
        synaptic_repo
    )
    single_hop = load_queries(SINGLE_QUERY_PATH.name)
    multi_hop = load_queries(MULTIHOP_QUERY_PATH.name)
    result = asyncio.run(run_synaptic(synaptic_repo, graph_path, single_hop, multi_hop))
    runtime_after, _backend_type, _search_type = _synaptic_runtime_binding(
        synaptic_repo
    )
    scorer_after = _scorer_records(synaptic_repo)
    inputs_after = _input_records()
    graph_after = _artifact_manifest(graph_path)

    assert_unchanged("worker finreg inputs", inputs_before, inputs_after)
    assert_unchanged("worker scorer bindings", scorer_before, scorer_after)
    assert_unchanged("worker runtime bindings", runtime_before, runtime_after)
    assert_unchanged(
        "worker synaptic graph durable files",
        _durable_artifact_identity(graph_before),
        _durable_artifact_identity(graph_after),
    )
    assert_unchanged(
        "worker executed synaptic module paths",
        {
            name: str(
                synaptic_repo.resolve() / str(record["path"]).replace("/", os.sep)
            )
            for name, record in runtime_before["synaptic_modules"].items()
        },
        result.get("module_paths"),
    )
    return {
        "schema": WORKER_SCHEMA,
        "schema_version": WORKER_SCHEMA_VERSION,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "environment": _runtime_environment(),
        "inputs": {"before": inputs_before, "after": inputs_after},
        "graph": {"before": graph_before, "after": graph_after},
        "scorer": {"before": scorer_before, "after": scorer_after},
        "runtime_binding": {"before": runtime_before, "after": runtime_after},
        "integrity": {
            "fresh_process": True,
            "isolated_python_flags_verified": True,
            "environment_and_sys_path_verified": True,
            "exact_python_executable_bound": True,
            "runtime_distributions_bound_pre_and_post": True,
            "kiwi_active_pre_and_post": True,
            "scorer_exact_path_and_hash_bound_pre_and_post": True,
            "inputs_unchanged": True,
            "graph_durable_files_unchanged": True,
            "graph_shm_transient_only": True,
            "worker_output_write_once": True,
        },
        "result": result,
    }


def _run_worker_mode(args: argparse.Namespace) -> int:
    if args.worker_out is None:
        raise ProvenanceError("--worker-out is required in synaptic worker mode")
    if args.synaptic_python is None:
        raise ProvenanceError("--synaptic-python is required in synaptic worker mode")
    ensure_output_absent(args.worker_out)
    payload = _worker_payload(
        synaptic_repo=args.synaptic_repo.resolve(),
        graph_path=args.synaptic_graph.resolve(),
        expected_python=args.synaptic_python,
    )
    _atomic_write_json(args.worker_out, payload)
    return 0


def _run_synaptic_worker(
    *, synaptic_python: Path, synaptic_repo: Path, graph_path: Path
) -> dict[str, Any]:
    executable = synaptic_python.resolve()
    _python_executable_record(executable)
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="omnifuse-finreg-worker-") as directory:
        worker_output = Path(directory) / "result.json"
        command = [
            str(executable),
            "-I",
            "-X",
            "utf8",
            str(SCRIPT_PATH),
            "--_synaptic-worker",
            "--synaptic-repo",
            str(synaptic_repo.resolve()),
            "--synaptic-graph",
            str(graph_path.resolve()),
            "--synaptic-python",
            str(executable),
            "--worker-out",
            str(worker_output),
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            if len(detail) > 2000:
                detail = detail[-2000:]
            raise RuntimeError(
                f"fresh synaptic worker failed with exit code "
                f"{completed.returncode}: {detail}"
            )
        if not worker_output.is_file():
            raise ProvenanceError("fresh synaptic worker produced no result artifact")
        try:
            payload = json.loads(worker_output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProvenanceError(
                "fresh synaptic worker produced invalid JSON"
            ) from exc
    if not isinstance(payload, dict):
        raise ProvenanceError("fresh synaptic worker result must be a JSON object")
    return payload


def _validate_worker_payload(
    payload: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ProvenanceError("unexpected synaptic worker result schema")
    if payload.get("schema_version") != WORKER_SCHEMA_VERSION:
        raise ProvenanceError("unexpected synaptic worker schema version")
    if payload.get("status") != "ok":
        raise ProvenanceError("synaptic worker did not report ok status")
    process_id = payload.get("process_id")
    if not isinstance(process_id, int) or process_id == os.getpid():
        raise ProvenanceError("synaptic result was not produced by a fresh process")

    runtime = payload.get("runtime_binding")
    if not isinstance(runtime, dict):
        raise ProvenanceError("synaptic worker omitted runtime bindings")
    before = runtime.get("before")
    after = runtime.get("after")
    assert_unchanged("synaptic worker runtime bindings", before, after)
    if not isinstance(before, dict):
        raise ProvenanceError("synaptic worker runtime binding is invalid")
    kiwi = before.get("kiwi")
    if not isinstance(kiwi, dict) or kiwi.get("active") is not True:
        raise ProvenanceError("canonical synaptic worker did not keep Kiwi active")
    package_path = Path(str(state["synaptic_module_paths"]["package"])).resolve()
    _validate_worker_isolation_record(
        before.get("isolation"), expected_synaptic_source=package_path.parents[1]
    )

    assert_unchanged(
        "synaptic worker Python executable",
        state["synaptic_python"],
        before.get("python", {}).get("executable"),
    )
    assert_unchanged(
        "synaptic worker query inputs",
        {"before": state["inputs"], "after": state["inputs"]},
        payload.get("inputs"),
    )
    assert_unchanged(
        "synaptic worker scorer",
        {"before": state["sources"]["scorer"], "after": state["sources"]["scorer"]},
        payload.get("scorer"),
    )
    assert_unchanged(
        "synaptic worker imported modules",
        state["sources"]["imported_synaptic_memory"],
        before.get("synaptic_modules"),
    )
    worker_graph = payload.get("graph")
    if not isinstance(worker_graph, dict):
        raise ProvenanceError("synaptic worker omitted graph manifests")
    for phase in ("before", "after"):
        manifest = worker_graph.get(phase)
        if not isinstance(manifest, dict):
            raise ProvenanceError(f"synaptic worker graph {phase} manifest is invalid")
        assert_unchanged(
            f"synaptic worker graph {phase} durable files",
            _durable_artifact_identity(state["graph_watcher_open"]),
            _durable_artifact_identity(manifest),
        )
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("scores"), dict):
        raise ProvenanceError("synaptic worker omitted benchmark scores")
    enriched = dict(result)
    enriched["worker_provenance"] = {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "process_id": process_id,
        "fresh_subprocess": True,
        "environment": payload.get("environment"),
        "inputs": payload.get("inputs"),
        "graph": payload.get("graph"),
        "scorer": payload.get("scorer"),
        "runtime_binding": runtime,
        "integrity": payload.get("integrity"),
    }
    return enriched


def _comparison_preflight(
    *,
    output: Path,
    doctor_manifest: Path,
    synaptic_repo: Path,
    graph_path: Path,
    synaptic_python: Path,
) -> dict[str, Any]:
    ensure_output_absent(output)
    inputs = _input_records()
    repositories = {
        "omnifuse": repository_fingerprint(REPOSITORY_ROOT),
        "synaptic_memory": repository_fingerprint(synaptic_repo),
    }
    sources, module_paths = _comparison_sources(synaptic_repo)
    python_record = _python_executable_record(synaptic_python)
    doctor, raw_links = load_doctor_manifest(
        doctor_manifest, _doctor_input_specs(inputs)
    )
    verify_doctor_runtime(
        doctor,
        omnifuse_repository=repositories["omnifuse"],
        synaptic_repository=repositories["synaptic_memory"],
        omnifuse_scorer=sources["scorer"]["active"],
        synaptic_scorer=sources["scorer"]["synaptic_checkout_copy"],
    )
    assert_unchanged("finreg inputs during preflight", inputs, _input_records())
    graph_guard = _graph_guard_preflight(graph_path)
    try:
        return {
            "inputs": inputs,
            **graph_guard,
            "repositories": repositories,
            "sources": sources,
            "synaptic_module_paths": module_paths,
            "synaptic_python": python_record,
            "doctor_manifest": doctor,
            "doctor_links": _group_doctor_links(raw_links),
        }
    except Exception:
        _close_graph_watcher(graph_guard)
        raise


def _verify_comparison_postflight(
    state: dict[str, Any],
    *,
    synaptic_repo: Path,
    graph_path: Path,
    synaptic_result: dict[str, Any],
) -> dict[str, Any]:
    sources, module_paths = _comparison_sources(synaptic_repo)
    graph_after, data_version_after = _verify_graph_guard(state, graph_path)
    after = {
        "inputs": _input_records(),
        "graph": graph_after,
        "graph_data_version": data_version_after,
        "repositories": {
            "omnifuse": repository_fingerprint(REPOSITORY_ROOT),
            "synaptic_memory": repository_fingerprint(synaptic_repo),
        },
        "sources": sources,
        "synaptic_python": _python_executable_record(
            Path(str(state["synaptic_python"]["path"]))
        ),
    }
    assert_unchanged("finreg inputs", state["inputs"], after["inputs"])
    assert_unchanged(
        "repository fingerprints", state["repositories"], after["repositories"]
    )
    assert_unchanged("benchmark source fingerprints", state["sources"], sources)
    assert_unchanged(
        "synaptic Python executable", state["synaptic_python"], after["synaptic_python"]
    )
    assert_unchanged(
        "selected synaptic module paths",
        state["synaptic_module_paths"],
        module_paths,
    )
    assert_unchanged(
        "executed synaptic module paths",
        state["synaptic_module_paths"],
        synaptic_result.get("module_paths"),
    )
    verify_doctor_manifest(state["doctor_manifest"])
    scorer = sources["scorer"]
    verify_doctor_runtime(
        state["doctor_manifest"],
        omnifuse_repository=after["repositories"]["omnifuse"],
        synaptic_repository=after["repositories"]["synaptic_memory"],
        omnifuse_scorer=scorer["active"],
        synaptic_scorer=scorer["synaptic_checkout_copy"],
    )
    return {
        **after,
        "integrity": {
            "preflight_completed_before_benchmarks": True,
            "inputs_unchanged": True,
            "graph_durable_files_unchanged": True,
            "graph_shm_recorded_as_transient": True,
            "graph_data_version_unchanged": True,
            "repository_states_unchanged": True,
            "benchmark_sources_unchanged": True,
            "executed_modules_bound_to_selected_checkouts": True,
            "synaptic_python_executable_unchanged": True,
            "fresh_synaptic_worker_verified": True,
            "isolated_worker_environment_verified": True,
            "kiwi_active_pre_and_post": True,
            "doctor_manifest_unchanged": True,
            "doctor_runtime_binding_reverified": True,
            "postflight_verified_before_publish": True,
        },
    }


def _interactive_preflight(
    *, synaptic_repo: Path, graph_path: Path, synaptic_python: Path
) -> dict[str, Any]:
    inputs = _input_records()
    sources, module_paths = _comparison_sources(synaptic_repo)
    graph_guard = _graph_guard_preflight(graph_path)
    return {
        "inputs": inputs,
        **graph_guard,
        "sources": sources,
        "synaptic_module_paths": module_paths,
        "synaptic_python": _python_executable_record(synaptic_python),
    }


def _verify_interactive_postflight(
    state: dict[str, Any],
    *,
    synaptic_repo: Path,
    graph_path: Path,
    synaptic_result: dict[str, Any],
) -> None:
    graph_after, _data_version_after = _verify_graph_guard(state, graph_path)
    sources, module_paths = _comparison_sources(synaptic_repo)
    assert_unchanged("finreg inputs", state["inputs"], _input_records())
    assert_unchanged(
        "synaptic graph durable files",
        _durable_artifact_identity(state["graph_watcher_open"]),
        _durable_artifact_identity(graph_after),
    )
    assert_unchanged("benchmark source fingerprints", state["sources"], sources)
    assert_unchanged(
        "selected synaptic module paths", state["synaptic_module_paths"], module_paths
    )
    assert_unchanged(
        "executed synaptic module paths",
        state["synaptic_module_paths"],
        synaptic_result.get("module_paths"),
    )
    assert_unchanged(
        "synaptic Python executable",
        state["synaptic_python"],
        _python_executable_record(Path(str(state["synaptic_python"]["path"]))),
    )


def _build_report(
    *,
    omnifuse_result: dict[str, Any],
    synaptic_result: dict[str, Any],
    state: dict[str, Any],
    postflight: dict[str, Any],
) -> dict[str, object]:
    return {
        "schema": "omnifuse.eval.finreg_comparison",
        "schema_version": 2,
        "provenance_level": PROVENANCE_LEVEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "environment": _runtime_environment(),
        "synaptic_worker_python": {
            "before": state["synaptic_python"],
            "after": postflight["synaptic_python"],
        },
        "repositories": {
            "before": state["repositories"],
            "after": postflight["repositories"],
        },
        "inputs": {
            "finreg": {
                "before": state["inputs"],
                "after": postflight["inputs"],
            },
            "synaptic_prebuilt_sqlite": {
                "before": state["graph"],
                "after": postflight["graph"],
                "watcher_open": state["graph_watcher_open"],
                "data_version": {
                    "before": state["graph_data_version_before"],
                    "after": postflight["graph_data_version"],
                },
            },
        },
        "provenance": {
            "level": PROVENANCE_LEVEL,
            "benchmark_sources": {
                "before": state["sources"],
                "after": postflight["sources"],
            },
            "doctor_manifest": state["doctor_manifest"],
            "doctor_targets": state["doctor_links"],
        },
        "integrity": postflight["integrity"],
        "evaluation_contract": {
            "k": K,
            "candidate_limit": CANDIDATE_LIMIT,
            "same_k_for_both_systems": True,
            "same_candidate_limit_for_both_systems": True,
            "dedupe": DEDUPE_POLICY,
            "score_after_dedupe": True,
            "retrieval": "single-shot; no LLM, embedder, or reranker",
            "synaptic_fts_seed_limit": 30,
        },
        "index_conditions": {
            "omnifuse": {
                "state": "rebuilt_in_process_from_tracked_corpus",
                "rebuild_included_in_total_timing": True,
                "backend": "zero-infrastructure in-memory",
            },
            "synaptic_memory": {
                "state": "caller_supplied_prebuilt_sqlite",
                "rebuild_performed_by_harness": False,
                "ingestion_provenance_verified_against_corpus": False,
                "executed_in_fresh_bound_python_subprocess": True,
                "backend_closed_in_finally": True,
            },
            "caveat": INDEX_CONDITION_CAVEAT,
            "timings_directly_comparable": False,
        },
        "protocol_scope": {
            "same_invocation": True,
            "same_query_files": True,
            "same_metric_implementation": True,
            "synaptic_graph_matches_finreg_corpus_verified": False,
            "same_graph_build_recipe_verified": False,
            "standalone_finreg_artifacts_combined": False,
            "boundary": PROTOCOL_BOUNDARY,
        },
        "systems": {
            "omnifuse": omnifuse_result,
            "synaptic_memory": synaptic_result,
        },
    }


def _print_report(report: dict[str, object]) -> None:
    systems: dict[str, Any] = report["systems"]  # type: ignore[assignment]
    omnifuse = systems["omnifuse"]["scores"]
    synaptic_system = systems["synaptic_memory"]
    synaptic = synaptic_system["scores"] if synaptic_system is not None else None

    print("\n" + "=" * 74)
    print(f"{'FINREG single-shot retrieval, no LLM, shared scorer, k=10':^74}")
    print("=" * 74)
    print(f"contract: k={K}, candidate_limit={CANDIDATE_LIMIT}, dedupe={DEDUPE_POLICY}")
    print(f"{'metric':26}{'synaptic-memory':>22}{'OmniFuse':>22}")
    print("-" * 74)

    def row(name: str, synaptic_value: object, omnifuse_value: object) -> None:
        if synaptic is None:
            synaptic_text = "not run"
        elif isinstance(synaptic_value, float):
            synaptic_text = f"{synaptic_value:.4f}"
        else:
            synaptic_text = str(synaptic_value)
        omnifuse_text = (
            f"{omnifuse_value:.4f}"
            if isinstance(omnifuse_value, float)
            else str(omnifuse_value)
        )
        print(f"{name:26}{synaptic_text:>22}{omnifuse_text:>22}")

    row(
        "single-hop MRR@10",
        synaptic["single_hop"]["mrr"] if synaptic else None,
        omnifuse["single_hop"]["mrr"],
    )
    row(
        "single-hop nDCG@10",
        synaptic["single_hop"]["mean_ndcg@k"] if synaptic else None,
        omnifuse["single_hop"]["mean_ndcg@k"],
    )
    row(
        "single-hop hit@10",
        (
            f"{synaptic['single_hop']['hits']}/{synaptic['single_hop']['n']}"
            if synaptic
            else None
        ),
        f"{omnifuse['single_hop']['hits']}/{omnifuse['single_hop']['n']}",
    )
    row(
        "multi-hop strict-solved",
        (
            f"{synaptic['multi_hop']['strict']}/{synaptic['multi_hop']['n']}"
            if synaptic
            else None
        ),
        f"{omnifuse['multi_hop']['strict']}/{omnifuse['multi_hop']['n']}",
    )
    row(
        "multi-hop R@10",
        synaptic["multi_hop"]["mean_recall@k"] if synaptic else None,
        omnifuse["multi_hop"]["mean_recall@k"],
    )
    print("=" * 74)
    print(INDEX_CONDITION_CAVEAT)
    print(PROTOCOL_BOUNDARY)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.out is not None and args.doctor_manifest is None:
        parser.error("--doctor-manifest is required when --out is used")
    if args.out is not None and args.synaptic_python is None:
        parser.error("--synaptic-python is required when --out is used")

    synaptic_repo = args.synaptic_repo.resolve()
    graph_path = args.synaptic_graph.resolve()
    if not synaptic_repo.is_dir():
        parser.error(f"synaptic checkout not found: {synaptic_repo}")
    if not graph_path.is_file():
        parser.error(f"synaptic SQLite graph not found: {graph_path}")
    if args._synaptic_worker:
        try:
            return _run_worker_mode(args)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    synaptic_python = (
        args.synaptic_python.resolve()
        if args.synaptic_python is not None
        else Path(sys.executable).resolve()
    )

    state: dict[str, Any] | None = None
    postflight: dict[str, Any] | None = None
    try:
        if args.out is not None:
            state = _comparison_preflight(
                output=args.out,
                doctor_manifest=args.doctor_manifest,
                synaptic_repo=synaptic_repo,
                graph_path=graph_path,
                synaptic_python=synaptic_python,
            )
        else:
            state = _interactive_preflight(
                synaptic_repo=synaptic_repo,
                graph_path=graph_path,
                synaptic_python=synaptic_python,
            )

        single_hop = load_queries(SINGLE_QUERY_PATH.name)
        multi_hop = load_queries(MULTIHOP_QUERY_PATH.name)
        omnifuse_result = run_omnifuse(single_hop, multi_hop)
        worker_payload = _run_synaptic_worker(
            synaptic_python=synaptic_python,
            synaptic_repo=synaptic_repo,
            graph_path=graph_path,
        )
        synaptic_result = _validate_worker_payload(worker_payload, state)
        if args.out is not None:
            postflight = _verify_comparison_postflight(
                state,
                synaptic_repo=synaptic_repo,
                graph_path=graph_path,
                synaptic_result=synaptic_result,
            )
        else:
            _verify_interactive_postflight(
                state,
                synaptic_repo=synaptic_repo,
                graph_path=graph_path,
                synaptic_result=synaptic_result,
            )

        display_report = {
            "systems": {
                "omnifuse": omnifuse_result,
                "synaptic_memory": synaptic_result,
            }
        }
        _print_report(display_report)
        if args.out is not None:
            assert postflight is not None
            report = _build_report(
                omnifuse_result=omnifuse_result,
                synaptic_result=synaptic_result,
                state=state,
                postflight=postflight,
            )
            _atomic_write_json(args.out, report)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        _close_graph_watcher(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
