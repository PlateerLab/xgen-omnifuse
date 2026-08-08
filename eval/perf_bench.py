"""Fair efficiency head-to-head with shared ranking and accuracy rules.

Accuracy is scored with the repository's copy of synaptic's ``BenchmarkResult``.
Latency is a separate measurement: both search calls are timed directly with
``time.perf_counter`` after identical warm-up rounds and over identical repeated,
deterministically ordered queries.

    python eval/perf_bench.py --data-dir <synaptic tests/benchmark/data> \
        --dataset nfcorpus.json --synaptic-repo <path>

By default each system runs in two fresh subprocesses with AB/BA order. The
controller writes one canonical, write-once input payload that every worker must
fingerprint exactly. Reported per system:

``ingest_s``
    Wall time from raw corpus to a queryable index.
``query_latency_{p50,p95,mean}_ms``
    Distribution over all measured query calls (warm-ups excluded).
``process_memory``
    Current and peak resident memory for the entire fresh worker process, including
    imports, index build, warm-up, and measured queries.
``mrr@K``
    Accuracy from the first measured pass after the same candidate limit and
    top-K truncation. The official protocol preserves the raw upstream ranking.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import re
import site
import sqlite3
import statistics
import subprocess
import sys
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
sys.path.insert(0, str(EVAL_DIR))

from metrics import BenchmarkResult  # noqa: E402 - shared accuracy scorer
from provenance import (  # noqa: E402 - sibling benchmark helper
    ProvenanceError,
    assert_artifact_unchanged,
    assert_unchanged,
    canonical_json_sha256,
    capture_worker_identity,
    default_worker_directory,
    ensure_output_absent,
    file_fingerprint,
    load_doctor_manifest,
    new_worker_run_id,
    read_bytes_artifact,
    read_json_artifact,
    repository_fingerprint,
    run_with_launcher_pid,
    sha256_file,
    validate_worker_identity,
    verify_doctor_manifest,
    verify_doctor_runtime,
    worker_process_summary,
    write_json_once,
)

DEFAULT_K = 10
DEFAULT_TRIALS = 2
PROTOCOL_SQLITE_NATIVE = "sqlite-native"
PROTOCOL_OFFICIAL_EXTERNAL_MEMORY = "official-external-memory"
PROTOCOLS = (PROTOCOL_SQLITE_NATIVE, PROTOCOL_OFFICIAL_EXTERNAL_MEMORY)
Query = tuple[str, str, set[str]]
CorpusRow = tuple[str, str, str]
SYNAPTIC_DRIVER_RELATIVE = Path("eval/run_all.py")
SYNAPTIC_EXTERNAL_DRIVER_RELATIVE = Path("tests/benchmark/test_external_datasets.py")
SYNAPTIC_SCORER_RELATIVE = Path("tests/benchmark/metrics.py")
FROZEN_INPUT_SCHEMA = "omnifuse.eval.performance.input"
FROZEN_INPUT_SCHEMA_VERSION = 1
WORKER_RESULT_SCHEMA = "omnifuse.eval.performance.worker"
WORKER_RESULT_SCHEMA_VERSION = 4
REPORT_SCHEMA_VERSION = 5
PROVENANCE_LEVEL = "strict-preflight-postflight-isolated-write-once-v4"
WORKER_INPUT_DISPLAY_PATH = "worker-input/performance.json"


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]
PROCESS_MEMORY_SCOPE = (
    "entire fresh worker process (imports, build, warmup, measured queries)"
)
SYSTEM_RESULT_NAMES = {"omnifuse": "OmniFuse", "synaptic": "synaptic"}
REQUIRED_SOURCE_BINDINGS = {
    "omnifuse": frozenset({"package", "build_inmemory", "retrieve"}),
    ("synaptic", PROTOCOL_SQLITE_NATIVE): frozenset(
        {"package", "sqlite_backend", "sqlite_backend_base", "graph"}
    ),
    ("synaptic", PROTOCOL_OFFICIAL_EXTERNAL_MEMORY): frozenset(
        {
            "package",
            "memory_backend",
            "sqlite_normalizer",
            "graph",
            "official_external_driver",
        }
    ),
}
TOKENIZER_MODULE_NAMES = ("kiwipiepy", "_kiwipiepy", "kiwipiepy_model")
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
DATASET_TARGET_IDS = {
    "hotpotqa_24.json": "hotpotqa_24",
    "hotpotqa.json": "hotpotqa_200",
    "allganize_rag_ko.json": "allganize_rag_ko",
    "allganize_rag_eval.json": "allganize_rag_eval",
    "publichealthqa_ko.json": "publichealthqa_ko",
    "autorag_retrieval.json": "autorag_retrieval",
    "klue_mrc.json": "klue_mrc",
    "ko_strategyqa.json": "ko_strategyqa",
    "2wiki_dev.json": "2wiki_dev",
    "musique_dev.json": "musique_dev",
    "trec_covid.json": "trec_covid",
    "scifact.json": "scifact",
    "xpqa_ko.json": "xpqa_ko",
    "nfcorpus.json": "nfcorpus",
    "miracl_retrieval_ko.json": "miracl_retrieval_ko",
    "fiqa.json": "fiqa",
    "multilongdoc_ko.json": "multilongdoc_ko",
}


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _bytes_fingerprint(payload: bytes, *, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _is_below(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _worker_directory(output: Path, configured: Path | None) -> Path:
    if configured is not None:
        return configured.resolve()
    return default_worker_directory(ROOT, output, kind="perf")


def _validate_worker_directory(
    worker_root: Path, *, output: Path, synaptic_repo: Path
) -> None:
    worker_root = worker_root.resolve()
    output = output.resolve()
    synaptic_repo = synaptic_repo.resolve()
    if worker_root.exists():
        raise ProvenanceError(
            f"refusing to reuse worker-artifact directory: {worker_root}"
        )
    if _is_below(output, worker_root):
        raise ProvenanceError(
            f"benchmark output must be outside the worker-artifact directory: {output}"
        )
    if _is_below(worker_root, synaptic_repo):
        raise ProvenanceError(
            "worker artifacts must not be written inside the immutable "
            f"synaptic-memory checkout: {worker_root}"
        )
    if _is_below(worker_root, ROOT):
        relative = worker_root.relative_to(ROOT).as_posix()
        ignored = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", "--", relative],
            check=False,
            capture_output=True,
        )
        if ignored.returncode != 0:
            raise ProvenanceError(
                "worker-artifact directory inside OmniFuse must be Git-ignored: "
                f"{worker_root}"
            )


def _git_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return repository_fingerprint(path)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_module_under(
    module_file: str | None, source_root: Path, package: str
) -> Path:
    if module_file is None:
        raise RuntimeError(f"{package} has no __file__; cannot verify benchmark source")
    package_path = Path(module_file).resolve()
    if source_root.resolve() not in package_path.parents:
        raise RuntimeError(
            f"loaded {package} from {package_path}, expected source below {source_root.resolve()}"
        )
    return package_path


def _module_binding(
    value: Any, *, source_root: Path, repository_root: Path, name: str
) -> dict[str, Any]:
    module = value if inspect.ismodule(value) else inspect.getmodule(value)
    if module is None:
        raise RuntimeError(f"cannot resolve imported module for {name}")
    path = _assert_module_under(getattr(module, "__file__", None), source_root, name)
    return {
        **file_fingerprint(
            path,
            display_path=path.relative_to(repository_root.resolve()).as_posix(),
        ),
        "resolved_path": str(path),
    }


def _external_module_binding(module_name: str) -> dict[str, Any] | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(
            f"loaded tokenizer dependency {module_name!r} has no file path"
        )
    path = Path(raw_path).resolve()
    return {
        **file_fingerprint(path, display_path=str(path)),
        "resolved_path": str(path),
    }


def _runtime_environment_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "utf8_mode": bool(sys.flags.utf8_mode),
        "sqlite": sqlite3.sqlite_version,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "packages": {
            "omnifuse": _package_version("omnifuse"),
            "synaptic_memory": _package_version("synaptic-memory"),
            "kiwipiepy": _package_version("kiwipiepy"),
            "kiwipiepy_model": _package_version("kiwipiepy-model"),
        },
    }


def _controller_utf8_mode_enabled() -> bool:
    return sys.flags.utf8_mode == 1


def _require_protocol_utf8_mode(
    protocol: str,
    *,
    context: str,
    environment: Mapping[str, Any] | None = None,
) -> None:
    if protocol != PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        return
    enabled = (
        _controller_utf8_mode_enabled()
        if environment is None
        else environment.get("utf8_mode") is True
    )
    if not enabled:
        raise ProvenanceError(
            "official external MemoryBackend protocol requires Python UTF-8 mode "
            f"for {context}; relaunch with python -X utf8"
        )


def _official_environment_probe(repo: Path) -> dict[str, Any]:
    import direct_external_bench as direct_external

    return direct_external._probe_worker_environment(
        Path(sys.executable).resolve(), repo.resolve()
    )


def _official_environment_lock_evidence(repo: Path) -> dict[str, Any]:
    import direct_external_bench as direct_external

    return direct_external._environment_lock_evidence(repo.resolve())


def _validate_official_environment_lock(record: Mapping[str, Any], repo: Path) -> None:
    import direct_external_bench as direct_external

    direct_external._validate_environment_lock_record(record, repo.resolve())


def _official_environment_contract(probe: Mapping[str, Any]) -> dict[str, Any]:
    lock = probe.get("environment_lock")
    if not isinstance(lock, dict):
        raise ProvenanceError("official environment probe has no lock evidence")
    lockfile = lock.get("lockfile")
    uv_sync = lock.get("uv_sync_check")
    if not isinstance(lockfile, dict) or not isinstance(uv_sync, dict):
        raise ProvenanceError("official environment lock evidence is incomplete")
    return {
        "lockfile_sha256": lockfile.get("sha256"),
        "installed_manifest_sha256": lock.get("installed_manifest_sha256"),
        "uv_sync_check": {
            key: uv_sync.get(key)
            for key in (
                "arguments",
                "selected_extras",
                "checked_package_count",
                "reported_no_changes",
                "virtual_environment",
            )
        },
    }


def _worker_environment_snapshot() -> dict[str, Any]:
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


def _isolated_worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _doctor_environment_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = record.get("snapshot")
    environment = snapshot.get("environment") if isinstance(snapshot, dict) else None
    required = {
        "python",
        "python_implementation",
        "python_executable",
        "platform",
    }
    if not isinstance(environment, dict) or set(environment) != required:
        raise ProvenanceError("doctor runtime environment fingerprint is invalid")
    if not all(
        isinstance(environment[key], str) and environment[key] for key in required
    ):
        raise ProvenanceError("doctor runtime environment fields are invalid")
    return dict(environment)


def _verify_doctor_environment(
    doctor_environment: Mapping[str, Any], runtime_environment: Mapping[str, Any]
) -> None:
    assert_unchanged(
        "runtime environment since doctor preflight",
        dict(doctor_environment),
        {key: runtime_environment[key] for key in doctor_environment},
    )


def _synaptic_tokenizer_evidence(sqlite_module: Any) -> dict[str, Any]:
    available = getattr(sqlite_module, "_kiwi_available", None)
    if available is not None and not isinstance(available, bool):
        raise RuntimeError("synaptic Kiwi availability state is invalid")
    if available is True and getattr(sqlite_module, "_kiwi_instance", None) is None:
        raise RuntimeError(
            "synaptic reported Kiwi available without an active instance"
        )
    mode = "unused" if available is None else "kiwi" if available else "regex_fallback"
    return {
        "mode": mode,
        "korean_normalization_used": available is not None,
        "kiwi_available": available,
        "kiwi_version": _package_version("kiwipiepy"),
        "kiwi_model_version": _package_version("kiwipiepy-model"),
        "modules": {
            name: _external_module_binding(name) for name in TOKENIZER_MODULE_NAMES
        },
    }


def _require_unique_nonempty_ids(
    rows: Sequence[tuple[Any, ...]], *, label: str, error_type: type[Exception]
) -> None:
    identifiers = [row[0] for row in rows]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise error_type(f"{label} IDs must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise error_type(f"{label} IDs must be unique")


def _frozen_input_payload(
    corpus: Sequence[CorpusRow], queries: Sequence[Query]
) -> bytes:
    _require_unique_nonempty_ids(corpus, label="corpus", error_type=ValueError)
    _require_unique_nonempty_ids(queries, label="query", error_type=ValueError)
    payload = {
        "schema": FROZEN_INPUT_SCHEMA,
        "schema_version": FROZEN_INPUT_SCHEMA_VERSION,
        "corpus": [
            {"id": doc_id, "title": title, "text": text}
            for doc_id, title, text in corpus
        ],
        "queries": [
            {
                "id": query_id,
                "text": text,
                "relevant": sorted(relevant),
            }
            for query_id, text, relevant in queries
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parse_frozen_input_value(raw: Any) -> tuple[list[CorpusRow], list[Query]]:
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != FROZEN_INPUT_SCHEMA
        or raw.get("schema_version") != FROZEN_INPUT_SCHEMA_VERSION
        or not isinstance(raw.get("corpus"), list)
        or not isinstance(raw.get("queries"), list)
    ):
        raise ProvenanceError("frozen worker input contract is invalid")

    corpus: list[CorpusRow] = []
    for row in raw["corpus"]:
        if not isinstance(row, dict) or set(row) != {"id", "title", "text"}:
            raise ProvenanceError("frozen worker corpus row is invalid")
        values = (row["id"], row["title"], row["text"])
        if not all(isinstance(value, str) for value in values):
            raise ProvenanceError("frozen worker corpus fields must be strings")
        corpus.append(values)

    queries: list[Query] = []
    for row in raw["queries"]:
        if not isinstance(row, dict) or set(row) != {"id", "text", "relevant"}:
            raise ProvenanceError("frozen worker query row is invalid")
        query_id = row["id"]
        text = row["text"]
        relevant = row["relevant"]
        if (
            not isinstance(query_id, str)
            or not isinstance(text, str)
            or not isinstance(relevant, list)
            or not relevant
            or not all(isinstance(value, str) and value for value in relevant)
            or relevant != sorted(set(relevant))
        ):
            raise ProvenanceError("frozen worker query fields are invalid")
        queries.append((query_id, text, set(relevant)))

    if not corpus or not queries:
        raise ProvenanceError(
            "frozen worker input must contain corpus and scored queries"
        )
    _require_unique_nonempty_ids(
        corpus, label="frozen worker corpus", error_type=ProvenanceError
    )
    _require_unique_nonempty_ids(
        queries, label="frozen worker query", error_type=ProvenanceError
    )
    return corpus, queries


def _parse_frozen_input(payload: bytes) -> tuple[list[CorpusRow], list[Query]]:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"invalid frozen worker input: {exc}") from exc
    corpus, queries = _parse_frozen_input_value(raw)
    if _frozen_input_payload(corpus, queries) != payload:
        raise ProvenanceError("frozen worker input bytes are not canonical")
    return corpus, queries


def _load_frozen_input_file(
    path: Path, *, display_path: str
) -> tuple[dict[str, Any], list[CorpusRow], list[Query]]:
    fingerprint = file_fingerprint(path, display_path=display_path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"invalid frozen worker input: {exc}") from exc
    corpus, queries = _parse_frozen_input_value(raw)

    digest = hashlib.sha256()
    byte_count = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for piece in encoder.iterencode(raw):
        encoded = piece.encode("utf-8")
        digest.update(encoded)
        byte_count += len(encoded)
    if (digest.hexdigest(), byte_count) != (
        fingerprint["sha256"],
        fingerprint["bytes"],
    ):
        raise ProvenanceError("frozen worker input bytes are not canonical")
    assert_unchanged(
        "worker input",
        fingerprint,
        file_fingerprint(path, display_path=display_path),
    )
    return fingerprint, corpus, queries


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ProvenanceError(
            f"refusing to overwrite write-once worker input: {path.resolve()}"
        ) from exc


def _atomic_write_json(path: Path, value: object) -> None:
    write_json_once(path, value)


def _process_memory_bytes() -> tuple[int | None, int | None, str | None]:
    """Current/peak resident bytes for the entire fresh worker, when available."""
    if sys.platform == "win32":
        counters = _WindowsProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if ok:
            return (
                int(counters.WorkingSetSize),
                int(counters.PeakWorkingSetSize),
                "Windows process working set",
            )
        return None, None, None

    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        current_bytes = None
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    current_bytes = int(line.split()[1]) * 1024
                    break
        return current_bytes, peak_bytes, "process RSS"
    except (ImportError, OSError, ValueError):
        return None, None, None


def _parse(data: dict[str, Any]) -> tuple[list[CorpusRow], list[Query]]:
    raw = data.get("corpus", data.get("documents", []))
    corpus: list[CorpusRow] = []
    if isinstance(raw, dict):
        for doc_id, doc in raw.items():
            corpus.append(
                (str(doc_id), str(doc.get("title", "")), str(doc.get("text", "")))
            )
    else:
        for doc in raw:
            doc_id = str(doc.get("doc_id", doc.get("_id", doc.get("id", ""))))
            corpus.append(
                (
                    doc_id,
                    str(doc.get("title", "")),
                    str(doc.get("text", doc.get("content", ""))),
                )
            )

    qrels = data.get("relevant_docs", data.get("qrels", {}))
    queries = data.get("queries", {})
    parsed_queries: list[Query] = []
    for query_id, text in queries.items() if isinstance(queries, dict) else []:
        relevant = qrels.get(query_id, [])
        relevant = set(
            map(str, relevant.keys() if isinstance(relevant, dict) else relevant)
        )
        if relevant and text:
            parsed_queries.append((str(query_id), str(text), relevant))

    corpus = sorted(corpus, key=lambda row: row[0])
    parsed_queries = sorted(parsed_queries, key=lambda row: row[0])
    _require_unique_nonempty_ids(corpus, label="corpus", error_type=ValueError)
    _require_unique_nonempty_ids(parsed_queries, label="query", error_type=ValueError)
    return corpus, parsed_queries


def _load_dataset(path: Path) -> tuple[list[CorpusRow], list[Query]]:
    try:
        return _parse(json.loads(path.read_bytes().decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset JSON {path}: {exc}") from exc


def _official_external_case(filename: str) -> Any:
    import direct_external_bench as direct_external

    matches = [case for case in direct_external.CASES if case.filename == filename]
    if len(matches) != 1:
        raise ProvenanceError(
            f"dataset {filename!r} is not one official external benchmark case"
        )
    return matches[0]


def _load_official_external_input(
    synaptic_repo: Path, dataset_path: Path
) -> tuple[list[CorpusRow], list[Query], dict[str, Any]]:
    """Reproduce the selected official external case before freezing worker input."""
    import direct_external_bench as direct_external

    case = _official_external_case(dataset_path.name)
    expected_dataset = (
        synaptic_repo / "tests" / "benchmark" / "data" / case.filename
    ).resolve()
    if dataset_path.resolve() != expected_dataset:
        raise ProvenanceError(
            f"official external protocol requires checkout dataset {expected_dataset}"
        )
    direct_external._validate_tag_checkout(synaptic_repo)
    driver, _scorer, runtime = direct_external._load_upstream_driver(synaptic_repo)
    prepared, selection = direct_external._prepare_case_data(driver, case)
    documents, truncation = direct_external._omnifuse_documents(prepared["corpus"])
    indexed_ids = {str(document["id"]) for document in documents}

    query_items = [
        (str(query_id), str(query_text))
        for query_id, query_text in prepared["queries"].items()
        if query_id in prepared["qrels"]
    ]
    if case.max_queries > 0 and len(query_items) > case.max_queries:
        query_items = random.Random(direct_external.SAMPLE_SEED).sample(
            query_items, case.max_queries
        )

    queries: list[Query] = []
    for query_id, query_text in query_items:
        raw_relevant = prepared["qrels"].get(query_id, {})
        if not isinstance(raw_relevant, dict):
            raise ProvenanceError(f"qrels[{query_id!r}] must be an object")
        relevant = {str(value) for value in raw_relevant if str(value) in indexed_ids}
        if relevant:
            queries.append((query_id, query_text, relevant))
    if not queries:
        raise ProvenanceError("official external case selected no scored queries")

    corpus = [
        (str(document["id"]), str(document["title"]), str(document["text"]))
        for document in documents
    ]
    selection = {
        **selection,
        "case_id": case.id,
        "case_name": case.name,
        "driver": runtime["upstream_driver"],
        "scorer": runtime["upstream_scorer"],
        "indexed_corpus": truncation,
        "scored_query_count": len(queries),
        "scored_query_ids_ordered_sha256": canonical_json_sha256(
            [query_id for query_id, _text, _relevant in queries]
        ),
    }
    return corpus, queries, selection


def _load_dataset_snapshot(
    path: Path, *, display_path: str
) -> tuple[dict[str, Any], list[CorpusRow], list[Query]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProvenanceError(f"could not read dataset {path}: {exc}") from exc
    fingerprint = _bytes_fingerprint(payload, path=display_path)
    try:
        corpus, queries = _parse(json.loads(payload.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"invalid dataset JSON {path}: {exc}") from exc
    assert_unchanged(
        "dataset input during preflight",
        fingerprint,
        file_fingerprint(path, display_path=display_path),
    )
    return fingerprint, corpus, queries


def _scorer_provenance(synaptic_repo: Path) -> dict[str, Any]:
    local = file_fingerprint(EVAL_DIR / "metrics.py", display_path="eval/metrics.py")
    upstream = file_fingerprint(
        synaptic_repo / SYNAPTIC_SCORER_RELATIVE,
        display_path=SYNAPTIC_SCORER_RELATIVE.as_posix(),
    )
    if (local["sha256"], local["bytes"]) != (
        upstream["sha256"],
        upstream["bytes"],
    ):
        raise ProvenanceError(
            "eval/metrics.py is not byte-identical to the selected synaptic-memory "
            "tests/benchmark/metrics.py"
        )
    return {
        "active": local,
        "synaptic_checkout_copy": upstream,
        "byte_identical": True,
    }


def _benchmark_sources(
    synaptic_repo: Path, protocol: str = PROTOCOL_SQLITE_NATIVE
) -> dict[str, Any]:
    driver_relative = (
        SYNAPTIC_EXTERNAL_DRIVER_RELATIVE
        if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
        else SYNAPTIC_DRIVER_RELATIVE
    )
    sources = {
        "harness": file_fingerprint(SCRIPT_PATH, display_path="eval/perf_bench.py"),
        "provenance_helper": file_fingerprint(
            EVAL_DIR / "provenance.py", display_path="eval/provenance.py"
        ),
        "scorer": _scorer_provenance(synaptic_repo),
        "synaptic_native_driver": file_fingerprint(
            synaptic_repo / driver_relative,
            display_path=driver_relative.as_posix(),
        ),
    }
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        sources["official_external_adapter"] = file_fingerprint(
            EVAL_DIR / "direct_external_bench.py",
            display_path="eval/direct_external_bench.py",
        )
    return sources


def _machine_dataset_identity(
    synaptic_repo: Path, dataset_path: Path
) -> tuple[str, str]:
    filename = dataset_path.name
    target_id = DATASET_TARGET_IDS.get(filename)
    if target_id is None:
        raise ProvenanceError(
            f"dataset {filename!r} has no strict doctor target binding"
        )
    relative_path = (Path("tests") / "benchmark" / "data" / filename).as_posix()
    expected = (synaptic_repo / relative_path).resolve()
    if dataset_path.resolve() != expected:
        raise ProvenanceError(
            "machine-readable performance evidence must use the selected "
            f"synaptic checkout dataset {expected}"
        )
    return target_id, relative_path


def _validate_official_artifact_paths(
    synaptic_repo: Path, *, output: Path, doctor_manifest: Path
) -> None:
    repo = synaptic_repo.resolve()
    for label, path in (("output", output), ("doctor manifest", doctor_manifest)):
        if _is_below(path.resolve(), repo):
            raise ProvenanceError(
                f"official performance {label} must be outside the immutable "
                f"synaptic-memory checkout: {repo}"
            )


def _machine_preflight(
    *,
    output: Path,
    doctor_manifest: Path,
    synaptic_repo: Path,
    dataset_path: Path,
    protocol: str = PROTOCOL_SQLITE_NATIVE,
) -> tuple[dict[str, Any], list[CorpusRow], list[Query]]:
    """Bind every machine-report input before either benchmark worker starts."""
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        _validate_official_artifact_paths(
            synaptic_repo,
            output=output,
            doctor_manifest=doctor_manifest,
        )
    ensure_output_absent(output)
    runtime_environment = _runtime_environment_snapshot()
    _require_protocol_utf8_mode(
        protocol,
        context="controller preflight",
        environment=runtime_environment,
    )
    target_id, relative_path = _machine_dataset_identity(synaptic_repo, dataset_path)
    repositories = {
        "omnifuse": repository_fingerprint(ROOT),
        "synaptic_memory": repository_fingerprint(synaptic_repo),
    }
    sources = (
        _benchmark_sources(synaptic_repo)
        if protocol == PROTOCOL_SQLITE_NATIVE
        else _benchmark_sources(synaptic_repo, protocol)
    )
    official_environment_probe = (
        _official_environment_probe(synaptic_repo)
        if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
        else None
    )
    dataset_fingerprint = file_fingerprint(dataset_path, display_path=relative_path)
    selection = None
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        corpus, queries, selection = _load_official_external_input(
            synaptic_repo, dataset_path
        )
        assert_unchanged(
            "official external dataset input during preflight",
            dataset_fingerprint,
            file_fingerprint(dataset_path, display_path=relative_path),
        )
    else:
        dataset_fingerprint, corpus, queries = _load_dataset_snapshot(
            dataset_path,
            display_path=relative_path,
        )
    if not queries:
        raise ValueError("dataset has no scored queries")

    dataset = {
        **dataset_fingerprint,
        "doctor_target_id": target_id,
        "documents": len(corpus),
        "scored_queries": len(queries),
        "relevance_judgments": sum(len(relevant) for _, _, relevant in queries),
    }
    if selection is not None:
        dataset["official_external_selection"] = selection
    doctor, links = load_doctor_manifest(
        doctor_manifest,
        [
            {
                "name": dataset_path.name,
                "target_id": target_id,
                "path": relative_path,
                "sha256": dataset_fingerprint["sha256"],
                "bytes": dataset_fingerprint["bytes"],
            }
        ],
    )
    doctor_environment = _doctor_environment_snapshot(doctor)
    _verify_doctor_environment(doctor_environment, runtime_environment)
    scorer = sources["scorer"]
    verify_doctor_runtime(
        doctor,
        omnifuse_repository=repositories["omnifuse"],
        synaptic_repository=repositories["synaptic_memory"],
        omnifuse_scorer=scorer["active"],
        synaptic_scorer=scorer["synaptic_checkout_copy"],
    )
    state = {
        "dataset": dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "repositories": repositories,
        "sources": sources,
        "runtime_environment": runtime_environment,
        "doctor_environment": doctor_environment,
        "doctor_manifest": doctor,
        "doctor_link": links[dataset_path.name],
    }
    if official_environment_probe is not None:
        state["official_environment_probe"] = official_environment_probe
    return state, corpus, queries


def _verify_machine_postflight(
    state: dict[str, Any],
    *,
    synaptic_repo: Path,
    dataset_path: Path,
    protocol: str = PROTOCOL_SQLITE_NATIVE,
) -> dict[str, Any]:
    """Fail closed if any evidence input changed during worker execution."""
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        import direct_external_bench as direct_external

        direct_external._validate_tag_checkout(synaptic_repo)
    repositories = {
        "omnifuse": repository_fingerprint(ROOT),
        "synaptic_memory": repository_fingerprint(synaptic_repo),
    }
    assert_unchanged("repository fingerprints", state["repositories"], repositories)
    sources = (
        _benchmark_sources(synaptic_repo)
        if protocol == PROTOCOL_SQLITE_NATIVE
        else _benchmark_sources(synaptic_repo, protocol)
    )
    assert_unchanged("benchmark source fingerprints", state["sources"], sources)
    dataset = file_fingerprint(
        dataset_path, display_path=state["dataset_fingerprint"]["path"]
    )
    assert_unchanged(
        "dataset input",
        state["dataset_fingerprint"],
        dataset,
    )
    runtime_environment = _runtime_environment_snapshot()
    _require_protocol_utf8_mode(
        protocol,
        context="controller postflight",
        environment=runtime_environment,
    )
    assert_unchanged(
        "runtime environment", state["runtime_environment"], runtime_environment
    )
    official_environment_probe = None
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        official_environment_probe = _official_environment_probe(synaptic_repo)
        assert_unchanged(
            "official worker environment",
            state["official_environment_probe"],
            official_environment_probe,
        )
    doctor = state["doctor_manifest"]
    verify_doctor_manifest(doctor)
    doctor_environment = _doctor_environment_snapshot(doctor)
    assert_unchanged(
        "doctor runtime environment",
        state["doctor_environment"],
        doctor_environment,
    )
    _verify_doctor_environment(doctor_environment, runtime_environment)
    scorer = sources["scorer"]
    verify_doctor_runtime(
        doctor,
        omnifuse_repository=repositories["omnifuse"],
        synaptic_repository=repositories["synaptic_memory"],
        omnifuse_scorer=scorer["active"],
        synaptic_scorer=scorer["synaptic_checkout_copy"],
    )
    after = {
        "repositories": repositories,
        "benchmark_sources": sources,
        "dataset_input": dataset,
        "runtime_environment": runtime_environment,
        "doctor_environment": doctor_environment,
    }
    checks = {
        "preflight_completed_before_workers": True,
        "repository_states_unchanged": True,
        "benchmark_sources_unchanged": True,
        "dataset_unchanged": True,
        "runtime_environment_unchanged": True,
        "doctor_manifest_unchanged": True,
        "doctor_environment_unchanged": True,
        "doctor_runtime_binding_reverified": True,
        "postflight_verified_before_publish": True,
    }
    if official_environment_probe is not None:
        after["official_environment_probe"] = official_environment_probe
        checks["official_environment_lock_unchanged"] = True
    return {"after": after, "checks": checks}


def _top_k_unique(doc_ids: Iterable[str], k: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(doc_id)
        if len(unique) == k:
            break
    return unique


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("latency samples must not be empty")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(samples),
        "p95": _percentile(samples, 0.95),
        "mean": statistics.fmean(samples),
    }


def _measure_sync(
    search: Callable[[str], Iterable[str]],
    queries: Sequence[Query],
    *,
    k: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, list[str]], list[float]]:
    for _ in range(warmup):
        for _, text, _ in queries:
            search(text)

    rankings: dict[str, list[str]] = {}
    samples: list[float] = []
    for repeat in range(repeats):
        for query_id, text, _ in queries:
            started = clock()
            doc_ids = search(text)
            samples.append((clock() - started) * 1000.0)
            if repeat == 0:
                rankings[query_id] = _top_k_unique(doc_ids, k)
    return rankings, samples


async def _measure_async(
    search: Callable[[str], Awaitable[Iterable[str]]],
    queries: Sequence[Query],
    *,
    k: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, list[str]], list[float]]:
    for _ in range(warmup):
        for _, text, _ in queries:
            await search(text)

    rankings: dict[str, list[str]] = {}
    samples: list[float] = []
    for repeat in range(repeats):
        for query_id, text, _ in queries:
            started = clock()
            doc_ids = await search(text)
            samples.append((clock() - started) * 1000.0)
            if repeat == 0:
                rankings[query_id] = _top_k_unique(doc_ids, k)
    return rankings, samples


def _materialize_candidate_ids(
    doc_ids: Iterable[str], candidate_limit: int
) -> list[str]:
    materialized: list[str] = []
    for raw_doc_id in doc_ids:
        materialized.append(str(raw_doc_id))
        if len(materialized) == candidate_limit:
            break
    return materialized


def _verify_canonical_ranking(
    canonical: dict[str, list[str]], query_id: str, ranking: list[str]
) -> None:
    expected = canonical.get(query_id)
    if expected is None:
        canonical[query_id] = ranking
    elif ranking != expected:
        raise RuntimeError(
            f"retrieval ranking changed during timing for query {query_id!r}"
        )


def _claim_grade_measurement_detail(
    queries: Sequence[Query],
    canonical: Mapping[str, Sequence[str]],
    *,
    candidate_limit: int,
    warmup: int,
    repeats: int,
    round_seconds: Sequence[float],
) -> dict[str, Any]:
    clock = time.get_clock_info("perf_counter")
    rows = [
        {"query_id": query_id, "retrieved_top_20": list(canonical[query_id])}
        for query_id, _text, _relevant in queries
    ]
    return {
        "clock": "time.perf_counter_ns",
        "clock_monotonic": clock.monotonic,
        "clock_adjustable": clock.adjustable,
        "clock_resolution_seconds": clock.resolution,
        "candidate_limit": candidate_limit,
        "canonical_rankings_sha256": canonical_json_sha256(rows),
        "canonical_query_count": len(rows),
        "warmup_calls_verified": warmup * len(rows),
        "measured_calls_verified": repeats * len(rows),
        "query_round_seconds": list(round_seconds),
    }


def _measure_sync_claim_grade(
    search: Callable[[str], Iterable[str]],
    queries: Sequence[Query],
    *,
    k: int,
    candidate_limit: int,
    warmup: int,
    repeats: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[dict[str, list[str]], list[float], dict[str, Any]]:
    canonical: dict[str, list[str]] = {}
    for _ in range(warmup):
        for query_id, text, _relevant in queries:
            ranking = _materialize_candidate_ids(search(text), candidate_limit)
            _verify_canonical_ranking(canonical, query_id, ranking)

    rankings: dict[str, list[str]] = {}
    samples: list[float] = []
    round_seconds: list[float] = []
    for repeat in range(repeats):
        round_elapsed_ns = 0
        for query_id, text, _relevant in queries:
            started_ns = clock_ns()
            ranking = _materialize_candidate_ids(search(text), candidate_limit)
            elapsed_ns = clock_ns() - started_ns
            if elapsed_ns < 0:
                raise RuntimeError("perf_counter_ns moved backwards")
            round_elapsed_ns += elapsed_ns
            samples.append(elapsed_ns / 1_000_000.0)
            _verify_canonical_ranking(canonical, query_id, ranking)
            if repeat == 0:
                rankings[query_id] = ranking[:k]
        round_seconds.append(round_elapsed_ns / 1_000_000_000.0)

    detail = _claim_grade_measurement_detail(
        queries,
        canonical,
        candidate_limit=candidate_limit,
        warmup=warmup,
        repeats=repeats,
        round_seconds=round_seconds,
    )
    return rankings, samples, detail


async def _measure_async_claim_grade(
    search: Callable[[str], Awaitable[Iterable[str]]],
    queries: Sequence[Query],
    *,
    k: int,
    candidate_limit: int,
    warmup: int,
    repeats: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[dict[str, list[str]], list[float], dict[str, Any]]:
    canonical: dict[str, list[str]] = {}
    for _ in range(warmup):
        for query_id, text, _relevant in queries:
            ranking = _materialize_candidate_ids(await search(text), candidate_limit)
            _verify_canonical_ranking(canonical, query_id, ranking)

    rankings: dict[str, list[str]] = {}
    samples: list[float] = []
    round_seconds: list[float] = []
    for repeat in range(repeats):
        round_elapsed_ns = 0
        for query_id, text, _relevant in queries:
            started_ns = clock_ns()
            ranking = _materialize_candidate_ids(await search(text), candidate_limit)
            elapsed_ns = clock_ns() - started_ns
            if elapsed_ns < 0:
                raise RuntimeError("perf_counter_ns moved backwards")
            round_elapsed_ns += elapsed_ns
            samples.append(elapsed_ns / 1_000_000.0)
            _verify_canonical_ranking(canonical, query_id, ranking)
            if repeat == 0:
                rankings[query_id] = ranking[:k]
        round_seconds.append(round_elapsed_ns / 1_000_000_000.0)

    detail = _claim_grade_measurement_detail(
        queries,
        canonical,
        candidate_limit=candidate_limit,
        warmup=warmup,
        repeats=repeats,
        round_seconds=round_seconds,
    )
    return rankings, samples, detail


def _score(queries: Sequence[Query], rankings: dict[str, list[str]], k: int) -> float:
    benchmark = BenchmarkResult()
    for query_id, text, relevant in queries:
        benchmark.add(
            query_id=query_id,
            query=text,
            retrieved=rankings[query_id],
            relevant=relevant,
            k=k,
        )
    return float(benchmark.summary()["mrr"])


def _result(
    system: str,
    ingest_s: float,
    queries: Sequence[Query],
    rankings: dict[str, list[str]],
    samples: Sequence[float],
    *,
    k: int,
    candidate_limit: int,
    warmup: int,
    repeats: int,
    claim_grade: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    latency = _latency_summary(samples)
    current_rss, peak_rss, memory_kind = _process_memory_bytes()
    result = {
        "system": system,
        "ingest_s": ingest_s,
        "query_latency_p50_ms": latency["p50"],
        "query_latency_p95_ms": latency["p95"],
        "query_latency_mean_ms": latency["mean"],
        "query_latency_samples": len(samples),
        "mrr_at_k": _score(queries, rankings, k),
        "k": k,
        "candidate_limit": candidate_limit,
        "warmup_rounds": warmup,
        "measurement_rounds": repeats,
        "process_memory": {
            "scope": PROCESS_MEMORY_SCOPE,
            "kind": memory_kind,
            "current_rss_mb": current_rss / 1_000_000
            if current_rss is not None
            else None,
            "peak_rss_mb": peak_rss / 1_000_000 if peak_rss is not None else None,
        },
    }
    if claim_grade is not None:
        round_seconds = [float(value) for value in claim_grade["query_round_seconds"]]
        query_round_mean = statistics.fmean(round_seconds)
        result["timing"] = {
            "ingest_seconds": ingest_s,
            "query": {
                "measured_total_seconds": sum(round_seconds),
                "first_round_seconds": round_seconds[0],
                "mean_round_seconds": query_round_mean,
                "round_seconds": round_seconds,
            },
            "end_to_end": {
                "ingest_plus_first_round_seconds": ingest_s + round_seconds[0],
                "ingest_plus_mean_round_seconds": ingest_s + query_round_mean,
            },
        }
        result["canonical_rankings"] = {
            key: claim_grade[key]
            for key in (
                "candidate_limit",
                "canonical_rankings_sha256",
                "canonical_query_count",
                "warmup_calls_verified",
                "measured_calls_verified",
                "clock",
                "clock_monotonic",
                "clock_adjustable",
                "clock_resolution_seconds",
            )
        }
    return result


def run_omnifuse(
    corpus: Sequence[CorpusRow],
    queries: Sequence[Query],
    *,
    k: int,
    candidate_limit: int,
    warmup: int,
    repeats: int,
    protocol: str = PROTOCOL_SQLITE_NATIVE,
) -> dict[str, Any]:
    source_root = (ROOT / "src").resolve()
    sys.path.insert(0, str(source_root))
    import omnifuse as omnifuse_package
    from omnifuse import build_inmemory

    package_path = _assert_module_under(
        omnifuse_package.__file__, source_root, "omnifuse"
    )
    source_bindings = {
        "package": _module_binding(
            omnifuse_package,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse",
        ),
        "build_inmemory": _module_binding(
            build_inmemory,
            source_root=source_root,
            repository_root=ROOT,
            name="omnifuse.build_inmemory",
        ),
    }

    prepared_documents = [
        {"id": doc_id, "title": title, "text": text} for doc_id, title, text in corpus
    ]
    claim_grade = protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
    started = time.perf_counter_ns() if claim_grade else time.perf_counter()
    omnifuse = build_inmemory(
        [],
        [],
        prepared_documents,
        vector_k=candidate_limit,
    )
    ingest_s = (
        (time.perf_counter_ns() - started) / 1_000_000_000.0
        if claim_grade
        else time.perf_counter() - started
    )
    source_bindings["retrieve"] = _module_binding(
        type(omnifuse).retrieve,
        source_root=source_root,
        repository_root=ROOT,
        name="omnifuse.OmniFuse.retrieve",
    )

    def search(text: str) -> Iterable[str]:
        return (chunk.id for chunk, _ in omnifuse.retrieve(text, limit=candidate_limit))

    measurement = None
    if claim_grade:
        rankings, samples, measurement = _measure_sync_claim_grade(
            search,
            queries,
            k=k,
            candidate_limit=candidate_limit,
            warmup=warmup,
            repeats=repeats,
        )
    else:
        rankings, samples = _measure_sync(
            search, queries, k=k, warmup=warmup, repeats=repeats
        )
    result = _result(
        "OmniFuse",
        ingest_s,
        queries,
        rankings,
        samples,
        k=k,
        candidate_limit=candidate_limit,
        warmup=warmup,
        repeats=repeats,
        claim_grade=measurement,
    )
    result["runtime"] = {
        "package_path": str(package_path),
        "package_version": getattr(omnifuse_package, "__version__", None)
        or _package_version("omnifuse"),
        "source_bindings": source_bindings,
        "tokenizer": None,
    }
    return result


async def _run_synaptic_official_external_memory(
    repo: Path,
    corpus: Sequence[CorpusRow],
    queries: Sequence[Query],
    *,
    k: int,
    candidate_limit: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import direct_external_bench as direct_external

    direct_external._validate_tag_checkout(repo)
    driver, _scorer, runtime = direct_external._load_upstream_driver(repo)
    source_root = (repo / "src").resolve()
    import synaptic
    from synaptic.backends import sqlite as sqlite_backend_base_module
    from synaptic.backends.memory import MemoryBackend
    from synaptic.graph import SynapticGraph

    package_path = _assert_module_under(synaptic.__file__, source_root, "synaptic")
    source_bindings = {
        "package": _module_binding(
            synaptic,
            source_root=source_root,
            repository_root=repo,
            name="synaptic",
        ),
        "memory_backend": _module_binding(
            MemoryBackend,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.memory.MemoryBackend",
        ),
        "sqlite_normalizer": _module_binding(
            sqlite_backend_base_module,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.sqlite",
        ),
        "graph": _module_binding(
            SynapticGraph,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.graph.SynapticGraph",
        ),
        "official_external_driver": _module_binding(
            driver,
            source_root=(repo / "tests").resolve(),
            repository_root=repo,
            name="tests.benchmark.test_external_datasets",
        ),
    }

    graph = None
    try:
        prepared_corpus = {
            doc_id: {"title": title, "text": text} for doc_id, title, text in corpus
        }
        started_ns = time.perf_counter_ns()
        graph, id_map = await driver._build_graph(prepared_corpus, no_embedding=True)
        ingest_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
        reverse_ids = {
            str(runtime_id): str(corpus_id) for corpus_id, runtime_id in id_map.items()
        }

        async def search(text: str) -> Iterable[str]:
            search_result = await graph.search(text, limit=candidate_limit)
            try:
                return [reverse_ids[str(hit.node.id)] for hit in search_result.nodes]
            except KeyError as exc:
                raise RuntimeError(
                    f"official MemoryBackend returned unknown node {exc.args[0]!r}"
                ) from exc

        rankings, samples, measurement = await _measure_async_claim_grade(
            search,
            queries,
            k=k,
            candidate_limit=candidate_limit,
            warmup=warmup,
            repeats=repeats,
        )
        result = _result(
            "synaptic",
            ingest_s,
            queries,
            rankings,
            samples,
            k=k,
            candidate_limit=candidate_limit,
            warmup=warmup,
            repeats=repeats,
            claim_grade=measurement,
        )
        result["runtime"] = {
            "package_path": str(package_path),
            "package_version": getattr(synaptic, "__version__", None)
            or _package_version("synaptic-memory"),
            "source_bindings": source_bindings,
            "tokenizer": _synaptic_tokenizer_evidence(sqlite_backend_base_module),
            "official_external_runtime": runtime,
        }
        return result
    finally:
        if graph is not None:
            await graph.backend.close()


async def run_synaptic(
    repo: Path,
    corpus: Sequence[CorpusRow],
    queries: Sequence[Query],
    *,
    k: int,
    candidate_limit: int,
    warmup: int,
    repeats: int,
    protocol: str = PROTOCOL_SQLITE_NATIVE,
) -> dict[str, Any]:
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        return await _run_synaptic_official_external_memory(
            repo,
            corpus,
            queries,
            k=k,
            candidate_limit=candidate_limit,
            warmup=warmup,
            repeats=repeats,
        )
    source_root = (repo / "src").resolve()
    sys.path[:0] = [str(source_root), str(repo)]
    import synaptic
    from synaptic.backends import sqlite as sqlite_backend_base_module
    from synaptic.backends.sqlite import SQLiteBackend
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    package_path = _assert_module_under(synaptic.__file__, source_root, "synaptic")
    source_bindings = {
        "package": _module_binding(
            synaptic,
            source_root=source_root,
            repository_root=repo,
            name="synaptic",
        ),
        "sqlite_backend": _module_binding(
            SqliteGraphBackend,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.sqlite_graph.SqliteGraphBackend",
        ),
        "sqlite_backend_base": _module_binding(
            SQLiteBackend,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.backends.sqlite.SQLiteBackend",
        ),
        "graph": _module_binding(
            SynapticGraph,
            source_root=source_root,
            repository_root=repo,
            name="synaptic.graph.SynapticGraph",
        ),
    }

    graph = None
    backend = None
    with TemporaryDirectory(prefix="omnifuse-perf-synaptic-") as temp_dir:
        db_path = Path(temp_dir) / "graph.sqlite"
        try:
            started = time.perf_counter()
            backend = SqliteGraphBackend(str(db_path))
            graph = SynapticGraph(backend, embedder=None, reranker=None)
            await graph.connect()
            for doc_id, title, text in corpus:
                if text or title:
                    await graph.add(
                        title=title or doc_id,
                        content=text,
                        properties={"doc_id": doc_id},
                    )
            ingest_s = time.perf_counter() - started

            async def search(text: str) -> Iterable[str]:
                result = await graph.search(text, limit=candidate_limit)
                return (
                    str((hit.node.properties or {}).get("doc_id", ""))
                    for hit in result.nodes
                )

            rankings, samples = await _measure_async(
                search, queries, k=k, warmup=warmup, repeats=repeats
            )
            result = _result(
                "synaptic",
                ingest_s,
                queries,
                rankings,
                samples,
                k=k,
                candidate_limit=candidate_limit,
                warmup=warmup,
                repeats=repeats,
            )
            result["runtime"] = {
                "package_path": str(package_path),
                "package_version": getattr(synaptic, "__version__", None)
                or _package_version("synaptic-memory"),
                "source_bindings": source_bindings,
                "tokenizer": _synaptic_tokenizer_evidence(sqlite_backend_base_module),
            }
        finally:
            if graph is not None:
                await graph.close()
            elif backend is not None:
                await backend.close()
    return result


def _validate_args(
    parser: argparse.ArgumentParser,
    *,
    k: int,
    candidate_limit: int,
    warmup: int,
    repeats: int,
    trials: int,
    machine_output: bool,
    protocol: str = PROTOCOL_SQLITE_NATIVE,
) -> None:
    if k < 1:
        parser.error("--k must be at least 1")
    if candidate_limit < k:
        parser.error("--candidate-limit must be at least --k")
    if warmup < 0:
        parser.error("--warmup must be non-negative")
    if repeats < 1:
        parser.error("--repeats must be at least 1")
    if trials < 1:
        parser.error("--trials must be at least 1")
    if machine_output and (trials < 2 or trials % 2):
        parser.error("machine output requires an even --trials value of at least 2")
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        if k != 10 or candidate_limit != 20:
            parser.error(
                "official external MemoryBackend protocol requires --k 10 and "
                "--candidate-limit 20"
            )
        if warmup < 1 or repeats < 2:
            parser.error(
                "official external MemoryBackend protocol requires at least one warm-up "
                "and two measured rounds"
            )
        if trials < 2 or trials % 2:
            parser.error(
                "official external MemoryBackend protocol requires an even --trials "
                "value of at least 2 for AB/BA order"
            )


def _run_worker(args: argparse.Namespace, candidate_limit: int) -> None:
    if args.input_file is None:
        raise ValueError("--input-file is required in worker mode")
    if args.worker_run_id is None:
        raise ValueError("--worker-run-id is required in worker mode")
    ensure_output_absent(args.result_file)
    input_path = args.input_file.resolve()
    try:
        input_payload = input_path.read_bytes()
    except OSError as exc:
        raise ProvenanceError(
            f"could not read frozen worker input {input_path}: {exc}"
        ) from exc
    input_fingerprint = _bytes_fingerprint(
        input_payload, path=WORKER_INPUT_DISPLAY_PATH
    )
    corpus, queries = _parse_frozen_input(input_payload)
    assert_unchanged(
        "frozen worker input after read",
        input_fingerprint,
        file_fingerprint(input_path, display_path=WORKER_INPUT_DISPLAY_PATH),
    )
    protocol = getattr(args, "protocol", PROTOCOL_SQLITE_NATIVE)
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY and args.synaptic_repo is None:
        raise ValueError(
            "official external MemoryBackend worker requires --synaptic-repo"
        )
    official_environment_before = (
        _official_environment_lock_evidence(args.synaptic_repo)
        if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
        else None
    )
    if args.worker == "omnifuse":
        result = run_omnifuse(
            corpus,
            queries,
            k=args.k,
            candidate_limit=candidate_limit,
            warmup=args.warmup,
            repeats=args.repeats,
            protocol=protocol,
        )
    else:
        if args.synaptic_repo is None:
            raise ValueError("--synaptic-repo is required for the synaptic worker")
        result = asyncio.run(
            run_synaptic(
                args.synaptic_repo,
                corpus,
                queries,
                k=args.k,
                candidate_limit=candidate_limit,
                warmup=args.warmup,
                repeats=args.repeats,
                protocol=protocol,
            )
        )
    if official_environment_before is not None:
        official_environment_after = _official_environment_lock_evidence(
            args.synaptic_repo
        )
        assert_unchanged(
            "official worker environment lock",
            official_environment_before,
            official_environment_after,
        )
        result["runtime"]["official_environment_lock"] = {
            "before": official_environment_before,
            "after": official_environment_after,
        }
    assert_unchanged(
        "frozen worker input after measurement",
        input_fingerprint,
        file_fingerprint(input_path, display_path=WORKER_INPUT_DISPLAY_PATH),
    )
    worker_identity = capture_worker_identity(args.worker_run_id)
    _atomic_write_json(
        args.result_file,
        {
            "schema": WORKER_RESULT_SCHEMA,
            "schema_version": WORKER_RESULT_SCHEMA_VERSION,
            "status": "ok",
            "system": args.worker,
            "protocol": protocol,
            "contract": {
                "k": args.k,
                "candidate_limit": candidate_limit,
                "warmup_rounds": args.warmup,
                "measurement_rounds": args.repeats,
            },
            "input": {
                **input_fingerprint,
                "documents": len(corpus),
                "scored_queries": len(queries),
                "relevance_judgments": sum(len(relevant) for _, _, relevant in queries),
            },
            "worker_identity": worker_identity,
            "environment": _worker_environment_snapshot(),
            "result": result,
        },
    )


def _worker_command(
    args: argparse.Namespace,
    system: str,
    input_file: Path,
    result_file: Path,
    candidate_limit: int,
    worker_run_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "--data-dir",
        str(args.data_dir),
        "--dataset",
        args.dataset,
        "--protocol",
        args.protocol,
        "--k",
        str(args.k),
        "--candidate-limit",
        str(candidate_limit),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--worker",
        system,
        "--input-file",
        str(input_file),
        "--result-file",
        str(result_file),
        "--worker-run-id",
        worker_run_id,
    ]
    if system == "synaptic" or args.protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        command.extend(["--synaptic-repo", str(args.synaptic_repo)])
    return command


def _validate_source_bindings(
    bindings: object,
    *,
    system: str,
    synaptic_repo: Path | None,
    protocol: str = PROTOCOL_SQLITE_NATIVE,
) -> dict[str, dict[str, Any]]:
    binding_key: object = system if system == "omnifuse" else (system, protocol)
    required = REQUIRED_SOURCE_BINDINGS[binding_key]
    if not isinstance(bindings, dict) or set(bindings) != required:
        actual = sorted(bindings) if isinstance(bindings, dict) else []
        raise ProvenanceError(
            f"{system} worker source bindings are incomplete: "
            f"expected={sorted(required)}, actual={actual}"
        )
    if system == "omnifuse":
        repository_root = ROOT.resolve()
        source_root = (repository_root / "src").resolve()
    else:
        if synaptic_repo is None:
            raise ProvenanceError(
                "synaptic worker source validation needs its checkout"
            )
        repository_root = synaptic_repo.resolve()
        source_root = (repository_root / "src").resolve()

    validated: dict[str, dict[str, Any]] = {}
    for name, raw_binding in bindings.items():
        if not isinstance(raw_binding, dict) or set(raw_binding) != {
            "path",
            "sha256",
            "bytes",
            "resolved_path",
        }:
            raise ProvenanceError(f"{system} worker source binding {name!r} is invalid")
        resolved_path = raw_binding["resolved_path"]
        if not isinstance(resolved_path, str):
            raise ProvenanceError(
                f"{system} worker source binding {name!r} has no resolved path"
            )
        path = Path(resolved_path).resolve()
        expected_root = (
            repository_root
            if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
            and name == "official_external_driver"
            else source_root
        )
        if not _is_below(path, expected_root):
            raise ProvenanceError(
                f"{system} worker imported {name!r} from {path}, outside {expected_root}"
            )
        if (
            protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
            and name == "official_external_driver"
            and path != (repository_root / SYNAPTIC_EXTERNAL_DRIVER_RELATIVE).resolve()
        ):
            raise ProvenanceError("official external driver binding path is invalid")
        expected = file_fingerprint(
            path, display_path=path.relative_to(repository_root).as_posix()
        )
        actual = {key: raw_binding[key] for key in ("path", "sha256", "bytes")}
        assert_unchanged(f"{system} worker imported source {name!r}", expected, actual)
        validated[name] = dict(raw_binding)
    return validated


def _number(
    value: object, *, label: str, minimum: float = 0.0, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ProvenanceError(f"{label} is outside its valid range")
    if maximum is not None and numeric > maximum:
        raise ProvenanceError(f"{label} is outside its valid range")
    return numeric


def _validate_worker_environment(environment: object, *, system: str) -> dict[str, Any]:
    if not isinstance(environment, dict) or set(environment) != WORKER_ENVIRONMENT_KEYS:
        raise ProvenanceError(f"{system} worker environment contract is invalid")
    if environment["python"] != platform.python_version():
        raise ProvenanceError(f"{system} worker used an unexpected Python version")
    if environment["python_executable"] != str(Path(sys.executable).resolve()):
        raise ProvenanceError(f"{system} worker used an unexpected Python executable")
    for key in (
        "isolated",
        "ignore_environment",
        "no_user_site",
        "safe_path",
        "utf8_mode",
    ):
        if environment[key] is not True:
            raise ProvenanceError(f"{system} worker did not enable {key}")
    if environment["user_site_enabled"] is not False:
        raise ProvenanceError(f"{system} worker left user-site packages enabled")
    for key in ("pythonpath", "pythonhome", "pythonusersite"):
        if environment[key] is not None:
            raise ProvenanceError(f"{system} worker inherited {key}")
    if environment["python_no_user_site_env"] != "1":
        raise ProvenanceError(
            f"{system} worker did not receive the no-user-site environment guard"
        )
    return dict(environment)


def _validate_external_module_binding(
    binding: object, *, module_name: str
) -> dict[str, Any] | None:
    if binding is None:
        return None
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "bytes",
        "resolved_path",
    }:
        raise ProvenanceError(
            f"tokenizer dependency binding {module_name!r} is invalid"
        )
    resolved_path = binding["resolved_path"]
    if not isinstance(resolved_path, str) or not resolved_path:
        raise ProvenanceError(
            f"tokenizer dependency binding {module_name!r} has no path"
        )
    path = Path(resolved_path).resolve()
    expected = file_fingerprint(path, display_path=str(path))
    actual = {key: binding[key] for key in ("path", "sha256", "bytes")}
    assert_unchanged(f"tokenizer dependency {module_name!r}", expected, actual)
    if resolved_path != str(path):
        raise ProvenanceError(
            f"tokenizer dependency binding {module_name!r} path is not canonical"
        )
    return dict(binding)


def _validate_tokenizer_evidence(
    evidence: object, *, system: str, require_kiwi: bool
) -> dict[str, Any] | None:
    if system == "omnifuse":
        if evidence is not None:
            raise ProvenanceError("OmniFuse worker reported a Synaptic tokenizer")
        return None
    required = {
        "mode",
        "korean_normalization_used",
        "kiwi_available",
        "kiwi_version",
        "kiwi_model_version",
        "modules",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ProvenanceError("synaptic worker tokenizer evidence is invalid")
    available = evidence["kiwi_available"]
    if available is not None and not isinstance(available, bool):
        raise ProvenanceError("synaptic worker Kiwi availability is invalid")
    used = evidence["korean_normalization_used"]
    if not isinstance(used, bool) or used is not (available is not None):
        raise ProvenanceError("synaptic worker Korean tokenizer usage is inconsistent")
    expected_mode = (
        "unused" if available is None else "kiwi" if available else "regex_fallback"
    )
    if evidence["mode"] != expected_mode:
        raise ProvenanceError("synaptic worker tokenizer mode is inconsistent")
    for key in ("kiwi_version", "kiwi_model_version"):
        value = evidence[key]
        if value is not None and (not isinstance(value, str) or not value):
            raise ProvenanceError(f"synaptic worker {key} is invalid")
    modules = evidence["modules"]
    if not isinstance(modules, dict) or set(modules) != set(TOKENIZER_MODULE_NAMES):
        raise ProvenanceError("synaptic worker tokenizer module bindings are invalid")
    validated_modules = {
        name: _validate_external_module_binding(modules[name], module_name=name)
        for name in TOKENIZER_MODULE_NAMES
    }
    if expected_mode == "kiwi":
        if (
            evidence["kiwi_version"] is None
            or evidence["kiwi_model_version"] is None
            or any(binding is None for binding in validated_modules.values())
        ):
            raise ProvenanceError(
                "synaptic Kiwi mode lacks dependency module or model evidence"
            )
    if require_kiwi and used and expected_mode != "kiwi":
        raise ProvenanceError(
            "strict machine output requires Kiwi when Synaptic Korean normalization runs"
        )
    return {**evidence, "modules": validated_modules}


def _validate_claim_grade_result(
    result: Mapping[str, Any],
    *,
    system: str,
    expected_queries: int,
    contract: Mapping[str, int],
) -> None:
    timing = result.get("timing")
    if not isinstance(timing, dict) or set(timing) != {
        "ingest_seconds",
        "query",
        "end_to_end",
    }:
        raise ProvenanceError(f"{system} claim-grade timing payload is invalid")
    ingest = _number(timing["ingest_seconds"], label=f"{system} timing ingest")
    if not math.isclose(ingest, float(result["ingest_s"]), rel_tol=0.0, abs_tol=1e-12):
        raise ProvenanceError(f"{system} ingest timing is inconsistent")

    query = timing["query"]
    if not isinstance(query, dict) or set(query) != {
        "measured_total_seconds",
        "first_round_seconds",
        "mean_round_seconds",
        "round_seconds",
    }:
        raise ProvenanceError(f"{system} query timing payload is invalid")
    rounds = query["round_seconds"]
    if not isinstance(rounds, list) or len(rounds) != contract["measurement_rounds"]:
        raise ProvenanceError(f"{system} query timing round count is invalid")
    round_values = [
        _number(value, label=f"{system} query round timing") for value in rounds
    ]
    expected_total = sum(round_values)
    expected_mean = statistics.fmean(round_values)
    for key, expected in (
        ("measured_total_seconds", expected_total),
        ("first_round_seconds", round_values[0]),
        ("mean_round_seconds", expected_mean),
    ):
        actual = _number(query[key], label=f"{system} query {key}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ProvenanceError(f"{system} query timing summary is inconsistent")

    end_to_end = timing["end_to_end"]
    if not isinstance(end_to_end, dict) or set(end_to_end) != {
        "ingest_plus_first_round_seconds",
        "ingest_plus_mean_round_seconds",
    }:
        raise ProvenanceError(f"{system} end-to-end timing payload is invalid")
    for key, expected in (
        ("ingest_plus_first_round_seconds", ingest + round_values[0]),
        ("ingest_plus_mean_round_seconds", ingest + expected_mean),
    ):
        actual = _number(end_to_end[key], label=f"{system} end-to-end {key}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ProvenanceError(f"{system} end-to-end timing is inconsistent")

    canonical = result.get("canonical_rankings")
    required_canonical = {
        "candidate_limit",
        "canonical_rankings_sha256",
        "canonical_query_count",
        "warmup_calls_verified",
        "measured_calls_verified",
        "clock",
        "clock_monotonic",
        "clock_adjustable",
        "clock_resolution_seconds",
    }
    if not isinstance(canonical, dict) or set(canonical) != required_canonical:
        raise ProvenanceError(f"{system} canonical ranking payload is invalid")
    expected_counts = {
        "candidate_limit": contract["candidate_limit"],
        "canonical_query_count": expected_queries,
        "warmup_calls_verified": expected_queries * contract["warmup_rounds"],
        "measured_calls_verified": expected_queries * contract["measurement_rounds"],
    }
    for key, expected in expected_counts.items():
        if canonical[key] != expected:
            raise ProvenanceError(f"{system} canonical ranking count is inconsistent")
    if (
        not isinstance(canonical["canonical_rankings_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", canonical["canonical_rankings_sha256"])
        or canonical["clock"] != "time.perf_counter_ns"
        or canonical["clock_monotonic"] is not True
        or canonical["clock_adjustable"] is not False
    ):
        raise ProvenanceError(f"{system} canonical ranking timer contract is invalid")
    _number(
        canonical["clock_resolution_seconds"],
        label=f"{system} perf_counter resolution",
    )


def _validate_worker_result(
    payload: object,
    *,
    system: str,
    expected_input: Mapping[str, Any],
    contract: Mapping[str, int],
    synaptic_repo: Path | None,
    trial_number: int,
    order_position: int,
    expected_worker_run_id: str,
    require_kiwi: bool = False,
    protocol: str = PROTOCOL_SQLITE_NATIVE,
    expected_official_environment_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "schema_version",
        "status",
        "system",
        "protocol",
        "contract",
        "input",
        "worker_identity",
        "environment",
        "result",
    }:
        raise ProvenanceError(f"{system} worker result must match the strict schema")
    if (
        payload["schema"] != WORKER_RESULT_SCHEMA
        or payload["schema_version"] != WORKER_RESULT_SCHEMA_VERSION
        or payload["status"] != "ok"
        or payload["system"] != system
        or payload["protocol"] != protocol
    ):
        raise ProvenanceError(f"{system} worker result contract is invalid")
    raw_contract = payload["contract"]
    if (
        not isinstance(raw_contract, dict)
        or set(raw_contract) != set(contract)
        or any(
            isinstance(raw_contract[key], bool)
            or not isinstance(raw_contract[key], int)
            or raw_contract[key] != expected
            for key, expected in contract.items()
        )
    ):
        raise ProvenanceError(f"{system} worker result contract is invalid")

    raw_input = payload["input"]
    if not isinstance(raw_input, dict) or set(raw_input) != {
        "path",
        "sha256",
        "bytes",
        "documents",
        "scored_queries",
        "relevance_judgments",
    }:
        raise ProvenanceError(f"{system} worker input fingerprint is invalid")
    if any(
        isinstance(raw_input[key], bool)
        or not isinstance(raw_input[key], int)
        or raw_input[key] < 0
        for key in ("bytes", "documents", "scored_queries", "relevance_judgments")
    ):
        raise ProvenanceError(f"{system} worker input counts are invalid")
    assert_unchanged(
        f"{system} worker consumed input",
        dict(expected_input),
        raw_input,
    )
    worker_identity = validate_worker_identity(
        payload["worker_identity"],
        expected_run_id=expected_worker_run_id,
        label=f"{system} worker identity",
    )
    worker_environment = _validate_worker_environment(
        payload["environment"], system=system
    )
    _require_protocol_utf8_mode(
        protocol,
        context=f"{system} worker",
        environment=worker_environment,
    )

    result = payload["result"]
    required_result_keys = {
        "system",
        "ingest_s",
        "query_latency_p50_ms",
        "query_latency_p95_ms",
        "query_latency_mean_ms",
        "query_latency_samples",
        "mrr_at_k",
        "k",
        "candidate_limit",
        "warmup_rounds",
        "measurement_rounds",
        "process_memory",
        "runtime",
    }
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        required_result_keys.update({"timing", "canonical_rankings"})
    if not isinstance(result, dict) or set(result) != required_result_keys:
        raise ProvenanceError(f"{system} worker metric payload is invalid")
    if result["system"] != SYSTEM_RESULT_NAMES[system]:
        raise ProvenanceError(f"{system} worker reported the wrong system name")
    for result_key, contract_key in (
        ("k", "k"),
        ("candidate_limit", "candidate_limit"),
        ("warmup_rounds", "warmup_rounds"),
        ("measurement_rounds", "measurement_rounds"),
    ):
        if (
            isinstance(result[result_key], bool)
            or not isinstance(result[result_key], int)
            or result[result_key] != contract[contract_key]
        ):
            raise ProvenanceError(
                f"{system} worker result {result_key} does not match its contract"
            )
    expected_samples = expected_input["scored_queries"] * contract["measurement_rounds"]
    if (
        isinstance(result["query_latency_samples"], bool)
        or not isinstance(result["query_latency_samples"], int)
        or result["query_latency_samples"] != expected_samples
    ):
        raise ProvenanceError(f"{system} worker latency sample count is inconsistent")

    _number(result["ingest_s"], label=f"{system} ingest_s")
    latency_values = {}
    for key in (
        "query_latency_p50_ms",
        "query_latency_p95_ms",
        "query_latency_mean_ms",
    ):
        latency_values[key] = _number(result[key], label=f"{system} {key}")
    if latency_values["query_latency_p95_ms"] < latency_values["query_latency_p50_ms"]:
        raise ProvenanceError(f"{system} worker latency percentiles are inconsistent")
    _number(result["mrr_at_k"], label=f"{system} mrr_at_k", maximum=1.0)
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        _validate_claim_grade_result(
            result,
            system=system,
            expected_queries=int(expected_input["scored_queries"]),
            contract=contract,
        )

    memory = result["process_memory"]
    if not isinstance(memory, dict) or set(memory) != {
        "scope",
        "kind",
        "current_rss_mb",
        "peak_rss_mb",
    }:
        raise ProvenanceError(f"{system} worker memory payload is invalid")
    if memory["scope"] != PROCESS_MEMORY_SCOPE:
        raise ProvenanceError(f"{system} worker memory scope is invalid")
    if memory["kind"] is not None and not isinstance(memory["kind"], str):
        raise ProvenanceError(f"{system} worker memory kind is invalid")
    memory_values: dict[str, float | None] = {}
    for key in ("current_rss_mb", "peak_rss_mb"):
        value = memory[key]
        memory_values[key] = (
            None if value is None else _number(value, label=f"{system} {key}")
        )
    if (
        memory_values["current_rss_mb"] is not None
        and memory_values["peak_rss_mb"] is not None
        and memory_values["current_rss_mb"] > memory_values["peak_rss_mb"]
    ):
        raise ProvenanceError(f"{system} worker current RSS exceeds peak RSS")

    runtime = result["runtime"]
    required_runtime_keys = {
        "package_path",
        "package_version",
        "source_bindings",
        "tokenizer",
    }
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        required_runtime_keys.add("official_environment_lock")
        if system == "synaptic":
            required_runtime_keys.add("official_external_runtime")
    if not isinstance(runtime, dict) or set(runtime) != required_runtime_keys:
        raise ProvenanceError(f"{system} worker runtime payload is invalid")
    bindings = _validate_source_bindings(
        runtime["source_bindings"],
        system=system,
        synaptic_repo=synaptic_repo,
        protocol=protocol,
    )
    if runtime["package_path"] != bindings["package"]["resolved_path"]:
        raise ProvenanceError(f"{system} worker package path is inconsistent")
    if runtime["package_version"] is not None and not isinstance(
        runtime["package_version"], str
    ):
        raise ProvenanceError(f"{system} worker package version is invalid")
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        if synaptic_repo is None or expected_official_environment_lock is None:
            raise ProvenanceError(
                "official worker validation has no controller environment lock"
            )
        environment_lock = runtime["official_environment_lock"]
        if not isinstance(environment_lock, dict) or set(environment_lock) != {
            "before",
            "after",
        }:
            raise ProvenanceError(
                f"{system} worker environment lock evidence is invalid"
            )
        before_lock = environment_lock["before"]
        after_lock = environment_lock["after"]
        if not isinstance(before_lock, dict) or not isinstance(after_lock, dict):
            raise ProvenanceError(
                f"{system} worker environment lock evidence is invalid"
            )
        _validate_official_environment_lock(before_lock, synaptic_repo)
        _validate_official_environment_lock(after_lock, synaptic_repo)
        assert_unchanged(
            f"{system} worker environment lock",
            before_lock,
            after_lock,
        )
        assert_unchanged(
            f"{system} worker environment versus controller preflight",
            dict(expected_official_environment_lock),
            before_lock,
        )
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY and system == "synaptic":
        official_runtime = runtime["official_external_runtime"]
        if not isinstance(official_runtime, dict) or set(official_runtime) != {
            "python_executable",
            "synaptic_package",
            "synaptic_version",
            "upstream_driver",
            "upstream_scorer",
        }:
            raise ProvenanceError("official external runtime evidence is invalid")
        expected_runtime_paths = {
            "python_executable": worker_environment["python_executable"],
            "synaptic_package": bindings["package"]["resolved_path"],
            "upstream_driver": bindings["official_external_driver"]["resolved_path"],
            "upstream_scorer": str(
                (synaptic_repo / SYNAPTIC_SCORER_RELATIVE).resolve()
            ),
        }
        if any(
            official_runtime[key] != expected
            for key, expected in expected_runtime_paths.items()
        ):
            raise ProvenanceError("official external runtime path evidence is invalid")
    tokenizer = _validate_tokenizer_evidence(
        runtime["tokenizer"], system=system, require_kiwi=require_kiwi
    )

    return {
        **result,
        "runtime": {**runtime, "source_bindings": bindings, "tokenizer": tokenizer},
        "trial": {
            "number": trial_number,
            "order_position": order_position,
        },
        "worker_input": dict(raw_input),
        "worker_identity": worker_identity,
        "worker_environment": worker_environment,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("trial distribution must not be empty")
    return {
        "count": len(values),
        "min": min(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _optional_distribution(
    values: Sequence[float | None],
) -> dict[str, float | int | None]:
    available = [value for value in values if value is not None]
    if not available:
        return {
            "count": 0,
            "missing": len(values),
            "min": None,
            "p50": None,
            "p95": None,
            "mean": None,
            "max": None,
        }
    return {**_distribution(available), "missing": len(values) - len(available)}


def _trial_schedule(systems: Sequence[str], trials: int) -> list[list[str]]:
    forward = list(systems)
    reverse = list(reversed(forward))
    return [forward if index % 2 == 0 else reverse for index in range(trials)]


def _aggregate_trial_results(
    trial_results: Mapping[str, Sequence[dict[str, Any]]], systems: Sequence[str]
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for system in systems:
        trials = list(trial_results[system])
        if not trials:
            raise ProvenanceError(f"{system} has no completed worker trials")
        accuracy_values = {trial["mrr_at_k"] for trial in trials}
        if len(accuracy_values) != 1:
            raise ProvenanceError(
                f"{system} accuracy changed across identical isolated trials"
            )
        distributions = {
            "ingest_s": _distribution([trial["ingest_s"] for trial in trials]),
            "query_latency_p50_ms": _distribution(
                [trial["query_latency_p50_ms"] for trial in trials]
            ),
            "query_latency_p95_ms": _distribution(
                [trial["query_latency_p95_ms"] for trial in trials]
            ),
            "query_latency_mean_ms": _distribution(
                [trial["query_latency_mean_ms"] for trial in trials]
            ),
            "peak_rss_mb": _optional_distribution(
                [trial["process_memory"]["peak_rss_mb"] for trial in trials]
            ),
            "current_rss_mb": _optional_distribution(
                [trial["process_memory"]["current_rss_mb"] for trial in trials]
            ),
            "mrr_at_k": _distribution([trial["mrr_at_k"] for trial in trials]),
        }
        claim_grade_flags = ["timing" in trial for trial in trials]
        if any(claim_grade_flags) and not all(claim_grade_flags):
            raise ProvenanceError(
                f"{system} mixed claim-grade and legacy timing trials"
            )
        aggregate = {
            "system": SYSTEM_RESULT_NAMES[system],
            "trial_count": len(trials),
            "order_positions": [trial["trial"]["order_position"] for trial in trials],
            "accuracy_consistent": True,
            "distributions": distributions,
            "trials": trials,
        }
        if all(claim_grade_flags):
            ranking_hashes = {
                trial["canonical_rankings"]["canonical_rankings_sha256"]
                for trial in trials
            }
            if len(ranking_hashes) != 1:
                raise ProvenanceError(
                    f"{system} canonical top-20 rankings changed across trials"
                )
            distributions["query_round_mean_s"] = _distribution(
                [trial["timing"]["query"]["mean_round_seconds"] for trial in trials]
            )
            distributions["end_to_end_mean_round_s"] = _distribution(
                [
                    trial["timing"]["end_to_end"]["ingest_plus_mean_round_seconds"]
                    for trial in trials
                ]
            )
            aggregate["canonical_rankings_consistent"] = True
            aggregate["canonical_rankings_sha256"] = next(iter(ranking_hashes))
        aggregates.append(aggregate)
    return aggregates


def _print_results(results: Sequence[dict[str, Any]], k: int) -> None:
    print(
        f"{'system':10}{'trials':>8}{'ingest_mean_s':>16}{'qlat_p50_mean':>16}"
        f"{'qlat_p95_mean':>16}{'peak_rss_mean':>16}{f'mrr@{k} mean':>13}"
    )
    for result in results:
        distributions = result["distributions"]
        peak = distributions["peak_rss_mb"]["mean"]
        peak_display = f"{peak:.1f}" if peak is not None else "n/a"
        print(
            f"{result['system']:10}{result['trial_count']:>8}"
            f"{distributions['ingest_s']['mean']:>16.2f}"
            f"{distributions['query_latency_p50_ms']['mean']:>16.2f}"
            f"{distributions['query_latency_p95_ms']['mean']:>16.2f}"
            f"{peak_display:>16}{distributions['mrr_at_k']['mean']:>13.4f}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="nfcorpus.json")
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=PROTOCOL_SQLITE_NATIVE,
        help=(
            "sqlite-native (default) or the official v0.27.0 external-test "
            "MemoryBackend path"
        ),
    )
    parser.add_argument("--synaptic-repo", type=Path, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="independent worker trials per system (default: 2, AB/BA order)",
    )
    parser.add_argument(
        "--out", type=Path, help="write a new immutable machine-readable result JSON"
    )
    parser.add_argument(
        "--workers-dir",
        type=Path,
        help="new durable directory for frozen input and raw worker JSON artifacts",
    )
    parser.add_argument(
        "--doctor-manifest",
        type=Path,
        help="strict eval/bench.py doctor JSON; required with --out",
    )
    parser.add_argument(
        "--worker", choices=("omnifuse", "synaptic"), help=argparse.SUPPRESS
    )
    parser.add_argument("--input-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    return parser


def _machine_report(
    *,
    args: argparse.Namespace,
    candidate_limit: int,
    results: Sequence[dict[str, Any]],
    schedule: Sequence[Sequence[str]],
    state: dict[str, Any],
    postflight: Mapping[str, Any],
    workers_directory: Path,
    frozen_input_artifact: Mapping[str, Any],
    worker_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    provenance_level = PROVENANCE_LEVEL
    protocol = getattr(args, "protocol", PROTOCOL_SQLITE_NATIVE)
    process_summary = worker_process_summary(
        worker_records, expected_count=sum(len(order) for order in schedule)
    )
    before = {
        "repositories": state["repositories"],
        "benchmark_sources": state["sources"],
        "dataset_input": state["dataset_fingerprint"],
        "runtime_environment": state["runtime_environment"],
        "doctor_environment": state["doctor_environment"],
    }
    if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
        before["official_environment_probe"] = state["official_environment_probe"]
    return {
        "schema": "omnifuse.eval.performance",
        "schema_version": REPORT_SCHEMA_VERSION,
        "provenance_level": provenance_level,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "dataset": {
            **state["dataset"],
            "doctor": state["doctor_link"],
        },
        "environment": state["runtime_environment"],
        "repositories": state["repositories"],
        "provenance": {
            "level": provenance_level,
            "before": before,
            "after": postflight["after"],
            "benchmark_sources": state["sources"],
            "frozen_worker_input": state["frozen_worker_input"],
            "frozen_input_artifact": dict(frozen_input_artifact),
        },
        "doctor_manifest": state["doctor_manifest"],
        "integrity": {
            **postflight["checks"],
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
            "protocol": protocol,
            "utf8_mode_contract": {
                "controller_required": protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
                "controller_enabled": state["runtime_environment"]["utf8_mode"],
                "workers_required": True,
            },
            **(
                {
                    "official_environment_lock": _official_environment_contract(
                        state["official_environment_probe"]
                    )
                }
                if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
                else {}
            ),
            "k": args.k,
            "candidate_limit": candidate_limit,
            "warmup_rounds": args.warmup,
            "measurement_rounds": args.repeats,
            "independent_trials_per_system": args.trials,
            "trial_order": [list(order) for order in schedule],
            "counterbalanced_ab_ba": True,
            "accuracy_scorer": "eval/metrics.py::BenchmarkResult",
            "latency_timer": (
                "time.perf_counter_ns with materialized top-20 in isolated fresh workers"
                if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
                else "time.perf_counter in isolated fresh workers"
            ),
            "worker_model": (
                "one fresh process per system and trial; two-system trials alternate "
                "AB/BA order"
            ),
            "trial_statistics": (
                "min/p50/p95/mean/max summarize independent worker-level observations; "
                "each latency observation first summarizes all measured query calls in "
                "that worker"
            ),
            "machine_output_policy": (
                "doctor-bound write-once frozen worker input; strict worker result and "
                "runtime-source validation; preflight before workers; "
                "postflight before atomic write-once publication"
            ),
            "ingest_capability_caveat": (
                (
                    "Official MemoryBackend stores nodes cheaply and computes corpus "
                    "statistics at query time, while OmniFuse eagerly builds a RAM index; "
                    "ingest, query, and ingest-plus-query are reported separately."
                )
                if protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
                else (
                    "Both ingest the same raw corpus to a queryable native index, but "
                    "the artifacts are not capability-equivalent: OmniFuse builds a RAM "
                    "index; synaptic builds a durable, disk-queryable SQLite graph."
                )
            ),
            "memory_scope": (
                "whole fresh worker process, including imports, build, warm-up, and queries"
            ),
        },
        "results": list(results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _require_protocol_utf8_mode(args.protocol, context="controller")
    except ProvenanceError as exc:
        parser.error(str(exc))

    candidate_limit = (
        args.k * 2 if args.candidate_limit is None else args.candidate_limit
    )
    _validate_args(
        parser,
        k=args.k,
        candidate_limit=candidate_limit,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        machine_output=args.out is not None,
        protocol=args.protocol,
    )

    if args.worker:
        if (
            args.result_file is None
            or args.input_file is None
            or args.worker_run_id is None
        ):
            parser.error(
                "--result-file, --input-file, and --worker-run-id are required "
                "in worker mode"
            )
        try:
            _run_worker(args, candidate_limit)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        return 0

    if args.out is not None and args.doctor_manifest is None:
        parser.error("--doctor-manifest is required when --out is used")
    if args.out is not None and args.synaptic_repo is None:
        parser.error("--synaptic-repo is required when --out is used")
    if args.workers_dir is not None and args.out is None:
        parser.error("--workers-dir requires --out")
    if (
        args.protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
        and args.synaptic_repo is None
    ):
        parser.error(
            "official external MemoryBackend protocol requires --synaptic-repo"
        )

    args.data_dir = args.data_dir.resolve()
    if args.out is not None:
        args.out = args.out.resolve()
    if args.workers_dir is not None:
        args.workers_dir = args.workers_dir.resolve()
    dataset_path = (args.data_dir / args.dataset).resolve()
    synaptic_repo = (
        args.synaptic_repo.resolve() if args.synaptic_repo is not None else None
    )
    if synaptic_repo is not None and not synaptic_repo.is_dir():
        parser.error(f"synaptic checkout not found: {synaptic_repo}")
    args.synaptic_repo = synaptic_repo

    machine_state: dict[str, Any] | None = None
    official_environment_before: dict[str, Any] | None = None
    worker_root: Path | None = None
    try:
        if args.out is not None:
            assert synaptic_repo is not None
            worker_root = _worker_directory(args.out, args.workers_dir)
            _validate_worker_directory(
                worker_root, output=args.out, synaptic_repo=synaptic_repo
            )
            machine_state, corpus, queries = _machine_preflight(
                output=args.out,
                doctor_manifest=args.doctor_manifest,
                synaptic_repo=synaptic_repo,
                dataset_path=dataset_path,
                protocol=args.protocol,
            )
            if args.protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
                official_environment_before = machine_state[
                    "official_environment_probe"
                ]
        else:
            if args.protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY:
                assert synaptic_repo is not None
                official_environment_before = _official_environment_probe(synaptic_repo)
                corpus, queries, _selection = _load_official_external_input(
                    synaptic_repo, dataset_path
                )
            else:
                corpus, queries = _load_dataset(dataset_path)
            if not queries:
                raise ValueError("dataset has no scored queries")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(
        f"{args.dataset}: corpus={len(corpus)} queries={len(queries)} "
        f"k={args.k} candidates={candidate_limit} warmup={args.warmup} "
        f"repeats={args.repeats} trials={args.trials}"
    )
    print(
        "accuracy=eval/metrics.py scorer; query latency="
        + (
            "perf_counter_ns claim-grade measurement"
            if args.protocol == PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
            else "direct perf_counter measurement"
        )
    )

    systems = ["omnifuse"]
    if args.synaptic_repo is not None:
        systems.append("synaptic")
    schedule = _trial_schedule(systems, args.trials)
    frozen_payload = _frozen_input_payload(corpus, queries)
    frozen_input = {
        **_bytes_fingerprint(frozen_payload, path=WORKER_INPUT_DISPLAY_PATH),
        "documents": len(corpus),
        "scored_queries": len(queries),
        "relevance_judgments": sum(len(relevant) for _, _, relevant in queries),
    }
    worker_contract = {
        "k": args.k,
        "candidate_limit": candidate_limit,
        "warmup_rounds": args.warmup,
        "measurement_rounds": args.repeats,
    }
    trial_results: dict[str, list[dict[str, Any]]] = {system: [] for system in systems}
    results: list[dict[str, Any]] = []
    postflight: dict[str, Any] | None = None
    worker_records: list[dict[str, Any]] = []
    frozen_input_artifact: dict[str, Any] = {}
    temporary_directory: TemporaryDirectory[str] | None = None
    try:
        if args.out is not None:
            assert worker_root is not None
            worker_root.mkdir(parents=True, exist_ok=False)
        else:
            temporary_directory = TemporaryDirectory(prefix="omnifuse-perf-workers-")
            worker_root = Path(temporary_directory.name)
        input_file = worker_root / "performance-input.json"
        _write_bytes_once(input_file, frozen_payload)
        _frozen_bytes, frozen_input_artifact = read_bytes_artifact(input_file)
        assert_unchanged(
            "published frozen worker input",
            {key: frozen_input[key] for key in ("path", "sha256", "bytes")},
            file_fingerprint(input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
        )
        for trial_number, order in enumerate(schedule, start=1):
            for order_position, system in enumerate(order, start=1):
                assert_unchanged(
                    "frozen worker input before launch",
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
                        system,
                        input_file,
                        result_file,
                        candidate_limit,
                        worker_run_id,
                    ),
                    cwd=ROOT,
                    check=True,
                    env=_isolated_worker_environment(),
                )
                raw_result, worker_artifact = read_json_artifact(result_file)
                validated = _validate_worker_result(
                    raw_result,
                    system=system,
                    expected_input=frozen_input,
                    contract=worker_contract,
                    synaptic_repo=synaptic_repo,
                    trial_number=trial_number,
                    order_position=order_position,
                    expected_worker_run_id=worker_run_id,
                    require_kiwi=machine_state is not None,
                    protocol=args.protocol,
                    expected_official_environment_lock=(
                        official_environment_before["environment_lock"]
                        if official_environment_before is not None
                        else None
                    ),
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
            "frozen worker input after all trials",
            {key: frozen_input[key] for key in ("path", "sha256", "bytes")},
            file_fingerprint(input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
        )
        worker_process_summary(
            worker_records, expected_count=sum(len(order) for order in schedule)
        )
        results = _aggregate_trial_results(trial_results, systems)
        if machine_state is not None:
            assert synaptic_repo is not None
            machine_state["frozen_worker_input"] = frozen_input
            postflight = _verify_machine_postflight(
                machine_state,
                synaptic_repo=synaptic_repo,
                dataset_path=dataset_path,
                protocol=args.protocol,
            )
        elif official_environment_before is not None:
            assert synaptic_repo is not None
            official_environment_after = _official_environment_probe(synaptic_repo)
            assert_unchanged(
                "official controller environment",
                official_environment_before,
                official_environment_after,
            )
        assert_artifact_unchanged(
            "frozen worker input artifact",
            Path(str(frozen_input_artifact["path"])),
            frozen_input_artifact,
        )
        for record in worker_records:
            artifact = record["artifact"]
            assert_artifact_unchanged(
                f"worker artifact {record['worker_run_id']}",
                Path(str(artifact["path"])),
                artifact,
            )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    _print_results(results, args.k)
    if args.out is not None:
        assert (
            machine_state is not None
            and postflight is not None
            and worker_root is not None
        )
        report = _machine_report(
            args=args,
            candidate_limit=candidate_limit,
            results=results,
            schedule=schedule,
            state=machine_state,
            postflight=postflight,
            workers_directory=worker_root,
            frozen_input_artifact=frozen_input_artifact,
            worker_records=worker_records,
        )
        try:
            assert_artifact_unchanged(
                "frozen worker input artifact before report publication",
                Path(str(frozen_input_artifact["path"])),
                frozen_input_artifact,
            )
            for record in worker_records:
                artifact = record["artifact"]
                assert_artifact_unchanged(
                    f"worker artifact {record['worker_run_id']} before report publication",
                    Path(str(artifact["path"])),
                    artifact,
                )
            _atomic_write_json(args.out, report)
        except ProvenanceError as exc:
            parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
