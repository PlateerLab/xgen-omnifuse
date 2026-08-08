"""Compare native persistence artifact sizes for explicitly named datasets.

Both systems receive the same ordered, normalized ``id/title/text`` payload in
fresh worker processes. The result remains a native-artifact measurement, not a
capability-equivalence claim: OmniFuse writes a compressed retrieval index, while
synaptic-memory writes a disk-queryable SQLite graph with additional metadata.

    python eval/footprint_bench.py --synaptic-repo PATH --dataset nfcorpus

Machine-readable output additionally requires a strict doctor manifest:

    python eval/footprint_bench.py --synaptic-repo PATH --dataset nfcorpus \
        --doctor-manifest eval/results/doctor.json --out footprint.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import site
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
OMNIFUSE_SOURCE = (REPOSITORY_ROOT / "src").resolve()
SYNAPTIC_DRIVER_RELATIVE = Path("eval/run_all.py")
SYNAPTIC_SCORER_RELATIVE = Path("tests/benchmark/metrics.py")
PROVENANCE_LEVEL = "strict-preflight-postflight-isolated-write-once-v1"
INTERACTIVE_PROVENANCE_LEVEL = "interactive-preflight-postflight-isolated-v1"
REQUIRED_SOURCE_BINDINGS = {
    "omnifuse": frozenset({"package", "build_inmemory", "save_index"}),
    "synaptic": frozenset(
        {"package", "sqlite_backend", "sqlite_backend_base", "graph"}
    ),
}
TOKENIZER_MODULE_NAMES = ("kiwipiepy", "_kiwipiepy", "kiwipiepy_model")
WORKER_INPUT_DISPLAY_PATH = "worker-input/normalized_documents.json"
WORKER_RESULT_SCHEMA = "omnifuse.eval.native_artifact_footprint.worker"
WORKER_RESULT_SCHEMA_VERSION = 2
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

sys.path.insert(0, str(EVAL_DIR))

from provenance import (  # noqa: E402 - sibling benchmark helper
    ProvenanceError,
    assert_unchanged,
    canonical_json_sha256,
    ensure_output_absent,
    file_fingerprint,
    load_doctor_manifest,
    repository_fingerprint,
    verify_doctor_manifest,
    verify_doctor_runtime,
    write_json_once,
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    relative_path: Path
    doctor_target_id: str | None


@dataclass(frozen=True)
class ArtifactMeasurement:
    bytes: int
    files: dict[str, int]


@dataclass(frozen=True)
class PreparedDataset:
    spec: DatasetSpec
    source: Path
    documents: list[dict[str, str]]
    provenance: dict[str, Any]


DEFAULT_DATASETS = {
    "nfcorpus": DatasetSpec(
        key="nfcorpus",
        label="NFCorpus",
        relative_path=Path("tests/benchmark/data/nfcorpus.json"),
        doctor_target_id="nfcorpus",
    ),
    "allganize_rag_ko": DatasetSpec(
        key="allganize_rag_ko",
        label="Allganize RAG-ko",
        relative_path=Path("tests/benchmark/data/allganize_rag_ko.json"),
        doctor_target_id="allganize_rag_ko",
    ),
}

SELECTION_CAVEAT = (
    "The default NFCorpus and Allganize RAG-ko inputs are explicitly named regression "
    "fixtures, not a statistically representative sample. Results apply only to the "
    "datasets listed in this report."
)
CAPABILITY_CAVEAT = (
    "The artifacts are not capability-equivalent: OmniFuse stores a compressed retrieval "
    "index that is loaded into RAM for querying; synaptic-memory stores a disk-queryable "
    "SQLite graph with additional graph metadata. This benchmark measures artifact bytes "
    "after native persistence and clean close, not total system capability or runtime memory."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptic-repo", type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DEFAULT_DATASETS),
        help=(
            "dataset to measure; repeat to select multiple datasets "
            "(default: nfcorpus and allganize_rag_ko)"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="optional additional JSON corpus for interactive, non-machine output only",
    )
    parser.add_argument(
        "--out", type=Path, help="write a new immutable machine-readable result JSON"
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
    return parser


def _bytes_fingerprint(payload: bytes, *, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


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
    value = str(path.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)
    importlib.invalidate_caches()


def _module_binding(
    value: Any, *, source_root: Path, repository_root: Path, name: str
) -> dict[str, Any]:
    module = value if inspect.ismodule(value) else inspect.getmodule(value)
    if module is None:
        raise RuntimeError(f"cannot resolve imported module for {name}")
    path = _verify_module_under(module, source_root, name)
    relative_path = path.relative_to(repository_root.resolve()).as_posix()
    return {
        **file_fingerprint(path, display_path=relative_path),
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
        "sqlite": sqlite3.sqlite_version,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "packages": {
            "omnifuse": _package_version("omnifuse"),
            "synaptic_memory": _package_version("synaptic-memory"),
            "kiwipiepy": _package_version("kiwipiepy"),
            "kiwipiepy_model": _package_version("kiwipiepy-model"),
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


def _load_omnifuse_api() -> tuple[Any, Any, dict[str, dict[str, Any]]]:
    package_init = OMNIFUSE_SOURCE / "omnifuse" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"OmniFuse source package not found: {package_init}")
    _prepend_import_path(OMNIFUSE_SOURCE)
    package = importlib.import_module("omnifuse")
    _verify_module_under(package, OMNIFUSE_SOURCE, "omnifuse")
    build_inmemory = package.build_inmemory
    save_index = package.save_index
    bindings = {
        "package": _module_binding(
            package,
            source_root=OMNIFUSE_SOURCE,
            repository_root=REPOSITORY_ROOT,
            name="omnifuse",
        ),
        "build_inmemory": _module_binding(
            build_inmemory,
            source_root=OMNIFUSE_SOURCE,
            repository_root=REPOSITORY_ROOT,
            name="omnifuse.build_inmemory",
        ),
        "save_index": _module_binding(
            save_index,
            source_root=OMNIFUSE_SOURCE,
            repository_root=REPOSITORY_ROOT,
            name="omnifuse.save_index",
        ),
    }
    return build_inmemory, save_index, bindings


def _load_synaptic_api(
    repo: Path,
) -> tuple[Any, Any, Any, dict[str, dict[str, Any]]]:
    repository = repo.resolve()
    source = (repository / "src").resolve()
    package_init = source / "synaptic" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            f"synaptic checkout source package not found: {package_init}"
        )

    _prepend_import_path(source)
    package = importlib.import_module("synaptic")
    _verify_module_under(package, source, "synaptic")
    base_backend_module = importlib.import_module("synaptic.backends.sqlite")
    backend_module = importlib.import_module("synaptic.backends.sqlite_graph")
    graph_module = importlib.import_module("synaptic.graph")
    backend_type = backend_module.SqliteGraphBackend
    graph_type = graph_module.SynapticGraph
    bindings = {
        "package": _module_binding(
            package,
            source_root=source,
            repository_root=repository,
            name="synaptic",
        ),
        "sqlite_backend": _module_binding(
            backend_type,
            source_root=source,
            repository_root=repository,
            name="synaptic.backends.sqlite_graph.SqliteGraphBackend",
        ),
        "sqlite_backend_base": _module_binding(
            base_backend_module.SQLiteBackend,
            source_root=source,
            repository_root=repository,
            name="synaptic.backends.sqlite.SQLiteBackend",
        ),
        "graph": _module_binding(
            graph_type,
            source_root=source,
            repository_root=repository,
            name="synaptic.graph.SynapticGraph",
        ),
    }
    return backend_type, graph_type, base_backend_module, bindings


def _normalize_documents(data: object) -> list[dict[str, str]]:
    if isinstance(data, dict):
        raw_documents = data.get("corpus", data.get("documents", []))
    else:
        raw_documents = data

    documents: list[dict[str, str]] = []
    if isinstance(raw_documents, dict):
        for document_id, raw_document in raw_documents.items():
            if isinstance(raw_document, dict):
                title = raw_document.get("title", "")
                text = raw_document.get("text", raw_document.get("content", ""))
            elif isinstance(raw_document, str):
                title, text = "", raw_document
            else:
                continue
            normalized_id = str(document_id)
            documents.append(
                {
                    "id": normalized_id,
                    "title": str(title) or normalized_id,
                    "text": str(text),
                }
            )
    elif isinstance(raw_documents, list):
        for raw_document in raw_documents:
            if not isinstance(raw_document, dict):
                continue
            document_id = raw_document.get(
                "id", raw_document.get("doc_id", raw_document.get("_id", ""))
            )
            title = raw_document.get("title", "")
            text = raw_document.get("text", raw_document.get("content", ""))
            normalized_id = str(document_id)
            documents.append(
                {
                    "id": normalized_id,
                    "title": str(title) or normalized_id,
                    "text": str(text),
                }
            )
    else:
        raise ValueError("corpus JSON must contain a corpus/documents mapping or list")

    documents.sort(key=lambda row: (row["id"], row["title"], row["text"]))
    if not documents:
        raise ValueError("corpus contains no usable documents")
    return documents


def _load_documents(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return _normalize_documents(json.load(stream))


def _normalized_payload(documents: list[dict[str, str]]) -> bytes:
    return json.dumps(
        documents,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_measurement(directory: Path) -> ArtifactMeasurement:
    files = {
        path.relative_to(directory).as_posix(): path.stat().st_size
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    return ArtifactMeasurement(bytes=sum(files.values()), files=files)


def _measure_omnifuse(
    documents: list[dict[str, str]], build_inmemory: Any, save_index: Any
) -> ArtifactMeasurement:
    with tempfile.TemporaryDirectory(prefix="omnifuse-footprint-index-") as directory:
        root = Path(directory)
        save_index(build_inmemory([], [], documents), root / "index.pkl.gz")
        return _artifact_measurement(root)


async def _measure_synaptic(
    documents: list[dict[str, str]], backend_type: Any, graph_type: Any
) -> ArtifactMeasurement:
    with tempfile.TemporaryDirectory(
        prefix="omnifuse-footprint-synaptic-"
    ) as directory:
        root = Path(directory)
        backend = backend_type(str(root / "graph.sqlite"))
        graph = graph_type(backend, embedder=None, reranker=None)
        try:
            await graph.connect()
            for document in documents:
                if document["title"] or document["text"]:
                    await graph.add(
                        title=document["title"],
                        content=document["text"],
                        properties={"doc_id": document["id"]},
                    )
        finally:
            await graph.close()
        return _artifact_measurement(root)


def _dataset_specs(
    repo: Path, selected: Sequence[str] | None, extra_corpus: Path | None
) -> list[tuple[DatasetSpec, Path]]:
    names = list(selected) if selected else list(DEFAULT_DATASETS)
    specs = [
        (DEFAULT_DATASETS[name], repo / DEFAULT_DATASETS[name].relative_path)
        for name in names
    ]
    if extra_corpus is not None:
        resolved = extra_corpus.resolve()
        specs.append(
            (
                DatasetSpec(
                    key=f"external:{resolved.stem}",
                    label=f"External corpus: {resolved.stem}",
                    relative_path=resolved,
                    doctor_target_id=None,
                ),
                resolved,
            )
        )
    for spec, path in specs:
        if not path.is_file():
            raise FileNotFoundError(f"dataset {spec.key!r} not found: {path}")
    return specs


def _source_tree_fingerprint(
    source_root: Path, *, repository_root: Path
) -> dict[str, Any]:
    resolved_source = source_root.resolve()
    resolved_repository = repository_root.resolve()
    files = [
        file_fingerprint(
            path,
            display_path=path.resolve().relative_to(resolved_repository).as_posix(),
        )
        for path in sorted(resolved_source.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    if not files:
        raise ProvenanceError(f"no Python sources found below {resolved_source}")
    return {
        "path": resolved_source.relative_to(resolved_repository).as_posix(),
        "sha256_kind": "file-manifest-v1",
        "sha256": canonical_json_sha256(files),
        "bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


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


def _benchmark_sources(synaptic_repo: Path) -> dict[str, Any]:
    repository = synaptic_repo.resolve()
    return {
        "harness": file_fingerprint(
            SCRIPT_PATH, display_path="eval/footprint_bench.py"
        ),
        "provenance_helper": file_fingerprint(
            EVAL_DIR / "provenance.py", display_path="eval/provenance.py"
        ),
        "omnifuse_python_sources": _source_tree_fingerprint(
            OMNIFUSE_SOURCE, repository_root=REPOSITORY_ROOT
        ),
        "synaptic_python_sources": _source_tree_fingerprint(
            repository / "src" / "synaptic", repository_root=repository
        ),
        "synaptic_native_driver": file_fingerprint(
            repository / SYNAPTIC_DRIVER_RELATIVE,
            display_path=SYNAPTIC_DRIVER_RELATIVE.as_posix(),
        ),
        "scorer": _scorer_provenance(repository),
    }


def _prepare_inputs(
    specs: Sequence[tuple[DatasetSpec, Path]],
) -> list[PreparedDataset]:
    prepared: list[PreparedDataset] = []
    for spec, source in specs:
        resolved = source.resolve()
        source_path = (
            spec.relative_path.as_posix()
            if spec.doctor_target_id is not None
            else str(resolved)
        )
        before = file_fingerprint(resolved, display_path=source_path)
        documents = _load_documents(resolved)
        after = file_fingerprint(resolved, display_path=source_path)
        assert_unchanged(f"dataset {spec.key!r} during input loading", before, after)
        normalized = _bytes_fingerprint(
            _normalized_payload(documents),
            path=f"normalized/{spec.key}.json",
        )
        prepared.append(
            PreparedDataset(
                spec=spec,
                source=resolved,
                documents=documents,
                provenance={
                    "key": spec.key,
                    "label": spec.label,
                    "doctor_target_id": spec.doctor_target_id,
                    "source": before,
                    "normalized_document_payload": normalized,
                    "documents": len(documents),
                },
            )
        )
    return prepared


def _repository_snapshot(synaptic_repo: Path) -> dict[str, Any]:
    return {
        "omnifuse": repository_fingerprint(REPOSITORY_ROOT),
        "synaptic_memory": repository_fingerprint(synaptic_repo),
    }


def _snapshot(
    synaptic_repo: Path, prepared: Sequence[PreparedDataset]
) -> dict[str, Any]:
    return {
        "repositories": _repository_snapshot(synaptic_repo),
        "benchmark_sources": _benchmark_sources(synaptic_repo),
        "inputs": [item.provenance for item in prepared],
        "environment": _runtime_environment_snapshot(),
    }


def _doctor_bindings(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    repositories = snapshot["repositories"]
    scorer = snapshot["benchmark_sources"]["scorer"]
    return {
        "omnifuse_repository": repositories["omnifuse"],
        "synaptic_repository": repositories["synaptic_memory"],
        "omnifuse_scorer": scorer["active"],
        "synaptic_scorer": scorer["synaptic_checkout_copy"],
    }


def _machine_preflight(
    *,
    output: Path,
    doctor_manifest: Path,
    synaptic_repo: Path,
    specs: Sequence[tuple[DatasetSpec, Path]],
) -> tuple[dict[str, Any], list[PreparedDataset]]:
    ensure_output_absent(output)
    prepared = _prepare_inputs(specs)
    if any(item.spec.doctor_target_id is None for item in prepared):
        raise ProvenanceError(
            "machine-readable footprint evidence only accepts doctor-declared datasets"
        )
    before = _snapshot(synaptic_repo, prepared)
    doctor_inputs = [
        {
            "name": item.spec.key,
            "target_id": item.spec.doctor_target_id,
            "path": item.spec.relative_path.as_posix(),
            "sha256": item.provenance["source"]["sha256"],
            "bytes": item.provenance["source"]["bytes"],
        }
        for item in prepared
    ]
    doctor, links = load_doctor_manifest(doctor_manifest, doctor_inputs)
    doctor_environment = _doctor_environment_snapshot(doctor)
    _verify_doctor_environment(doctor_environment, before["environment"])
    verify_doctor_runtime(doctor, **_doctor_bindings(before))
    return {
        "before": before,
        "doctor_manifest": doctor,
        "doctor_environment": doctor_environment,
        "doctor_links": links,
    }, prepared


def _verify_machine_postflight(
    state: Mapping[str, Any],
    *,
    synaptic_repo: Path,
    specs: Sequence[tuple[DatasetSpec, Path]],
) -> dict[str, Any]:
    prepared_after = _prepare_inputs(specs)
    after = _snapshot(synaptic_repo, prepared_after)
    before = state["before"]
    assert_unchanged(
        "repository fingerprints", before["repositories"], after["repositories"]
    )
    assert_unchanged(
        "benchmark source fingerprints",
        before["benchmark_sources"],
        after["benchmark_sources"],
    )
    assert_unchanged("dataset inputs", before["inputs"], after["inputs"])
    doctor = state["doctor_manifest"]
    verify_doctor_manifest(doctor)
    doctor_environment = _doctor_environment_snapshot(doctor)
    assert_unchanged(
        "doctor runtime environment",
        state["doctor_environment"],
        doctor_environment,
    )
    _verify_doctor_environment(doctor_environment, after["environment"])
    verify_doctor_runtime(doctor, **_doctor_bindings(after))
    return {
        "after": after,
        "checks": {
            "preflight_completed_before_workers": True,
            "repository_states_unchanged": True,
            "benchmark_sources_unchanged": True,
            "dataset_inputs_unchanged": True,
            "runtime_environment_unchanged": True,
            "doctor_manifest_unchanged": True,
            "doctor_environment_unchanged": True,
            "doctor_runtime_binding_reverified": True,
            "postflight_verified_before_publish": True,
        },
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    write_json_once(path, payload)


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _run_worker(args: argparse.Namespace) -> None:
    assert args.input_file is not None and args.result_file is not None
    ensure_output_absent(args.result_file)
    input_path = args.input_file.resolve()
    before = file_fingerprint(input_path, display_path=WORKER_INPUT_DISPLAY_PATH)
    documents = _load_documents(input_path)
    assert_unchanged(
        "worker normalized input",
        before,
        file_fingerprint(input_path, display_path=WORKER_INPUT_DISPLAY_PATH),
    )
    normalized = _bytes_fingerprint(
        _normalized_payload(documents), path=WORKER_INPUT_DISPLAY_PATH
    )
    assert_unchanged("worker normalized payload bytes", before, normalized)

    if args.worker == "omnifuse":
        build_inmemory, save_index, bindings = _load_omnifuse_api()
        measurement = _measure_omnifuse(documents, build_inmemory, save_index)
        tokenizer = None
    else:
        assert args.synaptic_repo is not None
        backend_type, graph_type, sqlite_module, bindings = _load_synaptic_api(
            args.synaptic_repo
        )
        measurement = asyncio.run(
            _measure_synaptic(documents, backend_type, graph_type)
        )
        tokenizer = _synaptic_tokenizer_evidence(sqlite_module)

    assert_unchanged(
        "worker normalized input after measurement",
        before,
        file_fingerprint(input_path, display_path=WORKER_INPUT_DISPLAY_PATH),
    )
    _atomic_write_json(
        args.result_file,
        {
            "schema": WORKER_RESULT_SCHEMA,
            "schema_version": WORKER_RESULT_SCHEMA_VERSION,
            "status": "ok",
            "system": args.worker,
            "input": {**before, "documents": len(documents)},
            "artifact": asdict(measurement),
            "source_bindings": bindings,
            "environment": _worker_environment_snapshot(),
            "tokenizer": tokenizer,
        },
    )


def _worker_command(
    *,
    synaptic_repo: Path,
    system: str,
    input_file: Path,
    result_file: Path,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(SCRIPT_PATH),
        "--worker",
        system,
        "--input-file",
        str(input_file),
        "--result-file",
        str(result_file),
        "--synaptic-repo",
        str(synaptic_repo),
    ]


def _validate_source_bindings(
    bindings: object,
    *,
    system: str,
    expected_source: Path,
    repository_root: Path,
) -> dict[str, dict[str, Any]]:
    required = REQUIRED_SOURCE_BINDINGS[system]
    if not isinstance(bindings, dict) or set(bindings) != required:
        actual = sorted(bindings) if isinstance(bindings, dict) else []
        raise ProvenanceError(
            f"{system} worker source bindings are incomplete: "
            f"expected={sorted(required)}, actual={actual}"
        )
    validated: dict[str, dict[str, Any]] = {}
    for name, raw_binding in bindings.items():
        if not isinstance(raw_binding, dict) or set(raw_binding) != {
            "path",
            "sha256",
            "bytes",
            "resolved_path",
        }:
            raise ProvenanceError(f"worker source binding {name!r} is invalid")
        resolved_path = raw_binding["resolved_path"]
        if not isinstance(resolved_path, str):
            raise ProvenanceError(
                f"worker source binding {name!r} has no resolved path"
            )
        path = Path(resolved_path).resolve()
        if resolved_path != str(path):
            raise ProvenanceError(
                f"worker source binding {name!r} path is not canonical"
            )
        if not _is_below(path, expected_source):
            raise ProvenanceError(
                f"worker imported {name!r} from {path}, outside {expected_source.resolve()}"
            )
        relative = path.relative_to(repository_root.resolve()).as_posix()
        expected = file_fingerprint(path, display_path=relative)
        actual = {key: raw_binding[key] for key in ("path", "sha256", "bytes")}
        assert_unchanged(f"worker imported source {name!r}", expected, actual)
        validated[name] = dict(raw_binding)
    return validated


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


def _validate_worker_result(
    result: object,
    *,
    system: str,
    prepared: PreparedDataset,
    synaptic_repo: Path,
    require_kiwi: bool = False,
) -> dict[str, Any]:
    required_result_keys = {
        "schema",
        "schema_version",
        "status",
        "system",
        "input",
        "artifact",
        "source_bindings",
        "environment",
        "tokenizer",
    }
    if not isinstance(result, dict) or set(result) != required_result_keys:
        raise ProvenanceError(f"{system} worker result must match the strict schema")
    if (
        result["schema"] != WORKER_RESULT_SCHEMA
        or result["schema_version"] != WORKER_RESULT_SCHEMA_VERSION
        or result["status"] != "ok"
        or result["system"] != system
    ):
        raise ProvenanceError(f"{system} worker result contract is invalid")
    expected_input = prepared.provenance["normalized_document_payload"]
    raw_input = result["input"]
    if not isinstance(raw_input, dict) or set(raw_input) != {
        "path",
        "sha256",
        "bytes",
        "documents",
    }:
        raise ProvenanceError(f"{system} worker input fingerprint is invalid")
    if raw_input["path"] != WORKER_INPUT_DISPLAY_PATH:
        raise ProvenanceError(f"{system} worker input path is invalid")
    if (
        isinstance(raw_input["bytes"], bool)
        or not isinstance(raw_input["bytes"], int)
        or raw_input["bytes"] < 1
        or isinstance(raw_input["documents"], bool)
        or not isinstance(raw_input["documents"], int)
        or raw_input["documents"] < 1
        or not isinstance(raw_input["sha256"], str)
        or len(raw_input["sha256"]) != 64
    ):
        raise ProvenanceError(f"{system} worker input fields are invalid")
    actual_input = {key: raw_input[key] for key in ("sha256", "bytes")}
    assert_unchanged(
        f"{system} worker input fingerprint",
        {key: expected_input[key] for key in ("sha256", "bytes")},
        actual_input,
    )
    if raw_input["documents"] != len(prepared.documents):
        raise ProvenanceError(f"{system} worker document count is inconsistent")

    artifact = result["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {"bytes", "files"}:
        raise ProvenanceError(f"{system} worker result has no artifact measurement")
    total = artifact["bytes"]
    files = artifact["files"]
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < 0
        or not isinstance(files, dict)
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or Path(path).as_posix() != path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            for path, size in files.items()
        )
        or sum(files.values()) != total
    ):
        raise ProvenanceError(f"{system} worker artifact measurement is invalid")

    if system == "omnifuse":
        expected_source = OMNIFUSE_SOURCE
        repository_root = REPOSITORY_ROOT
    else:
        expected_source = synaptic_repo.resolve() / "src"
        repository_root = synaptic_repo.resolve()
    bindings = _validate_source_bindings(
        result["source_bindings"],
        system=system,
        expected_source=expected_source,
        repository_root=repository_root,
    )
    environment = _validate_worker_environment(result["environment"], system=system)
    tokenizer = _validate_tokenizer_evidence(
        result["tokenizer"], system=system, require_kiwi=require_kiwi
    )
    return {
        **result,
        "input": dict(raw_input),
        "artifact": {"bytes": total, "files": dict(files)},
        "source_bindings": bindings,
        "environment": environment,
        "tokenizer": tokenizer,
    }


def _measure_prepared_dataset(
    prepared: PreparedDataset, *, synaptic_repo: Path, require_kiwi: bool = False
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="omnifuse-footprint-workers-") as directory:
        root = Path(directory)
        input_file = root / "normalized_documents.json"
        input_payload = _normalized_payload(prepared.documents)
        expected_input = _bytes_fingerprint(
            input_payload, path=WORKER_INPUT_DISPLAY_PATH
        )
        _write_bytes_once(input_file, input_payload)
        assert_unchanged(
            "published footprint worker input",
            expected_input,
            file_fingerprint(input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
        )
        worker_results: dict[str, dict[str, Any]] = {}
        for system in ("omnifuse", "synaptic"):
            assert_unchanged(
                "footprint worker input before launch",
                expected_input,
                file_fingerprint(input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
            )
            result_file = root / f"{system}.json"
            subprocess.run(
                _worker_command(
                    synaptic_repo=synaptic_repo,
                    system=system,
                    input_file=input_file,
                    result_file=result_file,
                ),
                cwd=REPOSITORY_ROOT,
                check=True,
                env=_isolated_worker_environment(),
            )
            try:
                raw_result = json.loads(result_file.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ProvenanceError(
                    f"invalid {system} worker result {result_file}: {exc}"
                ) from exc
            worker_results[system] = _validate_worker_result(
                raw_result,
                system=system,
                prepared=prepared,
                synaptic_repo=synaptic_repo,
                require_kiwi=require_kiwi,
            )
        assert_unchanged(
            "footprint worker input after all systems",
            expected_input,
            file_fingerprint(input_file, display_path=WORKER_INPUT_DISPLAY_PATH),
        )

    omnifuse = worker_results["omnifuse"]["artifact"]
    synaptic = worker_results["synaptic"]["artifact"]
    return {
        "label": prepared.spec.label,
        "input": prepared.provenance,
        "artifacts": {"omnifuse": omnifuse, "synaptic": synaptic},
        "runtime_source_bindings": {
            system: result["source_bindings"]
            for system, result in worker_results.items()
        },
        "worker_environments": {
            system: result["environment"] for system, result in worker_results.items()
        },
        "tokenizers": {
            system: result["tokenizer"] for system, result in worker_results.items()
        },
        "ratio_synaptic_over_omnifuse": (
            synaptic["bytes"] / omnifuse["bytes"] if omnifuse["bytes"] else None
        ),
    }


def _build_report(
    *,
    rows: Mapping[str, Any],
    state: Mapping[str, Any],
    postflight: Mapping[str, Any],
) -> dict[str, Any]:
    before = state["before"]
    after = postflight["after"]
    strict_machine_output = state["doctor_manifest"] is not None
    provenance_level = (
        PROVENANCE_LEVEL if strict_machine_output else INTERACTIVE_PROVENANCE_LEVEL
    )
    return {
        "schema": "omnifuse.eval.native_artifact_footprint",
        "schema_version": 3,
        "provenance_level": provenance_level,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "measurement": "native_persistence_artifact_bytes_after_clean_close",
        "input_contract": (
            "Each isolated worker receives the same ordered normalized "
            "id/title/text document list."
        ),
        "datasets_selected": list(rows),
        "dataset_selection_caveat": SELECTION_CAVEAT,
        "capability_caveat": CAPABILITY_CAVEAT,
        "environment": {
            **before["environment"],
            "worker_model": "one fresh write-once worker per system and dataset",
        },
        "repositories": before["repositories"],
        "provenance": {
            "level": provenance_level,
            "mode": "strict-machine" if strict_machine_output else "interactive",
            "before": before,
            "after": after,
            "doctor_manifest": state["doctor_manifest"],
            "doctor_environment": state.get("doctor_environment"),
            "native_driver_note": (
                "The selected synaptic native driver is fingerprinted for checkout and "
                "doctor binding. Footprint measurement calls the checkout's native "
                "SqliteGraphBackend/SynapticGraph persistence API directly; it does not "
                "claim that eval/run_all.py emits a footprint metric."
            ),
        },
        "integrity": postflight["checks"],
        "datasets": dict(rows),
        "historical_artifact_policy": (
            "Earlier 2026-07-13 schema-v1 measurements remain historical selected-host "
            "artifacts and are not relabeled by this stricter schema."
        ),
    }


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        f"{'dataset':24}{'payload MB':>12}{'OmniFuse MB':>14}"
        f"{'synaptic MB':>14}{'ratio':>10}"
    )
    for key, row in report["datasets"].items():
        payload_mb = row["input"]["normalized_document_payload"]["bytes"] / 1_000_000
        omnifuse_mb = row["artifacts"]["omnifuse"]["bytes"] / 1_000_000
        synaptic_mb = row["artifacts"]["synaptic"]["bytes"] / 1_000_000
        ratio = row["ratio_synaptic_over_omnifuse"]
        ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
        print(
            f"{key:24}{payload_mb:>12.2f}{omnifuse_mb:>14.2f}"
            f"{synaptic_mb:>14.2f}{ratio_text:>10}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.worker is not None:
        if args.input_file is None or args.result_file is None:
            parser.error("--input-file and --result-file are required in worker mode")
        if args.worker == "synaptic" and args.synaptic_repo is None:
            parser.error("--synaptic-repo is required for the synaptic worker")
        try:
            _run_worker(args)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        return 0

    if args.synaptic_repo is None:
        parser.error("--synaptic-repo is required")
    if args.out is not None and args.doctor_manifest is None:
        parser.error("--doctor-manifest is required when --out is used")
    if args.out is not None and args.corpus is not None:
        parser.error("--corpus cannot be used with strict machine output")
    if args.out is None and args.doctor_manifest is not None:
        parser.error("--doctor-manifest requires --out")

    synaptic_repo = args.synaptic_repo.resolve()
    if not synaptic_repo.is_dir():
        parser.error(f"synaptic checkout not found: {synaptic_repo}")

    try:
        specs = _dataset_specs(synaptic_repo, args.dataset, args.corpus)
        if args.out is not None:
            state, prepared = _machine_preflight(
                output=args.out,
                doctor_manifest=args.doctor_manifest,
                synaptic_repo=synaptic_repo,
                specs=specs,
            )
        else:
            prepared = _prepare_inputs(specs)
            state = {
                "before": _snapshot(synaptic_repo, prepared),
                "doctor_manifest": None,
                "doctor_environment": None,
                "doctor_links": {},
            }

        rows = {
            item.spec.key: {
                **_measure_prepared_dataset(
                    item,
                    synaptic_repo=synaptic_repo,
                    require_kiwi=args.out is not None,
                ),
                **(
                    {"doctor": state["doctor_links"][item.spec.key]}
                    if args.out is not None
                    else {}
                ),
            }
            for item in prepared
        }

        if args.out is not None:
            postflight = _verify_machine_postflight(
                state, synaptic_repo=synaptic_repo, specs=specs
            )
        else:
            after_prepared = _prepare_inputs(specs)
            after = _snapshot(synaptic_repo, after_prepared)
            assert_unchanged("interactive benchmark snapshot", state["before"], after)
            postflight = {
                "after": after,
                "checks": {
                    "preflight_completed_before_workers": True,
                    "repository_states_unchanged": True,
                    "benchmark_sources_unchanged": True,
                    "dataset_inputs_unchanged": True,
                    "doctor_manifest_unchanged": False,
                    "doctor_runtime_binding_reverified": False,
                    "postflight_verified_before_publish": False,
                },
            }
        report = _build_report(rows=rows, state=state, postflight=postflight)
        _print_report(report)
        if args.out is not None:
            _atomic_write_json(args.out, report)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
