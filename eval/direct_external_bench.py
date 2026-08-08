"""Run the official synaptic-memory external-dataset protocol against OmniFuse.

This is a deterministic FTS-only companion to the 14 tests declared by
``tests/benchmark/test_external_datasets.py`` at the official v0.27.0 tag.  Each
dataset runs in a fresh process using the selected tag environment.  Machine
evidence is doctor-bound, fail-closed, and published write-once.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import re
import shutil
import site
import subprocess
import sys
import time
import tomllib
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = (ROOT / "src").resolve()
sys.path[:0] = [str(SOURCE_ROOT), str(EVAL_DIR)]

import omnifuse as omnifuse_package  # noqa: E402
from metrics import (  # noqa: E402
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from omnifuse import build_inmemory  # noqa: E402
from provenance import (  # noqa: E402
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
    repository_fingerprint,
    read_json_artifact,
    run_with_launcher_pid,
    validate_worker_identity,
    verify_doctor_manifest,
    verify_doctor_runtime,
    worker_process_summary,
    write_json_once,
)

K = 10
CANDIDATE_LIMIT = 20
TEXT_LIMIT = 2000
SAMPLE_SEED = 42
EXPECTED_ORIGIN = "https://github.com/PlateerLab/synaptic-memory.git"
EXPECTED_TAG = "v0.27.0"
EXPECTED_TAG_SHA = "836d53640e520c88910dd57e098167a4defe50d2"
UPSTREAM_DRIVER_RELATIVE = Path("tests/benchmark/test_external_datasets.py")
UPSTREAM_SCORER_RELATIVE = Path("tests/benchmark/metrics.py")
UPSTREAM_PACKAGE_RELATIVE = Path("src/synaptic/__init__.py")
UPSTREAM_SQLITE_RELATIVE = Path("src/synaptic/backends/sqlite.py")
UPSTREAM_LOCK_RELATIVE = Path("uv.lock")
LOCAL_SCORER_RELATIVE = Path("eval/metrics.py")
UPSTREAM_DRIVER_SHA256 = (
    "8709d34c6c98436b0bd7bc7a2ab11efa47dd714d6b0129d83e59511adbbfa686"
)
UPSTREAM_SCORER_SHA256 = (
    "3634fe7d237c14d7975ed370dc7e328386c58ff43d04a917d88dced86f4ef978"
)
UPSTREAM_PACKAGE_SHA256 = (
    "9ad7a2e902686ac0e9509c30949a431121cd4dd847b606f2c6c99f65a0d37cbd"
)
UPSTREAM_SQLITE_SHA256 = (
    "8ae13d0fb779a35f1f0ab64b5b3bc62618c650e8fc775e7bd8216d00751b0a06"
)
UPSTREAM_NORMALIZE_KOREAN_SOURCE_SHA256 = (
    "684eb976d12daf9598bf532914bd2fc84956371eb39248b4af228a5a9abce124"
)
UPSTREAM_LOCK_SHA256 = (
    "1f3f3b2f9a625997ec5168e22e879e47b68f3045de50aa75748de832b0c41792"
)
WORKER_SCHEMA = "omnifuse.eval.synaptic_direct_external_worker"
ENVIRONMENT_PROBE_SCHEMA = "omnifuse.eval.synaptic_direct_external_environment"
REPORT_SCHEMA = "omnifuse.eval.synaptic_direct_external_comparison"
WORKER_SCHEMA_VERSION = 4
ENVIRONMENT_PROBE_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 4
PROVENANCE_LEVEL = "official-tag-direct-fts-isolated-write-once-v4"
METRIC_NAMES = (
    "mrr_at_20",
    "mrr_at_10",
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "ndcg_at_10",
)
WORKER_CONTRACT = {
    "upstream_driver": (
        "tests/benchmark/test_external_datasets.py at official v0.27.0"
    ),
    "synaptic_build": "_build_graph(..., no_embedding=True)",
    "synaptic_query_runner": "_run_benchmark",
    "k": K,
    "candidate_limit": CANDIDATE_LIMIT,
    "text_character_limit": TEXT_LIMIT,
    "sampling_seed": SAMPLE_SEED,
    "mrr_at_20": (
        "upstream BenchmarkResult MRR independently reproduced from captured "
        "retrieved_top_20"
    ),
    "mrr_at_10": "separately recomputed from captured retrieved_top_10",
}
REQUIRED_WORKER_ENVIRONMENT = {
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}
ENVIRONMENT_TOOL_EXCEPTIONS = frozenset({"pip", "setuptools", "wheel"})
UV_SELECTED_EXTRAS = ("sqlite", "embedding", "korean", "dev")
UV_SYNC_CHECK_ARGUMENTS = (
    "sync",
    "--active",
    "--frozen",
    "--no-install-project",
    "--no-dev",
    *(argument for extra in UV_SELECTED_EXTRAS for argument in ("--extra", extra)),
    "--check",
)
TOKENIZER_DISTRIBUTIONS = ("kiwipiepy", "kiwipiepy-model")
TOKENIZER_MODULES = {
    "kiwipiepy": "kiwipiepy",
    "kiwipiepy-model": "kiwipiepy_model",
}
TOKENIZER_PROBE_TEXT = "대한민국의 공중보건 정책은 무엇입니까"


@dataclass(frozen=True)
class DatasetCase:
    id: str
    name: str
    filename: str
    max_queries: int = 0
    klue_corpus_sample: int = 0
    hotpot_supporting: bool = False


CASES = (
    DatasetCase("ko_strategyqa", "Ko-StrategyQA", "ko_strategyqa.json", 100),
    DatasetCase("autorag_retrieval", "AutoRAGRetrieval", "autorag_retrieval.json"),
    DatasetCase("klue_mrc", "KLUE-MRC", "klue_mrc.json", 100, 500),
    DatasetCase(
        "allganize_rag_eval",
        "Allganize-RAG-Eval",
        "allganize_rag_eval.json",
    ),
    DatasetCase("allganize_rag_ko", "Allganize-rag-ko", "allganize_rag_ko.json"),
    DatasetCase(
        "hotpotqa_24",
        "HotPotQA-24",
        "hotpotqa_24.json",
        hotpot_supporting=True,
    ),
    DatasetCase(
        "hotpotqa_200",
        "HotPotQA-200",
        "hotpotqa.json",
        hotpot_supporting=True,
    ),
    DatasetCase("publichealthqa_ko", "PublicHealthQA-ko", "publichealthqa_ko.json"),
    DatasetCase("nfcorpus", "NFCorpus", "nfcorpus.json", 100),
    DatasetCase("scifact", "SciFact", "scifact.json", 100),
    DatasetCase("fiqa", "FiQA", "fiqa.json", 100),
    DatasetCase(
        "miracl_retrieval_ko",
        "MIRACLRetrieval-ko",
        "miracl_retrieval_ko.json",
        100,
    ),
    DatasetCase(
        "multilongdoc_ko",
        "MultiLongDocRetrieval-ko",
        "multilongdoc_ko.json",
        100,
    ),
    DatasetCase("xpqa_ko", "XPQARetrieval-ko", "xpqa_ko.json"),
)
CASE_BY_ID = {case.id: case for case in CASES}
if len(CASES) != 14 or len(CASE_BY_ID) != len(CASES):
    raise RuntimeError("direct external benchmark must declare exactly 14 unique cases")


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo.resolve()), *args],
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ProvenanceError(f"could not inspect Git checkout {repo}: {exc}") from exc
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ProvenanceError(f"git {' '.join(args)} failed for {repo}: {detail}")
    return completed


def _is_below(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _require_runtime_path(
    actual: str | os.PathLike[str], expected: Path, label: str
) -> str:
    actual_path = Path(actual).resolve()
    expected_path = expected.resolve()
    if actual_path != expected_path:
        raise ProvenanceError(
            f"loaded {label} from {actual_path}; expected {expected_path}"
        )
    return str(actual_path)


def _nul_paths(output: str) -> list[str]:
    return sorted(path.replace("\\", "/") for path in output.split("\0") if path)


def _allowed_ignored_dataset(path: str) -> bool:
    candidate = Path(path)
    return (
        candidate.suffix.lower() == ".json"
        and candidate.parent.as_posix() == "tests/benchmark/data"
    )


def _validate_tag_checkout(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    if not repo.is_dir():
        raise ProvenanceError(f"synaptic-memory checkout not found: {repo}")
    root = Path(_git(repo, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if root != repo:
        raise ProvenanceError(f"selected path is not the Git root: {repo} != {root}")

    fetch_origin = _git(repo, "remote", "get-url", "origin").stdout.strip()
    push_origin = _git(repo, "remote", "get-url", "--push", "origin").stdout.strip()
    if fetch_origin != EXPECTED_ORIGIN or push_origin != EXPECTED_ORIGIN:
        raise ProvenanceError(
            "selected synaptic-memory origin mismatch: "
            f"fetch={fetch_origin!r}, push={push_origin!r}"
        )

    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    tag_sha = _git(repo, "rev-parse", f"refs/tags/{EXPECTED_TAG}^{{}}").stdout.strip()
    exact_tag = _git(repo, "describe", "--tags", "--exact-match", "HEAD").stdout.strip()
    if (
        head != EXPECTED_TAG_SHA
        or tag_sha != EXPECTED_TAG_SHA
        or exact_tag != EXPECTED_TAG
    ):
        raise ProvenanceError(
            "selected synaptic-memory checkout is not the exact official tag: "
            f"head={head}, tag_sha={tag_sha}, exact_tag={exact_tag!r}"
        )
    if _git(repo, "diff", "--quiet", "HEAD", "--", check=False).returncode != 0:
        raise ProvenanceError("official tag checkout has tracked working-tree changes")

    untracked = _nul_paths(
        _git(
            repo,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).stdout
    )
    if untracked:
        raise ProvenanceError(
            "official tag checkout has untracked non-ignored files: "
            + ", ".join(untracked)
        )
    ignored = _nul_paths(
        _git(
            repo,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        ).stdout
    )
    unexpected_ignored = [
        path for path in ignored if not _allowed_ignored_dataset(path)
    ]
    if unexpected_ignored:
        raise ProvenanceError(
            "official tag checkout has ignored files outside benchmark data: "
            + ", ".join(unexpected_ignored)
        )

    expected_sources = {
        UPSTREAM_DRIVER_RELATIVE: UPSTREAM_DRIVER_SHA256,
        UPSTREAM_SCORER_RELATIVE: UPSTREAM_SCORER_SHA256,
        UPSTREAM_PACKAGE_RELATIVE: UPSTREAM_PACKAGE_SHA256,
        UPSTREAM_SQLITE_RELATIVE: UPSTREAM_SQLITE_SHA256,
        UPSTREAM_LOCK_RELATIVE: UPSTREAM_LOCK_SHA256,
    }
    sources: dict[str, dict[str, Any]] = {}
    for relative, expected_sha256 in expected_sources.items():
        relative_name = relative.as_posix()
        _git(repo, "ls-files", "--error-unmatch", "--", relative_name)
        fingerprint = file_fingerprint(repo / relative, display_path=relative_name)
        if fingerprint["sha256"] != expected_sha256:
            raise ProvenanceError(
                f"official tag source hash mismatch for {relative_name}: "
                f"{fingerprint['sha256']} != {expected_sha256}"
            )
        sources[relative_name] = fingerprint

    return {
        "repository": "PlateerLab/synaptic-memory",
        "git_root": str(root),
        "origin_fetch": fetch_origin,
        "origin_push": push_origin,
        "tag": EXPECTED_TAG,
        "tag_sha": tag_sha,
        "head": head,
        "detached": _git(repo, "symbolic-ref", "-q", "HEAD", check=False).returncode
        != 0,
        "tracked_tree_clean": True,
        "untracked_nonignored_files": [],
        "allowed_ignored_dataset_files": ignored,
        "validated_sources": sources,
    }


def _verify_shared_tag_identity(
    identity: Mapping[str, Any], repository: Mapping[str, Any]
) -> None:
    expected = {
        "path": str(Path(str(identity["git_root"])).resolve()),
        "git_root": str(Path(str(identity["git_root"])).resolve()),
        "sha": identity["head"],
        "origin_fetch_url": identity["origin_fetch"],
        "exact_tags": [identity["tag"]],
    }
    try:
        actual = {key: repository[key] for key in expected}
    except KeyError as exc:
        raise ProvenanceError(
            f"shared repository fingerprint is missing {exc.args[0]!r}"
        ) from exc
    assert_unchanged("official tag and shared repository identity", expected, actual)


def _tree_fingerprint(root: Path, *, display_root: str) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ProvenanceError(f"required source tree does not exist: {root}")
    files = [
        file_fingerprint(
            path,
            display_path=f"{display_root}/{path.relative_to(root).as_posix()}",
        )
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and "__pycache__" not in path.parts
    ]
    if not files:
        raise ProvenanceError(f"required source tree has no files: {root}")
    return {
        "root": str(root),
        "files": files,
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "manifest_sha256": canonical_json_sha256(files),
    }


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_package_versions(lock_path: Path) -> dict[str, list[str]]:
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ProvenanceError(
            f"could not parse official tag lockfile {lock_path}: {exc}"
        ) from exc
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ProvenanceError("official tag uv.lock has no package records")
    versions: dict[str, set[str]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ProvenanceError("official tag uv.lock has a malformed package record")
        name = package.get("name")
        version = package.get("version")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
        ):
            raise ProvenanceError("official tag uv.lock package identity is incomplete")
        versions.setdefault(_normalize_distribution_name(name), set()).add(version)
    return {name: sorted(values) for name, values in sorted(versions.items())}


def _installed_distribution_manifest() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(raw_name, str) or not raw_name or not version:
            raise ProvenanceError(
                "installed Python distribution identity is incomplete"
            )
        records.append(
            {
                "name": _normalize_distribution_name(raw_name),
                "version": str(version),
                "location": str(Path(distribution.locate_file("")).resolve()),
            }
        )
    records.sort(
        key=lambda record: (record["name"], record["version"], record["location"])
    )
    duplicates = sorted(
        name
        for name in {record["name"] for record in records}
        if sum(record["name"] == name for record in records) > 1
    )
    if duplicates:
        raise ProvenanceError(
            "worker environment has duplicate installed distributions: "
            + ", ".join(duplicates)
        )
    return records


def _validate_installed_distributions(
    lock_versions: Mapping[str, Sequence[str]],
    installed: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    matched: list[dict[str, Any]] = []
    exceptions: list[dict[str, str]] = []
    for distribution in installed:
        name = distribution.get("name")
        version = distribution.get("version")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
        ):
            raise ProvenanceError("installed distribution manifest is incomplete")
        allowed_versions = list(lock_versions.get(name, ()))
        if name in ENVIRONMENT_TOOL_EXCEPTIONS:
            exceptions.append({"name": name, "version": version})
            continue
        if not allowed_versions:
            raise ProvenanceError(
                f"installed distribution {name}=={version} is absent from official uv.lock"
            )
        if version not in allowed_versions:
            raise ProvenanceError(
                f"installed distribution {name}=={version} does not match official "
                f"uv.lock versions {allowed_versions}"
            )
        matched.append(
            {"name": name, "version": version, "allowed_versions": allowed_versions}
        )
    return {
        "status": "ok",
        "coverage": "installed-distribution-membership",
        "completeness_enforced_by": "uv sync --check",
        "matched_distributions": matched,
        "tool_exceptions": exceptions,
        "tool_exception_names": sorted(ENVIRONMENT_TOOL_EXCEPTIONS),
    }


def _uv_sync_check(repo: Path) -> dict[str, Any]:
    executable_name = shutil.which("uv")
    if executable_name is None:
        raise ProvenanceError("uv is required to verify the frozen worker environment")
    executable = Path(executable_name).resolve()
    prefix = Path(sys.prefix).resolve()
    environment = dict(os.environ)
    environment.update(
        {
            "VIRTUAL_ENV": str(prefix),
            "UV_PROJECT_ENVIRONMENT": str(prefix),
            "UV_NO_PROGRESS": "1",
        }
    )
    version = subprocess.run(
        [str(executable), "--version"],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if version.returncode:
        detail = version.stderr.strip() or version.stdout.strip()
        raise ProvenanceError(f"could not identify uv executable: {detail}")
    completed = subprocess.run(
        [str(executable), *UV_SYNC_CHECK_ARGUMENTS],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    detail = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode:
        raise ProvenanceError(
            "worker environment is not synchronized to the selected official-tag "
            f"extras (uv exit {completed.returncode}): {detail.strip()}"
        )
    environment_match = re.search(
        r"^Would use project environment at:\s*(.+)$", detail, flags=re.MULTILINE
    )
    package_match = re.search(
        r"^Checked\s+(\d+)\s+packages\b", detail, flags=re.MULTILINE
    )
    if (
        environment_match is None
        or Path(environment_match.group(1).strip()).resolve() != prefix
        or package_match is None
        or "Would make no changes" not in detail
    ):
        raise ProvenanceError(
            "uv frozen-environment check did not confirm the selected environment "
            "and an exact no-change package set"
        )
    return {
        "status": "ok",
        "returncode": completed.returncode,
        "executable": file_fingerprint(executable),
        "version": version.stdout.strip() or version.stderr.strip(),
        "cwd": str(repo.resolve()),
        "virtual_environment": str(prefix),
        "arguments": list(UV_SYNC_CHECK_ARGUMENTS),
        "selected_extras": list(UV_SELECTED_EXTRAS),
        "dependency_groups": {"dev": "excluded"},
        "install_project": False,
        "checked_package_count": int(package_match.group(1)),
        "reported_no_changes": True,
    }


def _validate_uv_sync_check_record(record: Mapping[str, Any], repo: Path) -> None:
    expected = {
        "status": "ok",
        "returncode": 0,
        "cwd": str(repo.resolve()),
        "arguments": list(UV_SYNC_CHECK_ARGUMENTS),
        "selected_extras": list(UV_SELECTED_EXTRAS),
        "dependency_groups": {"dev": "excluded"},
        "install_project": False,
        "reported_no_changes": True,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ProvenanceError(f"worker uv sync check has invalid {key!r}")
    if (
        not isinstance(record.get("checked_package_count"), int)
        or int(record["checked_package_count"]) < 1
    ):
        raise ProvenanceError("worker uv sync check has no package count")
    version = record.get("version")
    if not isinstance(version, str) or not version.startswith("uv "):
        raise ProvenanceError("worker uv sync check has no uv version")
    executable = record.get("executable")
    if not isinstance(executable, dict) or not isinstance(executable.get("path"), str):
        raise ProvenanceError("worker uv sync check has no executable fingerprint")
    if file_fingerprint(Path(executable["path"])) != executable:
        raise ProvenanceError("worker uv executable fingerprint changed")
    virtual_environment = record.get("virtual_environment")
    if not isinstance(virtual_environment, str) or not virtual_environment:
        raise ProvenanceError("worker uv sync check has no virtual environment")


def _environment_lock_evidence(repo: Path) -> dict[str, Any]:
    lock_path = (repo / UPSTREAM_LOCK_RELATIVE).resolve()
    lock_fingerprint = file_fingerprint(
        lock_path, display_path=UPSTREAM_LOCK_RELATIVE.as_posix()
    )
    if lock_fingerprint["sha256"] != UPSTREAM_LOCK_SHA256:
        raise ProvenanceError(
            "selected worker environment does not use the official tag uv.lock"
        )
    lock_versions = _lock_package_versions(lock_path)
    installed = _installed_distribution_manifest()
    validation = _validate_installed_distributions(lock_versions, installed)
    uv_sync_check = _uv_sync_check(repo)
    record = {
        "execution_mode": "official-tag-source-via-pythonpath",
        "project_distribution_install_required": False,
        "lockfile": lock_fingerprint,
        "lock_package_count": len(lock_versions),
        "lock_package_manifest_sha256": canonical_json_sha256(lock_versions),
        "installed_distributions": installed,
        "installed_manifest_sha256": canonical_json_sha256(installed),
        "validation": validation,
        "uv_sync_check": uv_sync_check,
    }
    _validate_environment_lock_record(record, repo)
    return record


def _validate_environment_lock_record(record: Mapping[str, Any], repo: Path) -> None:
    if record.get("execution_mode") != "official-tag-source-via-pythonpath":
        raise ProvenanceError("worker source execution mode is not declared accurately")
    if record.get("project_distribution_install_required") is not False:
        raise ProvenanceError("worker source execution contract is incomplete")
    lock_versions = _lock_package_versions(repo / UPSTREAM_LOCK_RELATIVE)
    if record.get("lock_package_count") != len(lock_versions):
        raise ProvenanceError("worker lock package count is inconsistent")
    if record.get("lock_package_manifest_sha256") != canonical_json_sha256(
        lock_versions
    ):
        raise ProvenanceError("worker lock package manifest hash is inconsistent")
    expected_lock = file_fingerprint(
        repo / UPSTREAM_LOCK_RELATIVE,
        display_path=UPSTREAM_LOCK_RELATIVE.as_posix(),
    )
    if record.get("lockfile") != expected_lock:
        raise ProvenanceError(
            "worker lockfile fingerprint differs from official checkout"
        )
    installed = record.get("installed_distributions")
    if not isinstance(installed, list):
        raise ProvenanceError("worker installed distribution manifest is missing")
    if record.get("installed_manifest_sha256") != canonical_json_sha256(installed):
        raise ProvenanceError(
            "worker installed distribution manifest hash is inconsistent"
        )
    validation = _validate_installed_distributions(lock_versions, installed)
    if record.get("validation") != validation:
        raise ProvenanceError("worker distribution lock validation is inconsistent")
    uv_sync_check = record.get("uv_sync_check")
    if not isinstance(uv_sync_check, dict):
        raise ProvenanceError("worker environment has no exact uv sync check")
    _validate_uv_sync_check_record(uv_sync_check, repo)


def _installed_distribution(
    environment_lock: Mapping[str, Any], name: str
) -> dict[str, str]:
    installed = environment_lock.get("installed_distributions")
    if not isinstance(installed, list):
        raise ProvenanceError("worker installed distribution manifest is missing")
    matches = [
        item
        for item in installed
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ProvenanceError(
            f"worker environment must contain exactly one {name} distribution"
        )
    record = matches[0]
    if not all(
        isinstance(record.get(key), str) and record[key]
        for key in ("name", "version", "location")
    ):
        raise ProvenanceError(f"worker {name} distribution record is incomplete")
    return record


def _tokenizer_runtime_evidence(
    repo: Path, environment_lock: Mapping[str, Any]
) -> dict[str, Any]:
    sqlite_module = importlib.import_module("synaptic.backends.sqlite")
    sqlite_path = _require_runtime_path(
        getattr(sqlite_module, "__file__", ""),
        repo / UPSTREAM_SQLITE_RELATIVE,
        "official Korean tokenizer",
    )
    sqlite_fingerprint = file_fingerprint(
        Path(sqlite_path), display_path=UPSTREAM_SQLITE_RELATIVE.as_posix()
    )
    if sqlite_fingerprint["sha256"] != UPSTREAM_SQLITE_SHA256:
        raise ProvenanceError("official Korean tokenizer source hash mismatch")
    get_kiwi = getattr(sqlite_module, "_get_kiwi", None)
    normalize_korean = getattr(sqlite_module, "_normalize_korean", None)
    if not callable(get_kiwi) or not callable(normalize_korean):
        raise ProvenanceError("official Korean tokenizer functions are unavailable")
    kiwi = get_kiwi()
    if kiwi is None or getattr(sqlite_module, "_kiwi_available", None) is not True:
        raise ProvenanceError(
            "official Korean tokenizer did not activate Kiwi; regex fallback is forbidden"
        )
    normalized = normalize_korean(TOKENIZER_PROBE_TEXT, query_mode=True)
    if not isinstance(normalized, str) or not normalized.strip():
        raise ProvenanceError("official Korean tokenizer functional probe failed")

    prefix = Path(sys.prefix).resolve()
    lock_versions = _lock_package_versions(repo / UPSTREAM_LOCK_RELATIVE)
    modules: dict[str, dict[str, Any]] = {}
    for distribution_name in TOKENIZER_DISTRIBUTIONS:
        distribution = _installed_distribution(environment_lock, distribution_name)
        versions = lock_versions.get(distribution_name, [])
        if versions != [distribution["version"]]:
            raise ProvenanceError(
                f"worker {distribution_name} version is not exact for the official lock"
            )
        metadata_version = importlib_metadata.version(distribution_name)
        if metadata_version != distribution["version"]:
            raise ProvenanceError(
                f"worker {distribution_name} import metadata version differs from manifest"
            )
        module = importlib.import_module(TOKENIZER_MODULES[distribution_name])
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ProvenanceError(
                f"worker {distribution_name} module has no source path"
            )
        module_path = Path(module_file).resolve()
        distribution_location = Path(distribution["location"]).resolve()
        if not _is_below(distribution_location, prefix):
            raise ProvenanceError(
                f"worker {distribution_name} distribution is outside the selected environment"
            )
        if not _is_below(module_path, distribution_location):
            raise ProvenanceError(
                f"worker {distribution_name} module is outside its distribution"
            )
        modules[distribution_name] = {
            "module": TOKENIZER_MODULES[distribution_name],
            "version": metadata_version,
            "distribution": distribution,
            "module_file": file_fingerprint(module_path),
        }

    normalize_source_sha256 = hashlib.sha256(
        inspect.getsource(normalize_korean).encode("utf-8")
    ).hexdigest()
    if normalize_source_sha256 != UPSTREAM_NORMALIZE_KOREAN_SOURCE_SHA256:
        raise ProvenanceError("official Korean tokenizer function source hash mismatch")
    return {
        "status": "ok",
        "python_prefix": str(prefix),
        "sqlite": {
            **sqlite_fingerprint,
            "runtime_path": sqlite_path,
            "module": sqlite_module.__name__,
            "normalize_function_source_sha256": normalize_source_sha256,
        },
        "kiwi_available": True,
        "kiwi_instance": {
            "module": type(kiwi).__module__,
            "qualname": type(kiwi).__qualname__,
        },
        "modules": modules,
        "functional_probe": {
            "input_sha256": hashlib.sha256(
                TOKENIZER_PROBE_TEXT.encode("utf-8")
            ).hexdigest(),
            "normalized_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "normalized_token_count": len(normalized.split()),
        },
    }


def _validate_tokenizer_runtime_record(
    record: Mapping[str, Any],
    *,
    repo: Path,
    python_prefix: Path,
    environment_lock: Mapping[str, Any],
) -> None:
    if record.get("status") != "ok" or record.get("kiwi_available") is not True:
        raise ProvenanceError("worker tokenizer did not prove active Kiwi mode")
    if Path(str(record.get("python_prefix"))).resolve() != python_prefix.resolve():
        raise ProvenanceError("worker tokenizer used an unexpected Python environment")
    sqlite = record.get("sqlite")
    if not isinstance(sqlite, dict):
        raise ProvenanceError("worker tokenizer has no sqlite source evidence")
    expected_sqlite = file_fingerprint(
        repo / UPSTREAM_SQLITE_RELATIVE,
        display_path=UPSTREAM_SQLITE_RELATIVE.as_posix(),
    )
    if expected_sqlite["sha256"] != UPSTREAM_SQLITE_SHA256:
        raise ProvenanceError("official Korean tokenizer source hash mismatch")
    for key, value in expected_sqlite.items():
        if sqlite.get(key) != value:
            raise ProvenanceError(
                "worker tokenizer sqlite source differs from official tag"
            )
    _require_runtime_path(
        str(sqlite.get("runtime_path")),
        repo / UPSTREAM_SQLITE_RELATIVE,
        "worker Korean tokenizer",
    )
    if sqlite.get("module") != "synaptic.backends.sqlite":
        raise ProvenanceError("worker tokenizer sqlite module binding is invalid")
    source_sha = sqlite.get("normalize_function_source_sha256")
    if source_sha != UPSTREAM_NORMALIZE_KOREAN_SOURCE_SHA256:
        raise ProvenanceError("worker tokenizer function source hash is invalid")

    modules = record.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(TOKENIZER_DISTRIBUTIONS):
        raise ProvenanceError("worker tokenizer module evidence is incomplete")
    lock_versions = _lock_package_versions(repo / UPSTREAM_LOCK_RELATIVE)
    for distribution_name in TOKENIZER_DISTRIBUTIONS:
        module_record = modules.get(distribution_name)
        if not isinstance(module_record, dict):
            raise ProvenanceError(
                f"worker tokenizer has no {distribution_name} module evidence"
            )
        distribution = _installed_distribution(environment_lock, distribution_name)
        if module_record.get("distribution") != distribution:
            raise ProvenanceError(
                f"worker {distribution_name} distribution evidence is inconsistent"
            )
        if module_record.get("version") != distribution["version"] or lock_versions.get(
            distribution_name
        ) != [distribution["version"]]:
            raise ProvenanceError(
                f"worker {distribution_name} version differs from official lock"
            )
        if module_record.get("module") != TOKENIZER_MODULES[distribution_name]:
            raise ProvenanceError(
                f"worker {distribution_name} module binding is invalid"
            )
        module_file = module_record.get("module_file")
        if not isinstance(module_file, dict) or not isinstance(
            module_file.get("path"), str
        ):
            raise ProvenanceError(f"worker {distribution_name} module file is missing")
        module_path = Path(module_file["path"]).resolve()
        distribution_location = Path(distribution["location"]).resolve()
        if not _is_below(distribution_location, python_prefix) or not _is_below(
            module_path, distribution_location
        ):
            raise ProvenanceError(
                f"worker {distribution_name} module escaped the selected environment"
            )
        if file_fingerprint(module_path) != module_file:
            raise ProvenanceError(f"worker {distribution_name} module file changed")
    instance = record.get("kiwi_instance")
    if (
        not isinstance(instance, dict)
        or instance.get("module") != "kiwipiepy"
        or instance.get("qualname") != "Kiwi"
    ):
        raise ProvenanceError("worker tokenizer Kiwi instance evidence is invalid")
    probe = record.get("functional_probe")
    if (
        not isinstance(probe, dict)
        or probe.get("input_sha256")
        != hashlib.sha256(TOKENIZER_PROBE_TEXT.encode("utf-8")).hexdigest()
        or not isinstance(probe.get("normalized_sha256"), str)
        or len(probe["normalized_sha256"]) != 64
        or not isinstance(probe.get("normalized_token_count"), int)
        or int(probe["normalized_token_count"]) < 1
    ):
        raise ProvenanceError("worker tokenizer functional probe evidence is invalid")


def _omnifuse_builder_provenance() -> dict[str, Any]:
    expected = (SOURCE_ROOT / "omnifuse" / "facade.py").resolve()
    source_path = inspect.getsourcefile(build_inmemory)
    if source_path is None:
        raise ProvenanceError("OmniFuse build_inmemory has no inspectable source")
    runtime_path = _require_runtime_path(
        source_path, expected, "OmniFuse build_inmemory"
    )
    if build_inmemory.__module__ != "omnifuse.facade":
        raise ProvenanceError(
            f"OmniFuse build_inmemory came from {build_inmemory.__module__!r}"
        )
    source = inspect.getsource(build_inmemory)
    return {
        **file_fingerprint(expected, display_path="src/omnifuse/facade.py"),
        "runtime_path": runtime_path,
        "module": build_inmemory.__module__,
        "qualname": build_inmemory.__qualname__,
        "function_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _worker_pythonpath_entries(repo: Path) -> list[str]:
    return [
        str((repo / "src").resolve()),
        str(repo.resolve()),
        str(SOURCE_ROOT),
        str(EVAL_DIR),
    ]


def _normalized_sys_path() -> list[str]:
    return [str(Path(entry or os.getcwd()).resolve()) for entry in sys.path]


def _user_site_paths() -> list[str]:
    configured = site.getusersitepackages()
    values = [configured] if isinstance(configured, str) else list(configured)
    return sorted(str(Path(value).resolve()) for value in values)


def _validate_process_environment_record(
    record: Mapping[str, Any],
    *,
    repo: Path,
    expected_python: Path,
    phase: str,
) -> None:
    if record.get("phase") != phase:
        raise ProvenanceError(
            f"worker environment phase mismatch: {record.get('phase')!r}"
        )
    if record.get("variables") != REQUIRED_WORKER_ENVIRONMENT:
        raise ProvenanceError(
            f"worker determinism environment mismatch: {record.get('variables')!r}"
        )
    if record.get("pythonpath_entries") != _worker_pythonpath_entries(repo):
        raise ProvenanceError(
            "worker PYTHONPATH does not match the isolated source contract"
        )
    flags = record.get("flags")
    if not isinstance(flags, dict) or flags != {
        "utf8_mode": 1,
        "no_user_site": 1,
        "dont_write_bytecode": True,
    }:
        raise ProvenanceError(f"worker Python flags violate isolation: {flags!r}")
    if record.get("user_site_enabled") is not False:
        raise ProvenanceError("worker user-site packages are enabled")
    if record.get("user_site_present_on_sys_path"):
        raise ProvenanceError("worker user-site directory is present on sys.path")
    if (
        Path(str(record.get("python_executable"))).resolve()
        != expected_python.resolve()
    ):
        raise ProvenanceError("worker environment used an unexpected Python executable")
    normalized = record.get("normalized_sys_path")
    if not isinstance(normalized, list) or not all(
        isinstance(entry, str) and entry for entry in normalized
    ):
        raise ProvenanceError("worker sys.path evidence is incomplete")
    required_roots = _worker_pythonpath_entries(repo)
    missing_roots = [root for root in required_roots if root not in normalized]
    if missing_roots:
        raise ProvenanceError(
            "worker sys.path is missing required source roots: "
            + ", ".join(missing_roots)
        )
    if phase == "runtime" and normalized[0] != str((repo / "src").resolve()):
        raise ProvenanceError("official tag source is not first on runtime sys.path")


def _process_environment_record(
    repo: Path, expected_python: Path, *, phase: str
) -> dict[str, Any]:
    variables = {key: os.environ.get(key) for key in REQUIRED_WORKER_ENVIRONMENT}
    pythonpath_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    normalized = _normalized_sys_path()
    user_sites = _user_site_paths()
    record = {
        "phase": phase,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "platform": platform.platform(),
        "variables": variables,
        "pythonpath_entries": pythonpath_entries,
        "flags": {
            "utf8_mode": sys.flags.utf8_mode,
            "no_user_site": sys.flags.no_user_site,
            "dont_write_bytecode": sys.dont_write_bytecode,
        },
        "sys_path": list(sys.path),
        "normalized_sys_path": normalized,
        "user_site_enabled": site.ENABLE_USER_SITE,
        "user_site_paths": user_sites,
        "user_site_present_on_sys_path": any(path in normalized for path in user_sites),
    }
    _validate_process_environment_record(
        record, repo=repo, expected_python=expected_python, phase=phase
    )
    return record


def _validate_runtime_environment_transition(
    before: Mapping[str, Any], after: Mapping[str, Any], *, repo: Path
) -> None:
    expected = dict(before)
    expected["phase"] = "runtime"
    source_root = str((repo / "src").resolve())
    raw_sys_path = before.get("sys_path")
    normalized_sys_path = before.get("normalized_sys_path")
    if not isinstance(raw_sys_path, list) or not isinstance(normalized_sys_path, list):
        raise ProvenanceError("worker startup sys.path evidence is incomplete")
    expected["sys_path"] = [
        source_root,
        *(entry for entry in raw_sys_path if entry != source_root),
    ]
    expected["normalized_sys_path"] = [
        source_root,
        *(entry for entry in normalized_sys_path if entry != source_root),
    ]
    assert_unchanged("worker runtime environment transition", expected, after)


def _environment_probe_payload(repo: Path, expected_python: Path) -> dict[str, Any]:
    process = _process_environment_record(repo, expected_python, phase="startup")
    environment_lock = _environment_lock_evidence(repo)
    tokenizer_runtime = _tokenizer_runtime_evidence(repo, environment_lock)
    _validate_tokenizer_runtime_record(
        tokenizer_runtime,
        repo=repo,
        python_prefix=Path(str(process["python_prefix"])),
        environment_lock=environment_lock,
    )
    return {
        "schema": ENVIRONMENT_PROBE_SCHEMA,
        "schema_version": ENVIRONMENT_PROBE_SCHEMA_VERSION,
        "provenance_level": PROVENANCE_LEVEL,
        "status": "ok",
        "process": process,
        "environment_lock": environment_lock,
        "tokenizer_runtime": tokenizer_runtime,
    }


def _scorer_provenance(repo: Path) -> dict[str, Any]:
    local = file_fingerprint(
        ROOT / LOCAL_SCORER_RELATIVE, display_path=LOCAL_SCORER_RELATIVE.as_posix()
    )
    upstream = file_fingerprint(
        repo / UPSTREAM_SCORER_RELATIVE,
        display_path=UPSTREAM_SCORER_RELATIVE.as_posix(),
    )
    if local["sha256"] != UPSTREAM_SCORER_SHA256:
        raise ProvenanceError(
            "local eval/metrics.py does not match the official v0.27.0 scorer"
        )
    if (local["sha256"], local["bytes"]) != (
        upstream["sha256"],
        upstream["bytes"],
    ):
        raise ProvenanceError(
            "local and official-tag benchmark scorers are not byte-identical"
        )
    return {
        "active": local,
        "synaptic_checkout_copy": upstream,
        "byte_identical": True,
    }


def _source_fingerprints(repo: Path) -> dict[str, Any]:
    return {
        "harness": file_fingerprint(
            SCRIPT_PATH, display_path="eval/direct_external_bench.py"
        ),
        "provenance_helper": file_fingerprint(
            EVAL_DIR / "provenance.py", display_path="eval/provenance.py"
        ),
        "scorer": _scorer_provenance(repo),
        "upstream_direct_driver": file_fingerprint(
            repo / UPSTREAM_DRIVER_RELATIVE,
            display_path=UPSTREAM_DRIVER_RELATIVE.as_posix(),
        ),
        "upstream_lockfile": file_fingerprint(
            repo / UPSTREAM_LOCK_RELATIVE,
            display_path=UPSTREAM_LOCK_RELATIVE.as_posix(),
        ),
        "omnifuse_builder": _omnifuse_builder_provenance(),
        "omnifuse_source_tree": _tree_fingerprint(
            SOURCE_ROOT / "omnifuse", display_root="src/omnifuse"
        ),
        "synaptic_source_tree": _tree_fingerprint(
            repo / "src" / "synaptic", display_root="src/synaptic"
        ),
    }


def _dataset_path(repo: Path, case: DatasetCase) -> Path:
    path = (repo / "tests" / "benchmark" / "data" / case.filename).resolve()
    expected_parent = (repo / "tests" / "benchmark" / "data").resolve()
    if path.parent != expected_parent:
        raise ProvenanceError(f"dataset path escaped official checkout: {path}")
    return path


def _input_fingerprint(repo: Path, case: DatasetCase) -> dict[str, Any]:
    relative = (Path("tests") / "benchmark" / "data" / case.filename).as_posix()
    path = _dataset_path(repo, case)
    tracked = (
        _git(
            repo,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
            check=False,
        ).returncode
        == 0
    )
    return {
        **file_fingerprint(path, display_path=relative),
        "case_id": case.id,
        "case_name": case.name,
        "git_tracked": tracked,
    }


def _input_fingerprints(repo: Path) -> dict[str, dict[str, Any]]:
    return {case.id: _input_fingerprint(repo, case) for case in CASES}


def _doctor_inputs(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": case.id,
            "target_id": case.id,
            "path": inputs[case.id]["path"],
            "sha256": inputs[case.id]["sha256"],
            "bytes": inputs[case.id]["bytes"],
        }
        for case in CASES
    ]


def _preflight(
    *,
    repo: Path,
    python: Path,
    doctor_path: Path,
    output: Path,
    workers_dir: Path,
) -> dict[str, Any]:
    ensure_output_absent(output)
    if _is_below(output, workers_dir):
        raise ProvenanceError(
            "suite output must be outside the worker-artifact directory: "
            f"{output.resolve()}"
        )
    if _is_below(output, repo) or _is_below(workers_dir, repo):
        raise ProvenanceError(
            "benchmark artifacts must not be written inside the immutable "
            f"synaptic-memory tag checkout: {repo.resolve()}"
        )
    if workers_dir.exists():
        raise ProvenanceError(
            f"refusing to reuse worker-artifact directory: {workers_dir.resolve()}"
        )
    if _is_below(workers_dir, ROOT):
        relative = workers_dir.resolve().relative_to(ROOT).as_posix()
        if (
            _git(ROOT, "check-ignore", "-q", "--", relative, check=False).returncode
            != 0
        ):
            raise ProvenanceError(
                "worker-artifact directory inside OmniFuse must be Git-ignored: "
                f"{workers_dir.resolve()}"
            )

    identity = _validate_tag_checkout(repo)
    repositories = {
        "omnifuse": repository_fingerprint(ROOT),
        "synaptic_memory": repository_fingerprint(repo),
    }
    _verify_shared_tag_identity(identity, repositories["synaptic_memory"])
    sources = _source_fingerprints(repo)
    inputs = _input_fingerprints(repo)
    worker_environment = _probe_worker_environment(python, repo)
    doctor, links = load_doctor_manifest(doctor_path, _doctor_inputs(inputs))
    scorer = sources["scorer"]
    verify_doctor_runtime(
        doctor,
        omnifuse_repository=repositories["omnifuse"],
        synaptic_repository=repositories["synaptic_memory"],
        omnifuse_scorer=scorer["active"],
        synaptic_scorer=scorer["synaptic_checkout_copy"],
    )
    return {
        "identity": identity,
        "repositories": repositories,
        "sources": sources,
        "inputs": inputs,
        "worker_environment": worker_environment,
        "doctor": doctor,
        "doctor_links": links,
    }


def _postflight(repo: Path, python: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    identity = _validate_tag_checkout(repo)
    repositories = {
        "omnifuse": repository_fingerprint(ROOT),
        "synaptic_memory": repository_fingerprint(repo),
    }
    _verify_shared_tag_identity(identity, repositories["synaptic_memory"])
    sources = _source_fingerprints(repo)
    inputs = _input_fingerprints(repo)
    worker_environment = _probe_worker_environment(python, repo)
    assert_unchanged("official tag identity", state["identity"], identity)
    assert_unchanged("repository fingerprints", state["repositories"], repositories)
    assert_unchanged("complete source fingerprints", state["sources"], sources)
    assert_unchanged("dataset input fingerprints", state["inputs"], inputs)
    assert_unchanged(
        "worker Python environment", state["worker_environment"], worker_environment
    )
    doctor = state["doctor"]
    verify_doctor_manifest(doctor)
    scorer = sources["scorer"]
    verify_doctor_runtime(
        doctor,
        omnifuse_repository=repositories["omnifuse"],
        synaptic_repository=repositories["synaptic_memory"],
        omnifuse_scorer=scorer["active"],
        synaptic_scorer=scorer["synaptic_checkout_copy"],
    )
    return {
        "official_tag_identity": {"before": state["identity"], "after": identity},
        "repositories": {"before": state["repositories"], "after": repositories},
        "sources": {"before": state["sources"], "after": sources},
        "inputs": {"before": state["inputs"], "after": inputs},
        "worker_environment": {
            "before": state["worker_environment"],
            "after": worker_environment,
        },
        "checks": {
            "preflight_completed_before_workers": True,
            "official_tag_identity_unchanged": True,
            "repository_states_unchanged": True,
            "complete_source_trees_unchanged": True,
            "all_input_files_unchanged": True,
            "worker_python_environment_unchanged": True,
            "doctor_manifest_unchanged": True,
            "doctor_runtime_binding_reverified": True,
            "postflight_completed_before_publication": True,
        },
    }


def _load_upstream_driver(
    repo: Path,
) -> tuple[types.ModuleType, types.ModuleType, dict[str, str]]:
    repo = repo.resolve()
    source_root = (repo / "src").resolve()
    source_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    synaptic = importlib.import_module("synaptic")
    package_file = getattr(synaptic, "__file__", None)
    if not package_file or not _is_below(Path(package_file), source_root):
        raise ProvenanceError(
            f"loaded synaptic from {package_file}; expected source below {source_root}"
        )

    package_name = "_omnifuse_official_synaptic_benchmark"
    benchmark_dir = (repo / "tests" / "benchmark").resolve()
    package = types.ModuleType(package_name)
    package.__path__ = [str(benchmark_dir)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    driver_name = f"{package_name}.test_external_datasets"
    driver_path = (repo / UPSTREAM_DRIVER_RELATIVE).resolve()
    spec = importlib.util.spec_from_file_location(driver_name, driver_path)
    if spec is None or spec.loader is None:
        raise ProvenanceError(
            f"could not load official benchmark driver: {driver_path}"
        )
    driver = importlib.util.module_from_spec(spec)
    sys.modules[driver_name] = driver
    spec.loader.exec_module(driver)
    _require_runtime_path(driver.__file__, driver_path, "official direct driver")

    scorer = sys.modules.get(f"{package_name}.metrics")
    if not isinstance(scorer, types.ModuleType):
        raise ProvenanceError("official direct driver did not load its relative scorer")
    _require_runtime_path(
        scorer.__file__, (repo / UPSTREAM_SCORER_RELATIVE).resolve(), "official scorer"
    )
    benchmark_result_source = inspect.getsourcefile(driver.BenchmarkResult)
    if benchmark_result_source is None:
        raise ProvenanceError("official BenchmarkResult has no inspectable source")
    _require_runtime_path(
        benchmark_result_source,
        (repo / UPSTREAM_SCORER_RELATIVE).resolve(),
        "official BenchmarkResult",
    )
    return (
        driver,
        scorer,
        {
            "python_executable": str(Path(sys.executable).resolve()),
            "synaptic_package": str(Path(package_file).resolve()),
            "synaptic_version": getattr(synaptic, "__version__", None),
            "upstream_driver": str(driver_path),
            "upstream_scorer": str((repo / UPSTREAM_SCORER_RELATIVE).resolve()),
        },
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"dataset {label} must be a JSON object")
    return value


def _prepare_case_data(
    driver: types.ModuleType, case: DatasetCase
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = driver._load_dataset(case.filename)
    if not isinstance(data, dict):
        raise FileNotFoundError(f"official loader could not load {case.filename}")
    corpus = _require_mapping(data.get("corpus"), "corpus")
    queries = _require_mapping(data.get("queries"), "queries")
    qrels = _require_mapping(data.get("qrels"), "qrels")

    original_corpus_count = len(corpus)
    original_query_count = len(queries)
    if case.klue_corpus_sample:
        corpus_items = list(corpus.items())
        random.seed(SAMPLE_SEED)
        sampled_ids = {
            key
            for key, _value in random.sample(
                corpus_items, min(case.klue_corpus_sample, len(corpus_items))
            )
        }
        corpus = {key: value for key, value in corpus.items() if key in sampled_ids}
        queries = {
            key: value
            for key, value in queries.items()
            if key.replace("klue_", "klue_doc_") in sampled_ids
        }
        qrels = {key: value for key, value in qrels.items() if key in queries}

    candidate_query_ids = [query_id for query_id in queries if query_id in qrels]
    prepared = {"corpus": corpus, "queries": queries, "qrels": qrels}
    selection = {
        "seed": SAMPLE_SEED,
        "original_corpus_count": original_corpus_count,
        "selected_corpus_count": len(corpus),
        "original_query_count": original_query_count,
        "eligible_query_count_before_max_queries": len(candidate_query_ids),
        "max_queries": case.max_queries,
        "klue_corpus_sample": case.klue_corpus_sample,
        "selected_corpus_ids_ordered_sha256": canonical_json_sha256(list(corpus)),
        "eligible_query_ids_ordered_sha256": canonical_json_sha256(candidate_query_ids),
    }
    return prepared, selection


def _omnifuse_documents(
    corpus: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    documents: list[dict[str, str]] = []
    skipped_empty = 0
    truncated = 0
    title_fallbacks = 0
    original_characters = 0
    indexed_characters = 0
    for corpus_id, raw_document in corpus.items():
        document = _require_mapping(raw_document, f"corpus[{corpus_id!r}]")
        title = document.get("title", "")
        text = document.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(
                f"dataset document {corpus_id!r} title/text must be strings"
            )
        if not text:
            skipped_empty += 1
            continue
        original_characters += len(text)
        if len(text) > TEXT_LIMIT:
            text = text[:TEXT_LIMIT]
            truncated += 1
        indexed_characters += len(text)
        if not title:
            title_fallbacks += 1
        documents.append(
            {"id": str(corpus_id), "title": title or text[:80], "text": text}
        )
    if not documents:
        raise ValueError(
            "dataset has no non-empty documents after official preprocessing"
        )
    indexed_ids = [document["id"] for document in documents]
    return documents, {
        "text_character_limit": TEXT_LIMIT,
        "input_documents": len(corpus),
        "indexed_documents": len(documents),
        "skipped_empty_text_documents": skipped_empty,
        "truncated_documents": truncated,
        "title_fallback_documents": title_fallbacks,
        "original_text_characters": original_characters,
        "indexed_text_characters": indexed_characters,
        "indexed_document_ids_ordered_sha256": canonical_json_sha256(indexed_ids),
    }


def _reciprocal_rank_at_k(
    retrieved: Sequence[str], relevant: set[str], k: int
) -> float:
    for index, document_id in enumerate(retrieved[:k], 1):
        if document_id in relevant:
            return 1.0 / index
    return 0.0


def _normalize_query_rows(
    benchmark: Any,
    *,
    retrieved_top_20: Sequence[Sequence[str]],
    reverse_ids: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    if len(retrieved_top_20) != len(benchmark.queries):
        raise RuntimeError("captured top-20 retrieval count differs from scorer rows")
    rows: list[dict[str, Any]] = []
    for raw, captured in zip(benchmark.queries, retrieved_top_20, strict=True):
        retrieved = [str(value) for value in raw["retrieved_top_k"]]
        full_retrieved = [str(value) for value in captured]
        relevant = [str(value) for value in raw["relevant"]]
        if len(full_retrieved) > CANDIDATE_LIMIT:
            raise RuntimeError(
                "captured retrieval exceeded the official candidate limit"
            )
        if full_retrieved[:K] != retrieved:
            raise RuntimeError(
                "captured top-20 prefix differs from official scorer top-10"
            )
        relevant_set = set(relevant)
        recomputed_mrr_at_20 = _reciprocal_rank_at_k(
            full_retrieved, relevant_set, CANDIDATE_LIMIT
        )
        upstream_mrr_at_20 = float(raw["mrr"])
        if not math.isclose(
            recomputed_mrr_at_20,
            upstream_mrr_at_20,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                "captured top-20 reciprocal rank differs from the official scorer"
            )
        if reverse_ids is not None:
            try:
                retrieved = [reverse_ids[value] for value in retrieved]
                full_retrieved = [reverse_ids[value] for value in full_retrieved]
                relevant = [reverse_ids[value] for value in relevant]
            except KeyError as exc:
                raise RuntimeError(
                    f"official result referenced an unknown runtime node id: {exc.args[0]}"
                ) from exc
        relevant_set = set(relevant)
        rows.append(
            {
                "query_id": str(raw["query_id"]),
                "retrieved_top_10": retrieved[:K],
                "retrieved_top_20": full_retrieved,
                "relevant": sorted(relevant_set),
                "reciprocal_rank_at_20": recomputed_mrr_at_20,
                "reciprocal_rank_at_10": _reciprocal_rank_at_k(
                    retrieved, relevant_set, K
                ),
                "search_time_ms": float(raw["search_time_ms"]),
            }
        )
    return rows


def _system_result(
    benchmark: Any,
    *,
    ingest_seconds: float,
    retrieved_top_20: Sequence[Sequence[str]],
    reverse_ids: Mapping[str, str] | None = None,
    hotpot_supporting: bool,
) -> dict[str, Any]:
    summary = benchmark.summary()
    if not summary or int(summary.get("total_queries", 0)) < 1:
        raise RuntimeError("official scorer produced no evaluated queries")
    query_rows = _normalize_query_rows(
        benchmark, retrieved_top_20=retrieved_top_20, reverse_ids=reverse_ids
    )
    mrr_at_20 = sum(row["reciprocal_rank_at_20"] for row in query_rows) / len(
        query_rows
    )
    if not math.isclose(mrr_at_20, float(summary["mrr"]), rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("recomputed MRR@20 differs from official scorer summary")
    mrr_at_10 = sum(row["reciprocal_rank_at_10"] for row in query_rows) / len(
        query_rows
    )
    result: dict[str, Any] = {
        "metrics": {
            "mrr_at_20": mrr_at_20,
            "mrr_at_10": mrr_at_10,
            "precision_at_10": float(summary["mean_precision@k"]),
            "recall_at_10": float(summary["mean_recall@k"]),
            "f1_at_10": float(summary["mean_f1@k"]),
            "ndcg_at_10": float(summary["mean_ndcg@k"]),
        },
        "evaluated_queries": len(query_rows),
        "ingest_seconds_observed": ingest_seconds,
        "mean_search_time_ms_observed": float(summary["mean_search_time_ms"]),
        "query_ids_ordered_sha256": canonical_json_sha256(
            [row["query_id"] for row in query_rows]
        ),
        "queries": query_rows,
    }
    if hotpot_supporting:
        hits = sum(
            len(set(row["retrieved_top_10"]) & set(row["relevant"]))
            for row in query_rows
        )
        total = sum(len(row["relevant"]) for row in query_rows)
        result["supporting_facts"] = {
            "hits_at_10": hits,
            "total": total,
            "micro_recall_at_10": hits / total if total else 0.0,
        }
    return result


class _SearchCaptureProxy:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, **kwargs: Any) -> Any:
        result = await self.graph.search(query, **kwargs)
        self.calls.append(
            {
                "query": query,
                "limit": kwargs.get("limit"),
                "retrieved": [str(item.node.id) for item in result.nodes],
            }
        )
        return result


async def _run_case(case: DatasetCase, repo: Path) -> dict[str, Any]:
    driver, _scorer, runtime = _load_upstream_driver(repo)
    prepared, selection = _prepare_case_data(driver, case)
    documents, truncation = _omnifuse_documents(prepared["corpus"])

    graph = None
    started = time.perf_counter()
    try:
        graph, id_map = await driver._build_graph(prepared["corpus"], no_embedding=True)
        synaptic_ingest = time.perf_counter() - started
        captured_search = _SearchCaptureProxy(graph)
        synaptic_benchmark = await driver._run_benchmark(
            case.name,
            captured_search,
            id_map,
            prepared["queries"],
            prepared["qrels"],
            max_queries=case.max_queries,
        )
    finally:
        if graph is not None:
            await graph.backend.close()

    reverse_ids = {
        str(node_id): str(corpus_id) for corpus_id, node_id in id_map.items()
    }
    if len(captured_search.calls) != len(synaptic_benchmark.queries):
        raise RuntimeError("official helper search calls differ from scored query rows")
    for capture, row in zip(
        captured_search.calls, synaptic_benchmark.queries, strict=True
    ):
        if capture["query"] != row["query"] or capture["limit"] != CANDIDATE_LIMIT:
            raise RuntimeError("official helper search capture contract changed")
    synaptic_top_20 = [capture["retrieved"] for capture in captured_search.calls]
    synaptic_result = _system_result(
        synaptic_benchmark,
        ingest_seconds=synaptic_ingest,
        retrieved_top_20=synaptic_top_20,
        reverse_ids=reverse_ids,
        hotpot_supporting=case.hotpot_supporting,
    )
    selected_query_ids = [str(row["query_id"]) for row in synaptic_benchmark.queries]
    if not selected_query_ids:
        raise RuntimeError("official direct runner selected no evaluable queries")

    started = time.perf_counter()
    omnifuse = build_inmemory([], [], documents, vector_k=CANDIDATE_LIMIT)
    omnifuse_ingest = time.perf_counter() - started
    omnifuse_benchmark = driver.BenchmarkResult()
    omnifuse_top_20: list[list[str]] = []
    indexed_ids = {document["id"] for document in documents}
    for raw_query in synaptic_benchmark.queries:
        query_id = str(raw_query["query_id"])
        query_text = str(raw_query["query"])
        raw_relevant = _require_mapping(
            prepared["qrels"].get(query_id, {}), f"qrels[{query_id!r}]"
        )
        relevant = {str(value) for value in raw_relevant if str(value) in indexed_ids}
        if not relevant:
            raise RuntimeError(
                f"OmniFuse lost the official runner's relevant set for query {query_id!r}"
            )
        search_started = time.time()
        ranked = [
            str(chunk.id)
            for chunk, _score in omnifuse.retrieve(query_text, limit=CANDIDATE_LIMIT)
        ]
        omnifuse_top_20.append(ranked)
        elapsed_ms = (time.time() - search_started) * 1000
        omnifuse_benchmark.add(
            query_id=query_id,
            query=query_text,
            retrieved=ranked,
            relevant=relevant,
            k=K,
            description=case.name,
            search_time_ms=elapsed_ms,
        )

    omnifuse_result = _system_result(
        omnifuse_benchmark,
        ingest_seconds=omnifuse_ingest,
        retrieved_top_20=omnifuse_top_20,
        hotpot_supporting=case.hotpot_supporting,
    )
    if (
        omnifuse_result["query_ids_ordered_sha256"]
        != synaptic_result["query_ids_ordered_sha256"]
    ):
        raise RuntimeError("system query selections differ")
    selection["scored_query_count"] = len(selected_query_ids)
    selection["scored_query_ids_ordered_sha256"] = canonical_json_sha256(
        selected_query_ids
    )
    metrics = (
        "mrr_at_20",
        "mrr_at_10",
        "precision_at_10",
        "recall_at_10",
        "f1_at_10",
        "ndcg_at_10",
    )
    winners = {}
    for metric in metrics:
        omni_value = omnifuse_result["metrics"][metric]
        synaptic_value = synaptic_result["metrics"][metric]
        winners[metric] = (
            "omnifuse"
            if omni_value > synaptic_value
            else "synaptic_memory"
            if synaptic_value > omni_value
            else "tie"
        )

    return {
        "case": {
            "id": case.id,
            "name": case.name,
            "filename": case.filename,
        },
        "selection": selection,
        "document_preprocessing": truncation,
        "systems": {
            "omnifuse": omnifuse_result,
            "synaptic_memory": synaptic_result,
        },
        "winners": winners,
        "runtime": {
            **runtime,
            "omnifuse_package": _require_runtime_path(
                omnifuse_package.__file__,
                SOURCE_ROOT / "omnifuse" / "__init__.py",
                "OmniFuse",
            ),
            "omnifuse_version": getattr(omnifuse_package, "__version__", None),
            "omnifuse_builder_source": _require_runtime_path(
                inspect.getsourcefile(build_inmemory) or "",
                SOURCE_ROOT / "omnifuse" / "facade.py",
                "OmniFuse build_inmemory",
            ),
        },
    }


def _worker_evidence(repo: Path, case: DatasetCase) -> dict[str, Any]:
    return {
        "official_tag_identity": _validate_tag_checkout(repo),
        "sources": _source_fingerprints(repo),
        "input": _input_fingerprint(repo, case),
        "environment_lock": _environment_lock_evidence(repo),
    }


def _run_worker(
    repo: Path,
    case: DatasetCase,
    result_path: Path,
    expected_python: Path,
    worker_run_id: str,
) -> None:
    ensure_output_absent(result_path)
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python.resolve():
        raise ProvenanceError(
            f"worker Python mismatch: {actual_python} != {expected_python.resolve()}"
        )
    environment_before = _process_environment_record(
        repo, expected_python, phase="startup"
    )
    evidence_before = _worker_evidence(repo, case)
    result = asyncio.run(_run_case(case, repo))
    environment_after = _process_environment_record(
        repo, expected_python, phase="runtime"
    )
    _validate_runtime_environment_transition(
        environment_before, environment_after, repo=repo
    )
    evidence_after = _worker_evidence(repo, case)
    tokenizer_runtime = _tokenizer_runtime_evidence(
        repo, evidence_after["environment_lock"]
    )
    _validate_tokenizer_runtime_record(
        tokenizer_runtime,
        repo=repo,
        python_prefix=Path(str(environment_after["python_prefix"])),
        environment_lock=evidence_after["environment_lock"],
    )
    assert_unchanged(
        "worker source and input evidence", evidence_before, evidence_after
    )
    worker_identity = capture_worker_identity(worker_run_id)
    payload = {
        "schema": WORKER_SCHEMA,
        "schema_version": WORKER_SCHEMA_VERSION,
        "provenance_level": PROVENANCE_LEVEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "worker_identity": worker_identity,
        "environment": {
            "before": environment_before,
            "after": environment_after,
            "tokenizer_runtime": tokenizer_runtime,
        },
        "contract": dict(WORKER_CONTRACT),
        "evidence": {"before": evidence_before, "after": evidence_after},
        "result": result,
    }
    write_json_once(result_path, payload)


def _worker_environment(repo: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(REQUIRED_WORKER_ENVIRONMENT)
    environment["PYTHONPATH"] = os.pathsep.join(_worker_pythonpath_entries(repo))
    return environment


def _probe_worker_environment(python: Path, repo: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(python.resolve()),
            str(SCRIPT_PATH),
            "--synaptic-repo",
            str(repo.resolve()),
            "--environment-probe",
            "--expected-python",
            str(python.resolve()),
        ],
        cwd=ROOT,
        env=_worker_environment(repo),
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ProvenanceError(
            f"worker environment probe failed with exit {completed.returncode}: {detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProvenanceError(
            f"worker environment probe returned invalid JSON: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != ENVIRONMENT_PROBE_SCHEMA
        or payload.get("schema_version") != ENVIRONMENT_PROBE_SCHEMA_VERSION
        or payload.get("provenance_level") != PROVENANCE_LEVEL
        or payload.get("status") != "ok"
    ):
        raise ProvenanceError("worker environment probe returned an invalid contract")
    process = payload.get("process")
    if not isinstance(process, dict):
        raise ProvenanceError("worker environment probe has no process evidence")
    _validate_process_environment_record(
        process, repo=repo, expected_python=python, phase="startup"
    )
    environment_lock = payload.get("environment_lock")
    if not isinstance(environment_lock, dict):
        raise ProvenanceError("worker environment probe has no lock evidence")
    _validate_environment_lock_record(environment_lock, repo)
    uv_sync_check = environment_lock["uv_sync_check"]
    if (
        Path(str(uv_sync_check["virtual_environment"])).resolve()
        != Path(str(process["python_prefix"])).resolve()
    ):
        raise ProvenanceError("worker uv sync check used an unexpected environment")
    tokenizer_runtime = payload.get("tokenizer_runtime")
    if not isinstance(tokenizer_runtime, dict):
        raise ProvenanceError("worker environment probe has no tokenizer evidence")
    _validate_tokenizer_runtime_record(
        tokenizer_runtime,
        repo=repo,
        python_prefix=Path(str(process["python_prefix"])),
        environment_lock=environment_lock,
    )
    return payload


def _worker_command(
    *,
    python: Path,
    repo: Path,
    case: DatasetCase,
    result_path: Path,
    worker_run_id: str,
) -> list[str]:
    return [
        str(python.resolve()),
        str(SCRIPT_PATH),
        "--synaptic-repo",
        str(repo.resolve()),
        "--worker-case",
        case.id,
        "--worker-result",
        str(result_path.resolve()),
        "--expected-python",
        str(python.resolve()),
        "--worker-run-id",
        worker_run_id,
    ]


def _strict_mapping(
    value: Any, *, label: str, keys: set[str] | frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ProvenanceError(f"{label} does not match the strict schema")
    return value


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProvenanceError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_number(
    value: Any,
    *,
    label: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ProvenanceError(f"{label} must be a finite number >= {minimum}")
    if maximum is not None and number > maximum:
        raise ProvenanceError(f"{label} must be <= {maximum}")
    return number


def _strict_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProvenanceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _assert_same_number(label: str, actual: Any, expected: float) -> None:
    number = _strict_number(actual, label=label, maximum=1.0)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-15):
        raise ProvenanceError(f"{label} differs from the independently recomputed value")


def _validate_worker_contract(value: Any, *, case: DatasetCase) -> None:
    if not isinstance(value, dict) or value != WORKER_CONTRACT:
        raise ProvenanceError(f"worker {case.id} contract differs from schema v4")


def _validate_selection(
    value: Any, *, case: DatasetCase, query_ids: Sequence[str]
) -> None:
    selection = _strict_mapping(
        value,
        label=f"worker {case.id} selection",
        keys={
            "seed",
            "original_corpus_count",
            "selected_corpus_count",
            "original_query_count",
            "eligible_query_count_before_max_queries",
            "max_queries",
            "klue_corpus_sample",
            "selected_corpus_ids_ordered_sha256",
            "eligible_query_ids_ordered_sha256",
            "scored_query_count",
            "scored_query_ids_ordered_sha256",
        },
    )
    if selection["seed"] != SAMPLE_SEED:
        raise ProvenanceError(f"worker {case.id} selection seed differs from contract")
    if selection["max_queries"] != case.max_queries:
        raise ProvenanceError(f"worker {case.id} max-query selection differs from case")
    if selection["klue_corpus_sample"] != case.klue_corpus_sample:
        raise ProvenanceError(f"worker {case.id} corpus sampling differs from case")

    original_corpus = _strict_int(
        selection["original_corpus_count"],
        label=f"worker {case.id} original corpus count",
        minimum=1,
    )
    selected_corpus = _strict_int(
        selection["selected_corpus_count"],
        label=f"worker {case.id} selected corpus count",
        minimum=1,
    )
    original_queries = _strict_int(
        selection["original_query_count"],
        label=f"worker {case.id} original query count",
        minimum=1,
    )
    eligible_queries = _strict_int(
        selection["eligible_query_count_before_max_queries"],
        label=f"worker {case.id} eligible query count",
        minimum=1,
    )
    if selected_corpus > original_corpus or eligible_queries > original_queries:
        raise ProvenanceError(f"worker {case.id} selection counts are inconsistent")
    expected_selected = (
        min(case.klue_corpus_sample, original_corpus)
        if case.klue_corpus_sample
        else original_corpus
    )
    if selected_corpus != expected_selected:
        raise ProvenanceError(f"worker {case.id} selected corpus count is inconsistent")
    scored_count = _strict_int(
        selection["scored_query_count"],
        label=f"worker {case.id} scored query count",
        minimum=1,
    )
    if scored_count != len(query_ids) or scored_count > eligible_queries:
        raise ProvenanceError(f"worker {case.id} scored query count is inconsistent")
    if case.max_queries and scored_count > case.max_queries:
        raise ProvenanceError(f"worker {case.id} exceeded its max-query selection")
    for key in (
        "selected_corpus_ids_ordered_sha256",
        "eligible_query_ids_ordered_sha256",
    ):
        _strict_sha256(selection[key], label=f"worker {case.id} {key}")
    expected_query_hash = canonical_json_sha256(list(query_ids))
    if selection["scored_query_ids_ordered_sha256"] != expected_query_hash:
        raise ProvenanceError(
            f"worker {case.id} scored query order hash is inconsistent"
        )


def _validate_document_preprocessing(
    value: Any, *, case: DatasetCase, selected_corpus_count: int
) -> None:
    preprocessing = _strict_mapping(
        value,
        label=f"worker {case.id} document preprocessing",
        keys={
            "text_character_limit",
            "input_documents",
            "indexed_documents",
            "skipped_empty_text_documents",
            "truncated_documents",
            "title_fallback_documents",
            "original_text_characters",
            "indexed_text_characters",
            "indexed_document_ids_ordered_sha256",
        },
    )
    if preprocessing["text_character_limit"] != TEXT_LIMIT:
        raise ProvenanceError(
            f"worker {case.id} document character limit differs from contract"
        )
    counts = {
        key: _strict_int(preprocessing[key], label=f"worker {case.id} {key}")
        for key in (
            "input_documents",
            "indexed_documents",
            "skipped_empty_text_documents",
            "truncated_documents",
            "title_fallback_documents",
            "original_text_characters",
            "indexed_text_characters",
        )
    }
    if counts["input_documents"] != selected_corpus_count:
        raise ProvenanceError(f"worker {case.id} preprocessing input count differs")
    if (
        counts["indexed_documents"] < 1
        or counts["indexed_documents"] + counts["skipped_empty_text_documents"]
        != counts["input_documents"]
        or counts["truncated_documents"] > counts["indexed_documents"]
        or counts["title_fallback_documents"] > counts["indexed_documents"]
        or counts["indexed_text_characters"] > counts["original_text_characters"]
    ):
        raise ProvenanceError(f"worker {case.id} preprocessing counts are inconsistent")
    _strict_sha256(
        preprocessing["indexed_document_ids_ordered_sha256"],
        label=f"worker {case.id} indexed document order hash",
    )


def _strict_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ProvenanceError(f"{label} must be a list of non-empty strings")
    return value


def _official_case_contract(
    *, case: DatasetCase, repo: Path, state: Mapping[str, Any]
) -> dict[str, Any]:
    inputs = state.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ProvenanceError("suite preflight has no dataset input evidence")
    expected_input = inputs.get(case.id)
    if not isinstance(expected_input, Mapping):
        raise ProvenanceError(f"suite preflight has no input for {case.id}")
    expected_relative = (
        Path("tests") / "benchmark" / "data" / case.filename
    ).as_posix()
    expected_fingerprint = {
        key: expected_input.get(key) for key in ("path", "sha256", "bytes")
    }
    if expected_fingerprint["path"] != expected_relative:
        raise ProvenanceError(f"suite preflight input path differs for {case.id}")
    dataset, actual_fingerprint = read_json_artifact(
        _dataset_path(repo, case), display_path=expected_relative
    )
    assert_unchanged(
        f"official dataset input {case.id}",
        expected_fingerprint,
        actual_fingerprint,
    )
    if not isinstance(dataset, dict):
        raise ProvenanceError(f"official dataset {case.id} is not a JSON object")
    corpus = dataset.get("corpus")
    queries = dataset.get("queries")
    qrels = dataset.get("qrels")
    if not all(isinstance(value, dict) for value in (corpus, queries, qrels)):
        raise ProvenanceError(f"official dataset {case.id} contract is incomplete")
    corpus = dict(corpus)
    queries = dict(queries)
    qrels = dict(qrels)
    original_corpus_count = len(corpus)
    original_query_count = len(queries)

    if case.klue_corpus_sample:
        sampled_items = random.Random(SAMPLE_SEED).sample(
            list(corpus.items()), min(case.klue_corpus_sample, len(corpus))
        )
        sampled_ids = {key for key, _value in sampled_items}
        corpus = {key: value for key, value in corpus.items() if key in sampled_ids}
        queries = {
            key: value
            for key, value in queries.items()
            if key.replace("klue_", "klue_doc_") in sampled_ids
        }
        qrels = {key: value for key, value in qrels.items() if key in queries}

    indexed_ids: set[str] = set()
    indexed_ids_ordered: list[str] = []
    skipped_empty = 0
    truncated = 0
    title_fallbacks = 0
    original_characters = 0
    indexed_characters = 0
    for corpus_id, raw_document in corpus.items():
        if not isinstance(raw_document, dict):
            raise ProvenanceError(
                f"official dataset {case.id} contains a non-object document"
            )
        title = raw_document.get("title", "")
        text = raw_document.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ProvenanceError(
                f"official dataset {case.id} contains non-string document text"
            )
        if not text:
            skipped_empty += 1
            continue
        document_id = str(corpus_id)
        indexed_ids.add(document_id)
        indexed_ids_ordered.append(document_id)
        original_characters += len(text)
        if len(text) > TEXT_LIMIT:
            text = text[:TEXT_LIMIT]
            truncated += 1
        indexed_characters += len(text)
        if not title:
            title_fallbacks += 1

    eligible_query_ids = [str(query_id) for query_id in queries if query_id in qrels]
    selected_query_ids = list(eligible_query_ids)
    if case.max_queries and len(selected_query_ids) > case.max_queries:
        selected_query_ids = random.Random(SAMPLE_SEED).sample(
            selected_query_ids, case.max_queries
        )
    expected_queries: list[dict[str, Any]] = []
    for query_id in selected_query_ids:
        raw_relevant = qrels.get(query_id)
        if not isinstance(raw_relevant, dict):
            raise ProvenanceError(
                f"official dataset {case.id} has invalid qrels for {query_id!r}"
            )
        relevant = sorted(
            {str(document_id) for document_id in raw_relevant if str(document_id) in indexed_ids}
        )
        if relevant:
            expected_queries.append({"query_id": query_id, "relevant": relevant})
    if not expected_queries:
        raise ProvenanceError(f"official dataset {case.id} has no evaluable queries")
    scored_query_ids = [row["query_id"] for row in expected_queries]
    return {
        "queries": expected_queries,
        "selection": {
            "seed": SAMPLE_SEED,
            "original_corpus_count": original_corpus_count,
            "selected_corpus_count": len(corpus),
            "original_query_count": original_query_count,
            "eligible_query_count_before_max_queries": len(eligible_query_ids),
            "max_queries": case.max_queries,
            "klue_corpus_sample": case.klue_corpus_sample,
            "selected_corpus_ids_ordered_sha256": canonical_json_sha256(list(corpus)),
            "eligible_query_ids_ordered_sha256": canonical_json_sha256(
                eligible_query_ids
            ),
            "scored_query_count": len(scored_query_ids),
            "scored_query_ids_ordered_sha256": canonical_json_sha256(
                scored_query_ids
            ),
        },
        "document_preprocessing": {
            "text_character_limit": TEXT_LIMIT,
            "input_documents": len(corpus),
            "indexed_documents": len(indexed_ids_ordered),
            "skipped_empty_text_documents": skipped_empty,
            "truncated_documents": truncated,
            "title_fallback_documents": title_fallbacks,
            "original_text_characters": original_characters,
            "indexed_text_characters": indexed_characters,
            "indexed_document_ids_ordered_sha256": canonical_json_sha256(
                indexed_ids_ordered
            ),
        },
    }


def _validate_query_row(value: Any, *, case: DatasetCase, system: str) -> dict[str, Any]:
    label = f"worker {case.id} {system} query row"
    row = _strict_mapping(
        value,
        label=label,
        keys={
            "query_id",
            "retrieved_top_10",
            "retrieved_top_20",
            "relevant",
            "reciprocal_rank_at_20",
            "reciprocal_rank_at_10",
            "search_time_ms",
        },
    )
    query_id = row["query_id"]
    if not isinstance(query_id, str) or not query_id:
        raise ProvenanceError(f"{label} has an invalid query ID")
    top_10 = _strict_string_list(row["retrieved_top_10"], label=f"{label} top-10")
    top_20 = _strict_string_list(row["retrieved_top_20"], label=f"{label} top-20")
    relevant = _strict_string_list(row["relevant"], label=f"{label} relevant")
    if len(top_10) > K or len(top_20) > CANDIDATE_LIMIT:
        raise ProvenanceError(f"{label} exceeds the retrieval contract")
    if top_10 != top_20[:K]:
        raise ProvenanceError(f"{label} top-10 is not the top-20 prefix")
    if len(top_20) != len(set(top_20)):
        raise ProvenanceError(f"{label} contains duplicate retrieved IDs")
    if relevant != sorted(set(relevant)):
        raise ProvenanceError(f"{label} relevant IDs are not sorted and unique")
    if not relevant:
        raise ProvenanceError(f"{label} relevant IDs must not be empty")
    relevant_set = set(relevant)
    rr_20 = _reciprocal_rank_at_k(top_20, relevant_set, CANDIDATE_LIMIT)
    rr_10 = _reciprocal_rank_at_k(top_10, relevant_set, K)
    _assert_same_number(f"{label} reciprocal rank at 20", row["reciprocal_rank_at_20"], rr_20)
    _assert_same_number(f"{label} reciprocal rank at 10", row["reciprocal_rank_at_10"], rr_10)
    search_time = _strict_number(
        row["search_time_ms"], label=f"{label} search time"
    )
    return {
        "query_id": query_id,
        "retrieved_top_10": top_10,
        "retrieved_top_20": top_20,
        "relevant": relevant,
        "rr_20": rr_20,
        "rr_10": rr_10,
        "search_time_ms": search_time,
    }


def _recompute_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    count = len(rows)
    return {
        "mrr_at_20": sum(float(row["rr_20"]) for row in rows) / count,
        "mrr_at_10": sum(float(row["rr_10"]) for row in rows) / count,
        "precision_at_10": sum(
            precision_at_k(
                list(row["retrieved_top_10"]), set(row["relevant"]), K
            )
            for row in rows
        )
        / count,
        "recall_at_10": sum(
            recall_at_k(list(row["retrieved_top_10"]), set(row["relevant"]), K)
            for row in rows
        )
        / count,
        "f1_at_10": sum(
            f1_at_k(list(row["retrieved_top_10"]), set(row["relevant"]), K)
            for row in rows
        )
        / count,
        "ndcg_at_10": sum(
            ndcg_at_k(list(row["retrieved_top_10"]), set(row["relevant"]), K)
            for row in rows
        )
        / count,
    }


def _validate_system_result(
    value: Any, *, case: DatasetCase, system: str
) -> dict[str, Any]:
    expected_keys = {
        "metrics",
        "evaluated_queries",
        "ingest_seconds_observed",
        "mean_search_time_ms_observed",
        "query_ids_ordered_sha256",
        "queries",
    }
    if case.hotpot_supporting:
        expected_keys.add("supporting_facts")
    result = _strict_mapping(
        value,
        label=f"worker {case.id} {system} result",
        keys=expected_keys,
    )
    queries = result["queries"]
    if not isinstance(queries, list) or not queries:
        raise ProvenanceError(f"worker {case.id} {system} has no query rows")
    rows = [
        _validate_query_row(row, case=case, system=system) for row in queries
    ]
    query_ids = [str(row["query_id"]) for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ProvenanceError(f"worker {case.id} {system} has duplicate query IDs")
    evaluated = _strict_int(
        result["evaluated_queries"],
        label=f"worker {case.id} {system} evaluated query count",
        minimum=1,
    )
    if evaluated != len(rows):
        raise ProvenanceError(
            f"worker {case.id} {system} evaluated query count is inconsistent"
        )
    expected_hash = canonical_json_sha256(query_ids)
    if result["query_ids_ordered_sha256"] != expected_hash:
        raise ProvenanceError(
            f"worker {case.id} {system} query order hash is inconsistent"
        )
    _strict_number(
        result["ingest_seconds_observed"],
        label=f"worker {case.id} {system} ingest time",
    )
    expected_mean_search = sum(float(row["search_time_ms"]) for row in rows) / len(
        rows
    )
    mean_search = _strict_number(
        result["mean_search_time_ms_observed"],
        label=f"worker {case.id} {system} mean search time",
    )
    if not math.isclose(
        mean_search, expected_mean_search, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ProvenanceError(
            f"worker {case.id} {system} mean search time is inconsistent"
        )
    metrics = _strict_mapping(
        result["metrics"],
        label=f"worker {case.id} {system} metrics",
        keys=set(METRIC_NAMES),
    )
    recomputed_metrics = _recompute_metrics(rows)
    for metric, expected in recomputed_metrics.items():
        _assert_same_number(
            f"worker {case.id} {system} metric {metric}", metrics[metric], expected
        )

    if case.hotpot_supporting:
        supporting = _strict_mapping(
            result["supporting_facts"],
            label=f"worker {case.id} {system} supporting facts",
            keys={"hits_at_10", "total", "micro_recall_at_10"},
        )
        hits = sum(
            len(set(row["retrieved_top_10"]) & set(row["relevant"]))
            for row in rows
        )
        total = sum(len(row["relevant"]) for row in rows)
        if (
            _strict_int(
                supporting["hits_at_10"],
                label=f"worker {case.id} {system} supporting hits",
            )
            != hits
            or _strict_int(
                supporting["total"],
                label=f"worker {case.id} {system} supporting total",
            )
            != total
        ):
            raise ProvenanceError(
                f"worker {case.id} {system} supporting fact counts are inconsistent"
            )
        _assert_same_number(
            f"worker {case.id} {system} supporting micro recall",
            supporting["micro_recall_at_10"],
            hits / total if total else 0.0,
        )
    return {
        "query_ids": query_ids,
        "relevant": [row["relevant"] for row in rows],
        "metrics": recomputed_metrics,
    }


def _validate_result_runtime(
    value: Any, *, case: DatasetCase, python: Path, repo: Path
) -> None:
    runtime = _strict_mapping(
        value,
        label=f"worker {case.id} runtime",
        keys={
            "python_executable",
            "synaptic_package",
            "synaptic_version",
            "upstream_driver",
            "upstream_scorer",
            "omnifuse_package",
            "omnifuse_version",
            "omnifuse_builder_source",
        },
    )
    expected_paths = {
        "python_executable": python,
        "synaptic_package": repo / UPSTREAM_PACKAGE_RELATIVE,
        "upstream_driver": repo / UPSTREAM_DRIVER_RELATIVE,
        "upstream_scorer": repo / UPSTREAM_SCORER_RELATIVE,
        "omnifuse_package": SOURCE_ROOT / "omnifuse" / "__init__.py",
        "omnifuse_builder_source": SOURCE_ROOT / "omnifuse" / "facade.py",
    }
    for key, expected in expected_paths.items():
        _require_runtime_path(str(runtime[key]), expected, f"worker {case.id} {key}")
    if runtime["synaptic_version"] != EXPECTED_TAG.removeprefix("v"):
        raise ProvenanceError(f"worker {case.id} synaptic version differs from tag")
    if runtime["omnifuse_version"] != getattr(omnifuse_package, "__version__", None):
        raise ProvenanceError(f"worker {case.id} OmniFuse version differs from runtime")


def _validate_worker_result(
    value: Any,
    *,
    case: DatasetCase,
    python: Path,
    repo: Path,
    official_contract: Mapping[str, Any] | None = None,
) -> None:
    result = _strict_mapping(
        value,
        label=f"worker {case.id} benchmark result",
        keys={
            "case",
            "selection",
            "document_preprocessing",
            "systems",
            "winners",
            "runtime",
        },
    )
    expected_case = {"id": case.id, "name": case.name, "filename": case.filename}
    if result["case"] != expected_case:
        raise ProvenanceError(f"worker {case.id} returned the wrong case")
    systems = _strict_mapping(
        result["systems"],
        label=f"worker {case.id} systems",
        keys={"omnifuse", "synaptic_memory"},
    )
    validated = {
        system: _validate_system_result(systems[system], case=case, system=system)
        for system in ("omnifuse", "synaptic_memory")
    }
    omni = validated["omnifuse"]
    synaptic = validated["synaptic_memory"]
    if omni["query_ids"] != synaptic["query_ids"]:
        raise ProvenanceError(f"worker {case.id} system query selections differ")
    if omni["relevant"] != synaptic["relevant"]:
        raise ProvenanceError(f"worker {case.id} system relevant judgments differ")
    if official_contract is not None:
        observed_queries = [
            {"query_id": query_id, "relevant": relevant}
            for query_id, relevant in zip(
                omni["query_ids"], omni["relevant"], strict=True
            )
        ]
        if observed_queries != official_contract.get("queries"):
            raise ProvenanceError(
                f"worker {case.id} query order or relevant judgments differ "
                "from the official input"
            )
    _validate_selection(result["selection"], case=case, query_ids=omni["query_ids"])
    selected_corpus_count = int(result["selection"]["selected_corpus_count"])
    _validate_document_preprocessing(
        result["document_preprocessing"],
        case=case,
        selected_corpus_count=selected_corpus_count,
    )
    if official_contract is not None:
        if result["selection"] != official_contract.get("selection"):
            raise ProvenanceError(
                f"worker {case.id} selection differs from the official input"
            )
        if result["document_preprocessing"] != official_contract.get(
            "document_preprocessing"
        ):
            raise ProvenanceError(
                f"worker {case.id} preprocessing differs from the official input"
            )
    winners = _strict_mapping(
        result["winners"],
        label=f"worker {case.id} winners",
        keys=set(METRIC_NAMES),
    )
    for metric in METRIC_NAMES:
        omni_value = omni["metrics"][metric]
        synaptic_value = synaptic["metrics"][metric]
        expected_winner = (
            "omnifuse"
            if omni_value > synaptic_value
            else "synaptic_memory"
            if synaptic_value > omni_value
            else "tie"
        )
        if winners[metric] != expected_winner:
            raise ProvenanceError(
                f"worker {case.id} winner for {metric} is inconsistent"
            )
    _validate_result_runtime(result["runtime"], case=case, python=python, repo=repo)


def _validate_worker_payload(
    payload: Any,
    *,
    case: DatasetCase,
    python: Path,
    repo: Path,
    state: Mapping[str, Any],
    expected_worker_run_id: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProvenanceError(f"worker {case.id} result is not a JSON object")
    if set(payload) != {
        "schema",
        "schema_version",
        "provenance_level",
        "generated_at",
        "status",
        "worker_identity",
        "environment",
        "contract",
        "evidence",
        "result",
    }:
        raise ProvenanceError(
            f"worker {case.id} result does not match the strict schema"
        )
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("schema_version") != WORKER_SCHEMA_VERSION
    ):
        raise ProvenanceError(f"worker {case.id} result has an unsupported schema")
    if payload.get("status") != "ok":
        raise ProvenanceError(f"worker {case.id} did not complete successfully")
    if payload.get("provenance_level") != PROVENANCE_LEVEL:
        raise ProvenanceError(f"worker {case.id} provenance level differs from suite")
    if not isinstance(payload.get("generated_at"), str) or not payload["generated_at"]:
        raise ProvenanceError(f"worker {case.id} has no generation timestamp")
    validate_worker_identity(
        payload.get("worker_identity"),
        expected_run_id=expected_worker_run_id,
        label=f"worker {case.id} identity",
    )
    _validate_worker_contract(payload.get("contract"), case=case)
    official_contract = _official_case_contract(case=case, repo=repo, state=state)
    _validate_worker_result(
        payload.get("result"),
        case=case,
        python=python,
        repo=repo,
        official_contract=official_contract,
    )
    environment = _strict_mapping(
        payload.get("environment"),
        label=f"worker {case.id} process environment evidence",
        keys={"before", "after", "tokenizer_runtime"},
    )
    environment_before = environment.get("before")
    environment_after = environment.get("after")
    if not isinstance(environment_before, dict) or not isinstance(
        environment_after, dict
    ):
        raise ProvenanceError(f"worker {case.id} process evidence is incomplete")
    _validate_process_environment_record(
        environment_before, repo=repo, expected_python=python, phase="startup"
    )
    _validate_process_environment_record(
        environment_after, repo=repo, expected_python=python, phase="runtime"
    )
    _validate_runtime_environment_transition(
        environment_before, environment_after, repo=repo
    )
    expected_probe = state.get("worker_environment")
    if not isinstance(expected_probe, dict):
        raise ProvenanceError("suite preflight has no worker environment evidence")
    assert_unchanged(
        f"worker {case.id} startup process environment",
        expected_probe.get("process"),
        environment_before,
    )
    tokenizer_runtime = environment.get("tokenizer_runtime")
    if not isinstance(tokenizer_runtime, dict):
        raise ProvenanceError(f"worker {case.id} has no tokenizer runtime evidence")
    expected_tokenizer = expected_probe.get("tokenizer_runtime")
    if not isinstance(expected_tokenizer, dict):
        raise ProvenanceError("suite preflight has no tokenizer runtime evidence")
    evidence = _strict_mapping(
        payload.get("evidence"),
        label=f"worker {case.id} source/input evidence",
        keys={"before", "after"},
    )
    evidence_before = evidence.get("before")
    evidence_after = evidence.get("after")
    if not isinstance(evidence_before, dict) or not isinstance(evidence_after, dict):
        raise ProvenanceError(f"worker {case.id} source/input evidence is incomplete")
    assert_unchanged(
        f"worker {case.id} source/input evidence", evidence_before, evidence_after
    )
    expected_evidence = {
        "official_tag_identity": state["identity"],
        "sources": state["sources"],
        "input": state["inputs"][case.id],
        "environment_lock": expected_probe["environment_lock"],
    }
    assert_unchanged(
        f"worker {case.id} evidence versus suite preflight",
        expected_evidence,
        evidence_before,
    )
    _validate_environment_lock_record(evidence_before["environment_lock"], repo)
    uv_sync_check = evidence_before["environment_lock"]["uv_sync_check"]
    if (
        Path(str(uv_sync_check["virtual_environment"])).resolve()
        != Path(str(environment_before["python_prefix"])).resolve()
    ):
        raise ProvenanceError(f"worker {case.id} uv check used the wrong environment")
    _validate_tokenizer_runtime_record(
        tokenizer_runtime,
        repo=repo,
        python_prefix=Path(str(environment_after["python_prefix"])),
        environment_lock=evidence_before["environment_lock"],
    )
    assert_unchanged(
        f"worker {case.id} tokenizer versus suite preflight",
        expected_tokenizer,
        tokenizer_runtime,
    )
    return payload


def _worker_directory(output: Path, configured: Path | None) -> Path:
    if configured is not None:
        return configured.resolve()
    return default_worker_directory(ROOT, output, kind="direct-external")


def _summary(workers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wins = {
        metric: {"omnifuse": 0, "synaptic_memory": 0, "tie": 0}
        for metric in METRIC_NAMES
    }
    macro = {
        metric: {"omnifuse": 0.0, "synaptic_memory": 0.0}
        for metric in METRIC_NAMES
    }
    for worker in workers:
        result = worker["payload"]["result"]
        for metric in METRIC_NAMES:
            wins[metric][result["winners"][metric]] += 1
            for system in ("omnifuse", "synaptic_memory"):
                macro[metric][system] += result["systems"][system]["metrics"][metric]
    for values in macro.values():
        for system in values:
            values[system] /= len(workers)
    return {
        "required_cases": len(CASES),
        "completed_cases": len(workers),
        "failed_cases": 0,
        "wins": wins,
        "macro_average": macro,
    }


def _run_suite(
    *,
    repo: Path,
    python: Path,
    doctor_path: Path,
    output: Path,
    workers_dir: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    python = python.resolve()
    doctor_path = doctor_path.resolve()
    output = output.resolve()
    workers_dir = workers_dir.resolve()
    if not python.is_file():
        raise ProvenanceError(f"synaptic tag Python executable not found: {python}")
    state = _preflight(
        repo=repo,
        python=python,
        doctor_path=doctor_path,
        output=output,
        workers_dir=workers_dir,
    )
    workers_dir.mkdir(parents=True, exist_ok=False)
    workers: list[dict[str, Any]] = []
    environment = _worker_environment(repo)
    for case in CASES:
        worker_run_id = new_worker_run_id()
        if any(worker["worker_run_id"] == worker_run_id for worker in workers):
            raise ProvenanceError("controller generated a duplicate worker run ID")
        result_path = workers_dir / f"{case.id}.json"
        ensure_output_absent(result_path)
        started = time.perf_counter()
        completed, launcher_pid = run_with_launcher_pid(
            _worker_command(
                python=python,
                repo=repo,
                case=case,
                result_path=result_path,
                worker_run_id=worker_run_id,
            ),
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - started
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ProvenanceError(
                f"isolated worker {case.id} failed with exit {completed.returncode}: {detail}"
            )
        payload, artifact = read_json_artifact(result_path)
        payload = _validate_worker_payload(
            payload,
            case=case,
            python=python,
            repo=repo,
            state=state,
            expected_worker_run_id=worker_run_id,
        )
        worker_identity = payload["worker_identity"]
        workers.append(
            {
                "case_id": case.id,
                "worker_run_id": worker_identity["worker_run_id"],
                "launcher_pid": launcher_pid,
                "worker_pid": worker_identity["worker_pid"],
                "same_process_id": launcher_pid == worker_identity["worker_pid"],
                "elapsed_seconds": elapsed,
                "artifact": artifact,
                "payload": payload,
            }
        )

    integrity = _postflight(repo, python, state)
    process_summary = worker_process_summary(workers, expected_count=len(CASES))
    integrity["checks"]["worker_run_ids_unique"] = True
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "provenance_level": PROVENANCE_LEVEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "scope": {
            "official_upstream_test_cases": 14,
            "case_ids": [case.id for case in CASES],
            "official_tag": EXPECTED_TAG,
            "official_tag_sha": EXPECTED_TAG_SHA,
            "mode": "deterministic FTS-only companion",
        },
        "environment": {
            "orchestrator_python": str(Path(sys.executable).resolve()),
            "worker_python": str(python),
            "platform": platform.platform(),
            "worker_variables": {
                key: environment[key]
                for key in (
                    "PYTHONUTF8",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONHASHSEED",
                    "PYTHONNOUSERSITE",
                )
            },
        },
        "contract": {
            "upstream_build": "_build_graph(..., no_embedding=True)",
            "upstream_query_runner": "_run_benchmark",
            "dataset_specific_sampling": "official test_external_datasets.py semantics",
            "k": K,
            "candidate_limit": CANDIDATE_LIMIT,
            "text_character_limit": TEXT_LIMIT,
            "sampling_seed": SAMPLE_SEED,
            "hotpot_supporting_recall": "micro recall over stored top-10 retrievals",
            "mrr_at_20": (
                "upstream reported MRR independently reproduced over captured "
                "retrieved_top_20"
            ),
            "mrr_at_10": "separately recomputed over captured retrieved_top_10",
            "precision_denominator": (
                "official scorer divides by the number returned in top 10, not always 10"
            ),
            "worker_model": "one fresh official-tag Python process per dataset case",
            "worker_source_mode": "official-tag source imported via isolated PYTHONPATH",
            "worker_lockfile_sha256": UPSTREAM_LOCK_SHA256,
            "worker_distribution_policy": (
                "uv sync --active --frozen --no-install-project --no-dev with the "
                "sqlite/embedding/korean/dev extras must report an exact no-change "
                "environment; installed distributions are also checked against uv.lock"
            ),
            "worker_tokenizer_policy": (
                "official sqlite.py plus active Kiwi, kiwipiepy, and kiwipiepy-model "
                "runtime evidence is required; regex fallback is forbidden"
            ),
            "failure_policy": "any failed case prevents suite artifact publication",
            "artifact_policy": "worker and suite JSON files are atomic write-once",
            "timing_caveat": (
                "ingest and search timings are observational direct-test timings, not the "
                "separately controlled performance benchmark"
            ),
        },
        "doctor_manifest": state["doctor"],
        "doctor_links": state["doctor_links"],
        "integrity": integrity,
        "workers_directory": str(workers_dir),
        "workers": workers,
        "worker_processes": [
            {
                key: worker[key]
                for key in (
                    "case_id",
                    "worker_run_id",
                    "launcher_pid",
                    "worker_pid",
                    "same_process_id",
                )
            }
            for worker in workers
        ],
        "worker_process_summary": process_summary,
        "summary": _summary(workers),
    }
    for worker in workers:
        assert_artifact_unchanged(
            f"worker artifact {worker['case_id']}",
            Path(worker["artifact"]["path"]),
            worker["artifact"],
        )
    integrity["checks"]["worker_artifacts_unchanged_before_publication"] = True
    write_json_once(output, report)
    return report


def _print_report(report: Mapping[str, Any]) -> None:
    print(f"official direct FTS-only cases: {report['summary']['completed_cases']}/14")
    print(f"{'case':24}{'syn_mrr@10':>12}{'omni_mrr@10':>13}{'winner':>18}")
    for worker in report["workers"]:
        result = worker["payload"]["result"]
        systems = result["systems"]
        print(
            f"{result['case']['id']:24}"
            f"{systems['synaptic_memory']['metrics']['mrr_at_10']:>12.4f}"
            f"{systems['omnifuse']['metrics']['mrr_at_10']:>13.4f}"
            f"{result['winners']['mrr_at_10']:>18}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptic-repo", type=Path)
    parser.add_argument("--synaptic-python", type=Path)
    parser.add_argument("--doctor-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers-dir", type=Path)
    parser.add_argument(
        "--worker-case", choices=tuple(CASE_BY_ID), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--environment-probe", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-python", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.environment_probe:
        missing = [
            flag
            for flag, value in (
                ("--synaptic-repo", args.synaptic_repo),
                ("--expected-python", args.expected_python),
            )
            if value is None
        ]
        if missing:
            parser.error(f"environment probe requires {', '.join(missing)}")
        try:
            payload = _environment_probe_payload(
                args.synaptic_repo.resolve(), args.expected_python.resolve()
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return 0
    if args.worker_case is not None:
        missing = [
            flag
            for flag, value in (
                ("--synaptic-repo", args.synaptic_repo),
                ("--worker-result", args.worker_result),
                ("--expected-python", args.expected_python),
                ("--worker-run-id", args.worker_run_id),
            )
            if value is None
        ]
        if missing:
            parser.error(f"worker mode requires {', '.join(missing)}")
        try:
            _run_worker(
                args.synaptic_repo.resolve(),
                CASE_BY_ID[args.worker_case],
                args.worker_result.resolve(),
                args.expected_python.resolve(),
                args.worker_run_id,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        return 0

    missing = [
        flag
        for flag, value in (
            ("--synaptic-repo", args.synaptic_repo),
            ("--synaptic-python", args.synaptic_python),
            ("--doctor-manifest", args.doctor_manifest),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error(f"suite mode requires {', '.join(missing)}")
    workers_dir = _worker_directory(args.out, args.workers_dir)
    try:
        report = _run_suite(
            repo=args.synaptic_repo,
            python=args.synaptic_python,
            doctor_path=args.doctor_manifest,
            output=args.out,
            workers_dir=workers_dir,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
