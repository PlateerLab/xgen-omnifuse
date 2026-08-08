"""Shared provenance primitives for benchmark manifests and result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DOCTOR_SCHEMA = "omnifuse.eval.doctor"
DOCTOR_SCHEMA_VERSION = 1
WORKER_IDENTITY_SCHEMA = "omnifuse.eval.worker-instance"
WORKER_IDENTITY_SCHEMA_VERSION = 1
WORKER_IDENTITY_CAPTURE_PHASE = "post_measurement"
_WORKER_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}")
STRICT_PUBLIC_TARGET_CONTRACTS = {
    "finreg_single": (
        "omnifuse",
        (
            ("corpus", "eval/data/finreg/raw.jsonl"),
            ("queries", "eval/data/queries/finreg.json"),
        ),
    ),
    "finreg_multi": (
        "omnifuse",
        (
            ("corpus", "eval/data/finreg/raw.jsonl"),
            ("queries", "eval/data/queries/finreg_multihop.json"),
        ),
    ),
    "hotpotqa_24": (
        "synaptic",
        (("dataset", "tests/benchmark/data/hotpotqa_24.json"),),
    ),
    "hotpotqa_200": (
        "synaptic",
        (("dataset", "tests/benchmark/data/hotpotqa.json"),),
    ),
    "allganize_rag_ko": (
        "synaptic",
        (("dataset", "tests/benchmark/data/allganize_rag_ko.json"),),
    ),
    "allganize_rag_eval": (
        "synaptic",
        (("dataset", "tests/benchmark/data/allganize_rag_eval.json"),),
    ),
    "publichealthqa_ko": (
        "synaptic",
        (("dataset", "tests/benchmark/data/publichealthqa_ko.json"),),
    ),
    "autorag_retrieval": (
        "synaptic",
        (("dataset", "tests/benchmark/data/autorag_retrieval.json"),),
    ),
    "klue_mrc": (
        "synaptic",
        (("dataset", "tests/benchmark/data/klue_mrc.json"),),
    ),
    "ko_strategyqa": (
        "synaptic",
        (("dataset", "tests/benchmark/data/ko_strategyqa.json"),),
    ),
    "2wiki_dev": (
        "synaptic",
        (("dataset", "tests/benchmark/data/2wiki_dev.json"),),
    ),
    "musique_dev": (
        "synaptic",
        (("dataset", "tests/benchmark/data/musique_dev.json"),),
    ),
    "trec_covid": (
        "synaptic",
        (("dataset", "tests/benchmark/data/trec_covid.json"),),
    ),
    "scifact": (
        "synaptic",
        (("dataset", "tests/benchmark/data/scifact.json"),),
    ),
    "xpqa_ko": (
        "synaptic",
        (("dataset", "tests/benchmark/data/xpqa_ko.json"),),
    ),
    "nfcorpus": (
        "synaptic",
        (("dataset", "tests/benchmark/data/nfcorpus.json"),),
    ),
    "miracl_retrieval_ko": (
        "synaptic",
        (("dataset", "tests/benchmark/data/miracl_retrieval_ko.json"),),
    ),
    "fiqa": (
        "synaptic",
        (("dataset", "tests/benchmark/data/fiqa.json"),),
    ),
    "multilongdoc_ko": (
        "synaptic",
        (("dataset", "tests/benchmark/data/multilongdoc_ko.json"),),
    ),
}
STRICT_PUBLIC_TARGET_IDS = frozenset(STRICT_PUBLIC_TARGET_CONTRACTS)


class ProvenanceError(RuntimeError):
    """Raised when benchmark evidence is incomplete, mutable, or inconsistent."""


class OutputExistsError(ProvenanceError):
    """Raised when a write-once result destination already exists."""


def new_worker_run_id() -> str:
    """Create a fixed-width controller launch identity without worker-side imports."""
    import uuid

    return uuid.uuid4().hex


def validate_worker_run_id(value: object, *, label: str = "worker_run_id") -> str:
    if not isinstance(value, str) or _WORKER_RUN_ID_PATTERN.fullmatch(value) is None:
        raise ProvenanceError(f"{label} must be a lowercase UUIDv4 hex value")
    return value


def capture_worker_identity(worker_run_id: object) -> dict[str, Any]:
    """Capture process identity only after the caller's measurement has finished."""
    return {
        "schema": WORKER_IDENTITY_SCHEMA,
        "schema_version": WORKER_IDENTITY_SCHEMA_VERSION,
        "worker_run_id": validate_worker_run_id(worker_run_id),
        "worker_pid": os.getpid(),
        "capture_phase": WORKER_IDENTITY_CAPTURE_PHASE,
    }


def validate_worker_identity(
    value: object,
    *,
    expected_run_id: object | None = None,
    label: str = "worker identity",
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "schema_version",
        "worker_run_id",
        "worker_pid",
        "capture_phase",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ProvenanceError(f"{label} must match the strict schema")
    if (
        value["schema"] != WORKER_IDENTITY_SCHEMA
        or value["schema_version"] != WORKER_IDENTITY_SCHEMA_VERSION
        or value["capture_phase"] != WORKER_IDENTITY_CAPTURE_PHASE
    ):
        raise ProvenanceError(f"{label} contract is invalid")
    run_id = validate_worker_run_id(value["worker_run_id"], label=f"{label} run ID")
    if expected_run_id is not None and run_id != validate_worker_run_id(
        expected_run_id, label=f"expected {label} run ID"
    ):
        raise ProvenanceError(f"{label} run ID does not match the controller launch")
    worker_pid = value["worker_pid"]
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid < 1
    ):
        raise ProvenanceError(f"{label} PID must be a positive integer")
    return dict(value)


def run_with_launcher_pid(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    capture_output: bool = False,
    encoding: str | None = None,
    errors: str | None = None,
) -> tuple[subprocess.CompletedProcess[Any], int]:
    """Run one worker and retain the launcher's PID independently of worker PID."""
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        encoding=encoding,
        errors=errors,
    )
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(
        list(command), process.returncode, stdout=stdout, stderr=stderr
    )
    if check:
        completed.check_returncode()
    return completed, process.pid


def worker_process_summary(
    records: Sequence[Mapping[str, Any]], *, expected_count: int
) -> dict[str, Any]:
    """Validate one launch record per worker while treating PID reuse as observation."""
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ProvenanceError("expected worker count must be an integer")
    if expected_count < 1 or len(records) != expected_count:
        raise ProvenanceError(
            f"worker process evidence count differs: {len(records)} != {expected_count}"
        )
    run_ids: list[str] = []
    launcher_pids: list[int] = []
    worker_pids: list[int] = []
    matches = 0
    for index, record in enumerate(records, start=1):
        run_id = validate_worker_run_id(
            record.get("worker_run_id"), label=f"worker process {index} run ID"
        )
        launcher_pid = record.get("launcher_pid")
        worker_pid = record.get("worker_pid")
        if (
            isinstance(launcher_pid, bool)
            or not isinstance(launcher_pid, int)
            or launcher_pid < 1
        ):
            raise ProvenanceError(f"worker process {index} launcher PID is invalid")
        if (
            isinstance(worker_pid, bool)
            or not isinstance(worker_pid, int)
            or worker_pid < 1
        ):
            raise ProvenanceError(f"worker process {index} worker PID is invalid")
        same_process_id = record.get("same_process_id")
        if not isinstance(same_process_id, bool) or same_process_id != (
            launcher_pid == worker_pid
        ):
            raise ProvenanceError(
                f"worker process {index} PID relationship is inconsistent"
            )
        run_ids.append(run_id)
        launcher_pids.append(launcher_pid)
        worker_pids.append(worker_pid)
        matches += int(same_process_id)
    if len(set(run_ids)) != len(run_ids):
        raise ProvenanceError(
            "worker run IDs must be unique across controller launches"
        )
    distinct_launcher_pids = len(set(launcher_pids))
    distinct_worker_pids = len(set(worker_pids))
    return {
        "expected_workers": expected_count,
        "completed_workers": len(records),
        "distinct_worker_run_ids": len(set(run_ids)),
        "worker_run_ids_unique": True,
        "distinct_launcher_pids": distinct_launcher_pids,
        "distinct_worker_pids": distinct_worker_pids,
        "launcher_pid_reuse_observed": distinct_launcher_pids != len(records),
        "worker_pid_reuse_observed": distinct_worker_pids != len(records),
        "launcher_worker_pid_match_count": matches,
        "launcher_worker_pid_mismatch_count": len(records) - matches,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ProvenanceError(f"required file does not exist: {resolved}")
    return {
        "path": display_path or str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def read_bytes_artifact(
    path: Path, *, display_path: str | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Read an artifact once and bind its fingerprint to those exact bytes."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise ProvenanceError(f"required file does not exist: {resolved}")
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ProvenanceError(f"could not read required file {resolved}: {exc}") from exc
    return payload, {
        "path": display_path or str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def read_json_artifact(
    path: Path, *, display_path: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Parse JSON from the same immutable byte snapshot that is fingerprinted."""
    payload, fingerprint = read_bytes_artifact(path, display_path=display_path)
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProvenanceError(
            f"JSON artifact is not valid UTF-8: {path.resolve()}"
        ) from exc
    try:
        return json.loads(decoded), fingerprint
    except json.JSONDecodeError as exc:
        raise ProvenanceError(
            f"JSON artifact is not valid JSON: {path.resolve()}: {exc.msg}"
        ) from exc


def assert_artifact_unchanged(
    label: str, path: Path, expected: Mapping[str, Any]
) -> None:
    """Re-read an artifact and compare it with a prior exact-byte fingerprint."""
    if not isinstance(expected, Mapping) or set(expected) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise ProvenanceError(f"{label} fingerprint must match the strict schema")
    expected_path = expected["path"]
    expected_sha256 = expected["sha256"]
    expected_bytes = expected["bytes"]
    if (
        not isinstance(expected_path, str)
        or not expected_path
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ProvenanceError(f"{label} fingerprint is invalid")
    if Path(expected_path).is_absolute() and str(path.resolve()) != expected_path:
        raise ProvenanceError(f"{label} changed while the benchmark was running")
    _payload, current = read_bytes_artifact(path, display_path=expected_path)
    assert_unchanged(label, dict(expected), current)


def default_worker_directory(root: Path, output: Path, *, kind: str) -> Path:
    """Derive a stable worker directory from the complete output-path identity."""
    if not isinstance(kind, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]*", kind) is None:
        raise ProvenanceError("worker-artifact directory kind is invalid")
    resolved_output = output.resolve()
    normalized_output = os.path.normcase(
        os.path.normpath(str(resolved_output))
    ).replace("\\", "/")
    identity = hashlib.sha256(normalized_output.encode("utf-8")).hexdigest()[:16]
    return (
        root.resolve()
        / "worklogs"
        / f"{resolved_output.stem}-{identity}-{kind}-workers"
    ).resolve()


def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _safe_remote_identity(value: str) -> str:
    remote = value.strip()
    if "://" in remote:
        parsed = urlsplit(remote)
        hostname = parsed.hostname or ""
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
    if "@" in remote and ":" in remote.rsplit("@", 1)[1]:
        return remote.rsplit("@", 1)[1]
    return remote


def repository_fingerprint(repo: Path) -> dict[str, Any]:
    resolved = repo.resolve()

    def git_bytes(*args: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", "-C", str(resolved), *args],
                check=False,
                capture_output=True,
            )
        except OSError as exc:
            raise ProvenanceError(
                f"could not inspect Git checkout {resolved}: {exc}"
            ) from exc
        if completed.returncode:
            detail_bytes = completed.stderr.strip() or completed.stdout.strip()
            detail = detail_bytes.decode("utf-8", errors="replace")
            raise ProvenanceError(
                f"git {' '.join(args)} failed for {resolved}: {detail}"
            )
        return completed.stdout

    def git_text(*args: str) -> str:
        payload = git_bytes(*args)
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProvenanceError(
                f"git {' '.join(args)} returned non-UTF-8 metadata for {resolved}"
            ) from exc

    revision = git_text("rev-parse", "HEAD").strip()
    git_root = Path(git_text("rev-parse", "--show-toplevel").strip()).resolve()
    remotes = set(git_text("remote").splitlines())
    origin = (
        _safe_remote_identity(git_text("remote", "get-url", "origin"))
        if "origin" in remotes
        else None
    )
    exact_tags = sorted(
        tag for tag in git_text("tag", "--points-at", "HEAD").splitlines() if tag
    )
    status = git_bytes(
        "-c",
        "core.quotepath=false",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    tracked_diff = git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked_paths = sorted(
        path
        for path in git_text(
            "-c",
            "core.quotepath=false",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split("\0")
        if path
    )
    untracked_manifest = [
        file_fingerprint(resolved / path, display_path=path) for path in untracked_paths
    ]
    tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
    untracked_manifest_sha256 = canonical_json_sha256(untracked_manifest)
    return {
        "path": str(resolved),
        "git_root": str(git_root),
        "sha": revision,
        "origin_fetch_url": origin,
        "exact_tags": exact_tags,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_manifest_sha256": untracked_manifest_sha256,
        "untracked_files": len(untracked_manifest),
        "untracked_bytes": sum(item["bytes"] for item in untracked_manifest),
        "dirty_content_sha256": canonical_json_sha256(
            {
                "tracked_diff_sha256": tracked_diff_sha256,
                "untracked_manifest_sha256": untracked_manifest_sha256,
            }
        ),
    }


def assert_unchanged(label: str, before: Any, after: Any) -> None:
    if before != after:
        raise ProvenanceError(f"{label} changed while the benchmark was running")


def ensure_output_absent(path: Path) -> None:
    destination = path.resolve()
    if destination.exists():
        raise OutputExistsError(
            f"refusing to overwrite write-once benchmark output: {destination}"
        )


def write_json_once(path: Path, payload: Any) -> None:
    """Atomically publish JSON only when the destination has never existed."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_output_absent(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise OutputExistsError(
                f"refusing to overwrite write-once benchmark output: {destination}"
            ) from exc
        except OSError as exc:
            raise ProvenanceError(
                f"could not atomically publish write-once output {destination}: {exc}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_target(
    manifest: Mapping[str, Any], *, target_id: str, relative_path: str, role: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    targets = manifest.get("targets")
    if not isinstance(targets, list):
        raise ProvenanceError("doctor manifest targets must be a list")
    matches = [
        target
        for target in targets
        if isinstance(target, dict) and target.get("id") == target_id
    ]
    if len(matches) != 1:
        raise ProvenanceError(
            f"doctor manifest must contain exactly one target {target_id!r}"
        )
    target = matches[0]
    if target.get("status") != "ok":
        raise ProvenanceError(
            f"doctor target {target_id!r} is not ok: {target.get('status')!r}"
        )

    normalized_path = relative_path.replace("\\", "/")
    files = target.get("files")
    if not isinstance(files, list):
        raise ProvenanceError(f"doctor target {target_id!r} files must be a list")
    file_matches = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("role") == role
        and str(item.get("path", "")).replace("\\", "/") == normalized_path
    ]
    if len(file_matches) != 1:
        raise ProvenanceError(
            f"doctor target {target_id!r} must contain {role} {normalized_path!r}"
        )
    item = file_matches[0]
    if item.get("status") != "ok":
        raise ProvenanceError(
            f"doctor {role} {normalized_path!r} is not ok: {item.get('status')!r}"
        )
    validation = item.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "ok":
        raise ProvenanceError(
            f"doctor {role} {normalized_path!r} has no successful input validation"
        )
    return target, item


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_target_file_contract(
    target_id: str,
    target: Mapping[str, Any],
    *,
    source_roots: Mapping[str, Path],
    fingerprint_cache: dict[Path, dict[str, Any]],
) -> None:
    expected_source, expected_files = STRICT_PUBLIC_TARGET_CONTRACTS[target_id]
    if target.get("source") != expected_source:
        raise ProvenanceError(
            f"doctor strict-public target {target_id!r} source is inconsistent"
        )
    files = target.get("files")
    if not isinstance(files, list) or not files:
        raise ProvenanceError(
            f"doctor strict-public target {target_id!r} must contain validated files"
        )

    content: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    total_bytes = 0
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {index} is not an object"
            )
        role = item.get("role")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("bytes")
        validation = item.get("validation")
        if not isinstance(role, str) or not role:
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {index} has no role"
            )
        if not isinstance(path, str) or not path:
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {index} has no path"
            )
        normalized_path = path.replace("\\", "/")
        identity = (role, normalized_path)
        if identity in identities:
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} contains duplicate file {identity}"
            )
        identities.add(identity)
        if item.get("status") != "ok":
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {path!r} is not ok"
            )
        if not _valid_sha256(digest):
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {path!r} has invalid sha256"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {path!r} has invalid bytes"
            )
        if not isinstance(validation, dict) or validation.get("status") != "ok":
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {path!r} is not validated"
            )
        root = source_roots[expected_source]
        actual_path = (root / normalized_path).resolve()
        try:
            actual_path.relative_to(root)
        except ValueError as exc:
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file escapes its repository: "
                f"{normalized_path!r}"
            ) from exc
        actual = fingerprint_cache.get(actual_path)
        if actual is None:
            actual = file_fingerprint(actual_path, display_path=normalized_path)
            fingerprint_cache[actual_path] = actual
        if actual["sha256"] != digest or actual["bytes"] != size:
            raise ProvenanceError(
                f"doctor strict-public target {target_id!r} file {normalized_path!r} "
                "does not match the current repository input"
            )
        content.append(
            {
                "role": role,
                "path": normalized_path,
                "sha256": digest,
                "bytes": size,
            }
        )
        total_bytes += size

    actual_files = tuple(
        (str(item["role"]), str(item["path"]).replace("\\", "/")) for item in files
    )
    if actual_files != expected_files:
        raise ProvenanceError(
            f"doctor strict-public target {target_id!r} file contract is inconsistent: "
            f"expected={expected_files!r}, actual={actual_files!r}"
        )

    if len(content) == 1:
        expected_sha256 = content[0]["sha256"]
        expected_kind = "file"
    else:
        expected_sha256 = canonical_json_sha256(content)
        expected_kind = "file-manifest-v1"
    if (
        target.get("sha256") != expected_sha256
        or target.get("sha256_kind") != expected_kind
        or target.get("bytes") != total_bytes
    ):
        raise ProvenanceError(
            f"doctor strict-public target {target_id!r} aggregate fingerprint is inconsistent"
        )


def _validate_strict_public_contract(manifest: Mapping[str, Any]) -> None:
    strict = manifest.get("strict_public")
    if (
        not isinstance(strict, dict)
        or strict.get("enabled") is not True
        or strict.get("would_pass") is not True
        or strict.get("passed") is not True
        or strict.get("blockers") != []
    ):
        raise ProvenanceError(
            "doctor manifest did not pass enabled strict-public validation"
        )

    targets = manifest.get("targets")
    if not isinstance(targets, list) or not all(
        isinstance(target, dict) and isinstance(target.get("id"), str)
        for target in targets
    ):
        raise ProvenanceError("doctor manifest targets must be identified objects")
    target_ids = [str(target["id"]) for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ProvenanceError("doctor manifest target ids must be unique")
    strict_targets = {
        str(target["id"]): target
        for target in targets
        if target.get("strict_public") is True
    }
    if set(strict_targets) != STRICT_PUBLIC_TARGET_IDS:
        missing = sorted(STRICT_PUBLIC_TARGET_IDS - set(strict_targets))
        unexpected = sorted(set(strict_targets) - STRICT_PUBLIC_TARGET_IDS)
        raise ProvenanceError(
            "doctor strict-public target contract mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    incomplete = sorted(
        target_id
        for target_id, target in strict_targets.items()
        if target.get("status") != "ok"
    )
    if incomplete:
        raise ProvenanceError(f"doctor strict-public targets are not ok: {incomplete}")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, dict):
        raise ProvenanceError("doctor manifest has no repository fingerprints")
    source_roots: dict[str, Path] = {}
    for source, repository_key in (
        ("omnifuse", "omnifuse"),
        ("synaptic", "synaptic_memory"),
    ):
        repository = repositories.get(repository_key)
        path = repository.get("path") if isinstance(repository, dict) else None
        if not isinstance(path, str) or not path:
            raise ProvenanceError(
                f"doctor manifest repository {repository_key!r} has no path"
            )
        root = Path(path).resolve()
        if not root.is_dir():
            raise ProvenanceError(
                f"doctor manifest repository {repository_key!r} does not exist: {root}"
            )
        source_roots[source] = root

    fingerprint_cache: dict[Path, dict[str, Any]] = {}
    for target_id, target in strict_targets.items():
        _validate_target_file_contract(
            target_id,
            target,
            source_roots=source_roots,
            fingerprint_cache=fingerprint_cache,
        )

    summary = manifest.get("summary")
    public = summary.get("public") if isinstance(summary, dict) else None
    expected_count = len(STRICT_PUBLIC_TARGET_IDS)
    if (
        not isinstance(public, dict)
        or public.get("total_targets") != expected_count
        or public.get("ok_targets") != expected_count
        or public.get("incomplete_targets") != []
    ):
        raise ProvenanceError("doctor strict-public summary is inconsistent")


def load_doctor_manifest(
    path: Path, inputs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = path.resolve()
    fingerprint = file_fingerprint(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(
            f"invalid doctor manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ProvenanceError("doctor manifest must be a JSON object")
    if manifest.get("schema") != DOCTOR_SCHEMA:
        raise ProvenanceError(f"unsupported doctor schema: {manifest.get('schema')!r}")
    if manifest.get("schema_version") != DOCTOR_SCHEMA_VERSION:
        raise ProvenanceError(
            f"unsupported doctor schema version: {manifest.get('schema_version')!r}"
        )
    _validate_strict_public_contract(manifest)

    links: dict[str, dict[str, Any]] = {}
    for benchmark_input in inputs:
        name = benchmark_input.get("name")
        target_id = benchmark_input.get("target_id")
        relative_path = benchmark_input.get("path")
        role = benchmark_input.get("role", "dataset")
        actual_hash = benchmark_input.get("sha256")
        actual_bytes = benchmark_input.get("bytes")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(target_id, str)
            or not target_id
            or not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(role, str)
            or not role
            or not _valid_sha256(actual_hash)
            or isinstance(actual_bytes, bool)
            or not isinstance(actual_bytes, int)
            or actual_bytes < 0
        ):
            raise ProvenanceError("benchmark input fingerprint is incomplete")
        if name in links:
            raise ProvenanceError(f"benchmark input names must be unique: {name!r}")
        target, item = _manifest_target(
            manifest,
            target_id=target_id,
            relative_path=relative_path,
            role=role,
        )
        doctor_hash = item.get("sha256")
        if doctor_hash != actual_hash:
            raise ProvenanceError(
                f"doctor hash mismatch for {relative_path}: "
                f"file={doctor_hash}, actual={actual_hash}"
            )
        if target.get("sha256_kind") == "file" and target.get("sha256") != actual_hash:
            raise ProvenanceError(
                f"doctor aggregate hash mismatch for {relative_path}: "
                f"target={target.get('sha256')}, actual={actual_hash}"
            )
        if item.get("bytes") != actual_bytes:
            raise ProvenanceError(
                f"doctor byte-size mismatch for {relative_path}: "
                f"file={item.get('bytes')}, actual={actual_bytes}"
            )
        links[name] = {
            "target_id": target_id,
            "target_name": target.get("name"),
            "target_status": "ok",
            "role": role,
            "input_sha256": actual_hash,
            "input_bytes": actual_bytes,
            "dataset_sha256": actual_hash,
        }

    return (
        {
            **fingerprint,
            "schema": DOCTOR_SCHEMA,
            "schema_version": DOCTOR_SCHEMA_VERSION,
            "strict_public_passed": True,
            "canonical_sha256": canonical_json_sha256(manifest),
            "snapshot": manifest,
            "targets": links,
        },
        links,
    )


def verify_doctor_runtime(
    record: Mapping[str, Any],
    *,
    omnifuse_repository: Mapping[str, Any],
    synaptic_repository: Mapping[str, Any],
    omnifuse_scorer: Mapping[str, Any],
    synaptic_scorer: Mapping[str, Any],
) -> None:
    snapshot = record.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ProvenanceError("doctor provenance has no embedded snapshot")
    repositories = snapshot.get("repositories")
    if not isinstance(repositories, dict):
        raise ProvenanceError("doctor manifest has no repository fingerprints")

    doctor_omnifuse = repositories.get("omnifuse")
    doctor_synaptic = repositories.get("synaptic_memory")
    if not isinstance(doctor_omnifuse, dict) or not isinstance(doctor_synaptic, dict):
        raise ProvenanceError("doctor manifest repository fingerprints are incomplete")

    shared_repository_keys = (
        "path",
        "sha",
        "dirty",
        "status_sha256",
        "tracked_diff_sha256",
        "untracked_manifest_sha256",
        "untracked_files",
        "untracked_bytes",
        "dirty_content_sha256",
    )
    identity_keys = ("git_root", "origin_fetch_url", "exact_tags")
    for label, repository in (
        ("OmniFuse", doctor_omnifuse),
        ("synaptic-memory", doctor_synaptic),
    ):
        present = {key for key in identity_keys if key in repository}
        if present and present != set(identity_keys):
            raise ProvenanceError(
                f"doctor {label} repository identity fingerprint is incomplete"
            )
    omnifuse_repository_keys = (
        *shared_repository_keys,
        *(
            identity_keys
            if all(key in doctor_omnifuse for key in identity_keys)
            else ()
        ),
    )
    synaptic_repository_keys = (
        *shared_repository_keys,
        *(
            identity_keys
            if all(key in doctor_synaptic for key in identity_keys)
            else ()
        ),
    )
    try:
        assert_unchanged(
            "OmniFuse repository since doctor preflight",
            {key: doctor_omnifuse[key] for key in omnifuse_repository_keys},
            {key: omnifuse_repository[key] for key in omnifuse_repository_keys},
        )
        assert_unchanged(
            "synaptic-memory repository since doctor preflight",
            {key: doctor_synaptic[key] for key in synaptic_repository_keys},
            {key: synaptic_repository[key] for key in synaptic_repository_keys},
        )
    except KeyError as exc:
        raise ProvenanceError(
            f"doctor repository fingerprint is missing {exc.args[0]!r}"
        ) from exc

    scorer = snapshot.get("scorer")
    if not isinstance(scorer, dict) or scorer.get("equal") is not True:
        raise ProvenanceError("doctor manifest did not verify byte-identical scorers")
    doctor_omnifuse_scorer = scorer.get("omnifuse")
    doctor_synaptic_scorer = scorer.get("synaptic_memory")
    if not isinstance(doctor_omnifuse_scorer, dict) or not isinstance(
        doctor_synaptic_scorer, dict
    ):
        raise ProvenanceError("doctor manifest scorer fingerprints are incomplete")
    scorer_keys = ("path", "sha256", "bytes")
    try:
        expected_omnifuse_scorer = {
            key: doctor_omnifuse_scorer[key] for key in scorer_keys
        }
        expected_synaptic_scorer = {
            key: doctor_synaptic_scorer[key] for key in scorer_keys
        }
        current_omnifuse_scorer = {key: omnifuse_scorer[key] for key in scorer_keys}
        current_synaptic_scorer = {key: synaptic_scorer[key] for key in scorer_keys}
    except KeyError as exc:
        raise ProvenanceError(
            f"doctor scorer fingerprint is missing {exc.args[0]!r}"
        ) from exc
    assert_unchanged(
        "OmniFuse scorer since doctor preflight",
        expected_omnifuse_scorer,
        current_omnifuse_scorer,
    )
    assert_unchanged(
        "synaptic-memory scorer since doctor preflight",
        expected_synaptic_scorer,
        current_synaptic_scorer,
    )
    if any(
        current_omnifuse_scorer[key] != current_synaptic_scorer[key]
        for key in ("sha256", "bytes")
    ):
        raise ProvenanceError("current OmniFuse and synaptic-memory scorers differ")


def verify_doctor_manifest(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    current_fingerprint = file_fingerprint(path)
    assert_unchanged(
        "doctor manifest file",
        {key: record[key] for key in ("path", "sha256", "bytes")},
        current_fingerprint,
    )
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"invalid doctor manifest {path}: {exc}") from exc
    if not isinstance(current, dict):
        raise ProvenanceError("doctor manifest must be a JSON object")
    _validate_strict_public_contract(current)
    assert_unchanged(
        "doctor manifest canonical content",
        record["canonical_sha256"],
        canonical_json_sha256(current),
    )
