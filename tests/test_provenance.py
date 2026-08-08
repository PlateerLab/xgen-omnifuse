from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

import provenance  # noqa: E402


WORKER_RUN_IDS = (
    "00000000000040008000000000000001",
    "00000000000040008000000000000002",
)


def _process_record(
    run_id: str, *, launcher_pid: int, worker_pid: int
) -> dict[str, object]:
    return {
        "worker_run_id": run_id,
        "launcher_pid": launcher_pid,
        "worker_pid": worker_pid,
        "same_process_id": launcher_pid == worker_pid,
    }


def test_worker_identity_contract_is_strict_and_controller_bound() -> None:
    identity = provenance.capture_worker_identity(WORKER_RUN_IDS[0])

    assert (
        provenance.validate_worker_identity(identity, expected_run_id=WORKER_RUN_IDS[0])
        == identity
    )
    assert identity["capture_phase"] == "post_measurement"
    assert identity["worker_pid"] > 0

    changed = copy.deepcopy(identity)
    changed["worker_run_id"] = WORKER_RUN_IDS[1]
    with pytest.raises(provenance.ProvenanceError, match="does not match"):
        provenance.validate_worker_identity(changed, expected_run_id=WORKER_RUN_IDS[0])

    for invalid in (
        "aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa".upper(),
        "0" * 32,
        "00000000000040007000000000000001",
        "not-a-run-id",
    ):
        with pytest.raises(provenance.ProvenanceError, match="UUIDv4"):
            provenance.validate_worker_run_id(invalid)

    invalid_pid = copy.deepcopy(identity)
    invalid_pid["worker_pid"] = True
    with pytest.raises(provenance.ProvenanceError, match="positive integer"):
        provenance.validate_worker_identity(invalid_pid)


def test_worker_launcher_pid_is_captured_separately() -> None:
    completed, launcher_pid = provenance.run_with_launcher_pid(
        [sys.executable, "-c", "print('worker-ok')"],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert completed.stdout.strip() == "worker-ok"
    assert launcher_pid > 0


def test_process_summary_allows_pid_mismatch_and_reuse_but_not_duplicate_runs() -> None:
    records = [
        _process_record(WORKER_RUN_IDS[0], launcher_pid=100, worker_pid=200),
        _process_record(WORKER_RUN_IDS[1], launcher_pid=100, worker_pid=201),
    ]

    summary = provenance.worker_process_summary(records, expected_count=2)

    assert summary["distinct_worker_run_ids"] == 2
    assert summary["launcher_pid_reuse_observed"] is True
    assert summary["worker_pid_reuse_observed"] is False
    assert summary["launcher_worker_pid_mismatch_count"] == 2

    duplicate = [records[0], {**records[1], "worker_run_id": WORKER_RUN_IDS[0]}]
    with pytest.raises(provenance.ProvenanceError, match="must be unique"):
        provenance.worker_process_summary(duplicate, expected_count=2)


def _manifest(tmp_path: Path) -> dict:
    roots = {
        "omnifuse": tmp_path / "omnifuse",
        "synaptic": tmp_path / "synaptic",
    }
    targets = []
    for target_id, (source, file_contract) in sorted(
        provenance.STRICT_PUBLIC_TARGET_CONTRACTS.items()
    ):
        files = []
        for role, relative_path in file_contract:
            path = roots[source] / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(f"{source}:{relative_path}".encode())
            fingerprint = provenance.file_fingerprint(path, display_path=relative_path)
            files.append(
                {
                    "role": role,
                    "path": relative_path,
                    "status": "ok",
                    "sha256": fingerprint["sha256"],
                    "bytes": fingerprint["bytes"],
                    "validation": {"status": "ok"},
                }
            )
        content = [
            {
                "role": item["role"],
                "path": item["path"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
            for item in files
        ]
        targets.append(
            {
                "id": target_id,
                "name": target_id,
                "source": source,
                "strict_public": True,
                "status": "ok",
                "sha256": (
                    content[0]["sha256"]
                    if len(content) == 1
                    else provenance.canonical_json_sha256(content)
                ),
                "sha256_kind": "file" if len(content) == 1 else "file-manifest-v1",
                "bytes": sum(item["bytes"] for item in content),
                "files": files,
            }
        )
    return {
        "schema": "omnifuse.eval.doctor",
        "schema_version": 1,
        "strict_public": {
            "enabled": True,
            "would_pass": True,
            "passed": True,
            "blockers": [],
        },
        "summary": {
            "public": {
                "total_targets": len(provenance.STRICT_PUBLIC_TARGET_IDS),
                "ok_targets": len(provenance.STRICT_PUBLIC_TARGET_IDS),
                "incomplete_targets": [],
            }
        },
        "repositories": {
            "omnifuse": {"path": str(roots["omnifuse"])},
            "synaptic_memory": {"path": str(roots["synaptic"])},
        },
        "targets": targets,
    }


def _target(manifest: dict, target_id: str) -> dict:
    return next(target for target in manifest["targets"] if target["id"] == target_id)


def _selected_hash(manifest: dict) -> str:
    return _target(manifest, "hotpotqa_24")["sha256"]


def _input(manifest: dict) -> dict[str, object]:
    item = _target(manifest, "hotpotqa_24")["files"][0]
    return {
        "name": "HotPotQA-24",
        "target_id": "hotpotqa_24",
        "path": "tests/benchmark/data/hotpotqa_24.json",
        "sha256": item["sha256"],
        "bytes": item["bytes"],
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_manifest_provenance_embeds_verified_snapshot_and_two_hashes(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "doctor.json"
    _write_manifest(path, manifest)

    record, links = provenance.load_doctor_manifest(path, [_input(manifest)])

    assert record["snapshot"] == manifest
    assert record["sha256"] == provenance.sha256_file(path)
    assert record["canonical_sha256"] == provenance.canonical_json_sha256(manifest)
    assert record["strict_public_passed"] is True
    assert links["HotPotQA-24"]["target_status"] == "ok"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong.schema", "unsupported doctor schema"),
        ("schema_version", 2, "unsupported doctor schema version"),
    ],
)
def test_manifest_rejects_unsupported_schema_or_version(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest = _manifest(tmp_path)
    manifest[field] = value
    path = tmp_path / "doctor.json"
    _write_manifest(path, manifest)

    with pytest.raises(provenance.ProvenanceError, match=message):
        provenance.load_doctor_manifest(path, [_input(manifest)])


def test_manifest_requires_strict_public_pass_and_ok_target(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "doctor.json"

    manifest["strict_public"]["passed"] = False
    _write_manifest(path, manifest)
    with pytest.raises(provenance.ProvenanceError, match="strict-public"):
        provenance.load_doctor_manifest(path, [_input(manifest)])

    manifest = _manifest(tmp_path)
    _target(manifest, "hotpotqa_24")["status"] = "skipped_missing_external"
    _write_manifest(path, manifest)
    with pytest.raises(provenance.ProvenanceError, match="not ok"):
        provenance.load_doctor_manifest(path, [_input(manifest)])


def test_manifest_rejects_partial_or_inconsistent_strict_public_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doctor.json"
    manifest = _manifest(tmp_path)
    benchmark_input = _input(manifest)
    manifest["targets"] = manifest["targets"][:1]
    _write_manifest(path, manifest)
    with pytest.raises(provenance.ProvenanceError, match="target contract mismatch"):
        provenance.load_doctor_manifest(path, [benchmark_input])

    manifest = _manifest(tmp_path)
    manifest["summary"]["public"]["ok_targets"] = 18
    _write_manifest(path, manifest)
    with pytest.raises(provenance.ProvenanceError, match="summary is inconsistent"):
        provenance.load_doctor_manifest(path, [_input(manifest)])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda target: target.update(files=[]), "must contain validated files"),
        (
            lambda target: target["files"][0].update(validation={"status": "error"}),
            "is not validated",
        ),
        (lambda target: target["files"][0].update(bytes=-1), "invalid bytes"),
        (
            lambda target: target.update(bytes=2),
            "aggregate fingerprint is inconsistent",
        ),
        (
            lambda target: target.update(sha256_kind="file-manifest-v1"),
            "aggregate fingerprint is inconsistent",
        ),
    ],
)
def test_manifest_rejects_unvalidated_or_inconsistent_strict_target_files(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest = _manifest(tmp_path)
    mutation(manifest["targets"][0])
    path = tmp_path / "doctor.json"
    _write_manifest(path, manifest)

    with pytest.raises(provenance.ProvenanceError, match=message):
        provenance.load_doctor_manifest(path, [_input(manifest)])


def test_manifest_requires_matching_target_and_file_hashes(tmp_path: Path) -> None:
    path = tmp_path / "doctor.json"
    for location in ("target", "file"):
        manifest = copy.deepcopy(_manifest(tmp_path))
        if location == "target":
            manifest["targets"][0]["sha256"] = "0" * 64
        else:
            manifest["targets"][0]["files"][0]["sha256"] = "0" * 64
        _write_manifest(path, manifest)

        message = (
            "aggregate fingerprint is inconsistent"
            if location == "target"
            else "current repository input"
        )
        with pytest.raises(provenance.ProvenanceError, match=message):
            provenance.load_doctor_manifest(path, [_input(manifest)])


def test_manifest_requires_matching_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "doctor.json"
    manifest = _manifest(tmp_path)
    _write_manifest(path, manifest)
    benchmark_input = _input(manifest)
    benchmark_input["bytes"] = 2

    with pytest.raises(provenance.ProvenanceError, match="byte-size mismatch"):
        provenance.load_doctor_manifest(path, [benchmark_input])


def test_manifest_revalidates_unrequested_strict_inputs_on_disk(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "doctor.json"
    _write_manifest(path, manifest)
    repository = Path(manifest["repositories"]["synaptic_memory"]["path"])
    trec = _target(manifest, "trec_covid")["files"][0]
    (repository / trec["path"]).write_bytes(b"changed after doctor capture")

    with pytest.raises(provenance.ProvenanceError, match="current repository input"):
        provenance.load_doctor_manifest(path, [_input(manifest)])


def test_manifest_binds_target_source_and_file_contract(tmp_path: Path) -> None:
    path = tmp_path / "doctor.json"
    manifest = _manifest(tmp_path)
    _target(manifest, "hotpotqa_24")["source"] = "omnifuse"
    _write_manifest(path, manifest)

    with pytest.raises(provenance.ProvenanceError, match="source is inconsistent"):
        provenance.load_doctor_manifest(path, [_input(manifest)])


def test_manifest_links_one_file_from_multi_file_target(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "doctor.json"
    _write_manifest(path, manifest)
    target = _target(manifest, "finreg_single")
    corpus = target["files"][0]

    _, links = provenance.load_doctor_manifest(
        path,
        [
            {
                "name": "finreg corpus",
                "target_id": "finreg_single",
                "role": "corpus",
                "path": corpus["path"],
                "sha256": corpus["sha256"],
                "bytes": corpus["bytes"],
            }
        ],
    )

    assert links["finreg corpus"]["role"] == "corpus"
    assert links["finreg corpus"]["input_sha256"] == corpus["sha256"]


def test_manifest_requires_complete_input_fingerprint(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "doctor.json"
    _write_manifest(path, manifest)
    benchmark_input = _input(manifest)
    del benchmark_input["bytes"]

    with pytest.raises(provenance.ProvenanceError, match="fingerprint is incomplete"):
        provenance.load_doctor_manifest(path, [benchmark_input])


def test_doctor_revalidation_detects_post_capture_mutation(tmp_path: Path) -> None:
    path = tmp_path / "doctor.json"
    manifest = _manifest(tmp_path)
    _write_manifest(path, manifest)
    record, _ = provenance.load_doctor_manifest(path, [_input(manifest)])

    manifest["generated_at"] = "later"
    _write_manifest(path, manifest)

    with pytest.raises(provenance.ProvenanceError, match="changed"):
        provenance.verify_doctor_manifest(record)


def test_doctor_runtime_binds_repository_and_scorer_fingerprints(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    def repository(path: str, sha: str) -> dict:
        return {
            "path": path,
            "sha": sha,
            "dirty": False,
            "status_sha256": "1" * 64,
            "tracked_diff_sha256": "2" * 64,
            "untracked_manifest_sha256": "3" * 64,
            "untracked_files": 0,
            "untracked_bytes": 0,
            "dirty_content_sha256": "4" * 64,
        }

    omnifuse_repository = repository("omnifuse", "a" * 40)
    synaptic_repository = repository("synaptic", "b" * 40)
    omnifuse_scorer = {
        "path": "eval/metrics.py",
        "sha256": "c" * 64,
        "bytes": 100,
    }
    synaptic_scorer = {
        "path": "tests/benchmark/metrics.py",
        "sha256": "c" * 64,
        "bytes": 100,
    }
    manifest["repositories"] = {
        "omnifuse": {**omnifuse_repository, "status": "ok"},
        "synaptic_memory": {**synaptic_repository, "status": "ok"},
    }
    manifest["scorer"] = {
        "omnifuse": {**omnifuse_scorer, "status": "ok"},
        "synaptic_memory": {**synaptic_scorer, "status": "ok"},
        "equal": True,
    }
    record = {"snapshot": manifest}

    provenance.verify_doctor_runtime(
        record,
        omnifuse_repository=omnifuse_repository,
        synaptic_repository=synaptic_repository,
        omnifuse_scorer=omnifuse_scorer,
        synaptic_scorer=synaptic_scorer,
    )

    changed_synaptic = {**synaptic_repository, "sha": "d" * 40}
    with pytest.raises(provenance.ProvenanceError, match="repository since doctor"):
        provenance.verify_doctor_runtime(
            record,
            omnifuse_repository=omnifuse_repository,
            synaptic_repository=changed_synaptic,
            omnifuse_scorer=omnifuse_scorer,
            synaptic_scorer=synaptic_scorer,
        )

    changed_omnifuse = {
        **omnifuse_repository,
        "untracked_manifest_sha256": "e" * 64,
        "dirty_content_sha256": "f" * 64,
    }
    with pytest.raises(provenance.ProvenanceError, match="repository since doctor"):
        provenance.verify_doctor_runtime(
            record,
            omnifuse_repository=changed_omnifuse,
            synaptic_repository=synaptic_repository,
            omnifuse_scorer=omnifuse_scorer,
            synaptic_scorer=synaptic_scorer,
        )

    changed_scorer = {**synaptic_scorer, "sha256": "e" * 64}
    with pytest.raises(provenance.ProvenanceError, match="scorer since doctor"):
        provenance.verify_doctor_runtime(
            record,
            omnifuse_repository=omnifuse_repository,
            synaptic_repository=synaptic_repository,
            omnifuse_scorer=omnifuse_scorer,
            synaptic_scorer=changed_scorer,
        )


def test_write_once_preserves_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    provenance.write_json_once(path, {"first": True})

    with pytest.raises(provenance.OutputExistsError, match="refusing to overwrite"):
        provenance.write_json_once(path, {"first": False})

    assert json.loads(path.read_text(encoding="utf-8")) == {"first": True}
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_json_artifact_fingerprint_is_bound_to_the_parsed_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker.json"
    raw = b'{"answer": 42}\n'
    path.write_bytes(raw)

    payload, fingerprint = provenance.read_json_artifact(path)

    assert payload == {"answer": 42}
    assert fingerprint == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    provenance.assert_artifact_unchanged("worker artifact", path, fingerprint)

    same_bytes_elsewhere = tmp_path / "other-worker.json"
    same_bytes_elsewhere.write_bytes(raw)
    with pytest.raises(provenance.ProvenanceError, match="changed"):
        provenance.assert_artifact_unchanged(
            "worker artifact", same_bytes_elsewhere, fingerprint
        )

    path.write_bytes(b'{"answer": 43}\n')
    with pytest.raises(provenance.ProvenanceError, match="changed"):
        provenance.assert_artifact_unchanged("worker artifact", path, fingerprint)


@pytest.mark.parametrize("raw", [b"\xff", b"{not-json}"])
def test_json_artifact_rejects_non_utf8_or_invalid_json(
    tmp_path: Path, raw: bytes
) -> None:
    path = tmp_path / "worker.json"
    path.write_bytes(raw)

    with pytest.raises(provenance.ProvenanceError, match="not valid"):
        provenance.read_json_artifact(path)


def test_default_worker_directory_binds_the_full_output_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    first = tmp_path / "one" / "result.json"
    second = tmp_path / "two" / "result.json"

    first_directory = provenance.default_worker_directory(
        root, first, kind="perf"
    )

    assert first_directory == provenance.default_worker_directory(
        root, first, kind="perf"
    )
    assert first_directory != provenance.default_worker_directory(
        root, second, kind="perf"
    )
    assert first_directory.parent == (root / "worklogs").resolve()
    assert re.fullmatch(r"result-[0-9a-f]{16}-perf-workers", first_directory.name)


def test_default_worker_directory_rejects_unsafe_kind(tmp_path: Path) -> None:
    with pytest.raises(provenance.ProvenanceError, match="kind"):
        provenance.default_worker_directory(
            tmp_path, tmp_path / "result.json", kind="../escape"
        )


def test_repository_fingerprint_hashes_dirty_file_contents(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    tracked = repo / "tracked.py"
    untracked = repo / "untracked.py"
    tracked.write_text("base\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Provenance Test")
    _git(repo, "config", "user.email", "provenance@example.invalid")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "fixture")
    _git(repo, "remote", "add", "origin", "https://secret@example.invalid/org/repo.git")
    _git(repo, "tag", "v-test")

    tracked.write_text("one\n", encoding="utf-8")
    untracked.write_text("aaa\n", encoding="utf-8")
    before = provenance.repository_fingerprint(repo)

    tracked.write_text("two\n", encoding="utf-8")
    untracked.write_text("bbb\n", encoding="utf-8")
    after = provenance.repository_fingerprint(repo)

    assert before["status_sha256"] == after["status_sha256"]
    assert before["tracked_diff_sha256"] != after["tracked_diff_sha256"]
    assert before["untracked_manifest_sha256"] != after["untracked_manifest_sha256"]
    assert before["dirty_content_sha256"] != after["dirty_content_sha256"]
    assert before["untracked_files"] == after["untracked_files"] == 1
    assert before["untracked_bytes"] == after["untracked_bytes"]
    assert before["untracked_bytes"] > 0
    assert before["git_root"] == str(repo.resolve())
    assert before["origin_fetch_url"] == "https://example.invalid/org/repo.git"
    assert before["exact_tags"] == ["v-test"]


def test_repository_fingerprint_hashes_raw_git_diff_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    tracked = repo / "tracked.bin"
    tracked.write_bytes(b"base\n")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Provenance Test")
    _git(repo, "config", "user.email", "provenance@example.invalid")
    _git(repo, "add", "tracked.bin")
    _git(repo, "commit", "-m", "fixture")
    tracked.write_bytes(b"\xff\n")
    raw_diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout

    fingerprint = provenance.repository_fingerprint(repo)

    assert b"\xff" in raw_diff
    assert fingerprint["tracked_diff_sha256"] == hashlib.sha256(raw_diff).hexdigest()
