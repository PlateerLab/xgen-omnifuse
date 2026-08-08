from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "bench.py"
SPEC = importlib.util.spec_from_file_location("omnifuse_eval_bench", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def _write(root: Path, relative: str, content: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_omnifuse(root: Path) -> None:
    _write(root, "eval/data/finreg/raw.jsonl", b'{"doc_id":"1","text":"body"}\n')
    queries = b'{"queries":[{"id":"q1","query":"body","relevant_docs":["1"]}]}\n'
    _write(root, "eval/data/queries/finreg.json", queries)
    _write(root, "eval/data/queries/finreg_multihop.json", queries)
    _write(root, "eval/metrics.py", b"# shared scorer\n")


def _public_data() -> bytes:
    return (
        json.dumps(
            {
                "corpus": {"d1": {"title": "title", "text": "body"}},
                "queries": {"q1": "body"},
                "qrels": {"q1": {"d1": 1}},
            }
        ).encode()
        + b"\n"
    )


def _enterprise_data() -> bytes:
    return (
        json.dumps(
            {
                "knowledge_sources": [
                    {"id": "d1", "title": "title", "content": "body"}
                ],
                "evaluation_queries": [
                    {"id": "q1", "query": "body", "relevant_ids": ["d1"]}
                ],
            }
        ).encode()
        + b"\n"
    )


def _write_synaptic_target(root: Path, spec: object) -> None:
    content = _enterprise_data() if spec.group == "enterprise" else _public_data()
    for _, relative in spec.files:
        _write(root, relative, content)


def _commit_fixture(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Fixture"], check=True
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)


def _target(manifest: dict, target_id: str) -> dict:
    return next(target for target in manifest["targets"] if target["id"] == target_id)


def test_manifest_enumerates_all_targets_and_keeps_missing_in_denominator(
    tmp_path: Path,
) -> None:
    omni = tmp_path / "omnifuse"
    synaptic = tmp_path / "synaptic-memory"
    _make_omnifuse(omni)
    _write(synaptic, "tests/benchmark/metrics.py", b"# shared scorer\n")
    _write(synaptic, "tests/benchmark/data/hotpotqa_24.json", _public_data())

    manifest = bench.build_manifest(
        omnifuse_repo=omni,
        synaptic_repo=synaptic,
        kra_golden=None,
        strict_public=False,
    )

    assert manifest["summary"]["total_targets"] == 22
    assert manifest["summary"]["public"]["total_targets"] == 19
    assert sum(manifest["summary"]["status_counts"].values()) == 22
    assert _target(manifest, "finreg_single")["status"] == "ok"
    assert (
        _target(manifest, "finreg_single")["files"][0]["validation"]["documents"] == 1
    )
    assert len(_target(manifest, "finreg_single")["sha256"]) == 64
    assert _target(manifest, "hotpotqa_24")["status"] == "ok"
    assert _target(manifest, "hotpotqa_200")["status"] == "error"
    assert _target(manifest, "scifact")["status"] == "skipped_missing_external"
    assert _target(manifest, "kra_golden")["status"] == "skipped_private"
    assert manifest["scorer"]["equal"] is True
    assert "hotpotqa_200" in manifest["summary"]["public"]["incomplete_targets"]


def test_absent_synaptic_checkout_is_external_skip_not_false_error(
    tmp_path: Path,
) -> None:
    omni = tmp_path / "omnifuse"
    _make_omnifuse(omni)

    manifest = bench.build_manifest(
        omnifuse_repo=omni,
        synaptic_repo=None,
        kra_golden=None,
        strict_public=False,
    )

    assert _target(manifest, "hotpotqa_24")["status"] == "skipped_missing_external"
    assert (
        _target(manifest, "enterprise_scenario")["status"] == "skipped_missing_external"
    )
    assert _target(manifest, "qa_combined")["status"] == "skipped_missing_external"
    assert manifest["scorer"]["equal"] is None


def test_strict_public_passes_with_all_public_data_and_atomic_output(
    tmp_path: Path,
) -> None:
    omni = tmp_path / "omnifuse"
    synaptic = tmp_path / "synaptic-memory"
    _make_omnifuse(omni)
    _write(synaptic, "tests/benchmark/metrics.py", b"# shared scorer\n")
    for spec in bench.TARGETS:
        if spec.strict_public and spec.source == "synaptic":
            _write_synaptic_target(synaptic, spec)
    _commit_fixture(omni)
    _commit_fixture(synaptic)

    output = tmp_path / "results" / "doctor.json"
    exit_code = bench.main(
        [
            "doctor",
            "--omnifuse-repo",
            str(omni),
            "--synaptic-repo",
            str(synaptic),
            "--strict-public",
            "--out",
            str(output),
        ]
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["strict_public"]["passed"] is True
    assert manifest["summary"]["public"] == {
        "total_targets": 19,
        "ok_targets": 19,
        "incomplete_targets": [],
    }
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_doctor_output_is_write_once_and_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "doctor.json"
    output.write_bytes(b"immutable\n")

    exit_code = bench.main(["doctor", "--out", str(output)])

    assert exit_code == 2
    assert output.read_bytes() == b"immutable\n"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_strict_public_writes_manifest_before_nonzero_exit(tmp_path: Path) -> None:
    omni = tmp_path / "omnifuse"
    synaptic = tmp_path / "synaptic-memory"
    _make_omnifuse(omni)
    _write(synaptic, "tests/benchmark/metrics.py", b"# shared scorer\n")
    for spec in bench.TARGETS:
        if spec.strict_public and spec.source == "synaptic" and spec.id != "fiqa":
            _write_synaptic_target(synaptic, spec)
    _commit_fixture(omni)
    _commit_fixture(synaptic)

    output = tmp_path / "doctor.json"
    exit_code = bench.main(
        [
            "doctor",
            "--omnifuse-repo",
            str(omni),
            "--synaptic-repo",
            str(synaptic),
            "--strict-public",
            "--out",
            str(output),
        ]
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert manifest["strict_public"]["passed"] is False
    assert {blocker.get("id") for blocker in manifest["strict_public"]["blockers"]} >= {
        "fiqa"
    }
    assert manifest["summary"]["public"]["ok_targets"] == 18


def test_invalid_public_json_is_not_ready(tmp_path: Path) -> None:
    omni = tmp_path / "omnifuse"
    synaptic = tmp_path / "synaptic-memory"
    _make_omnifuse(omni)
    _write(synaptic, "tests/benchmark/metrics.py", b"# shared scorer\n")
    _write(synaptic, "tests/benchmark/data/hotpotqa_24.json", b"{}\n")

    manifest = bench.build_manifest(
        omnifuse_repo=omni,
        synaptic_repo=synaptic,
        kra_golden=None,
        strict_public=False,
    )

    target = _target(manifest, "hotpotqa_24")
    assert target["status"] == "error"
    assert target["files"][0]["validation"]["status"] == "error"


def test_strict_dirty_requires_explicit_override(tmp_path: Path) -> None:
    omni = tmp_path / "omnifuse"
    synaptic = tmp_path / "synaptic-memory"
    _make_omnifuse(omni)
    _write(synaptic, "tests/benchmark/metrics.py", b"# shared scorer\n")
    for spec in bench.TARGETS:
        if spec.strict_public and spec.source == "synaptic":
            _write_synaptic_target(synaptic, spec)
    _commit_fixture(omni)
    _commit_fixture(synaptic)
    _write(omni, "dirty.txt", b"dirty\n")

    blocked = bench.build_manifest(
        omnifuse_repo=omni,
        synaptic_repo=synaptic,
        kra_golden=None,
        strict_public=True,
    )
    allowed = bench.build_manifest(
        omnifuse_repo=omni,
        synaptic_repo=synaptic,
        kra_golden=None,
        strict_public=True,
        allow_dirty=True,
    )

    assert blocked["strict_public"]["passed"] is False
    assert {item.get("status") for item in blocked["strict_public"]["blockers"]} >= {
        "dirty"
    }
    assert allowed["strict_public"]["passed"] is True
