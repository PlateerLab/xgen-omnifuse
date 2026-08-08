from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "idf_pow_bench.py"
SPEC = importlib.util.spec_from_file_location("omnifuse_idf_pow_bench", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
idf_bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = idf_bench
SPEC.loader.exec_module(idf_bench)

import provenance as provenance_helper  # noqa: E402


def _dataset_input(path: Path, name: str = "HotPotQA-24") -> dict:
    return {
        "name": name,
        "filename": path.name,
        "path": path,
        "fingerprint": idf_bench._file_fingerprint(
            path, display_path=f"tests/benchmark/data/{path.name}"
        ),
    }


def _make_synaptic_runtime_sources(repo: Path) -> None:
    scorer = repo / idf_bench.public_bench_driver.SYNAPTIC_SCORER_RELATIVE
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_bytes((idf_bench.EVAL_ROOT / "metrics.py").read_bytes())
    driver = repo / idf_bench.public_bench_driver.SYNAPTIC_DRIVER_RELATIVE
    driver.parent.mkdir(parents=True, exist_ok=True)
    driver.write_text("# native benchmark driver fixture\n", encoding="utf-8")


def test_declares_exact_public_ir_targets_and_five_float_arms() -> None:
    assert len(idf_bench.DATASETS) == 17
    assert len({name for name, _ in idf_bench.DATASETS}) == 17
    assert len({filename for _, filename in idf_bench.DATASETS}) == 17
    assert idf_bench.ARMS == (1.0, 1.1, 1.2, 1.3, 1.5)
    assert all(type(arm) is float for arm in idf_bench.ARMS)
    assert idf_bench.SHIPPED_ARM == 1.2


def test_run_benchmark_preserves_unrounded_floats_and_all_arms(tmp_path: Path) -> None:
    dataset = tmp_path / "hotpotqa_24.json"
    dataset.write_text("{}\n", encoding="utf-8")
    exact = {
        1.0: 0.12345678901234566,
        1.1: 0.22345678901234567,
        1.2: 0.3234567890123457,
        1.3: 0.4234567890123457,
        1.5: 0.5234567890123457,
    }
    calls: list[tuple[Path, Path, str]] = []

    def omni_runner(path: Path, arm: float) -> float:
        assert path == dataset
        return exact[arm]

    async def synaptic_runner(repo: Path, path: Path, name: str) -> float:
        calls.append((repo, path, name))
        return 0.23456789012345678

    rows = idf_bench.run_benchmark(
        tmp_path / "synaptic",
        [_dataset_input(dataset)],
        omni_runner=omni_runner,
        synaptic_runner=synaptic_runner,
        emit=lambda _line: None,
    )

    row = rows["HotPotQA-24"]
    assert row["synaptic"] == 0.23456789012345678
    for arm in idf_bench.ARMS:
        assert row[f"omnifuse_idf_pow_{arm}"] == exact[arm]
    assert row["delta_1.5_minus_1.0"] == exact[1.5] - exact[1.0]
    assert len(row["input"]["sha256"]) == 64
    assert calls == [(tmp_path / "synaptic", dataset, "HotPotQA-24")]


def test_omni_scoring_discards_empty_document_ids(monkeypatch) -> None:
    graph = SimpleNamespace(
        vector=SimpleNamespace(_bm25=SimpleNamespace(idf={"term": 1.0})),
        retrieve=lambda _query, limit: [
            (SimpleNamespace(id=""), 1.0),
            (SimpleNamespace(id="doc"), 0.5),
        ][:limit],
    )
    monkeypatch.setattr(idf_bench, "build_inmemory", lambda *_args, **_kwargs: graph)

    score = idf_bench._omni_mrr_loaded([], [("q", "term", {"doc"})], 1.2)

    assert score == 1.0


def test_run_benchmark_reports_ties_without_calling_them_losses(tmp_path: Path) -> None:
    dataset = tmp_path / "hotpotqa_24.json"
    dataset.write_text("{}\n", encoding="utf-8")
    output: list[str] = []

    async def synaptic_runner(_repo: Path, _path: Path, _name: str) -> float:
        return 0.5

    idf_bench.run_benchmark(
        tmp_path / "synaptic",
        [_dataset_input(dataset)],
        omni_runner=lambda _path, _arm: 0.5,
        synaptic_runner=synaptic_runner,
        emit=output.append,
    )

    assert "p=1.0 ties" in output[0]
    assert "LOSES" not in output[0]


def test_no_available_rows_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(idf_bench.BenchmarkError, match="required dataset file"):
        idf_bench._dataset_inputs(
            tmp_path / "missing-synaptic",
            {"HotPotQA-24"},
            lambda _line: None,
        )


def test_default_scope_fails_if_even_one_declared_dataset_is_missing(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "tests" / "benchmark" / "data"
    data_root.mkdir(parents=True)
    for name, filename in idf_bench.DATASETS[:-1]:
        assert name != "MultiLongDoc-ko"
        (data_root / filename).write_text("{}\n", encoding="utf-8")

    with pytest.raises(idf_bench.BenchmarkError, match="MultiLongDoc-ko"):
        idf_bench._dataset_inputs(tmp_path, None, lambda _line: None)


def test_atomic_json_creates_parent_refuses_overwrite_and_leaves_no_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "results" / "idf.json"
    payload = {"exact": 0.12345678901234566}

    idf_bench._atomic_write_json(output, payload)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        idf_bench._atomic_write_json(output, {"exact": 0.0})

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_main_accepts_argv_and_creates_output_parent(
    tmp_path: Path, monkeypatch
) -> None:
    payload = {"exact": 0.9876543210987654}
    monkeypatch.setattr(idf_bench, "execute_benchmark", lambda **_kwargs: payload)
    output = tmp_path / "new" / "parent" / "report.json"
    doctor = tmp_path / "doctor.json"

    exit_code = idf_bench.main(
        [
            "--synaptic-repo",
            str(tmp_path / "synaptic"),
            "--doctor-manifest",
            str(doctor),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_main_requires_doctor_for_machine_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        idf_bench.main(
            [
                "--synaptic-repo",
                str(tmp_path / "synaptic"),
                "--out",
                str(tmp_path / "report.json"),
            ]
        )

    assert raised.value.code == 2


def test_main_refuses_existing_output_before_benchmark(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "report.json"
    output.write_text("preserve me\n", encoding="utf-8")
    monkeypatch.setattr(
        idf_bench,
        "execute_benchmark",
        lambda **_kwargs: pytest.fail("benchmark should not start"),
    )

    exit_code = idf_bench.main(
        [
            "--synaptic-repo",
            str(tmp_path / "synaptic"),
            "--doctor-manifest",
            str(tmp_path / "doctor.json"),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 2
    assert output.read_text(encoding="utf-8") == "preserve me\n"


def test_driver_and_scorer_sources_are_validated(tmp_path: Path, monkeypatch) -> None:
    synaptic = tmp_path / "synaptic"
    _make_synaptic_runtime_sources(synaptic)
    provenance = idf_bench._source_provenance(synaptic)
    assert provenance["synaptic_driver_wrapper"]["path"] == "eval/public_bench.py"
    assert provenance["synaptic_native_driver"]["path"] == "eval/run_all.py"
    assert provenance["scorer"]["active"]["path"] == "eval/metrics.py"
    assert provenance["scorer"]["byte_identical"] is True

    async def wrong_driver(_repo: Path, _path: Path, _name: str) -> float:
        return 0.0

    monkeypatch.setattr(idf_bench.public_bench_driver, "synaptic_mrr", wrong_driver)
    with pytest.raises(idf_bench.BenchmarkError, match="loaded from"):
        idf_bench._source_provenance(synaptic)


def test_source_provenance_rejects_a_different_upstream_scorer(
    tmp_path: Path,
) -> None:
    synaptic = tmp_path / "synaptic"
    _make_synaptic_runtime_sources(synaptic)
    scorer = synaptic / idf_bench.public_bench_driver.SYNAPTIC_SCORER_RELATIVE
    scorer.write_text("# different scorer\n", encoding="utf-8")

    with pytest.raises(idf_bench.BenchmarkError, match="not byte-identical"):
        idf_bench._source_provenance(synaptic)


def test_doctor_manifest_links_target_and_requires_matching_dataset_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance_helper, "_validate_strict_public_contract", lambda manifest: None
    )
    dataset = tmp_path / "hotpotqa_24.json"
    dataset.write_text('{"corpus": {}}\n', encoding="utf-8")
    dataset_input = _dataset_input(dataset)
    manifest = {
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
                "total_targets": len(provenance_helper.STRICT_PUBLIC_TARGET_IDS),
                "ok_targets": len(provenance_helper.STRICT_PUBLIC_TARGET_IDS),
                "incomplete_targets": [],
            }
        },
        "targets": [
            {
                "id": "hotpotqa_24",
                "name": "HotPotQA-24",
                "strict_public": True,
                "status": "ok",
                "sha256": dataset_input["fingerprint"]["sha256"],
                "sha256_kind": "file",
                "bytes": dataset_input["fingerprint"]["bytes"],
                "files": [
                    {
                        "role": "dataset",
                        "path": "tests/benchmark/data/hotpotqa_24.json",
                        "status": "ok",
                        "sha256": dataset_input["fingerprint"]["sha256"],
                        "bytes": dataset_input["fingerprint"]["bytes"],
                        "validation": {"status": "ok"},
                    }
                ],
            },
            *[
                {
                    "id": target_id,
                    "name": target_id,
                    "strict_public": True,
                    "status": "ok",
                    "sha256": "f" * 64,
                    "sha256_kind": "file",
                    "bytes": 1,
                    "files": [
                        {
                            "role": "dataset",
                            "path": f"fixtures/{target_id}.json",
                            "status": "ok",
                            "sha256": "f" * 64,
                            "bytes": 1,
                            "validation": {"status": "ok"},
                        }
                    ],
                }
                for target_id in sorted(
                    provenance_helper.STRICT_PUBLIC_TARGET_IDS - {"hotpotqa_24"}
                )
            ],
        ],
    }
    doctor = tmp_path / "doctor.json"
    doctor.write_text(json.dumps(manifest), encoding="utf-8")

    provenance, links = idf_bench._doctor_provenance(doctor, [dataset_input])

    assert provenance["path"] == str(doctor.resolve())
    assert len(provenance["sha256"]) == 64
    assert len(provenance["canonical_sha256"]) == 64
    assert provenance["snapshot"] == manifest
    assert links["HotPotQA-24"] == {
        "target_id": "hotpotqa_24",
        "target_name": "HotPotQA-24",
        "target_status": "ok",
        "role": "dataset",
        "input_sha256": dataset_input["fingerprint"]["sha256"],
        "input_bytes": dataset_input["fingerprint"]["bytes"],
        "dataset_sha256": dataset_input["fingerprint"]["sha256"],
    }

    manifest["targets"][0]["files"][0]["sha256"] = "0" * 64
    doctor.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(idf_bench.BenchmarkError, match="doctor hash mismatch"):
        idf_bench._doctor_provenance(doctor, [dataset_input])


def test_execute_preflights_native_runner_before_dataset_work(
    tmp_path: Path, monkeypatch
) -> None:
    synaptic = tmp_path / "synaptic"
    _make_synaptic_runtime_sources(synaptic)
    stable_repository = {
        "path": "fixture",
        "sha": "a" * 40,
        "dirty": False,
        "status_sha256": "b" * 64,
    }
    monkeypatch.setattr(idf_bench, "_repository_state", lambda _repo: stable_repository)
    monkeypatch.setattr(
        idf_bench.public_bench_driver,
        "preflight_synaptic_runner",
        lambda _repo: (_ for _ in ()).throw(RuntimeError("broken import")),
    )
    monkeypatch.setattr(
        idf_bench,
        "_dataset_inputs",
        lambda *_args, **_kwargs: pytest.fail("dataset work must not begin"),
    )

    with pytest.raises(idf_bench.BenchmarkError, match="preflight failed"):
        idf_bench.execute_benchmark(synaptic_repo=synaptic, emit=lambda _line: None)


def test_execute_records_repositories_runtime_sources_and_dataset_hash(
    tmp_path: Path, monkeypatch
) -> None:
    synaptic = tmp_path / "synaptic"
    _make_synaptic_runtime_sources(synaptic)
    dataset = synaptic / "tests" / "benchmark" / "data" / "hotpotqa_24.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")

    def repository_state(repo: Path) -> dict:
        return {
            "path": str(repo.resolve()),
            "sha": "a" * 40,
            "dirty": True,
            "status_sha256": "b" * 64,
        }

    monkeypatch.setattr(idf_bench, "_repository_state", repository_state)

    def omni_runner(_path: Path, arm: float) -> float:
        return arm / 10.0

    async def synaptic_runner(_repo: Path, _path: Path, _name: str) -> float:
        return 0.05

    report = idf_bench.execute_benchmark(
        synaptic_repo=synaptic,
        requested={"HotPotQA-24"},
        omni_runner=omni_runner,
        synaptic_runner=synaptic_runner,
        emit=lambda _line: None,
    )

    assert report["arms"] == [1.0, 1.1, 1.2, 1.3, 1.5]
    assert report["repositories"]["omnifuse"]["sha"] == "a" * 40
    assert report["repositories"]["synaptic_memory"]["dirty"] is True
    assert report["repositories_after"] == report["repositories"]
    assert report["environment"]["python_executable"] == sys.executable
    assert report["environment"]["platform"]
    assert report["sources"].keys() == {
        "harness",
        "provenance_helper",
        "synaptic_driver_wrapper",
        "synaptic_native_driver",
        "omnifuse_imports",
        "scorer",
    }
    assert report["sources_after"] == report["sources"]
    assert len(report["datasets"]["HotPotQA-24"]["input"]["sha256"]) == 64
    assert report["doctor_manifest"] is None


def test_execute_rejects_dataset_mutation_after_scoring(
    tmp_path: Path, monkeypatch
) -> None:
    synaptic = tmp_path / "synaptic"
    _make_synaptic_runtime_sources(synaptic)
    dataset = synaptic / "tests" / "benchmark" / "data" / "hotpotqa_24.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")
    stable_repository = {
        "path": "fixture",
        "sha": "a" * 40,
        "dirty": False,
        "status_sha256": "b" * 64,
    }
    monkeypatch.setattr(idf_bench, "_repository_state", lambda _repo: stable_repository)
    mutated = False

    def omni_runner(path: Path, arm: float) -> float:
        nonlocal mutated
        if not mutated:
            path.write_text('{"changed": true}\n', encoding="utf-8")
            mutated = True
        return arm

    async def synaptic_runner(_repo: Path, _path: Path, _name: str) -> float:
        return 0.0

    with pytest.raises(idf_bench.BenchmarkError, match="dataset HotPotQA-24 changed"):
        idf_bench.execute_benchmark(
            synaptic_repo=synaptic,
            requested={"HotPotQA-24"},
            omni_runner=omni_runner,
            synaptic_runner=synaptic_runner,
            emit=lambda _line: None,
        )
