import asyncio
import json
import pathlib
import subprocess
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))

import public_bench  # noqa: E402
import provenance  # noqa: E402


def _dataset_payload() -> dict:
    return {
        "corpus": {"d1": {"title": "one", "text": "body"}},
        "queries": {"q1": "body"},
        "qrels": {"q1": {"d1": 1}},
    }


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _commit_fixture(repo: pathlib.Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.name", "Benchmark Test")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")


def test_scope_is_exactly_the_eight_tracked_public_datasets() -> None:
    assert len(public_bench.DATASETS) == 8
    assert {spec.filename for spec in public_bench.DATASETS} == {
        "hotpotqa_24.json",
        "hotpotqa.json",
        "allganize_rag_ko.json",
        "allganize_rag_eval.json",
        "publichealthqa_ko.json",
        "autorag_retrieval.json",
        "klue_mrc.json",
        "ko_strategyqa.json",
    }
    assert public_bench.CANDIDATE_LIMIT == public_bench.K * 2 == 20


def test_parse_public_rejects_a_dataset_with_no_scored_rows() -> None:
    with pytest.raises(ValueError, match="scored queries"):
        public_bench.parse_public(
            {
                "corpus": {"d1": {"text": "body"}},
                "queries": {"q1": "question"},
                "qrels": {},
            }
        )


def test_omnifuse_uses_shared_candidate_dedupe_and_top_k(monkeypatch) -> None:
    calls: list[int] = []

    class Chunk:
        def __init__(self, document_id: str) -> None:
            self.id = document_id

    class Graph:
        def retrieve(self, _query: str, *, limit: int):
            calls.append(limit)
            return [(Chunk(value), 1.0) for value in ["x", "x", "", "d1"]]

    monkeypatch.setattr(
        public_bench, "build_inmemory", lambda *_args, **_kwargs: Graph()
    )
    metrics = public_bench._score_omnifuse(
        [("d1", "title", "body")], [("q1", "body", {"d1"})]
    )

    assert calls == [20]
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["recall_at_10"] == 1.0


def test_dataset_input_records_exact_hash_counts_and_requires_git_tracking(
    tmp_path: pathlib.Path,
) -> None:
    repo = tmp_path / "synaptic"
    data_dir = repo / "tests" / "benchmark" / "data"
    data_dir.mkdir(parents=True)
    tracked = data_dir / "tracked.json"
    tracked.write_text(json.dumps(_dataset_payload()), encoding="utf-8")
    untracked = data_dir / "untracked.json"
    untracked.write_text(json.dumps(_dataset_payload()), encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", tracked.relative_to(repo).as_posix())

    record, corpus, queries = public_bench._dataset_input(repo, tracked)

    assert record["git_tracked"] is True
    assert record["documents"] == len(corpus) == 1
    assert record["scored_queries"] == len(queries) == 1
    assert record["relevance_judgments"] == 1
    assert len(record["sha256"]) == 64
    with pytest.raises(ValueError, match="not tracked"):
        public_bench._dataset_input(repo, untracked)


def test_execute_fails_before_scoring_when_any_required_dataset_is_missing(
    tmp_path: pathlib.Path,
) -> None:
    repo = tmp_path / "synaptic"
    source = repo / "src" / "synaptic" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixture\n", encoding="utf-8")
    driver = repo / public_bench.SYNAPTIC_DRIVER_RELATIVE
    driver.parent.mkdir(parents=True)
    driver.write_text("# fixture\n", encoding="utf-8")
    scorer = repo / public_bench.SYNAPTIC_SCORER_RELATIVE
    scorer.parent.mkdir(parents=True)
    scorer.write_bytes(public_bench.SCORER_PATH.read_bytes())
    data_dir = repo / "tests" / "benchmark" / "data"
    data_dir.mkdir(parents=True)
    for spec in public_bench.DATASETS:
        if spec.id != "ko_strategyqa":
            (data_dir / spec.filename).write_text(
                json.dumps(_dataset_payload()), encoding="utf-8"
            )
    _commit_fixture(repo)

    with pytest.raises(RuntimeError, match="Ko-StrategyQA"):
        public_bench.execute_benchmark(repo)


def test_atomic_json_is_write_once_and_leaves_no_temporary_file(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "nested" / "report.json"
    payload = {"exact": 0.12345678901234566}

    public_bench._atomic_write_json(output, payload)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        public_bench._atomic_write_json(output, {"exact": 0.0})

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_native_loader_verifies_package_driver_and_active_scorer_sources(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "synaptic"
    package_init = repo / "src" / "synaptic" / "__init__.py"
    driver = repo / "eval" / "run_all.py"
    scorer = repo / "tests" / "benchmark" / "metrics.py"
    for path in (package_init, driver, scorer):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    package = types.SimpleNamespace(__file__=str(package_init))
    benchmark_result = type("BenchmarkResult", (), {})
    runner = types.SimpleNamespace(
        __file__=str(driver),
        DatasetConfig=object,
        run_public_dataset=object(),
        BenchmarkResult=benchmark_result,
    )

    def import_module(name: str):
        return package if name == "synaptic" else runner

    monkeypatch.setattr(public_bench.importlib, "import_module", import_module)
    monkeypatch.setattr(
        public_bench.inspect,
        "getsourcefile",
        lambda value: str(scorer) if value is benchmark_result else None,
    )

    _config, _runner, paths = public_bench._load_synaptic_runner(repo)

    assert paths == {
        "package": str(package_init.resolve()),
        "native_driver": str(driver.resolve()),
        "native_scorer": str(scorer.resolve()),
    }
    assert sys.dont_write_bytecode is True


def test_omnifuse_import_provenance_rejects_cached_external_package(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    external_package = tmp_path / "omnifuse" / "__init__.py"
    external_package.parent.mkdir()
    external_package.write_text("# wrong package\n", encoding="utf-8")
    monkeypatch.setattr(
        public_bench.omnifuse_package, "__file__", str(external_package)
    )

    with pytest.raises(RuntimeError, match="omnifuse package loaded from"):
        public_bench._omnifuse_import_provenance()


def test_doctor_provenance_embeds_full_snapshot_and_canonical_hash(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provenance, "_validate_strict_public_contract", lambda manifest: None
    )
    inputs = {}
    targets = []
    for index, spec in enumerate(public_bench.DATASETS, 1):
        sha256 = f"{index:064x}"
        relative = f"tests/benchmark/data/{spec.filename}"
        inputs[spec.id] = (
            {"path": relative, "sha256": sha256, "bytes": index},
            [],
            [],
        )
        targets.append(
            {
                "id": spec.id,
                "name": spec.name,
                "strict_public": True,
                "status": "ok",
                "sha256": sha256,
                "sha256_kind": "file",
                "bytes": index,
                "files": [
                    {
                        "role": "dataset",
                        "path": relative,
                        "status": "ok",
                        "sha256": sha256,
                        "bytes": index,
                        "validation": {"status": "ok"},
                    }
                ],
            }
        )
    targets.extend(
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
            provenance.STRICT_PUBLIC_TARGET_IDS
            - {spec.id for spec in public_bench.DATASETS}
        )
    )
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
                "total_targets": len(provenance.STRICT_PUBLIC_TARGET_IDS),
                "ok_targets": len(provenance.STRICT_PUBLIC_TARGET_IDS),
                "incomplete_targets": [],
            }
        },
        "targets": targets,
    }
    doctor_path = tmp_path / "doctor.json"
    doctor_path.write_text(json.dumps(manifest), encoding="utf-8")

    doctor, links = public_bench._doctor_provenance(doctor_path, inputs)

    assert doctor is not None
    assert doctor["snapshot"] == manifest
    assert len(doctor["sha256"]) == 64
    assert len(doctor["canonical_sha256"]) == 64
    assert set(links) == {spec.name for spec in public_bench.DATASETS}


def test_synaptic_native_error_is_not_converted_to_a_zero_score(monkeypatch) -> None:
    class Config:
        def __init__(self, **_kwargs) -> None:
            pass

    async def runner(*_args, **_kwargs):
        return types.SimpleNamespace(error="no valid queries")

    monkeypatch.setattr(
        public_bench,
        "_load_synaptic_runner",
        lambda _repo: (Config, runner, {}),
    )

    with pytest.raises(RuntimeError, match="no valid queries"):
        asyncio.run(
            public_bench.synaptic_metrics(
                pathlib.Path("repo"), pathlib.Path("dataset.json"), "fixture"
            )
        )


def test_execute_reports_source_mutation_and_keeps_all_required_rows(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "synaptic"
    source_file = repo / "src" / "synaptic" / "__init__.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("VERSION = 1\n", encoding="utf-8")
    driver = repo / public_bench.SYNAPTIC_DRIVER_RELATIVE
    driver.parent.mkdir(parents=True)
    driver.write_text("# native runner\n", encoding="utf-8")
    scorer = repo / public_bench.SYNAPTIC_SCORER_RELATIVE
    scorer.parent.mkdir(parents=True)
    scorer.write_bytes(public_bench.SCORER_PATH.read_bytes())
    data_dir = repo / "tests" / "benchmark" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for spec in public_bench.DATASETS:
        (data_dir / spec.filename).write_text(
            json.dumps(_dataset_payload()), encoding="utf-8"
        )

    _git(repo, "init")
    _git(repo, "config", "user.name", "Benchmark Test")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    monkeypatch.setattr(
        public_bench, "_load_synaptic_runner", lambda _repo: (None, None, {})
    )

    mutated = False

    def score(_corpus, _queries):
        nonlocal mutated
        if not mutated:
            source_file.write_text("VERSION = 2\n", encoding="utf-8")
            mutated = True
        return {
            "mrr_at_10": 1.0,
            "precision_at_10": 1.0,
            "recall_at_10": 1.0,
            "ndcg_at_10": 1.0,
        }

    async def synaptic(_repo, _path, _name):
        return {
            "mrr_at_10": 0.5,
            "precision_at_10": 0.5,
            "recall_at_10": 0.5,
            "ndcg_at_10": 0.5,
            "reported_corpus_size": 1,
            "module_paths": {},
        }

    monkeypatch.setattr(public_bench, "_score_omnifuse", score)
    monkeypatch.setattr(public_bench, "synaptic_metrics", synaptic)

    report = public_bench.execute_benchmark(repo)

    assert report["status"] == "error"
    assert report["summary"]["completed"] == 8
    assert report["summary"]["wins"]["omnifuse"] == 8
    assert report["source_integrity"]["unchanged"] is False
    assert report["source_integrity"]["diff"]["modified"] == ["synaptic/__init__.py"]
    assert report["repositories_after"]["synaptic_memory"]["dirty"] is True
    assert report["provenance_after"] == report["provenance"]


def test_main_accepts_argv_and_writes_machine_report(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "synaptic"
    repo.mkdir()
    output = tmp_path / "new" / "report.json"
    report = {
        "status": "ok",
        "datasets": [],
        "summary": {
            "required": 0,
            "completed": 0,
            "failed": 0,
            "wins": {"omnifuse": 0, "synaptic_memory": 0, "tie": 0},
        },
        "exact": 0.9876543210987654,
    }
    monkeypatch.setattr(
        public_bench,
        "execute_benchmark",
        lambda _repo, doctor_manifest=None: report,
    )
    doctor = tmp_path / "doctor.json"

    exit_code = public_bench.main(
        [
            "--synaptic-repo",
            str(repo),
            "--doctor-manifest",
            str(doctor),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_main_requires_doctor_for_machine_output(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "synaptic"
    repo.mkdir()

    with pytest.raises(SystemExit) as raised:
        public_bench.main(
            ["--synaptic-repo", str(repo), "--out", str(tmp_path / "report.json")]
        )

    assert raised.value.code == 2


def test_main_refuses_existing_output_before_benchmark(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "synaptic"
    repo.mkdir()
    output = tmp_path / "report.json"
    output.write_text("preserve me\n", encoding="utf-8")
    monkeypatch.setattr(
        public_bench,
        "execute_benchmark",
        lambda *_args, **_kwargs: pytest.fail("benchmark should not start"),
    )

    with pytest.raises(SystemExit) as raised:
        public_bench.main(
            [
                "--synaptic-repo",
                str(repo),
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(output),
            ]
        )

    assert raised.value.code == 2
    assert output.read_text(encoding="utf-8") == "preserve me\n"
