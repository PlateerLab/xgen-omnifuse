import asyncio
import json
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))

import footprint_bench  # noqa: E402


def _worker_environment() -> dict:
    return {
        "python": footprint_bench.platform.python_version(),
        "python_executable": str(pathlib.Path(sys.executable).resolve()),
        "isolated": True,
        "ignore_environment": True,
        "no_user_site": True,
        "safe_path": True,
        "utf8_mode": True,
        "user_site_enabled": False,
        "pythonpath": None,
        "pythonhome": None,
        "pythonusersite": None,
        "python_no_user_site_env": "1",
    }


def _doctor_environment() -> dict:
    runtime = footprint_bench._runtime_environment_snapshot()
    return {
        key: runtime[key]
        for key in (
            "python",
            "python_implementation",
            "python_executable",
            "platform",
        )
    }


def _doctor_record(**extra) -> dict:
    return {
        "snapshot": {"environment": _doctor_environment()},
        **extra,
    }


def _write_public_dataset(repo: pathlib.Path) -> pathlib.Path:
    source = repo / "tests" / "benchmark" / "data" / "nfcorpus.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "corpus": {
                    "doc-b": {"title": "B", "text": "second"},
                    "doc-a": {"title": "A", "text": "first"},
                }
            }
        ),
        encoding="utf-8",
    )
    return source


def _prepared(source: pathlib.Path) -> footprint_bench.PreparedDataset:
    documents = footprint_bench._load_documents(source)
    payload = footprint_bench._normalized_payload(documents)
    spec = footprint_bench.DEFAULT_DATASETS["nfcorpus"]
    return footprint_bench.PreparedDataset(
        spec=spec,
        source=source.resolve(),
        documents=documents,
        provenance={
            "key": spec.key,
            "label": spec.label,
            "doctor_target_id": spec.doctor_target_id,
            "source": footprint_bench.file_fingerprint(
                source, display_path=spec.relative_path.as_posix()
            ),
            "normalized_document_payload": footprint_bench._bytes_fingerprint(
                payload, path="normalized/nfcorpus.json"
            ),
            "documents": len(documents),
        },
    )


def _snapshot() -> dict:
    scorer = {
        "active": {"path": "eval/metrics.py", "sha256": "1", "bytes": 1},
        "synaptic_checkout_copy": {
            "path": "tests/benchmark/metrics.py",
            "sha256": "1",
            "bytes": 1,
        },
        "byte_identical": True,
    }
    return {
        "repositories": {
            "omnifuse": {"path": "omni", "sha": "a"},
            "synaptic_memory": {"path": "synaptic", "sha": "b"},
        },
        "benchmark_sources": {
            "harness": {"sha256": "2"},
            "synaptic_native_driver": {"sha256": "3"},
            "scorer": scorer,
        },
        "inputs": [],
        "environment": footprint_bench._runtime_environment_snapshot(),
    }


def _measurement_row(prepared: footprint_bench.PreparedDataset) -> dict:
    return {
        "label": prepared.spec.label,
        "input": prepared.provenance,
        "artifacts": {
            "omnifuse": {"bytes": 41, "files": {"index.pkl.gz": 41}},
            "synaptic": {"bytes": 82, "files": {"graph.sqlite": 82}},
        },
        "runtime_source_bindings": {
            "omnifuse": {"package": {"path": "src/omnifuse/__init__.py"}},
            "synaptic": {"package": {"path": "src/synaptic/__init__.py"}},
        },
        "ratio_synaptic_over_omnifuse": 2.0,
    }


def test_normalized_payload_is_deterministic_and_shared():
    documents = footprint_bench._normalize_documents(
        {
            "corpus": {
                "doc-b": {"title": "B", "text": "second"},
                "doc-a": {"title": "", "text": "first"},
            }
        }
    )

    assert [document["id"] for document in documents] == ["doc-a", "doc-b"]
    assert documents[0]["title"] == "doc-a"
    assert json.loads(footprint_bench._normalized_payload(documents)) == documents


def test_atomic_json_output_is_write_once_and_preserves_existing_content(tmp_path):
    output = tmp_path / "nested" / "result.json"
    footprint_bench._atomic_write_json(output, {"ready": True})
    assert output.read_text(encoding="utf-8") == '{\n  "ready": true\n}\n'

    with pytest.raises(footprint_bench.ProvenanceError, match="refusing to overwrite"):
        footprint_bench._atomic_write_json(output, {"ready": False})

    assert output.read_text(encoding="utf-8") == '{\n  "ready": true\n}\n'
    assert not list(output.parent.glob(".*.tmp"))


def test_machine_output_requires_synaptic_checkout_and_doctor(tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as missing_repo:
        footprint_bench.main(["--out", str(output)])
    assert missing_repo.value.code == 2

    with pytest.raises(SystemExit) as missing_doctor:
        footprint_bench.main(
            ["--synaptic-repo", str(tmp_path / "synaptic"), "--out", str(output)]
        )
    assert missing_doctor.value.code == 2


def test_strict_machine_output_rejects_unbound_external_corpus(tmp_path):
    repo = tmp_path / "synaptic"
    repo.mkdir()
    with pytest.raises(SystemExit) as error:
        footprint_bench.main(
            [
                "--synaptic-repo",
                str(repo),
                "--corpus",
                str(tmp_path / "external.json"),
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(tmp_path / "result.json"),
            ]
        )
    assert error.value.code == 2


def test_existing_machine_output_is_rejected_before_input_loading(
    tmp_path, monkeypatch
):
    repo = tmp_path / "synaptic"
    _write_public_dataset(repo)
    output = tmp_path / "result.json"
    output.write_bytes(b"original artifact\n")

    def unexpected_prepare(_specs):
        raise AssertionError("input must not load after output preflight fails")

    monkeypatch.setattr(footprint_bench, "_prepare_inputs", unexpected_prepare)
    with pytest.raises(SystemExit) as error:
        footprint_bench.main(
            [
                "--synaptic-repo",
                str(repo),
                "--dataset",
                "nfcorpus",
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(output),
            ]
        )

    assert error.value.code == 2
    assert output.read_bytes() == b"original artifact\n"


def test_machine_preflight_binds_doctor_to_input_repositories_and_scorer(
    tmp_path, monkeypatch
):
    repo = tmp_path / "synaptic"
    source = _write_public_dataset(repo)
    prepared = _prepared(source)
    snapshot = _snapshot()
    snapshot["inputs"] = [prepared.provenance]
    captured = {}

    monkeypatch.setattr(footprint_bench, "_prepare_inputs", lambda _specs: [prepared])
    monkeypatch.setattr(footprint_bench, "_snapshot", lambda _repo, _prepared: snapshot)

    def load_doctor(path, inputs):
        captured["doctor_path"] = path
        captured["inputs"] = inputs
        return _doctor_record(path=str(path)), {
            "nfcorpus": {"target_id": "nfcorpus", "target_status": "ok"}
        }

    def verify_runtime(record, **bindings):
        captured["doctor"] = record
        captured["bindings"] = bindings

    monkeypatch.setattr(footprint_bench, "load_doctor_manifest", load_doctor)
    monkeypatch.setattr(footprint_bench, "verify_doctor_runtime", verify_runtime)

    state, actual_prepared = footprint_bench._machine_preflight(
        output=tmp_path / "result.json",
        doctor_manifest=tmp_path / "doctor.json",
        synaptic_repo=repo,
        specs=[(prepared.spec, source)],
    )

    assert actual_prepared == [prepared]
    assert state["before"] == snapshot
    assert captured["inputs"] == [
        {
            "name": "nfcorpus",
            "target_id": "nfcorpus",
            "path": "tests/benchmark/data/nfcorpus.json",
            "sha256": prepared.provenance["source"]["sha256"],
            "bytes": prepared.provenance["source"]["bytes"],
        }
    ]
    assert captured["bindings"] == {
        "omnifuse_repository": snapshot["repositories"]["omnifuse"],
        "synaptic_repository": snapshot["repositories"]["synaptic_memory"],
        "omnifuse_scorer": snapshot["benchmark_sources"]["scorer"]["active"],
        "synaptic_scorer": snapshot["benchmark_sources"]["scorer"][
            "synaptic_checkout_copy"
        ],
    }


def test_postflight_serializes_after_snapshot_and_rechecks_doctor(
    tmp_path, monkeypatch
):
    repo = tmp_path / "synaptic"
    source = _write_public_dataset(repo)
    prepared = _prepared(source)
    snapshot = _snapshot()
    snapshot["inputs"] = [prepared.provenance]
    doctor = _doctor_record(path="doctor")
    state = {
        "before": snapshot,
        "doctor_manifest": doctor,
        "doctor_environment": _doctor_environment(),
    }
    calls = []

    monkeypatch.setattr(footprint_bench, "_prepare_inputs", lambda _specs: [prepared])
    monkeypatch.setattr(footprint_bench, "_snapshot", lambda _repo, _prepared: snapshot)
    monkeypatch.setattr(
        footprint_bench,
        "verify_doctor_manifest",
        lambda doctor: calls.append(("manifest", doctor)),
    )
    monkeypatch.setattr(
        footprint_bench,
        "verify_doctor_runtime",
        lambda doctor, **bindings: calls.append(("runtime", doctor, bindings)),
    )

    postflight = footprint_bench._verify_machine_postflight(
        state, synaptic_repo=repo, specs=[(prepared.spec, source)]
    )

    assert postflight["after"] == snapshot
    assert postflight["checks"]["doctor_runtime_binding_reverified"] is True
    assert [call[0] for call in calls] == ["manifest", "runtime"]


def test_postflight_fails_closed_when_input_changes(tmp_path, monkeypatch):
    repo = tmp_path / "synaptic"
    source = _write_public_dataset(repo)
    before_prepared = _prepared(source)
    before = _snapshot()
    before["inputs"] = [before_prepared.provenance]
    state = {"before": before, "doctor_manifest": {}}
    source.write_text('{"corpus":{"different":"body"}}', encoding="utf-8")

    monkeypatch.setattr(
        footprint_bench,
        "_repository_snapshot",
        lambda _repo: before["repositories"],
    )
    monkeypatch.setattr(
        footprint_bench,
        "_benchmark_sources",
        lambda _repo: before["benchmark_sources"],
    )

    with pytest.raises(footprint_bench.ProvenanceError, match="dataset inputs changed"):
        footprint_bench._verify_machine_postflight(
            state, synaptic_repo=repo, specs=[(before_prepared.spec, source)]
        )


def test_benchmark_sources_fingerprint_driver_and_byte_identical_scorer(
    tmp_path, monkeypatch
):
    repo = tmp_path / "synaptic"
    scorer = repo / footprint_bench.SYNAPTIC_SCORER_RELATIVE
    driver = repo / footprint_bench.SYNAPTIC_DRIVER_RELATIVE
    package_file = repo / "src" / "synaptic" / "__init__.py"
    scorer.parent.mkdir(parents=True)
    driver.parent.mkdir(parents=True)
    package_file.parent.mkdir(parents=True)
    scorer.write_bytes((footprint_bench.EVAL_DIR / "metrics.py").read_bytes())
    driver.write_text("# native driver\n", encoding="utf-8")
    package_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        footprint_bench,
        "_source_tree_fingerprint",
        lambda source, **_kwargs: {"path": str(source), "sha256": "tree"},
    )

    before = footprint_bench._benchmark_sources(repo)
    assert before["scorer"]["byte_identical"] is True
    assert before["synaptic_native_driver"]["path"] == "eval/run_all.py"

    driver.write_text("# changed native driver\n", encoding="utf-8")
    assert footprint_bench._benchmark_sources(repo) != before

    scorer.write_text("# different scorer\n", encoding="utf-8")
    with pytest.raises(footprint_bench.ProvenanceError, match="not byte-identical"):
        footprint_bench._benchmark_sources(repo)


def test_worker_result_is_write_once_before_input_or_import(tmp_path, monkeypatch):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]", encoding="utf-8")
    result_file = tmp_path / "worker.json"
    result_file.write_text("original\n", encoding="utf-8")

    def unexpected_load(_path):
        raise AssertionError("worker must reject its result before loading input")

    monkeypatch.setattr(footprint_bench, "_load_documents", unexpected_load)
    with pytest.raises(footprint_bench.ProvenanceError, match="refusing to overwrite"):
        footprint_bench._run_worker(
            SimpleNamespace(
                worker="omnifuse",
                input_file=input_file,
                result_file=result_file,
                synaptic_repo=None,
            )
        )

    assert result_file.read_text(encoding="utf-8") == "original\n"


def test_omnifuse_worker_records_actual_checkout_source_bindings(tmp_path):
    documents = [{"id": "doc", "title": "Title", "text": "body"}]
    input_file = tmp_path / "input.json"
    input_file.write_bytes(footprint_bench._normalized_payload(documents))
    result_file = tmp_path / "worker.json"

    footprint_bench._run_worker(
        SimpleNamespace(
            worker="omnifuse",
            input_file=input_file,
            result_file=result_file,
            synaptic_repo=None,
        )
    )

    result = json.loads(result_file.read_text(encoding="utf-8"))
    bindings = result["source_bindings"]
    assert set(bindings) == {"package", "build_inmemory", "save_index"}
    for binding in bindings.values():
        assert pathlib.Path(binding["resolved_path"]).is_relative_to(
            footprint_bench.OMNIFUSE_SOURCE
        )
        assert len(binding["sha256"]) == 64
    assert result["artifact"]["bytes"] == sum(result["artifact"]["files"].values())


def test_worker_result_binds_actual_imported_source_and_detects_mutation(tmp_path):
    repo = tmp_path / "synaptic"
    source = _write_public_dataset(repo)
    prepared = _prepared(source)
    imported = repo / "src" / "synaptic" / "__init__.py"
    imported.parent.mkdir(parents=True)
    imported.write_text("# source\n", encoding="utf-8")
    binding = {
        **footprint_bench.file_fingerprint(
            imported, display_path="src/synaptic/__init__.py"
        ),
        "resolved_path": str(imported.resolve()),
    }
    expected_input = prepared.provenance["normalized_document_payload"]
    result = {
        "schema": footprint_bench.WORKER_RESULT_SCHEMA,
        "schema_version": footprint_bench.WORKER_RESULT_SCHEMA_VERSION,
        "status": "ok",
        "system": "synaptic",
        "input": {
            "path": footprint_bench.WORKER_INPUT_DISPLAY_PATH,
            "sha256": expected_input["sha256"],
            "bytes": expected_input["bytes"],
            "documents": len(prepared.documents),
        },
        "artifact": {"bytes": 2, "files": {"graph.sqlite": 2}},
        "source_bindings": {
            "package": binding,
            "sqlite_backend": binding,
            "sqlite_backend_base": binding,
            "graph": binding,
        },
        "environment": _worker_environment(),
        "tokenizer": {
            "mode": "regex_fallback",
            "korean_normalization_used": True,
            "kiwi_available": False,
            "kiwi_version": None,
            "kiwi_model_version": None,
            "modules": {name: None for name in footprint_bench.TOKENIZER_MODULE_NAMES},
        },
    }

    incomplete = json.loads(json.dumps(result))
    del incomplete["source_bindings"]["graph"]
    with pytest.raises(
        footprint_bench.ProvenanceError, match="bindings are incomplete"
    ):
        footprint_bench._validate_worker_result(
            incomplete,
            system="synaptic",
            prepared=prepared,
            synaptic_repo=repo,
        )

    extra = json.loads(json.dumps(result))
    extra["unexpected"] = True
    with pytest.raises(footprint_bench.ProvenanceError, match="strict schema"):
        footprint_bench._validate_worker_result(
            extra,
            system="synaptic",
            prepared=prepared,
            synaptic_repo=repo,
        )

    boolean_count = json.loads(json.dumps(result))
    boolean_count["input"]["documents"] = True
    with pytest.raises(footprint_bench.ProvenanceError, match="input fields"):
        footprint_bench._validate_worker_result(
            boolean_count,
            system="synaptic",
            prepared=prepared,
            synaptic_repo=repo,
        )

    assert (
        footprint_bench._validate_worker_result(
            result,
            system="synaptic",
            prepared=prepared,
            synaptic_repo=repo,
        )
        == result
    )

    imported.write_text("# mutated source\n", encoding="utf-8")
    with pytest.raises(footprint_bench.ProvenanceError, match="imported source"):
        footprint_bench._validate_worker_result(
            result,
            system="synaptic",
            prepared=prepared,
            synaptic_repo=repo,
        )


def test_strict_machine_worker_rejects_korean_regex_fallback():
    evidence = {
        "mode": "regex_fallback",
        "korean_normalization_used": True,
        "kiwi_available": False,
        "kiwi_version": None,
        "kiwi_model_version": None,
        "modules": {name: None for name in footprint_bench.TOKENIZER_MODULE_NAMES},
    }
    with pytest.raises(footprint_bench.ProvenanceError, match="requires Kiwi"):
        footprint_bench._validate_tokenizer_evidence(
            evidence, system="synaptic", require_kiwi=True
        )


def test_worker_environment_rejects_inherited_pythonhome():
    environment = _worker_environment()
    environment["pythonhome"] = "untrusted"
    with pytest.raises(footprint_bench.ProvenanceError, match="inherited pythonhome"):
        footprint_bench._validate_worker_environment(environment, system="synaptic")


def test_machine_report_contains_before_after_snapshots_and_new_schema():
    snapshot = _snapshot()
    state = {"before": snapshot, "doctor_manifest": {"schema": "doctor"}}
    postflight = {
        "after": snapshot,
        "checks": {"postflight_verified_before_publish": True},
    }
    report = footprint_bench._build_report(
        rows={"nfcorpus": {}}, state=state, postflight=postflight
    )

    assert report["schema_version"] == 3
    assert report["provenance_level"] == footprint_bench.PROVENANCE_LEVEL
    assert report["provenance"]["before"] == snapshot
    assert report["provenance"]["after"] == snapshot
    assert report["provenance"]["mode"] == "strict-machine"
    assert "not relabeled" in report["historical_artifact_policy"]


def test_interactive_report_does_not_claim_strict_or_write_once_provenance():
    snapshot = _snapshot()
    state = {"before": snapshot, "doctor_manifest": None}
    postflight = {
        "after": snapshot,
        "checks": {
            "postflight_verified_before_publish": False,
            "doctor_manifest_unchanged": False,
        },
    }

    report = footprint_bench._build_report(
        rows={"nfcorpus": {}}, state=state, postflight=postflight
    )

    assert report["provenance_level"] == footprint_bench.INTERACTIVE_PROVENANCE_LEVEL
    assert report["provenance"]["mode"] == "interactive"
    assert "strict" not in report["provenance_level"]
    assert "write-once" not in report["provenance_level"]


def test_main_writes_strict_report_without_relabeling_historical_artifacts(
    tmp_path, monkeypatch
):
    repo = tmp_path / "synaptic"
    source = _write_public_dataset(repo)
    prepared = _prepared(source)
    snapshot = _snapshot()
    snapshot["inputs"] = [prepared.provenance]
    state = {
        "before": snapshot,
        "doctor_manifest": {"schema": "doctor"},
        "doctor_links": {"nfcorpus": {"target_id": "nfcorpus"}},
    }
    postflight = {
        "after": snapshot,
        "checks": {"postflight_verified_before_publish": True},
    }
    output = tmp_path / "report.json"

    monkeypatch.setattr(
        footprint_bench,
        "_machine_preflight",
        lambda **_kwargs: (state, [prepared]),
    )
    monkeypatch.setattr(
        footprint_bench,
        "_measure_prepared_dataset",
        lambda _prepared, **_kwargs: _measurement_row(prepared),
    )
    monkeypatch.setattr(
        footprint_bench,
        "_verify_machine_postflight",
        lambda *_args, **_kwargs: postflight,
    )

    assert (
        footprint_bench.main(
            [
                "--synaptic-repo",
                str(repo),
                "--dataset",
                "nfcorpus",
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["datasets"]["nfcorpus"]["doctor"]["target_id"] == "nfcorpus"
    assert report["datasets"]["nfcorpus"]["ratio_synaptic_over_omnifuse"] == 2.0
    assert report["integrity"]["postflight_verified_before_publish"] is True
    assert "2026-07-13" in report["historical_artifact_policy"]


def test_synaptic_loader_rejects_an_already_loaded_wrong_source(tmp_path, monkeypatch):
    repo = tmp_path / "synaptic-checkout"
    package_init = repo / "src" / "synaptic" / "__init__.py"
    package_init.parent.mkdir(parents=True)
    package_init.write_text("", encoding="utf-8")
    wrong_module = types.ModuleType("synaptic")
    wrong_module.__file__ = str(tmp_path / "site-packages" / "synaptic" / "__init__.py")
    monkeypatch.setitem(sys.modules, "synaptic", wrong_module)

    with pytest.raises(RuntimeError, match="expected checkout source"):
        footprint_bench._load_synaptic_api(repo)


def test_synaptic_measurement_closes_graph_and_removes_tempdir_on_failure():
    state = {"closed": False, "database": None}

    class FakeBackend:
        def __init__(self, path):
            state["database"] = pathlib.Path(path)

    class FakeGraph:
        def __init__(self, backend, **_kwargs):
            self.backend = backend

        async def connect(self):
            state["database"].write_bytes(b"sqlite")

        async def add(self, **_kwargs):
            raise RuntimeError("ingest failed")

        async def close(self):
            state["closed"] = True

    with pytest.raises(RuntimeError, match="ingest failed"):
        asyncio.run(
            footprint_bench._measure_synaptic(
                [{"id": "d", "title": "title", "text": "text"}],
                FakeBackend,
                FakeGraph,
            )
        )

    assert state["closed"] is True
    assert state["database"] is not None
    assert not state["database"].parent.exists()


def test_synaptic_measurement_counts_files_after_clean_close_and_cleans_tempdir():
    state = {"database": None}

    class FakeBackend:
        def __init__(self, path):
            state["database"] = pathlib.Path(path)

    class FakeGraph:
        def __init__(self, backend, **_kwargs):
            self.backend = backend

        async def connect(self):
            state["database"].write_bytes(b"db")

        async def add(self, **_kwargs):
            return None

        async def close(self):
            state["database"].with_suffix(".sqlite-wal").write_bytes(b"wal")

    measurement = asyncio.run(
        footprint_bench._measure_synaptic(
            [{"id": "d", "title": "title", "text": "text"}],
            FakeBackend,
            FakeGraph,
        )
    )

    assert measurement.bytes == 5
    assert measurement.files == {"graph.sqlite": 2, "graph.sqlite-wal": 3}
    assert not state["database"].parent.exists()


def test_module_import_does_not_parse_process_arguments():
    assert callable(footprint_bench.main)
    assert footprint_bench.sys.dont_write_bytecode is True
