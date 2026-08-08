import asyncio
import hashlib
import json
import pathlib
import sqlite3
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))

import compare_synaptic  # noqa: E402
import finreg_bench  # noqa: E402
from provenance import OutputExistsError, ProvenanceError  # noqa: E402


def _fake_scores():
    return {
        "single_hop": {
            "mrr": 1.0,
            "mean_ndcg@k": 1.0,
            "mean_recall@k": 1.0,
            "hits": 1,
            "n": 1,
        },
        "multi_hop": {
            "mrr": 1.0,
            "mean_ndcg@k": 1.0,
            "mean_recall@k": 1.0,
            "strict": 1,
            "n": 1,
        },
    }


def _fake_omnifuse_result():
    return {
        "corpus": {"documents": 2, "reference_edges": 1},
        "queries": {"single_hop": 1, "multi_hop": 1},
        "scores": _fake_scores(),
        "timing_seconds": {
            "rebuild": 0.01,
            "scoring_all_queries": 0.02,
            "total": 0.03,
        },
    }


def _fake_comparison_result():
    return {
        "scores": _fake_scores(),
        "timing_seconds": {
            "rebuild": 0.01,
            "query_and_score": 0.02,
            "total": 0.03,
        },
    }


def _fingerprint(path, digest):
    return {"path": path, "bytes": 1, "sha256": digest * 64}


def _make_sqlite_graph(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE records (value TEXT)")
    connection.commit()
    connection.close()
    return path


def _finreg_state(tmp_path):
    inputs = {
        "corpus": _fingerprint("eval/data/finreg/raw.jsonl", "a"),
        "single_hop_queries": _fingerprint("eval/data/queries/finreg.json", "b"),
        "multi_hop_queries": _fingerprint(
            "eval/data/queries/finreg_multihop.json", "c"
        ),
    }
    repositories = {
        "omnifuse": {"path": "omnifuse", "sha": "d" * 40},
        "synaptic_memory_doctor_reference": {
            "path": "synaptic",
            "sha": "e" * 40,
        },
    }
    sources = {
        "entrypoint": _fingerprint("eval/finreg_bench.py", "f"),
        "scorer": {
            "active": _fingerprint("eval/metrics.py", "1"),
            "synaptic_checkout_copy": _fingerprint("tests/benchmark/metrics.py", "1"),
            "byte_identical": True,
        },
    }
    doctor = tmp_path / "doctor.json"
    doctor.write_text("{}", encoding="utf-8")
    return {
        "inputs": inputs,
        "repositories": repositories,
        "sources": sources,
        "doctor_manifest": {"path": str(doctor), "snapshot": {}},
        "doctor_links": {
            "finreg_single": {"corpus": {}, "queries": {}},
            "finreg_multi": {"corpus": {}, "queries": {}},
        },
        "synaptic_repo": tmp_path / "synaptic",
    }


def _postflight_from(state):
    return {
        "inputs": state["inputs"],
        "repositories": state["repositories"],
        "sources": state["sources"],
        "integrity": {
            "preflight_completed_before_benchmark": True,
            "inputs_unchanged": True,
            "repository_states_unchanged": True,
            "benchmark_sources_unchanged": True,
            "doctor_manifest_unchanged": True,
            "doctor_runtime_binding_reverified": True,
            "postflight_verified_before_publish": True,
        },
    }


def test_finreg_machine_report_is_doctor_bound_write_once_and_serializes_both_states(
    tmp_path, monkeypatch
):
    output = tmp_path / "reports" / "finreg.json"
    doctor = tmp_path / "doctor.json"
    doctor.write_text("{}", encoding="utf-8")
    state = _finreg_state(tmp_path)
    postflight = _postflight_from(state)
    calls = []

    def fake_preflight(*, output, doctor_manifest):
        calls.append((output, doctor_manifest))
        return state

    monkeypatch.setattr(finreg_bench, "_machine_preflight", fake_preflight)
    monkeypatch.setattr(
        finreg_bench,
        "_run_omnifuse",
        lambda *, graph_fusion: _fake_omnifuse_result(),
    )
    monkeypatch.setattr(
        finreg_bench, "_verify_machine_postflight", lambda current: postflight
    )

    assert (
        finreg_bench.main(["--out", str(output), "--doctor-manifest", str(doctor)]) == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [(output, doctor)]
    assert report["schema"] == "omnifuse.eval.finreg"
    assert report["schema_version"] == 2
    assert report["provenance_level"] == finreg_bench.PROVENANCE_LEVEL
    assert report["inputs"] == {
        "before": state["inputs"],
        "after": state["inputs"],
    }
    assert report["repositories"] == {
        "before": state["repositories"],
        "after": state["repositories"],
    }
    assert report["provenance"]["benchmark_sources"] == {
        "before": state["sources"],
        "after": state["sources"],
    }
    assert set(report["provenance"]["doctor_targets"]) == {
        "finreg_single",
        "finreg_multi",
    }
    assert report["protocol_scope"] == {
        "kind": "omnifuse_local_only",
        "synaptic_memory_executed": False,
        "doctor_synaptic_checkout_role": "input_and_scorer_reference_only",
        "cross_artifact_score_comparison_valid": False,
    }
    assert report["integrity"]["postflight_verified_before_publish"] is True
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_finreg_machine_output_requires_doctor_manifest(tmp_path):
    with pytest.raises(SystemExit) as caught:
        finreg_bench.main(["--out", str(tmp_path / "result.json")])
    assert caught.value.code == 2


def test_finreg_existing_output_is_rejected_before_benchmark(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("old", encoding="utf-8")
    doctor = tmp_path / "doctor.json"
    called = False

    def should_not_run(*, graph_fusion):
        nonlocal called
        called = True
        return _fake_omnifuse_result()

    monkeypatch.setattr(finreg_bench, "_run_omnifuse", should_not_run)
    with pytest.raises(SystemExit) as caught:
        finreg_bench.main(["--out", str(output), "--doctor-manifest", str(doctor)])
    assert caught.value.code == 2
    assert output.read_text(encoding="utf-8") == "old"
    assert called is False


def test_write_once_json_never_replaces_an_existing_result(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("old", encoding="utf-8")

    with pytest.raises(OutputExistsError, match="write-once"):
        finreg_bench._atomic_write_json(output, {"complete": True})

    assert output.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_finreg_doctor_specs_bind_both_roles_for_both_targets():
    inputs = {
        "corpus": _fingerprint("eval/data/finreg/raw.jsonl", "a"),
        "single_hop_queries": _fingerprint("eval/data/queries/finreg.json", "b"),
        "multi_hop_queries": _fingerprint(
            "eval/data/queries/finreg_multihop.json", "c"
        ),
    }

    specs = finreg_bench._doctor_input_specs(inputs)

    assert [
        (item["name"], item["target_id"], item["role"], item["path"]) for item in specs
    ] == [
        (
            "finreg_single_corpus",
            "finreg_single",
            "corpus",
            "eval/data/finreg/raw.jsonl",
        ),
        (
            "finreg_single_queries",
            "finreg_single",
            "queries",
            "eval/data/queries/finreg.json",
        ),
        (
            "finreg_multi_corpus",
            "finreg_multi",
            "corpus",
            "eval/data/finreg/raw.jsonl",
        ),
        (
            "finreg_multi_queries",
            "finreg_multi",
            "queries",
            "eval/data/queries/finreg_multihop.json",
        ),
    ]


def test_finreg_postflight_fails_closed_when_a_source_changes(tmp_path, monkeypatch):
    state = _finreg_state(tmp_path)
    state["synaptic_repo"].mkdir()
    changed_sources = {**state["sources"], "entrypoint": _fingerprint("changed", "9")}
    monkeypatch.setattr(finreg_bench, "_input_records", lambda: state["inputs"])
    monkeypatch.setattr(
        finreg_bench,
        "repository_fingerprint",
        lambda repo: (
            state["repositories"]["omnifuse"]
            if repo == finreg_bench.REPOSITORY_ROOT
            else state["repositories"]["synaptic_memory_doctor_reference"]
        ),
    )
    monkeypatch.setattr(
        finreg_bench, "_benchmark_sources", lambda _repo: changed_sources
    )

    with pytest.raises(ProvenanceError, match="source fingerprints changed"):
        finreg_bench._verify_machine_postflight(state)


def test_finreg_postflight_fails_closed_when_an_input_changes(tmp_path, monkeypatch):
    state = _finreg_state(tmp_path)
    state["synaptic_repo"].mkdir()
    changed_inputs = {
        **state["inputs"],
        "single_hop_queries": _fingerprint("eval/data/queries/finreg.json", "9"),
    }
    monkeypatch.setattr(finreg_bench, "_input_records", lambda: changed_inputs)
    monkeypatch.setattr(
        finreg_bench,
        "repository_fingerprint",
        lambda repo: (
            state["repositories"]["omnifuse"]
            if repo == finreg_bench.REPOSITORY_ROOT
            else state["repositories"]["synaptic_memory_doctor_reference"]
        ),
    )
    monkeypatch.setattr(
        finreg_bench, "_benchmark_sources", lambda _repo: state["sources"]
    )

    with pytest.raises(ProvenanceError, match="finreg inputs changed"):
        finreg_bench._verify_machine_postflight(state)


def _comparison_state(tmp_path, graph):
    finreg = _finreg_state(tmp_path)
    repositories = {
        "omnifuse": finreg["repositories"]["omnifuse"],
        "synaptic_memory": finreg["repositories"]["synaptic_memory_doctor_reference"],
    }
    module_paths = {
        "package": str(tmp_path / "synaptic" / "src" / "synaptic" / "__init__.py"),
        "sqlite_backend_base": str(
            tmp_path / "synaptic" / "src" / "synaptic" / "backends" / "sqlite.py"
        ),
        "sqlite_backend": str(
            tmp_path / "synaptic" / "src" / "synaptic" / "backends" / "sqlite_graph.py"
        ),
        "evidence_search": str(
            tmp_path
            / "synaptic"
            / "src"
            / "synaptic"
            / "extensions"
            / "evidence_search.py"
        ),
    }
    sources = {
        **finreg["sources"],
        "imported_synaptic_memory": {
            name: _fingerprint(
                pathlib.Path(path).relative_to(tmp_path / "synaptic").as_posix(), "7"
            )
            for name, path in module_paths.items()
        },
    }
    graph_guard = compare_synaptic._graph_guard_preflight(graph)
    return {
        "inputs": finreg["inputs"],
        **graph_guard,
        "repositories": repositories,
        "sources": sources,
        "synaptic_module_paths": module_paths,
        "synaptic_python": compare_synaptic._python_executable_record(
            pathlib.Path(sys.executable)
        ),
        "doctor_manifest": finreg["doctor_manifest"],
        "doctor_links": finreg["doctor_links"],
    }


def test_compare_machine_report_requires_both_system_inputs_and_preserves_boundary(
    tmp_path, monkeypatch
):
    synaptic_repo = tmp_path / "synaptic"
    synaptic_repo.mkdir()
    graph = tmp_path / "finreg.sqlite"
    _make_sqlite_graph(graph)
    output = tmp_path / "comparison.json"
    doctor = tmp_path / "doctor.json"
    doctor.write_text("{}", encoding="utf-8")
    state = _comparison_state(tmp_path, graph)
    synaptic_result = {
        "scores": _fake_scores(),
        "timing_seconds": {
            "connect_query_and_score": 0.02,
            "query_only": 0.01,
            "prebuilt_index_rebuild": None,
        },
        "module_paths": state["synaptic_module_paths"],
    }
    postflight = {
        "inputs": state["inputs"],
        "graph": state["graph"],
        "graph_data_version": state["graph_data_version_before"],
        "repositories": state["repositories"],
        "sources": state["sources"],
        "synaptic_python": state["synaptic_python"],
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
            "kiwi_active_pre_and_post": True,
            "doctor_manifest_unchanged": True,
            "doctor_runtime_binding_reverified": True,
            "postflight_verified_before_publish": True,
        },
    }
    single = [{"qid": "s1", "query": "single", "relevant_docs": ["d1"]}]
    multi = [{"qid": "m1", "query": "multi", "relevant_docs": ["d1", "d2"]}]
    monkeypatch.setattr(
        compare_synaptic,
        "_comparison_preflight",
        lambda **_kwargs: state,
    )
    monkeypatch.setattr(
        compare_synaptic,
        "load_queries",
        lambda name: (
            single if name == compare_synaptic.SINGLE_QUERY_PATH.name else multi
        ),
    )
    monkeypatch.setattr(
        compare_synaptic,
        "run_omnifuse",
        lambda _single, _multi: _fake_comparison_result(),
    )

    monkeypatch.setattr(
        compare_synaptic, "_run_synaptic_worker", lambda **_kwargs: {"worker": True}
    )
    monkeypatch.setattr(
        compare_synaptic,
        "_validate_worker_payload",
        lambda _payload, _state: synaptic_result,
    )
    monkeypatch.setattr(
        compare_synaptic,
        "_verify_comparison_postflight",
        lambda *_args, **_kwargs: postflight,
    )

    assert (
        compare_synaptic.main(
            [
                "--synaptic-repo",
                str(synaptic_repo),
                "--synaptic-graph",
                str(graph),
                "--synaptic-python",
                sys.executable,
                "--doctor-manifest",
                str(doctor),
                "--out",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["inputs"]["finreg"] == {
        "before": state["inputs"],
        "after": state["inputs"],
    }
    graph_report = report["inputs"]["synaptic_prebuilt_sqlite"]
    assert graph_report["before"] == state["graph"]
    assert graph_report["after"] == state["graph"]
    assert graph_report["watcher_open"] == state["graph_watcher_open"]
    assert graph_report["data_version"] == {
        "before": state["graph_data_version_before"],
        "after": state["graph_data_version_before"],
    }
    assert report["repositories"] == {
        "before": state["repositories"],
        "after": state["repositories"],
    }
    assert report["provenance"]["benchmark_sources"] == {
        "before": state["sources"],
        "after": state["sources"],
    }
    assert (
        report["index_conditions"]["synaptic_memory"][
            "ingestion_provenance_verified_against_corpus"
        ]
        is False
    )
    assert report["index_conditions"]["timings_directly_comparable"] is False
    assert report["protocol_scope"]["standalone_finreg_artifacts_combined"] is False
    assert report["protocol_scope"]["same_query_files"] is True
    assert (
        report["protocol_scope"]["synaptic_graph_matches_finreg_corpus_verified"]
        is False
    )
    assert report["protocol_scope"]["same_graph_build_recipe_verified"] is False
    assert "same_finreg_input_files" not in report["protocol_scope"]
    assert "must not be paired" in report["protocol_scope"]["boundary"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--synaptic-repo", "repo"],
        ["--synaptic-graph", "graph.sqlite"],
        [],
    ],
)
def test_compare_requires_checkout_and_graph(arguments):
    with pytest.raises(SystemExit) as caught:
        compare_synaptic.main(arguments)
    assert caught.value.code == 2


def test_compare_machine_output_requires_doctor_manifest(tmp_path):
    repo = tmp_path / "synaptic"
    repo.mkdir()
    graph = tmp_path / "graph.sqlite"
    graph.write_bytes(b"graph")
    with pytest.raises(SystemExit) as caught:
        compare_synaptic.main(
            [
                "--synaptic-repo",
                str(repo),
                "--synaptic-graph",
                str(graph),
                "--out",
                str(tmp_path / "result.json"),
            ]
        )
    assert caught.value.code == 2


def test_compare_machine_output_requires_synaptic_python(tmp_path):
    repo = tmp_path / "synaptic"
    repo.mkdir()
    graph = tmp_path / "graph.sqlite"
    graph.write_bytes(b"graph")
    doctor = tmp_path / "doctor.json"
    doctor.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        compare_synaptic.main(
            [
                "--synaptic-repo",
                str(repo),
                "--synaptic-graph",
                str(graph),
                "--doctor-manifest",
                str(doctor),
                "--out",
                str(tmp_path / "result.json"),
            ]
        )

    assert caught.value.code == 2


def test_compare_existing_output_is_rejected_before_either_system_runs(
    tmp_path, monkeypatch
):
    repo = tmp_path / "synaptic"
    repo.mkdir()
    graph = tmp_path / "graph.sqlite"
    graph.write_bytes(b"graph")
    output = tmp_path / "result.json"
    output.write_text("old", encoding="utf-8")
    called = False

    def should_not_run(_single, _multi):
        nonlocal called
        called = True
        return _fake_comparison_result()

    monkeypatch.setattr(compare_synaptic, "run_omnifuse", should_not_run)
    with pytest.raises(SystemExit) as caught:
        compare_synaptic.main(
            [
                "--synaptic-repo",
                str(repo),
                "--synaptic-graph",
                str(graph),
                "--synaptic-python",
                sys.executable,
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(output),
            ]
        )
    assert caught.value.code == 2
    assert output.read_text(encoding="utf-8") == "old"
    assert called is False


def test_compare_postflight_fails_closed_when_graph_changes(tmp_path, monkeypatch):
    repo = tmp_path / "synaptic"
    repo.mkdir()
    graph = tmp_path / "graph.sqlite"
    _make_sqlite_graph(graph)
    state = _comparison_state(tmp_path, graph)
    with graph.open("ab") as handle:
        handle.write(b"changed")
    monkeypatch.setattr(
        compare_synaptic,
        "_comparison_sources",
        lambda _repo: (state["sources"], state["synaptic_module_paths"]),
    )
    monkeypatch.setattr(compare_synaptic, "_input_records", lambda: state["inputs"])
    monkeypatch.setattr(
        compare_synaptic,
        "repository_fingerprint",
        lambda selected: (
            state["repositories"]["omnifuse"]
            if selected == compare_synaptic.REPOSITORY_ROOT
            else state["repositories"]["synaptic_memory"]
        ),
    )

    try:
        with pytest.raises(
            ProvenanceError, match="synaptic graph durable files changed"
        ):
            compare_synaptic._verify_comparison_postflight(
                state,
                synaptic_repo=repo,
                graph_path=graph,
                synaptic_result={"module_paths": state["synaptic_module_paths"]},
            )
    finally:
        compare_synaptic._close_graph_watcher(state)


def test_artifact_manifest_fingerprints_database_and_sidecars(tmp_path):
    graph = tmp_path / "graph.sqlite"
    wal = pathlib.Path(f"{graph}-wal")
    shm = pathlib.Path(f"{graph}-shm")
    graph.write_bytes(b"database")
    wal.write_bytes(b"wal")
    shm.write_bytes(b"shm")

    manifest = compare_synaptic._artifact_manifest(graph)

    assert [
        pathlib.Path(item["path"]).name for item in manifest["durable"]["files"]
    ] == [
        "graph.sqlite",
        "graph.sqlite-wal",
    ]
    assert [
        pathlib.Path(item["path"]).name for item in manifest["transient"]["files"]
    ] == ["graph.sqlite-shm"]
    assert (
        manifest["durable"]["files"][0]["sha256"]
        == hashlib.sha256(b"database").hexdigest()
    )
    assert len(manifest["durable"]["artifact_manifest_sha256"]) == 64
    assert len(manifest["transient"]["artifact_manifest_sha256"]) == 64


def test_shm_change_is_recorded_but_not_part_of_durable_graph_identity(tmp_path):
    graph = tmp_path / "graph.sqlite"
    shm = pathlib.Path(f"{graph}-shm")
    graph.write_bytes(b"database")
    shm.write_bytes(b"before")
    before = compare_synaptic._artifact_manifest(graph)

    shm.write_bytes(b"after reader lock state")
    after = compare_synaptic._artifact_manifest(graph)

    assert before["transient"] != after["transient"]
    assert compare_synaptic._durable_artifact_identity(
        before
    ) == compare_synaptic._durable_artifact_identity(after)


def test_sqlite_watcher_rejects_a_concurrent_commit(tmp_path):
    graph = _make_sqlite_graph(tmp_path / "graph.sqlite")
    state = compare_synaptic._graph_guard_preflight(graph)
    writer = sqlite3.connect(graph)
    try:
        writer.execute("INSERT INTO records VALUES ('concurrent write')")
        writer.commit()
        with pytest.raises(ProvenanceError, match="data_version advanced"):
            compare_synaptic._verify_graph_guard(state, graph)
    finally:
        writer.close()
        compare_synaptic._close_graph_watcher(state)


def test_synaptic_loader_rejects_an_already_loaded_wrong_checkout(
    tmp_path, monkeypatch
):
    repo = tmp_path / "synaptic-checkout"
    package_init = repo / "src" / "synaptic" / "__init__.py"
    package_init.parent.mkdir(parents=True)
    package_init.write_text("", encoding="utf-8")
    wrong_module = types.ModuleType("synaptic")
    wrong_module.__file__ = str(tmp_path / "site-packages" / "synaptic" / "__init__.py")
    monkeypatch.setitem(sys.modules, "synaptic", wrong_module)

    with pytest.raises(RuntimeError, match="expected checkout source"):
        compare_synaptic._load_synaptic_api(repo)


def test_synaptic_backend_closes_in_finally_when_search_fails(tmp_path, monkeypatch):
    state = {"connected": False, "closed": False, "k": None}

    class FakeBackend:
        def __init__(self, path):
            self.path = path

        async def connect(self):
            state["connected"] = True

        async def close(self):
            state["closed"] = True

    class FakeSearch:
        def __init__(self, **_kwargs):
            pass

        async def search(self, _query, *, k, fts_seed_limit):
            state["k"] = k
            assert fts_seed_limit == 30
            raise RuntimeError("search failed")

    monkeypatch.setattr(
        compare_synaptic,
        "_load_synaptic_api",
        lambda _repo: (FakeBackend, FakeSearch, {"package": "fake"}),
    )
    queries = [{"qid": "q1", "query": "query", "relevant_docs": ["d1"]}]

    with pytest.raises(RuntimeError, match="search failed"):
        asyncio.run(
            compare_synaptic.run_synaptic(
                tmp_path,
                tmp_path / "graph.sqlite",
                queries,
                [],
            )
        )

    assert state == {"connected": True, "closed": True, "k": 20}


def test_selected_synaptic_scorer_must_be_byte_identical(tmp_path):
    scorer = tmp_path / "tests" / "benchmark" / "metrics.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("different", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="not byte-identical"):
        compare_synaptic._scorer_records(tmp_path)


def test_active_scorer_is_exactly_bound_to_metrics_and_common_files():
    records = finreg_bench._active_scorer_records()

    assert records["module"]["path"] == "eval/metrics.py"
    assert records["benchmark_result"]["path"] == "eval/metrics.py"
    assert records["score_mrr"]["path"] == "eval/_common.py"
    assert records["score_strict"]["path"] == "eval/_common.py"
    assert records["module"]["sha256"] == records["benchmark_result"]["sha256"]
    assert records["score_mrr"]["sha256"] == records["score_strict"]["sha256"]


def test_active_scorer_rejects_a_preloaded_wrong_metrics_module(tmp_path, monkeypatch):
    wrong = types.ModuleType("metrics")
    wrong.__file__ = str(tmp_path / "site-packages" / "metrics.py")
    wrong.BenchmarkResult = finreg_bench.metrics_module.BenchmarkResult
    monkeypatch.setitem(sys.modules, "metrics", wrong)

    with pytest.raises(ProvenanceError, match="exact bound eval.metrics"):
        finreg_bench._active_scorer_records()


def test_worker_payload_validation_fails_closed_when_kiwi_is_inactive():
    binding = {"kiwi": {"active": False}}
    payload = {
        "schema": compare_synaptic.WORKER_SCHEMA,
        "schema_version": compare_synaptic.WORKER_SCHEMA_VERSION,
        "status": "ok",
        "process_id": -1,
        "runtime_binding": {"before": binding, "after": binding},
    }

    with pytest.raises(ProvenanceError, match="Kiwi active"):
        compare_synaptic._validate_worker_payload(payload, {})


def test_synaptic_worker_uses_selected_python_and_isolated_environment(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setenv("PYTHONPATH", "untrusted-pythonpath")
    monkeypatch.setenv("PYTHONHOME", "untrusted-pythonhome")
    monkeypatch.setenv("PYTHONUSERBASE", "untrusted-userbase")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = pathlib.Path(command[command.index("--worker-out") + 1])
        output.write_text('{"worker": true}', encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compare_synaptic.subprocess, "run", fake_run)
    payload = compare_synaptic._run_synaptic_worker(
        synaptic_python=pathlib.Path(sys.executable),
        synaptic_repo=tmp_path / "synaptic",
        graph_path=tmp_path / "graph.sqlite",
    )

    assert payload == {"worker": True}
    assert (
        pathlib.Path(captured["command"][0]).resolve()
        == pathlib.Path(sys.executable).resolve()
    )
    assert captured["command"][1:4] == ["-I", "-X", "utf8"]
    assert "--_synaptic-worker" in captured["command"]
    assert captured["kwargs"]["env"]["PYTHONNOUSERSITE"] == "1"
    assert captured["kwargs"]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert "PYTHONHOME" not in captured["kwargs"]["env"]
    assert "PYTHONUSERBASE" not in captured["kwargs"]["env"]


def test_worker_isolation_validation_rejects_non_isolated_flags(tmp_path):
    source = tmp_path / "synaptic" / "src"
    record = {
        "isolated": False,
        "ignore_environment": True,
        "no_user_site": True,
        "safe_path": True,
        "utf8_mode": True,
        "dont_write_bytecode": True,
        "user_site_enabled": False,
        "pythonpath": None,
        "pythonhome": None,
        "pythonuserbase": None,
        "python_no_user_site_env": "1",
        "sys_path": [str(source)],
        "normalized_sys_path": [str(source.resolve())],
        "required_source_roots": [
            str(source.resolve()),
            str(compare_synaptic.EVAL_DIR.resolve()),
            str(compare_synaptic.SOURCE_ROOT.resolve()),
        ],
        "user_site_present_on_sys_path": False,
    }

    with pytest.raises(ProvenanceError, match="did not enable isolated"):
        compare_synaptic._validate_worker_isolation_record(
            record, expected_synaptic_source=source
        )


def test_imported_omnifuse_sources_are_bound_to_this_checkout():
    sources = finreg_bench._omnifuse_import_sources()

    assert sources["package"]["path"] == "src/omnifuse/__init__.py"
    assert str(sources["build_inmemory"]["path"]).startswith("src/omnifuse/")
    assert sources["shared_finreg_module"]["path"] == "eval/_common.py"


def test_import_does_not_parse_process_arguments_or_enable_pyc_writes():
    assert callable(finreg_bench.main)
    assert callable(compare_synaptic.main)
    assert finreg_bench.sys.dont_write_bytecode is True
    assert compare_synaptic.sys.dont_write_bytecode is True
