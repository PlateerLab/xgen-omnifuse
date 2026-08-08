import asyncio
import json
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))

import perf_bench  # noqa: E402
import provenance  # noqa: E402


WORKER_RUN_ID = "00000000000040008000000000000001"


def _worker_identity(run_id: str = WORKER_RUN_ID, pid: int = 123) -> dict:
    return {
        "schema": provenance.WORKER_IDENTITY_SCHEMA,
        "schema_version": provenance.WORKER_IDENTITY_SCHEMA_VERSION,
        "worker_run_id": run_id,
        "worker_pid": pid,
        "capture_phase": provenance.WORKER_IDENTITY_CAPTURE_PHASE,
    }


def _worker_environment() -> dict:
    return {
        "python": perf_bench.platform.python_version(),
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
    runtime = perf_bench._runtime_environment_snapshot()
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


def test_cli_defaults_to_two_counterbalanced_trials(tmp_path):
    args = perf_bench._build_parser().parse_args(["--data-dir", str(tmp_path)])
    assert args.trials == 2


def test_worker_command_binds_exact_controller_run_id(tmp_path):
    args = SimpleNamespace(
        data_dir=tmp_path,
        dataset="nfcorpus.json",
        protocol=perf_bench.PROTOCOL_SQLITE_NATIVE,
        k=10,
        warmup=0,
        repeats=1,
        synaptic_repo=None,
    )

    command = perf_bench._worker_command(
        args,
        "omnifuse",
        tmp_path / "input.json",
        tmp_path / "result.json",
        20,
        WORKER_RUN_ID,
    )

    assert command[command.index("--worker-run-id") + 1] == WORKER_RUN_ID


def test_machine_worker_directory_is_new_and_outside_immutable_repo(tmp_path):
    repo = tmp_path / "synaptic"
    repo.mkdir()
    output = tmp_path / "result.json"
    existing = tmp_path / "existing-workers"
    existing.mkdir()

    with pytest.raises(perf_bench.ProvenanceError, match="refusing to reuse"):
        perf_bench._validate_worker_directory(
            existing, output=output, synaptic_repo=repo
        )
    with pytest.raises(perf_bench.ProvenanceError, match="immutable"):
        perf_bench._validate_worker_directory(
            repo / "workers", output=output, synaptic_repo=repo
        )
    default_directory = perf_bench._worker_directory(output, None)
    assert default_directory.parent == (perf_bench.ROOT / "worklogs").resolve()
    assert re.fullmatch(
        r"result-[0-9a-f]{16}-perf-workers", default_directory.name
    )
    assert default_directory != perf_bench._worker_directory(
        tmp_path / "other" / "result.json", None
    )
    assert perf_bench.REPORT_SCHEMA_VERSION == 5
    assert perf_bench.WORKER_RESULT_SCHEMA_VERSION == 4


def test_existing_worker_directory_fails_before_machine_preflight(
    tmp_path, monkeypatch, capsys
):
    synaptic_repo = tmp_path / "synaptic"
    synaptic_repo.mkdir()
    worker_root = tmp_path / "existing-workers"
    worker_root.mkdir()

    def unexpected_preflight(**_kwargs):
        pytest.fail("heavy machine preflight must not run after a worker collision")

    monkeypatch.setattr(perf_bench, "_machine_preflight", unexpected_preflight)
    with pytest.raises(SystemExit) as error:
        perf_bench.main(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--synaptic-repo",
                str(synaptic_repo),
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(tmp_path / "result.json"),
                "--workers-dir",
                str(worker_root),
            ]
        )

    assert error.value.code == 2
    assert "refusing to reuse worker-artifact directory" in capsys.readouterr().err


def test_official_controller_requires_utf8_before_input_access(
    tmp_path, monkeypatch, capsys
):
    synaptic_repo = tmp_path / "synaptic"
    synaptic_repo.mkdir()
    monkeypatch.setattr(perf_bench, "_controller_utf8_mode_enabled", lambda: False)

    def unexpected_load(*_args):
        raise AssertionError("official input must not be opened without UTF-8 mode")

    monkeypatch.setattr(perf_bench, "_load_official_external_input", unexpected_load)
    with pytest.raises(SystemExit) as error:
        perf_bench.main(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--protocol",
                perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
                "--synaptic-repo",
                str(synaptic_repo),
            ]
        )

    assert error.value.code == 2
    assert "relaunch with python -X utf8" in capsys.readouterr().err


def test_sqlite_native_does_not_require_controller_utf8():
    perf_bench._require_protocol_utf8_mode(
        perf_bench.PROTOCOL_SQLITE_NATIVE,
        context="controller",
        environment={"utf8_mode": False},
    )


def test_parse_uses_deterministic_document_and_query_order():
    corpus, queries = perf_bench._parse(
        {
            "corpus": {
                "doc-b": {"title": "B", "text": "body-b"},
                "doc-a": {"title": "A", "text": "body-a"},
            },
            "queries": {"q-2": "second", "q-1": "first"},
            "qrels": {"q-1": {"doc-a": 1}, "q-2": ["doc-b"]},
        }
    )

    assert [row[0] for row in corpus] == ["doc-a", "doc-b"]
    assert [row[0] for row in queries] == ["q-1", "q-2"]


def test_frozen_worker_input_is_canonical_and_round_trips_exactly():
    corpus = [("doc-b", "B", "body-b"), ("doc-a", "A", "body-a")]
    queries = [("q-2", "second", {"doc-b"}), ("q-1", "first", {"doc-a"})]
    corpus = sorted(corpus)
    queries = sorted(queries)

    payload = perf_bench._frozen_input_payload(corpus, queries)

    assert perf_bench._parse_frozen_input(payload) == (corpus, queries)
    assert (
        perf_bench._frozen_input_payload(*perf_bench._parse_frozen_input(payload))
        == payload
    )


def test_frozen_worker_file_loader_streams_canonical_validation(tmp_path):
    corpus = [("doc-a", "A", "body-a")]
    queries = [("q-1", "first", {"doc-a"})]
    payload = perf_bench._frozen_input_payload(corpus, queries)
    path = tmp_path / "input.json"
    path.write_bytes(payload)

    fingerprint, actual_corpus, actual_queries = perf_bench._load_frozen_input_file(
        path, display_path="worker-input/persistence.json"
    )

    assert (actual_corpus, actual_queries) == (corpus, queries)
    assert fingerprint["bytes"] == len(payload)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(json.loads(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(provenance.ProvenanceError, match="not canonical"):
        perf_bench._load_frozen_input_file(
            noncanonical, display_path="worker-input/persistence.json"
        )


def test_frozen_worker_input_preserves_official_source_order():
    corpus = [("doc-b", "B", "body-b"), ("doc-a", "A", "body-a")]
    queries = [("q-2", "second", {"doc-b"}), ("q-1", "first", {"doc-a"})]

    payload = perf_bench._frozen_input_payload(corpus, queries)

    assert perf_bench._parse_frozen_input(payload) == (corpus, queries)


def test_official_external_input_preserves_case_selection_and_corpus_order(
    tmp_path, monkeypatch
):
    import direct_external_bench as direct_external

    repo = tmp_path / "synaptic"
    dataset = repo / "tests" / "benchmark" / "data" / "synthetic.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}", encoding="utf-8")
    case = SimpleNamespace(
        id="synthetic",
        name="Synthetic",
        filename="synthetic.json",
        max_queries=0,
        klue_corpus_sample=0,
    )
    data = {
        "corpus": {
            "doc-b": {"title": "B", "text": "body-b"},
            "empty": {"title": "ignored", "text": ""},
            "doc-a": {"title": "", "text": "body-a"},
        },
        "queries": {"q-2": "second", "q-1": "first", "skip": "none"},
        "qrels": {
            "q-2": {"doc-b": 1},
            "q-1": {"doc-a": 1},
            "skip": {"empty": 1},
        },
    }
    driver = SimpleNamespace(_load_dataset=lambda _filename: data)
    monkeypatch.setattr(perf_bench, "_official_external_case", lambda _name: case)
    monkeypatch.setattr(direct_external, "_validate_tag_checkout", lambda _repo: {})
    monkeypatch.setattr(
        direct_external,
        "_load_upstream_driver",
        lambda _repo: (
            driver,
            SimpleNamespace(),
            {"upstream_driver": "driver.py", "upstream_scorer": "metrics.py"},
        ),
    )

    corpus, queries, selection = perf_bench._load_official_external_input(repo, dataset)

    assert [row[0] for row in corpus] == ["doc-b", "doc-a"]
    assert [row[0] for row in queries] == ["q-2", "q-1"]
    assert selection["case_id"] == "synthetic"
    assert selection["scored_query_count"] == 2


@pytest.mark.parametrize(
    ("corpus", "queries", "message"),
    [
        (
            [("doc", "A", "one"), ("doc", "B", "two")],
            [("q", "query", {"doc"})],
            "corpus IDs must be unique",
        ),
        (
            [("doc", "A", "one")],
            [("q", "first", {"doc"}), ("q", "second", {"doc"})],
            "query IDs must be unique",
        ),
    ],
)
def test_frozen_worker_input_rejects_duplicate_ids(corpus, queries, message):
    with pytest.raises(ValueError, match=message):
        perf_bench._frozen_input_payload(corpus, queries)


def test_top_k_unique_applies_one_shared_deduplication_rule():
    assert perf_bench._top_k_unique(["a", "a", "", "b", "c"], 2) == ["a", "b"]


def test_sync_measurement_excludes_warmup_and_repeats_in_query_order():
    queries = [("q-1", "first", {"b"}), ("q-2", "second", {"b"})]
    calls = []
    ticks = iter([0.0, 0.001, 1.0, 1.002, 2.0, 2.003, 3.0, 3.004])

    def search(text):
        calls.append(text)
        return ["a", "a", "b", "c"]

    rankings, samples = perf_bench._measure_sync(
        search,
        queries,
        k=2,
        warmup=1,
        repeats=2,
        clock=lambda: next(ticks),
    )

    assert calls == ["first", "second"] * 3
    assert rankings == {"q-1": ["a", "b"], "q-2": ["a", "b"]}
    assert samples == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert perf_bench._score(queries, rankings, k=2) == pytest.approx(0.5)


def test_async_measurement_uses_the_same_rules():
    queries = [("q", "query", {"b"})]
    ticks = iter([0.0, 0.002])

    async def search(_text):
        return ["a", "a", "b"]

    rankings, samples = asyncio.run(
        perf_bench._measure_async(
            search,
            queries,
            k=2,
            warmup=0,
            repeats=1,
            clock=lambda: next(ticks),
        )
    )

    assert rankings == {"q": ["a", "b"]}
    assert samples == pytest.approx([2.0])


def test_claim_grade_sync_measurement_materializes_top20_inside_ns_timer():
    queries = [("q", "query", {"d0"})]
    events = []
    ticks = iter([0, 2_000_000, 10_000_000, 13_000_000])

    def search(_text):
        def results():
            for index in range(25):
                events.append(f"id-{index}")
                yield f"d{index}"

        return results()

    def clock_ns():
        events.append("clock")
        return next(ticks)

    rankings, samples, detail = perf_bench._measure_sync_claim_grade(
        search,
        queries,
        k=10,
        candidate_limit=20,
        warmup=1,
        repeats=2,
        clock_ns=clock_ns,
    )

    assert rankings["q"] == [f"d{index}" for index in range(10)]
    assert samples == pytest.approx([2.0, 3.0])
    assert detail["query_round_seconds"] == pytest.approx([0.002, 0.003])
    assert detail["warmup_calls_verified"] == 1
    assert detail["measured_calls_verified"] == 2
    measured_events = events[20:]
    assert measured_events[0] == "clock"
    assert measured_events[1:21] == [f"id-{index}" for index in range(20)]
    assert measured_events[21] == "clock"


def test_claim_grade_measurement_fails_closed_when_ranking_changes():
    calls = 0

    def search(_text):
        nonlocal calls
        calls += 1
        return ["a", "b"] if calls == 1 else ["b", "a"]

    with pytest.raises(RuntimeError, match="ranking changed during timing"):
        perf_bench._measure_sync_claim_grade(
            search,
            [("q", "query", {"a"})],
            k=1,
            candidate_limit=2,
            warmup=1,
            repeats=1,
            clock_ns=iter([0, 1]).__next__,
        )


def test_official_protocol_result_separates_ingest_query_and_end_to_end():
    result = perf_bench.run_omnifuse(
        [("d1", "Title", "body"), ("d2", "Other", "text")],
        [("q", "body", {"d1"})],
        k=10,
        candidate_limit=20,
        warmup=1,
        repeats=2,
        protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
    )

    perf_bench._validate_claim_grade_result(
        result,
        system="omnifuse",
        expected_queries=1,
        contract={
            "k": 10,
            "candidate_limit": 20,
            "warmup_rounds": 1,
            "measurement_rounds": 2,
        },
    )
    assert result["canonical_rankings"]["clock"] == "time.perf_counter_ns"
    assert result["timing"]["end_to_end"]["ingest_plus_mean_round_seconds"] == (
        pytest.approx(
            result["timing"]["ingest_seconds"]
            + result["timing"]["query"]["mean_round_seconds"]
        )
    )


def test_omnifuse_ingest_timer_excludes_frozen_input_adapter(monkeypatch):
    events = []

    class CorpusRows(list):
        def __iter__(self):
            events.append("adapt")
            return super().__iter__()

    original_clock = perf_bench.time.perf_counter_ns

    def recording_clock():
        events.append("clock")
        return original_clock()

    monkeypatch.setattr(perf_bench.time, "perf_counter_ns", recording_clock)
    perf_bench.run_omnifuse(
        CorpusRows([("d1", "Title", "body")]),
        [("q", "body", {"d1"})],
        k=10,
        candidate_limit=20,
        warmup=1,
        repeats=2,
        protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
    )

    assert events[0] == "adapt"
    assert events[1] == "clock"


def test_latency_summary_names_actual_distribution_statistics():
    assert perf_bench._latency_summary([1.0, 2.0, 3.0, 4.0]) == pytest.approx(
        {"p50": 2.5, "p95": 3.85, "mean": 2.5}
    )


def test_process_memory_reports_whole_worker_rss_when_supported():
    current, peak, kind = perf_bench._process_memory_bytes()
    if peak is None:
        pytest.skip("resident-memory counters are unavailable on this platform")
    assert kind
    assert peak > 0
    if current is not None:
        assert 0 < current <= peak


def test_atomic_json_output_is_write_once_and_preserves_existing_content(tmp_path):
    output = tmp_path / "nested" / "result.json"
    perf_bench._atomic_write_json(output, {"ready": True})
    assert output.read_text(encoding="utf-8") == '{\n  "ready": true\n}\n'

    with pytest.raises(perf_bench.ProvenanceError, match="refusing to overwrite"):
        perf_bench._atomic_write_json(output, {"ready": False})

    assert output.read_text(encoding="utf-8") == '{\n  "ready": true\n}\n'
    assert not list(output.parent.glob(".*.tmp"))


def test_worker_consumes_one_frozen_payload_and_records_its_exact_fingerprint(
    tmp_path, monkeypatch
):
    corpus = [("doc", "Title", "body")]
    queries = [("q", "body", {"doc"})]
    input_file = tmp_path / "input.json"
    input_file.write_bytes(perf_bench._frozen_input_payload(corpus, queries))
    result_file = tmp_path / "worker.json"
    measurement_finished = False

    def fake_run(_corpus, _queries, **kwargs):
        nonlocal measurement_finished
        assert _corpus == corpus
        assert _queries == queries
        measurement_finished = True
        return {
            "system": "OmniFuse",
            "ingest_s": 1.0,
            "query_latency_p50_ms": 2.0,
            "query_latency_p95_ms": 2.0,
            "query_latency_mean_ms": 2.0,
            "query_latency_samples": kwargs["repeats"],
            "mrr_at_k": 1.0,
            "k": kwargs["k"],
            "candidate_limit": kwargs["candidate_limit"],
            "warmup_rounds": kwargs["warmup"],
            "measurement_rounds": kwargs["repeats"],
            "process_memory": {
                "scope": perf_bench.PROCESS_MEMORY_SCOPE,
                "kind": None,
                "current_rss_mb": None,
                "peak_rss_mb": None,
            },
            "runtime": {
                "package_path": "package",
                "package_version": None,
                "source_bindings": {},
                "tokenizer": None,
            },
        }

    monkeypatch.setattr(perf_bench, "run_omnifuse", fake_run)
    monkeypatch.setattr(
        perf_bench,
        "capture_worker_identity",
        lambda run_id: (
            _worker_identity(run_id)
            if measurement_finished
            else pytest.fail("worker identity was captured during measurement")
        ),
    )
    perf_bench._run_worker(
        SimpleNamespace(
            worker="omnifuse",
            input_file=input_file,
            result_file=result_file,
            synaptic_repo=None,
            worker_run_id=WORKER_RUN_ID,
            k=10,
            warmup=0,
            repeats=1,
        ),
        candidate_limit=20,
    )

    worker = json.loads(result_file.read_text(encoding="utf-8"))
    expected = perf_bench._bytes_fingerprint(
        input_file.read_bytes(), path=perf_bench.WORKER_INPUT_DISPLAY_PATH
    )
    assert worker["schema"] == perf_bench.WORKER_RESULT_SCHEMA
    assert worker["schema_version"] == perf_bench.WORKER_RESULT_SCHEMA_VERSION
    assert worker["protocol"] == perf_bench.PROTOCOL_SQLITE_NATIVE
    assert {key: worker["input"][key] for key in expected} == expected
    assert worker["input"]["documents"] == 1
    assert worker["input"]["scored_queries"] == 1
    assert worker["worker_identity"] == _worker_identity()


def test_official_worker_binds_environment_lock_before_and_after(tmp_path, monkeypatch):
    corpus = [("doc", "Title", "body")]
    queries = [("q", "body", {"doc"})]
    input_file = tmp_path / "input.json"
    input_file.write_bytes(perf_bench._frozen_input_payload(corpus, queries))
    result_file = tmp_path / "worker.json"
    lock = {
        "lockfile": {"sha256": "a" * 64},
        "installed_manifest_sha256": "b" * 64,
        "uv_sync_check": {"reported_no_changes": True},
    }
    calls = []

    def lock_evidence(repo):
        calls.append(repo)
        return json.loads(json.dumps(lock))

    def fake_run(_corpus, _queries, **_kwargs):
        return {"runtime": {}}

    monkeypatch.setattr(
        perf_bench, "_official_environment_lock_evidence", lock_evidence
    )
    monkeypatch.setattr(perf_bench, "run_omnifuse", fake_run)
    synaptic_repo = tmp_path / "synaptic"
    perf_bench._run_worker(
        SimpleNamespace(
            worker="omnifuse",
            input_file=input_file,
            result_file=result_file,
            synaptic_repo=synaptic_repo,
            worker_run_id=WORKER_RUN_ID,
            protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
            k=10,
            warmup=1,
            repeats=2,
        ),
        candidate_limit=20,
    )

    worker = json.loads(result_file.read_text(encoding="utf-8"))
    assert calls == [synaptic_repo, synaptic_repo]
    assert worker["result"]["runtime"]["official_environment_lock"] == {
        "before": lock,
        "after": lock,
    }


def test_worker_result_strictly_binds_input_contract_metrics_and_imports():
    corpus = [("doc", "Title", "body")]
    queries = [("q", "body", {"doc"})]
    frozen = perf_bench._frozen_input_payload(corpus, queries)
    expected_input = {
        **perf_bench._bytes_fingerprint(
            frozen, path=perf_bench.WORKER_INPUT_DISPLAY_PATH
        ),
        "documents": 1,
        "scored_queries": 1,
        "relevance_judgments": 1,
    }
    contract = {
        "k": 10,
        "candidate_limit": 20,
        "warmup_rounds": 0,
        "measurement_rounds": 1,
    }
    result = perf_bench.run_omnifuse(
        corpus,
        queries,
        k=10,
        candidate_limit=20,
        warmup=0,
        repeats=1,
    )
    payload = {
        "schema": perf_bench.WORKER_RESULT_SCHEMA,
        "schema_version": perf_bench.WORKER_RESULT_SCHEMA_VERSION,
        "status": "ok",
        "system": "omnifuse",
        "protocol": perf_bench.PROTOCOL_SQLITE_NATIVE,
        "contract": contract,
        "input": expected_input,
        "worker_identity": _worker_identity(),
        "environment": _worker_environment(),
        "result": result,
    }

    validated = perf_bench._validate_worker_result(
        payload,
        system="omnifuse",
        expected_input=expected_input,
        contract=contract,
        synaptic_repo=None,
        trial_number=1,
        order_position=1,
        expected_worker_run_id=WORKER_RUN_ID,
    )
    assert validated["worker_input"] == expected_input
    assert set(validated["runtime"]["source_bindings"]) == {
        "package",
        "build_inmemory",
        "retrieve",
    }
    assert validated["worker_identity"]["worker_run_id"] == WORKER_RUN_ID

    wrong_identity = json.loads(json.dumps(payload))
    wrong_identity["worker_identity"]["worker_run_id"] = (
        "00000000000040008000000000000002"
    )
    with pytest.raises(perf_bench.ProvenanceError, match="does not match"):
        perf_bench._validate_worker_result(
            wrong_identity,
            system="omnifuse",
            expected_input=expected_input,
            contract=contract,
            synaptic_repo=None,
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
        )

    wrong_input = json.loads(json.dumps(payload))
    wrong_input["input"]["sha256"] = "0" * 64
    with pytest.raises(perf_bench.ProvenanceError, match="consumed input changed"):
        perf_bench._validate_worker_result(
            wrong_input,
            system="omnifuse",
            expected_input=expected_input,
            contract=contract,
            synaptic_repo=None,
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
        )

    missing_binding = json.loads(json.dumps(payload))
    del missing_binding["result"]["runtime"]["source_bindings"]["retrieve"]
    with pytest.raises(perf_bench.ProvenanceError, match="bindings are incomplete"):
        perf_bench._validate_worker_result(
            missing_binding,
            system="omnifuse",
            expected_input=expected_input,
            contract=contract,
            synaptic_repo=None,
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
        )

    wrong_scope = json.loads(json.dumps(payload))
    wrong_scope["result"]["process_memory"]["scope"] = "worker"
    with pytest.raises(perf_bench.ProvenanceError, match="memory scope"):
        perf_bench._validate_worker_result(
            wrong_scope,
            system="omnifuse",
            expected_input=expected_input,
            contract=contract,
            synaptic_repo=None,
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
        )


def test_official_worker_result_binds_lock_to_controller_preflight(monkeypatch):
    corpus = [("doc", "Title", "body")]
    queries = [("q", "body", {"doc"})]
    frozen = perf_bench._frozen_input_payload(corpus, queries)
    expected_input = {
        **perf_bench._bytes_fingerprint(
            frozen, path=perf_bench.WORKER_INPUT_DISPLAY_PATH
        ),
        "documents": 1,
        "scored_queries": 1,
        "relevance_judgments": 1,
    }
    contract = {
        "k": 10,
        "candidate_limit": 20,
        "warmup_rounds": 1,
        "measurement_rounds": 2,
    }
    lock = {
        "lockfile": {"sha256": "a" * 64},
        "installed_manifest_sha256": "b" * 64,
        "uv_sync_check": {"reported_no_changes": True},
    }
    result = perf_bench.run_omnifuse(
        corpus,
        queries,
        k=10,
        candidate_limit=20,
        warmup=1,
        repeats=2,
        protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
    )
    result["runtime"]["official_environment_lock"] = {
        "before": lock,
        "after": json.loads(json.dumps(lock)),
    }
    payload = {
        "schema": perf_bench.WORKER_RESULT_SCHEMA,
        "schema_version": perf_bench.WORKER_RESULT_SCHEMA_VERSION,
        "status": "ok",
        "system": "omnifuse",
        "protocol": perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        "contract": contract,
        "input": expected_input,
        "worker_identity": _worker_identity(),
        "environment": _worker_environment(),
        "result": result,
    }
    monkeypatch.setattr(
        perf_bench, "_validate_official_environment_lock", lambda *_args: None
    )

    validated = perf_bench._validate_worker_result(
        payload,
        system="omnifuse",
        expected_input=expected_input,
        contract=contract,
        synaptic_repo=pathlib.Path("synaptic"),
        trial_number=1,
        order_position=1,
        expected_worker_run_id=WORKER_RUN_ID,
        protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        expected_official_environment_lock=lock,
    )
    assert validated["runtime"]["official_environment_lock"]["before"] == lock

    changed = json.loads(json.dumps(payload))
    changed["result"]["runtime"]["official_environment_lock"]["after"][
        "installed_manifest_sha256"
    ] = "c" * 64
    with pytest.raises(perf_bench.ProvenanceError, match="environment lock changed"):
        perf_bench._validate_worker_result(
            changed,
            system="omnifuse",
            expected_input=expected_input,
            contract=contract,
            synaptic_repo=pathlib.Path("synaptic"),
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
            protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
            expected_official_environment_lock=lock,
        )


def test_strict_machine_worker_rejects_korean_regex_fallback():
    evidence = {
        "mode": "regex_fallback",
        "korean_normalization_used": True,
        "kiwi_available": False,
        "kiwi_version": None,
        "kiwi_model_version": None,
        "modules": {name: None for name in perf_bench.TOKENIZER_MODULE_NAMES},
    }
    with pytest.raises(perf_bench.ProvenanceError, match="requires Kiwi"):
        perf_bench._validate_tokenizer_evidence(
            evidence, system="synaptic", require_kiwi=True
        )


def test_worker_environment_rejects_inherited_pythonpath():
    environment = _worker_environment()
    environment["pythonpath"] = "untrusted"
    with pytest.raises(perf_bench.ProvenanceError, match="inherited pythonpath"):
        perf_bench._validate_worker_environment(environment, system="synaptic")


def test_official_worker_validation_explicitly_requires_utf8(monkeypatch):
    expected_input = {
        "path": perf_bench.WORKER_INPUT_DISPLAY_PATH,
        "sha256": "a" * 64,
        "bytes": 1,
        "documents": 1,
        "scored_queries": 1,
        "relevance_judgments": 1,
    }
    contract = {
        "k": 10,
        "candidate_limit": 20,
        "warmup_rounds": 1,
        "measurement_rounds": 2,
    }
    payload = {
        "schema": perf_bench.WORKER_RESULT_SCHEMA,
        "schema_version": perf_bench.WORKER_RESULT_SCHEMA_VERSION,
        "status": "ok",
        "system": "omnifuse",
        "protocol": perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        "contract": contract,
        "input": expected_input,
        "worker_identity": _worker_identity(),
        "environment": {},
        "result": {},
    }
    monkeypatch.setattr(
        perf_bench,
        "_validate_worker_environment",
        lambda *_args, **_kwargs: {"utf8_mode": False},
    )

    with pytest.raises(perf_bench.ProvenanceError, match="omnifuse worker"):
        perf_bench._validate_worker_result(
            payload,
            system="omnifuse",
            expected_input=expected_input,
            contract=contract,
            synaptic_repo=pathlib.Path("synaptic"),
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
            protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
            expected_official_environment_lock={},
        )


def test_machine_output_requires_doctor_and_synaptic_checkout(tmp_path):
    common = [
        "--data-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "result.json"),
    ]
    with pytest.raises(SystemExit) as missing_doctor:
        perf_bench.main(common)
    assert missing_doctor.value.code == 2

    with pytest.raises(SystemExit) as missing_synaptic:
        perf_bench.main([*common, "--doctor-manifest", str(tmp_path / "doctor.json")])
    assert missing_synaptic.value.code == 2


def test_existing_machine_output_is_rejected_before_dataset_loading(
    tmp_path, monkeypatch
):
    synaptic_repo = tmp_path / "synaptic"
    synaptic_repo.mkdir()
    output = tmp_path / "result.json"
    output.write_bytes(b"original artifact\n")

    def unexpected_load(_path):
        raise AssertionError("dataset must not be loaded after output preflight fails")

    monkeypatch.setattr(perf_bench, "_load_dataset", unexpected_load)
    with pytest.raises(SystemExit) as error:
        perf_bench.main(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--dataset",
                "nfcorpus.json",
                "--synaptic-repo",
                str(synaptic_repo),
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(output),
            ]
        )

    assert error.value.code == 2

    assert output.read_bytes() == b"original artifact\n"


@pytest.mark.parametrize("inside_path", ["output", "doctor"])
def test_official_machine_artifacts_must_stay_outside_synaptic_checkout(
    tmp_path, monkeypatch, inside_path
):
    synaptic_repo = tmp_path / "synaptic"
    synaptic_repo.mkdir()
    output = tmp_path / "evidence" / "result.json"
    doctor = tmp_path / "evidence" / "doctor.json"
    if inside_path == "output":
        output = synaptic_repo / "result.json"
    else:
        doctor = synaptic_repo / "doctor.json"

    def unexpected_snapshot():
        raise AssertionError("artifact paths must fail before environment preflight")

    monkeypatch.setattr(
        perf_bench, "_runtime_environment_snapshot", unexpected_snapshot
    )
    with pytest.raises(
        perf_bench.ProvenanceError, match="outside the immutable synaptic-memory"
    ):
        perf_bench._machine_preflight(
            output=output,
            doctor_manifest=doctor,
            synaptic_repo=synaptic_repo,
            dataset_path=(
                synaptic_repo / "tests" / "benchmark" / "data" / "nfcorpus.json"
            ),
            protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        )


def test_official_machine_preflight_requires_utf8_before_dataset_access(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        perf_bench,
        "_runtime_environment_snapshot",
        lambda: {"utf8_mode": False},
    )

    def unexpected_identity(*_args):
        raise AssertionError("dataset identity must not run without UTF-8 mode")

    monkeypatch.setattr(perf_bench, "_machine_dataset_identity", unexpected_identity)
    with pytest.raises(perf_bench.ProvenanceError, match="controller preflight"):
        perf_bench._machine_preflight(
            output=tmp_path / "result.json",
            doctor_manifest=tmp_path / "doctor.json",
            synaptic_repo=tmp_path / "synaptic",
            dataset_path=tmp_path / "dataset.json",
            protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        )


def test_machine_preflight_binds_doctor_to_dataset_repositories_and_scorers(
    tmp_path, monkeypatch
):
    synaptic_repo = tmp_path / "synaptic"
    dataset = synaptic_repo / "tests" / "benchmark" / "data" / "nfcorpus.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            {
                "corpus": {"d": {"title": "", "text": "body"}},
                "queries": {"q": "query"},
                "qrels": {"q": {"d": 1}},
            }
        ),
        encoding="utf-8",
    )
    repositories = {
        "omnifuse": {"path": "omni", "sha": "a"},
        "synaptic_memory": {"path": "synaptic", "sha": "b"},
    }
    scorer = {
        "active": {"path": "eval/metrics.py", "sha256": "1", "bytes": 1},
        "synaptic_checkout_copy": {
            "path": "tests/benchmark/metrics.py",
            "sha256": "1",
            "bytes": 1,
        },
        "byte_identical": True,
    }
    sources = {"scorer": scorer, "harness": {"sha256": "2"}}
    monkeypatch.setattr(
        perf_bench,
        "repository_fingerprint",
        lambda path: (
            repositories["omnifuse"]
            if path.resolve() == perf_bench.ROOT
            else repositories["synaptic_memory"]
        ),
    )
    monkeypatch.setattr(perf_bench, "_benchmark_sources", lambda _repo: sources)
    captured = {}

    def load_doctor(path, inputs):
        captured["doctor_path"] = path
        captured["inputs"] = inputs
        return _doctor_record(path=str(path)), {
            "nfcorpus.json": {"target_id": "nfcorpus", "target_status": "ok"}
        }

    def verify_runtime(record, **bindings):
        captured["doctor"] = record
        captured["bindings"] = bindings

    monkeypatch.setattr(perf_bench, "load_doctor_manifest", load_doctor)
    monkeypatch.setattr(perf_bench, "verify_doctor_runtime", verify_runtime)

    state, corpus, queries = perf_bench._machine_preflight(
        output=tmp_path / "result.json",
        doctor_manifest=tmp_path / "doctor.json",
        synaptic_repo=synaptic_repo,
        dataset_path=dataset,
    )

    assert len(corpus) == len(queries) == 1
    assert state["dataset"]["doctor_target_id"] == "nfcorpus"
    assert captured["inputs"] == [
        {
            "name": "nfcorpus.json",
            "target_id": "nfcorpus",
            "path": "tests/benchmark/data/nfcorpus.json",
            "sha256": state["dataset"]["sha256"],
            "bytes": state["dataset"]["bytes"],
        }
    ]
    assert captured["bindings"] == {
        "omnifuse_repository": repositories["omnifuse"],
        "synaptic_repository": repositories["synaptic_memory"],
        "omnifuse_scorer": scorer["active"],
        "synaptic_scorer": scorer["synaptic_checkout_copy"],
    }


def test_official_machine_preflight_binds_environment_before_input(
    tmp_path, monkeypatch
):
    synaptic_repo = tmp_path / "synaptic"
    dataset = synaptic_repo / "tests" / "benchmark" / "data" / "nfcorpus.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}", encoding="utf-8")
    runtime = perf_bench._runtime_environment_snapshot()
    runtime["utf8_mode"] = True
    probe = {"environment_lock": {"installed_manifest_sha256": "a" * 64}}
    scorer = {
        "active": {"path": "eval/metrics.py", "sha256": "1", "bytes": 1},
        "synaptic_checkout_copy": {
            "path": "tests/benchmark/metrics.py",
            "sha256": "1",
            "bytes": 1,
        },
    }
    calls = []
    monkeypatch.setattr(perf_bench, "_runtime_environment_snapshot", lambda: runtime)
    monkeypatch.setattr(
        perf_bench, "repository_fingerprint", lambda _path: {"sha": "repo"}
    )
    monkeypatch.setattr(
        perf_bench,
        "_benchmark_sources",
        lambda *_args: {"scorer": scorer, "harness": {"sha256": "2"}},
    )

    def environment_probe(_repo):
        calls.append("environment")
        return probe

    def official_input(_repo, _dataset):
        calls.append("input")
        return (
            [("d", "Title", "body")],
            [("q", "query", {"d"})],
            {"case_id": "nfcorpus"},
        )

    monkeypatch.setattr(perf_bench, "_official_environment_probe", environment_probe)
    monkeypatch.setattr(perf_bench, "_load_official_external_input", official_input)
    monkeypatch.setattr(
        perf_bench,
        "load_doctor_manifest",
        lambda path, _inputs: (
            _doctor_record(path=str(path)),
            {"nfcorpus.json": {"target_id": "nfcorpus"}},
        ),
    )
    monkeypatch.setattr(perf_bench, "verify_doctor_runtime", lambda *_a, **_k: None)

    state, corpus, queries = perf_bench._machine_preflight(
        output=tmp_path / "evidence" / "result.json",
        doctor_manifest=tmp_path / "evidence" / "doctor.json",
        synaptic_repo=synaptic_repo,
        dataset_path=dataset,
        protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
    )

    assert calls == ["environment", "input"]
    assert state["official_environment_probe"] == probe
    assert len(corpus) == len(queries) == 1


def test_benchmark_sources_include_upstream_scorer_and_native_driver(tmp_path):
    synaptic_repo = tmp_path / "synaptic"
    upstream_scorer = synaptic_repo / perf_bench.SYNAPTIC_SCORER_RELATIVE
    upstream_driver = synaptic_repo / perf_bench.SYNAPTIC_DRIVER_RELATIVE
    upstream_scorer.parent.mkdir(parents=True)
    upstream_driver.parent.mkdir(parents=True)
    upstream_scorer.write_bytes((perf_bench.EVAL_DIR / "metrics.py").read_bytes())
    upstream_driver.write_text("# native runner\n", encoding="utf-8")

    before = perf_bench._benchmark_sources(synaptic_repo)
    assert before["scorer"]["byte_identical"] is True
    assert before["synaptic_native_driver"]["path"] == "eval/run_all.py"

    upstream_driver.write_text("# changed native runner\n", encoding="utf-8")
    after = perf_bench._benchmark_sources(synaptic_repo)
    assert before != after

    upstream_scorer.write_text("# different scorer\n", encoding="utf-8")
    with pytest.raises(perf_bench.ProvenanceError, match="not byte-identical"):
        perf_bench._benchmark_sources(synaptic_repo)


def test_postflight_fails_closed_when_dataset_changes(tmp_path, monkeypatch):
    dataset = tmp_path / "nfcorpus.json"
    dataset.write_text("before\n", encoding="utf-8")
    fingerprint = perf_bench.file_fingerprint(
        dataset, display_path="tests/benchmark/data/nfcorpus.json"
    )
    repositories = {
        "omnifuse": {"sha": "a"},
        "synaptic_memory": {"sha": "b"},
    }
    sources = {"harness": {"sha256": "source"}}
    state = {
        "repositories": repositories,
        "sources": sources,
        "dataset_fingerprint": fingerprint,
        "doctor_manifest": {},
    }
    monkeypatch.setattr(
        perf_bench,
        "repository_fingerprint",
        lambda path: (
            repositories["omnifuse"]
            if path.resolve() == perf_bench.ROOT
            else repositories["synaptic_memory"]
        ),
    )
    monkeypatch.setattr(perf_bench, "_benchmark_sources", lambda _repo: sources)
    dataset.write_text("after\n", encoding="utf-8")

    with pytest.raises(perf_bench.ProvenanceError, match="dataset input changed"):
        perf_bench._verify_machine_postflight(
            state, synaptic_repo=tmp_path, dataset_path=dataset
        )


def test_official_postflight_fails_when_environment_lock_changes(tmp_path, monkeypatch):
    import direct_external_bench as direct_external

    repo = tmp_path / "synaptic"
    repo.mkdir()
    dataset = tmp_path / "nfcorpus.json"
    dataset.write_text("{}", encoding="utf-8")
    dataset_fingerprint = perf_bench.file_fingerprint(
        dataset, display_path="tests/benchmark/data/nfcorpus.json"
    )
    runtime = perf_bench._runtime_environment_snapshot()
    runtime["utf8_mode"] = True
    repositories = {
        "omnifuse": {"sha": "a"},
        "synaptic_memory": {"sha": "b"},
    }
    sources = {"harness": {"sha256": "source"}}
    before_probe = {"environment_lock": {"installed_manifest_sha256": "a" * 64}}
    after_probe = {"environment_lock": {"installed_manifest_sha256": "b" * 64}}
    state = {
        "repositories": repositories,
        "sources": sources,
        "dataset_fingerprint": dataset_fingerprint,
        "runtime_environment": runtime,
        "official_environment_probe": before_probe,
    }
    monkeypatch.setattr(direct_external, "_validate_tag_checkout", lambda _repo: {})
    monkeypatch.setattr(
        perf_bench,
        "repository_fingerprint",
        lambda path: (
            repositories["omnifuse"]
            if path.resolve() == perf_bench.ROOT
            else repositories["synaptic_memory"]
        ),
    )
    monkeypatch.setattr(perf_bench, "_benchmark_sources", lambda *_args: sources)
    monkeypatch.setattr(perf_bench, "_runtime_environment_snapshot", lambda: runtime)
    monkeypatch.setattr(
        perf_bench, "_official_environment_probe", lambda _repo: after_probe
    )

    with pytest.raises(perf_bench.ProvenanceError, match="worker environment changed"):
        perf_bench._verify_machine_postflight(
            state,
            synaptic_repo=repo,
            dataset_path=dataset,
            protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        )


def test_machine_report_declares_new_schema_and_provenance_level():
    runtime_environment = perf_bench._runtime_environment_snapshot()
    runtime_environment["utf8_mode"] = True
    doctor_environment = _doctor_environment()
    environment_probe = {
        "environment_lock": {
            "lockfile": {"sha256": "1" * 64},
            "installed_manifest_sha256": "2" * 64,
            "uv_sync_check": {
                "arguments": ["sync", "--check"],
                "selected_extras": ["sqlite", "korean"],
                "checked_package_count": 25,
                "reported_no_changes": True,
                "virtual_environment": "venv",
            },
        }
    }
    state = {
        "dataset": {"path": "dataset.json", "sha256": "a", "bytes": 1},
        "dataset_fingerprint": {
            "path": "dataset.json",
            "sha256": "a",
            "bytes": 1,
        },
        "doctor_link": {"target_id": "nfcorpus"},
        "repositories": {"omnifuse": {}, "synaptic_memory": {}},
        "sources": {"harness": {}},
        "doctor_manifest": {"schema": "omnifuse.eval.doctor"},
        "runtime_environment": runtime_environment,
        "doctor_environment": doctor_environment,
        "frozen_worker_input": {"sha256": "f", "bytes": 1},
        "official_environment_probe": environment_probe,
    }
    after = {
        "repositories": state["repositories"],
        "benchmark_sources": state["sources"],
        "dataset_input": state["dataset_fingerprint"],
        "runtime_environment": runtime_environment,
        "doctor_environment": doctor_environment,
        "official_environment_probe": environment_probe,
    }
    postflight = {
        "after": after,
        "checks": {"postflight_verified_before_publish": True},
    }
    worker_records = [
        {
            "trial_number": ((index - 1) // 2) + 1,
            "order_position": ((index - 1) % 2) + 1,
            "system": "omnifuse" if index in (1, 4) else "synaptic",
            "worker_run_id": f"000000000000400080000000000000{index:02x}",
            "launcher_pid": 100 + index,
            "worker_pid": 200 + index,
            "same_process_id": False,
            "artifact": {"path": f"worker-{index}.json", "sha256": "a", "bytes": 1},
        }
        for index in range(1, 5)
    ]
    report = perf_bench._machine_report(
        args=SimpleNamespace(
            k=10,
            warmup=1,
            repeats=5,
            trials=2,
            protocol=perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
        ),
        candidate_limit=20,
        results=[{"system": "OmniFuse"}, {"system": "synaptic"}],
        schedule=[["omnifuse", "synaptic"], ["synaptic", "omnifuse"]],
        state=state,
        postflight=postflight,
        workers_directory=pathlib.Path("workers"),
        frozen_input_artifact={"path": "input.json", "sha256": "f", "bytes": 1},
        worker_records=worker_records,
    )

    assert report["schema"] == "omnifuse.eval.performance"
    assert report["schema_version"] == 5
    assert report["provenance_level"] == perf_bench.PROVENANCE_LEVEL
    assert report["provenance"]["benchmark_sources"] == state["sources"]
    assert report["provenance"]["frozen_worker_input"] == state["frozen_worker_input"]
    assert report["doctor_manifest"] == state["doctor_manifest"]
    assert report["integrity"]["postflight_verified_before_publish"] is True
    assert report["integrity"]["worker_run_ids_unique"] is True
    assert report["worker_process_summary"]["distinct_worker_run_ids"] == 4
    assert report["worker_process_summary"]["launcher_worker_pid_mismatch_count"] == 4
    assert report["provenance"]["after"] == after
    assert report["contract"]["trial_order"] == [
        ["omnifuse", "synaptic"],
        ["synaptic", "omnifuse"],
    ]
    assert report["contract"]["utf8_mode_contract"] == {
        "controller_required": True,
        "controller_enabled": True,
        "workers_required": True,
    }
    assert report["contract"]["official_environment_lock"] == {
        "lockfile_sha256": "1" * 64,
        "installed_manifest_sha256": "2" * 64,
        "uv_sync_check": environment_probe["environment_lock"]["uv_sync_check"],
    }
    assert report["provenance"]["before"]["official_environment_probe"] == (
        environment_probe
    )


def test_trial_schedule_counterbalances_two_systems_and_aggregates_distributions():
    schedule = perf_bench._trial_schedule(["omnifuse", "synaptic"], 2)
    assert schedule == [
        ["omnifuse", "synaptic"],
        ["synaptic", "omnifuse"],
    ]

    def trial(number, position, ingest, latency, peak, mrr):
        return {
            "trial": {"number": number, "order_position": position},
            "ingest_s": ingest,
            "query_latency_p50_ms": latency,
            "query_latency_p95_ms": latency + 1,
            "query_latency_mean_ms": latency + 0.5,
            "process_memory": {"current_rss_mb": peak - 1, "peak_rss_mb": peak},
            "mrr_at_k": mrr,
        }

    results = perf_bench._aggregate_trial_results(
        {
            "omnifuse": [
                trial(1, 1, 1.0, 2.0, 10.0, 0.5),
                trial(2, 2, 3.0, 4.0, 12.0, 0.5),
            ],
            "synaptic": [
                trial(1, 2, 2.0, 3.0, 20.0, 0.4),
                trial(2, 1, 4.0, 5.0, 22.0, 0.4),
            ],
        },
        ["omnifuse", "synaptic"],
    )

    assert results[0]["order_positions"] == [1, 2]
    assert results[0]["distributions"]["ingest_s"]["mean"] == pytest.approx(2.0)
    assert results[1]["order_positions"] == [2, 1]
    assert results[1]["distributions"]["peak_rss_mb"]["count"] == 2


def test_machine_output_requires_even_counterbalanced_trials(tmp_path):
    with pytest.raises(SystemExit) as error:
        perf_bench.main(
            [
                "--data-dir",
                str(tmp_path),
                "--trials",
                "1",
                "--out",
                str(tmp_path / "result.json"),
            ]
        )
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        perf_bench.main(
            [
                "--data-dir",
                str(tmp_path),
                "--trials",
                "3",
                "--out",
                str(tmp_path / "result.json"),
            ]
        )
    assert error.value.code == 2


def test_official_memory_protocol_enforces_claim_grade_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_bench, "_controller_utf8_mode_enabled", lambda: True)
    base = [
        "--data-dir",
        str(tmp_path),
        "--synaptic-repo",
        str(tmp_path),
        "--protocol",
        perf_bench.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY,
    ]
    with pytest.raises(SystemExit) as error:
        perf_bench.main([*base, "--trials", "1"])
    assert error.value.code == 2

    with pytest.raises(SystemExit) as error:
        perf_bench.main([*base, "--k", "5"])
    assert error.value.code == 2


def test_trial_aggregation_fails_closed_when_accuracy_changes():
    def trial(number, position, mrr):
        return {
            "trial": {"number": number, "order_position": position},
            "ingest_s": 1.0,
            "query_latency_p50_ms": 1.0,
            "query_latency_p95_ms": 1.0,
            "query_latency_mean_ms": 1.0,
            "process_memory": {"current_rss_mb": 1.0, "peak_rss_mb": 1.0},
            "mrr_at_k": mrr,
        }

    with pytest.raises(perf_bench.ProvenanceError, match="accuracy changed"):
        perf_bench._aggregate_trial_results(
            {
                "omnifuse": [trial(1, 1, 0.5), trial(2, 2, 0.6)],
                "synaptic": [trial(1, 2, 0.4), trial(2, 1, 0.4)],
            },
            ["omnifuse", "synaptic"],
        )


def test_trial_aggregation_fails_closed_when_top20_changes():
    def trial(number, position, ranking_hash):
        return {
            "trial": {"number": number, "order_position": position},
            "ingest_s": 1.0,
            "query_latency_p50_ms": 1.0,
            "query_latency_p95_ms": 1.0,
            "query_latency_mean_ms": 1.0,
            "process_memory": {"current_rss_mb": 1.0, "peak_rss_mb": 1.0},
            "mrr_at_k": 0.5,
            "timing": {
                "query": {"mean_round_seconds": 0.1},
                "end_to_end": {"ingest_plus_mean_round_seconds": 1.1},
            },
            "canonical_rankings": {
                "canonical_rankings_sha256": ranking_hash,
            },
        }

    with pytest.raises(perf_bench.ProvenanceError, match="top-20 rankings changed"):
        perf_bench._aggregate_trial_results(
            {
                "omnifuse": [trial(1, 1, "a" * 64), trial(2, 2, "b" * 64)],
                "synaptic": [trial(1, 2, "c" * 64), trial(2, 1, "c" * 64)],
            },
            ["omnifuse", "synaptic"],
        )


def test_module_source_guard_rejects_a_different_checkout(tmp_path):
    expected = tmp_path / "expected" / "src"
    expected.mkdir(parents=True)
    actual = tmp_path / "other" / "synaptic" / "__init__.py"
    actual.parent.mkdir(parents=True)
    actual.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected source below"):
        perf_bench._assert_module_under(str(actual), expected, "synaptic")


def test_git_state_records_dirty_status_and_diff_hash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Fixture"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    clean = perf_bench._git_state(repo)
    tracked.write_text("after\n", encoding="utf-8")
    dirty = perf_bench._git_state(repo)

    assert clean is not None and clean["dirty"] is False
    assert dirty is not None and dirty["dirty"] is True
    assert dirty["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
