import asyncio
import json
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))

import cdc_bench as cdc  # noqa: E402
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


def _corpus(size: int = 12):
    return [
        (
            f"doc-{index}",
            f"Title {index}",
            f"alpha beta document {index} has final searchable text",
        )
        for index in range(size)
    ]


def _queries():
    return [
        ("q-alpha", "alpha", {"doc-0"}),
        ("q-final", "final searchable", {"doc-1", "doc-2"}),
    ]


def _trace() -> cdc.MutationTrace:
    return cdc.MutationTrace(
        seed=42,
        group_fraction=0.1,
        insert=("doc-3",),
        update=("doc-1",),
        delete=("doc-2",),
        noop=("doc-0",),
    )


def _worker_environment() -> dict:
    return {
        "python": "3.12.0",
        "python_executable": "python",
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


def _validated_result(trace: cdc.MutationTrace) -> dict:
    queries = [("q", "query", {"doc"})]
    ranking_map = {"q": [("doc", (1.0).hex())]}
    rankings = cdc._ranking_rows(queries, ranking_map)
    ranking_sha256 = cdc.canonical_json_sha256(rankings)
    query_contract = cdc._active_query_contract(queries)
    checkpoints = [
        {
            "phase": phase,
            **query_contract,
            "ordered_external_ids_and_float_hex": True,
            "candidate_equals_full_rebuild_oracle": True,
            "difference_count": 0,
            "candidate_rankings_sha256": ranking_sha256,
            "oracle_rankings_sha256": ranking_sha256,
            "verification_only": {
                "candidate_query_seconds": 0.01,
                "oracle_rebuild_seconds": 0.02,
                "oracle_query_seconds": 0.01,
            },
        }
        for phase in ("insert", "update", "delete", "noop")
    ]
    cold_latency = {
        "count": 1,
        "min_ms": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "mean_ms": 1.0,
        "max_ms": 1.0,
    }
    steady_latency = {**cold_latency, "count": 2}
    return {
        "system": "OmniFuse",
        "trace_sha256": trace.sha256,
        "mutation": {
            "seconds": 0.01,
            "phase_seconds": {
                "insert": 0.002,
                "update": 0.003,
                "delete": 0.004,
                "noop": 0.001,
            },
            "native_document_writes": trace.group_size * 3,
            "inserted": trace.group_size,
            "updated": trace.group_size,
            "deleted": trace.group_size,
            "unchanged": trace.group_size,
            "missing": 0,
            "incremental": True,
            "reindexed_documents": 0,
            "final_revision": 4,
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "initial_ingest_seconds": 0.02,
            "mutation_seconds": 0.01,
            "cold_first": {"round_seconds": 0.03, "latency": cold_latency},
            "steady": {
                "rounds": 2,
                "round_seconds": [0.02, 0.02],
                "mean_round_seconds": 0.02,
                "latency": steady_latency,
            },
            "end_to_end": {
                "incremental_mutation_plus_cold_seconds": 0.04,
                "initial_ingest_plus_mutation_plus_cold_seconds": 0.06,
            },
            "oracle_full_rebuild_seconds_verification_only": 0.02,
        },
        "memory": {
            "scope": cdc.MEMORY_SCOPE,
            "before_initial_ingest": {
                "kind": "RSS",
                "current_rss_mb": 8.0,
                "peak_rss_mb": 9.0,
            },
            "after_initial_ingest": {
                "kind": "RSS",
                "current_rss_mb": 9.0,
                "peak_rss_mb": 10.0,
            },
            "after_mutation": {
                "kind": "RSS",
                "current_rss_mb": 10.0,
                "peak_rss_mb": 12.0,
            },
            "after_measured_queries": {
                "kind": "RSS",
                "current_rss_mb": 10.5,
                "peak_rss_mb": 12.5,
            },
        },
        "metrics": cdc._metrics(queries, ranking_map),
        "active_queries": query_contract,
        "exactness": {
            "ordered_external_ids_and_float_hex": True,
            "candidate_equals_full_rebuild_oracle": True,
            "query_count": 1,
            "rankings_sha256": ranking_sha256,
            "rankings": rankings,
        },
        "phase_checkpoints": checkpoints,
        "runtime": {
            "package_version": None,
            "source_bindings": {},
            "tokenizer": None,
        },
    }


def test_trace_is_deterministic_disjoint_and_exact_size():
    corpus = _corpus(40)
    first = cdc.build_mutation_trace(corpus, seed=7, group_fraction=0.1)
    second = cdc.build_mutation_trace(
        list(reversed(corpus)), seed=7, group_fraction=0.1
    )

    assert first == second
    assert first.group_size == 4
    selected = first.insert + first.update + first.delete + first.noop
    assert len(selected) == 16
    assert len(set(selected)) == 16
    assert len(first.sha256) == 64
    summary = cdc._trace_summary(first)
    assert json.loads(json.dumps(summary)) == summary


def test_trace_selection_depends_on_ids_not_document_content_or_qrels():
    corpus = _corpus(20)
    changed_content = [
        (document_id, f"changed {title}", f"unrelated {index}")
        for index, (document_id, title, _text) in enumerate(corpus)
    ]

    assert cdc.build_mutation_trace(
        corpus, seed=9, group_fraction=0.1
    ) == cdc.build_mutation_trace(changed_content, seed=9, group_fraction=0.1)
    assert (
        "qrel"
        not in cdc._trace_summary(
            cdc.build_mutation_trace(corpus, seed=9, group_fraction=0.1)
        )["algorithm"].lower()
    )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.25001])
def test_trace_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="group_fraction"):
        cdc.build_mutation_trace(_corpus(), seed=1, group_fraction=fraction)


def test_trace_rejects_too_small_or_duplicate_corpus():
    with pytest.raises(ValueError, match="at least four"):
        cdc.build_mutation_trace(_corpus(3), seed=1, group_fraction=0.1)
    duplicate = _corpus(4)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="unique"):
        cdc.build_mutation_trace(duplicate, seed=1, group_fraction=0.1)


def test_corpus_states_match_incremental_tie_order():
    corpus = _corpus(4)
    initial, final = cdc._corpus_states(corpus, _trace())

    assert [row[0] for row in initial] == ["doc-0", "doc-1", "doc-2"]
    assert initial[1][2] == corpus[1][2][: len(corpus[1][2]) // 2]
    assert [row[0] for row in final] == ["doc-0", "doc-1", "doc-3"]
    assert final[-1] == corpus[3]


def test_active_qrels_are_intersected_with_live_ids_and_empty_queries_drop():
    corpus = _corpus(4)
    _initial, final = cdc._corpus_states(corpus, _trace())
    queries = [
        ("deleted-only", "deleted", {"doc-2"}),
        ("mixed", "mixed", {"doc-1", "doc-2"}),
        ("inserted", "inserted", {"doc-3"}),
    ]

    active = cdc._active_queries(queries, final)

    assert active == [
        ("mixed", "mixed", {"doc-1"}),
        ("inserted", "inserted", {"doc-3"}),
    ]
    contracts = cdc._checkpoint_contracts(corpus, queries, _trace())
    assert [row["phase"] for row in contracts] == [
        "insert",
        "update",
        "delete",
        "noop",
    ]
    assert [row["count"] for row in contracts] == [3, 3, 2, 2]


def test_half_text_handles_a_single_character_without_equal_preimage():
    assert cdc._half_text("x") == ""
    with pytest.raises(ValueError, match="non-empty"):
        cdc._half_text("")


def test_synaptic_fields_match_upstream_title_fallback_and_nfc():
    title, text = cdc._official_synaptic_fields("", "e\u0301vidence body")

    assert title == "évidence body"
    assert text == "évidence body"
    explicit_title, _text = cdc._official_synaptic_fields("T\u0301", "body")
    assert explicit_title == "T́"


def test_normalize_hits_preserves_exact_float_hex_and_limit():
    hits = [(f"doc-{index}", index / 10) for index in range(25)]

    normalized = cdc._normalize_hits(hits)

    assert len(normalized) == cdc.CANDIDATE_LIMIT
    assert normalized[3] == ("doc-3", float(0.3).hex())


@pytest.mark.parametrize(
    "hits, message",
    [
        ([("doc", 1.0), ("doc", 0.5)], "duplicate"),
        ([("doc", float("nan"))], "non-finite"),
        ([("", 1.0)], "empty"),
    ],
)
def test_normalize_hits_fails_closed(hits, message):
    with pytest.raises(RuntimeError, match=message):
        cdc._normalize_hits(hits)


def test_exact_ranking_comparison_includes_score_bits():
    expected = {"q": [("doc", float(0.5).hex())]}
    same_id_different_score = {"q": [("doc", float(0.5000000000000001).hex())]}

    with pytest.raises(RuntimeError, match="ranking/score mismatch"):
        cdc._assert_exact_rankings(
            expected, same_id_different_score, label="candidate vs oracle"
        )


def test_exactness_row_validator_checks_query_order_unique_ids_and_canonical_hex():
    queries = [("q-1", "first", {"doc"}), ("q-2", "second", {"doc"})]
    rows = [
        {"query_id": "q-1", "hits": [{"document_id": "doc", "score_hex": (1.0).hex()}]},
        {"query_id": "q-2", "hits": []},
    ]

    assert cdc._validate_exactness_rows(rows, expected_queries=queries) == {
        "q-1": [("doc", (1.0).hex())],
        "q-2": [],
    }
    wrong_order = json.loads(json.dumps(rows))
    wrong_order[0]["query_id"] = "q-2"
    with pytest.raises(cdc.ProvenanceError, match="ID/order"):
        cdc._validate_exactness_rows(wrong_order, expected_queries=queries)
    duplicate = json.loads(json.dumps(rows))
    duplicate[0]["hits"].append(duplicate[0]["hits"][0])
    with pytest.raises(cdc.ProvenanceError, match="fields"):
        cdc._validate_exactness_rows(duplicate, expected_queries=queries)
    noncanonical = json.loads(json.dumps(rows))
    noncanonical[0]["hits"][0]["score_hex"] = "0x1p+0"
    with pytest.raises(cdc.ProvenanceError, match="not canonical"):
        cdc._validate_exactness_rows(noncanonical, expected_queries=queries)


def test_official_six_metrics_use_top20_mrr_and_top10_other_metrics():
    queries = [("q", "query", {"relevant"})]
    rankings = {"q": [("other", (1.0).hex()), ("relevant", (0.5).hex())]}

    metrics = cdc._metrics(queries, rankings)

    assert set(metrics) == set(cdc.METRIC_NAMES)
    assert metrics["mrr_at_20"] == 0.5
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["precision_at_10"] == 0.5
    assert metrics["recall_at_10"] == 1.0
    assert metrics["f1_at_10"] == pytest.approx(2 / 3)


def test_sync_and_async_rounds_materialize_scores():
    queries = [("q", "text", {"doc"})]

    sync_rankings, sync_samples, sync_seconds = cdc._measure_sync_round(
        lambda _text: [("doc", 0.75)], queries
    )

    async def search(_text):
        return [("doc", 0.75)]

    async_rankings, async_samples, async_seconds = asyncio.run(
        cdc._measure_async_round(search, queries)
    )
    assert sync_rankings == async_rankings == {"q": [("doc", (0.75).hex())]}
    assert len(sync_samples) == len(async_samples) == 1
    assert sync_seconds >= 0.0
    assert async_seconds >= 0.0


def test_native_build_boundaries_prepare_before_clock_and_release_after(monkeypatch):
    class Adapter:
        def __del__(self):
            events.append("release")

    def clock():
        events.append("clock")
        return next(ticks)

    def prepare(_corpus):
        events.append("prepare")
        return Adapter()

    def build(_adapter):
        events.append("build")
        return "sync-result"

    events = []
    ticks = iter((100, 160))
    monkeypatch.setattr(cdc.time, "perf_counter_ns", clock)
    monkeypatch.setattr(cdc.gc, "collect", lambda: events.append("collect"))

    result, seconds = cdc._timed_sync_native_build(
        _corpus(4), prepare=prepare, build=build
    )

    assert result == "sync-result"
    assert seconds == 60 / 1_000_000_000
    assert events == ["prepare", "clock", "build", "clock", "release", "collect"]

    async def async_build(_adapter):
        events.append("build")
        return "raw-result"

    def finalize(_adapter, raw):
        events.append("finalize")
        return f"final-{raw}"

    events.clear()
    ticks = iter((200, 290))
    result, seconds = asyncio.run(
        cdc._timed_async_native_build(
            _corpus(4),
            prepare=prepare,
            build=async_build,
            finalize=finalize,
        )
    )

    assert result == "final-raw-result"
    assert seconds == 90 / 1_000_000_000
    assert events == [
        "prepare",
        "clock",
        "build",
        "clock",
        "finalize",
        "release",
        "collect",
    ]


def test_native_input_adapters_preserve_the_same_corpus_rows():
    corpus = [("doc", "title", "text")]

    assert cdc._prepare_omnifuse_documents(corpus) == [
        {"id": "doc", "title": "title", "text": "text"}
    ]
    assert cdc._prepare_synaptic_documents(corpus) == {
        "doc": {"title": "title", "text": "text"}
    }


def test_memory_snapshot_collects_garbage_before_sampling(monkeypatch):
    events = []
    monkeypatch.setattr(cdc.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(
        cdc,
        "_memory_snapshot",
        lambda: events.append("snapshot") or {"current_rss_mb": 1.0},
    )

    assert cdc._stabilized_memory_snapshot() == {"current_rss_mb": 1.0}
    assert events == ["collect", "snapshot"]


def test_omnifuse_mutable_integration_is_exact_against_static_rebuild():
    corpus = _corpus()
    trace = cdc.build_mutation_trace(corpus, seed=42, group_fraction=0.1)

    result = cdc.run_omnifuse_cdc(corpus, _queries(), trace, steady_repeats=2)

    assert result["exactness"]["candidate_equals_full_rebuild_oracle"] is True
    assert result["exactness"]["ordered_external_ids_and_float_hex"] is True
    assert result["mutation"]["native_document_writes"] == 3
    assert result["mutation"]["unchanged"] == 1
    assert result["mutation"]["incremental"] is True
    assert result["mutation"]["reindexed_documents"] == 0
    assert result["timing"]["steady"]["rounds"] == 2
    assert set(result["memory"]) == {
        "scope",
        "before_initial_ingest",
        "after_initial_ingest",
        "after_mutation",
        "after_measured_queries",
    }
    assert set(result["metrics"]) == set(cdc.METRIC_NAMES)
    assert [row["phase"] for row in result["phase_checkpoints"]] == [
        "insert",
        "update",
        "delete",
        "noop",
    ]
    assert all(row["difference_count"] == 0 for row in result["phase_checkpoints"])


def test_worker_environment_contract_is_fail_closed():
    cdc._worker_environment_contract(_worker_environment())
    for key in ("isolated", "ignore_environment", "no_user_site", "safe_path"):
        environment = _worker_environment()
        environment[key] = False
        with pytest.raises(cdc.ProvenanceError, match="not isolated"):
            cdc._worker_environment_contract(environment)
    environment = _worker_environment()
    environment["pythonpath"] = "untrusted"
    with pytest.raises(cdc.ProvenanceError, match="path override"):
        cdc._worker_environment_contract(environment)


def test_worker_command_uses_isolated_utf8_process(tmp_path):
    args = SimpleNamespace(
        data_dir=tmp_path,
        dataset="nfcorpus.json",
        synaptic_repo=tmp_path / "synaptic",
        steady_repeats=3,
        seed=8,
        mutation_group_fraction=0.1,
    )

    command = cdc._worker_command(
        args,
        system="omnifuse",
        input_file=tmp_path / "input.json",
        result_file=tmp_path / "result.json",
        worker_run_id=WORKER_RUN_ID,
    )

    assert command[1:4] == ["-I", "-X", "utf8"]
    assert command[4] == str(cdc.SCRIPT_PATH)
    assert command[command.index("--worker") + 1] == "omnifuse"
    assert command[command.index("--worker-run-id") + 1] == WORKER_RUN_ID


def test_cdc_worker_directory_defaults_to_durable_worklogs_path(tmp_path):
    output = tmp_path / "claim.json"
    configured = tmp_path / "raw-workers"

    assert cdc._worker_directory(output, configured) == configured.resolve()
    default_directory = cdc._worker_directory(output, None)
    assert default_directory.parent == (cdc.ROOT / "worklogs").resolve()
    assert re.fullmatch(
        r"claim-[0-9a-f]{16}-cdc-workers", default_directory.name
    )
    assert default_directory != cdc._worker_directory(
        tmp_path / "other" / "claim.json", None
    )
    assert cdc.SCHEMA_VERSION == 3
    assert cdc.WORKER_SCHEMA_VERSION == 3


def test_existing_cdc_worker_directory_fails_before_machine_preflight(
    tmp_path, monkeypatch, capsys
):
    synaptic_repo = tmp_path / "synaptic"
    synaptic_repo.mkdir()
    worker_root = tmp_path / "existing-workers"
    worker_root.mkdir()
    monkeypatch.setattr(
        cdc.perf, "_require_protocol_utf8_mode", lambda *_args, **_kwargs: None
    )

    def unexpected_preflight(**_kwargs):
        pytest.fail("heavy machine preflight must not run after a worker collision")

    monkeypatch.setattr(cdc.perf, "_machine_preflight", unexpected_preflight)
    with pytest.raises(SystemExit) as error:
        cdc.main(
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


def test_synaptic_official_driver_binding_allows_only_exact_tests_path(
    tmp_path, monkeypatch
):
    driver = tmp_path / cdc.perf.SYNAPTIC_EXTERNAL_DRIVER_RELATIVE
    driver.parent.mkdir(parents=True)
    driver.write_text("driver = True\n", encoding="utf-8")
    fingerprint = cdc.file_fingerprint(
        driver, display_path=cdc.perf.SYNAPTIC_EXTERNAL_DRIVER_RELATIVE.as_posix()
    )
    binding = {**fingerprint, "resolved_path": str(driver.resolve())}
    monkeypatch.setattr(
        cdc, "SYNAPTIC_BINDINGS", frozenset({"official_external_driver"})
    )

    assert (
        cdc._validate_bindings(
            {"official_external_driver": binding},
            system="synaptic",
            synaptic_repo=tmp_path,
        )["official_external_driver"]
        == binding
    )

    wrong = tmp_path / "tests" / "wrong.py"
    wrong.write_text("wrong = True\n", encoding="utf-8")
    wrong_binding = {
        **cdc.file_fingerprint(wrong, display_path="tests/wrong.py"),
        "resolved_path": str(wrong.resolve()),
    }
    with pytest.raises(cdc.ProvenanceError, match="driver binding path"):
        cdc._validate_bindings(
            {"official_external_driver": wrong_binding},
            system="synaptic",
            synaptic_repo=tmp_path,
        )


def test_parser_defaults_and_controller_contract(tmp_path, capsys):
    args = cdc._parser().parse_args(
        ["--data-dir", str(tmp_path), "--synaptic-repo", str(tmp_path)]
    )
    assert args.trials == 2
    assert args.steady_repeats == 5
    assert args.mutation_group_fraction == 0.01
    with pytest.raises(SystemExit) as error:
        cdc.main(["--data-dir", str(tmp_path), "--synaptic-repo", str(tmp_path)])
    assert error.value.code == 2
    assert "requires --out and --doctor-manifest" in capsys.readouterr().err


def test_validate_worker_result_accepts_contract_and_rejects_tampering(
    tmp_path, monkeypatch
):
    trace = cdc.MutationTrace(42, 0.1, ("i",), ("u",), ("d",), ("n",))
    result = _validated_result(trace)
    expected_queries = [("q", "query", {"doc"})]
    active_contract = cdc._active_query_contract(expected_queries)
    expected_checkpoints = [
        {"phase": phase, **active_contract}
        for phase in ("insert", "update", "delete", "noop")
    ]
    expected_input = {
        "path": cdc.WORKER_INPUT_DISPLAY_PATH,
        "sha256": "a" * 64,
        "bytes": 10,
        "documents": 4,
        "source_scored_queries": 1,
        "active_scored_queries": 1,
        "active_query_ids_ordered_sha256": active_contract["query_ids_ordered_sha256"],
        "active_relevance_judgments": 1,
    }
    args = SimpleNamespace(
        steady_repeats=2,
        seed=42,
        mutation_group_fraction=0.1,
        synaptic_repo=tmp_path,
    )
    envelope = {
        "schema": cdc.WORKER_SCHEMA,
        "schema_version": cdc.WORKER_SCHEMA_VERSION,
        "status": "ok",
        "system": "omnifuse",
        "protocol": cdc.PROTOCOL,
        "contract": {
            "k": cdc.K,
            "candidate_limit": cdc.CANDIDATE_LIMIT,
            "steady_repeats": 2,
            "seed": 42,
            "mutation_group_fraction": 0.1,
        },
        "input": expected_input,
        "trace": cdc._trace_summary(trace),
        "worker_identity": _worker_identity(),
        "environment": _worker_environment(),
        "official_environment_lock": {"before": {"lock": 1}, "after": {"lock": 1}},
        "result": result,
    }
    envelope = json.loads(json.dumps(envelope))
    monkeypatch.setattr(cdc, "_validate_bindings", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cdc.perf, "_validate_official_environment_lock", lambda *_args: None
    )

    validated = cdc._validate_worker_result(
        envelope,
        system="omnifuse",
        expected_input=expected_input,
        expected_trace=trace,
        expected_queries=expected_queries,
        expected_checkpoints=expected_checkpoints,
        args=args,
        trial_number=1,
        order_position=2,
        expected_worker_run_id=WORKER_RUN_ID,
        official_environment_lock={"lock": 1},
    )
    assert validated["worker_identity"]["worker_run_id"] == WORKER_RUN_ID

    wrong_identity = json.loads(json.dumps(envelope))
    wrong_identity["worker_identity"]["worker_run_id"] = (
        "00000000000040008000000000000002"
    )
    with pytest.raises(cdc.ProvenanceError, match="does not match"):
        cdc._validate_worker_result(
            wrong_identity,
            system="omnifuse",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=2,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )

    assert validated["trial"] == {"number": 1, "order_position": 2}

    tampered_input = json.loads(json.dumps(envelope))
    tampered_input["input"]["unexpected"] = True
    with pytest.raises(cdc.ProvenanceError, match="input binding"):
        cdc._validate_worker_result(
            tampered_input,
            system="omnifuse",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=2,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )

    missing_revision = json.loads(json.dumps(envelope))
    del missing_revision["result"]["mutation"]["final_revision"]
    with pytest.raises(cdc.ProvenanceError, match="mutation accounting"):
        cdc._validate_worker_result(
            missing_revision,
            system="omnifuse",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=2,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )

    invalid_checkpoint = json.loads(json.dumps(envelope))
    invalid_checkpoint["result"]["phase_checkpoints"][0][
        "candidate_rankings_sha256"
    ] = "z" * 64
    invalid_checkpoint["result"]["phase_checkpoints"][0][
        "oracle_rankings_sha256"
    ] = "z" * 64
    with pytest.raises(cdc.ProvenanceError, match="phase exactness"):
        cdc._validate_worker_result(
            invalid_checkpoint,
            system="omnifuse",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=2,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )

    for mutation, message in (
        (("scope", None, "wrong scope"), "memory evidence"),
        (("after_initial_ingest", "current_rss_mb", 11.0), "exceeds peak"),
        (("after_measured_queries", "peak_rss_mb", 11.5), "peak RSS decreased"),
        (("after_mutation", "kind", "other RSS"), "kind changed"),
    ):
        invalid_memory = json.loads(json.dumps(envelope))
        phase, field, value = mutation
        if field is None:
            invalid_memory["result"]["memory"][phase] = value
        else:
            invalid_memory["result"]["memory"][phase][field] = value
        with pytest.raises(cdc.ProvenanceError, match=message):
            cdc._validate_worker_result(
                invalid_memory,
                system="omnifuse",
                expected_input=expected_input,
                expected_trace=trace,
                expected_queries=expected_queries,
                expected_checkpoints=expected_checkpoints,
                args=args,
                trial_number=1,
                order_position=2,
                expected_worker_run_id=WORKER_RUN_ID,
                official_environment_lock={"lock": 1},
            )

    extra_runtime = json.loads(json.dumps(envelope))
    extra_runtime["result"]["runtime"]["unexpected"] = True
    with pytest.raises(cdc.ProvenanceError, match="runtime is invalid"):
        cdc._validate_worker_result(
            extra_runtime,
            system="omnifuse",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=2,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )

    envelope["result"]["mutation"]["reindexed_documents"] = 4
    with pytest.raises(cdc.ProvenanceError, match="mutation accounting"):
        cdc._validate_worker_result(
            envelope,
            system="omnifuse",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=2,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )


def test_synaptic_worker_runtime_and_noop_contract_are_exact(tmp_path, monkeypatch):
    trace = cdc.MutationTrace(42, 0.1, ("i",), ("u",), ("d",), ("n",))
    expected_queries = [("q", "query", {"doc"})]
    active_contract = cdc._active_query_contract(expected_queries)
    expected_checkpoints = [
        {"phase": phase, **active_contract}
        for phase in ("insert", "update", "delete", "noop")
    ]
    expected_input = {
        "path": cdc.WORKER_INPUT_DISPLAY_PATH,
        "sha256": "a" * 64,
        "bytes": 10,
        "documents": 4,
        "source_scored_queries": 1,
        "active_scored_queries": 1,
        "active_query_ids_ordered_sha256": active_contract[
            "query_ids_ordered_sha256"
        ],
        "active_relevance_judgments": 1,
    }
    source = tmp_path / "src" / "synaptic" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("__version__ = '0.27.0'\n", encoding="utf-8")
    driver = tmp_path / cdc.perf.SYNAPTIC_EXTERNAL_DRIVER_RELATIVE
    driver.parent.mkdir(parents=True)
    driver.write_text("driver = True\n", encoding="utf-8")
    scorer = tmp_path / cdc.perf.SYNAPTIC_SCORER_RELATIVE
    scorer.write_text("scorer = True\n", encoding="utf-8")

    def binding(path):
        fingerprint = cdc.file_fingerprint(path, display_path=str(path.resolve()))
        return {**fingerprint, "resolved_path": str(path.resolve())}

    bindings = {
        name: binding(
            driver if name == "official_external_driver" else source
        )
        for name in cdc.SYNAPTIC_BINDINGS
    }
    result = _validated_result(trace)
    result["system"] = "synaptic"
    del result["mutation"]["final_revision"]
    result["mutation"][
        "noop_adapter_semantics"
    ] = cdc.SYNAPTIC_NOOP_ADAPTER_SEMANTICS
    result["runtime"] = {
        "package_version": "0.27.0",
        "source_bindings": bindings,
        "tokenizer": {
            "mode": "unused",
            "korean_normalization_used": False,
            "kiwi_available": None,
            "kiwi_version": None,
            "kiwi_model_version": None,
            "modules": {
                name: None for name in cdc.perf.TOKENIZER_MODULE_NAMES
            },
        },
        "official_external_runtime": {
            "python_executable": "python",
            "synaptic_package": str(source.resolve()),
            "synaptic_version": "0.27.0",
            "upstream_driver": str(driver.resolve()),
            "upstream_scorer": str(scorer.resolve()),
        },
    }
    envelope = {
        "schema": cdc.WORKER_SCHEMA,
        "schema_version": cdc.WORKER_SCHEMA_VERSION,
        "status": "ok",
        "system": "synaptic",
        "protocol": cdc.PROTOCOL,
        "contract": {
            "k": cdc.K,
            "candidate_limit": cdc.CANDIDATE_LIMIT,
            "steady_repeats": 2,
            "seed": 42,
            "mutation_group_fraction": 0.1,
        },
        "input": expected_input,
        "trace": cdc._trace_summary(trace),
        "worker_identity": _worker_identity(),
        "environment": _worker_environment(),
        "official_environment_lock": {
            "before": {"lock": 1},
            "after": {"lock": 1},
        },
        "result": result,
    }
    args = SimpleNamespace(
        steady_repeats=2,
        seed=42,
        mutation_group_fraction=0.1,
        synaptic_repo=tmp_path,
    )
    monkeypatch.setattr(
        cdc.perf, "_validate_official_environment_lock", lambda *_args: None
    )

    validated = cdc._validate_worker_result(
        envelope,
        system="synaptic",
        expected_input=expected_input,
        expected_trace=trace,
        expected_queries=expected_queries,
        expected_checkpoints=expected_checkpoints,
        args=args,
        trial_number=1,
        order_position=1,
        expected_worker_run_id=WORKER_RUN_ID,
        official_environment_lock={"lock": 1},
    )
    assert validated["runtime"]["official_external_runtime"][
        "upstream_scorer"
    ] == str(scorer.resolve())

    wrong_noop = json.loads(json.dumps(envelope))
    wrong_noop["result"]["mutation"]["noop_adapter_semantics"] = "skip everything"
    with pytest.raises(cdc.ProvenanceError, match="no-op adapter semantics"):
        cdc._validate_worker_result(
            wrong_noop,
            system="synaptic",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )

    incomplete_runtime = json.loads(json.dumps(envelope))
    del incomplete_runtime["result"]["runtime"]["official_external_runtime"][
        "upstream_scorer"
    ]
    with pytest.raises(cdc.ProvenanceError, match="runtime evidence is invalid"):
        cdc._validate_worker_result(
            incomplete_runtime,
            system="synaptic",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )

    wrong_runtime_path = json.loads(json.dumps(envelope))
    wrong_runtime_path["result"]["runtime"]["official_external_runtime"][
        "upstream_scorer"
    ] = str((tmp_path / "wrong.py").resolve())
    with pytest.raises(cdc.ProvenanceError, match="runtime path evidence"):
        cdc._validate_worker_result(
            wrong_runtime_path,
            system="synaptic",
            expected_input=expected_input,
            expected_trace=trace,
            expected_queries=expected_queries,
            expected_checkpoints=expected_checkpoints,
            args=args,
            trial_number=1,
            order_position=1,
            expected_worker_run_id=WORKER_RUN_ID,
            official_environment_lock={"lock": 1},
        )


def test_aggregate_requires_cross_trial_accuracy_identity():
    trace = cdc.MutationTrace(42, 0.1, ("i",), ("u",), ("d",), ("n",))
    omni = _validated_result(trace)
    synaptic = _validated_result(trace)
    omni["trial"] = {"order_position": 1}
    synaptic["trial"] = {"order_position": 2}
    trials = {"omnifuse": [omni, dict(omni)], "synaptic": [synaptic, dict(synaptic)]}

    aggregates = cdc._aggregate(trials)

    assert [row["trial_count"] for row in aggregates] == [2, 2]
    assert aggregates[0]["distributions"]["rss_initial_ingest_delta_mb"]["p50"] == 1.0
    assert (
        aggregates[0]["distributions"]["rss_after_measured_queries_delta_mb"]["p50"]
        == 2.5
    )
    changed = json.loads(json.dumps(omni))
    changed["metrics"]["mrr_at_20"] = 0.25
    trials["omnifuse"] = [omni, changed]
    with pytest.raises(cdc.ProvenanceError, match="accuracy changed"):
        cdc._aggregate(trials)


def test_run_worker_binds_input_and_refuses_result_overwrite(tmp_path, monkeypatch):
    corpus = _corpus()
    queries = _queries()
    payload = cdc.perf._frozen_input_payload(corpus, queries)
    input_file = tmp_path / "input.json"
    input_file.write_bytes(payload)
    result_file = tmp_path / "result.json"
    measurement_finished = False
    args = SimpleNamespace(
        input_file=input_file,
        result_file=result_file,
        worker="omnifuse",
        worker_run_id=WORKER_RUN_ID,
        synaptic_repo=tmp_path,
        seed=42,
        mutation_group_fraction=0.1,
        steady_repeats=2,
    )
    monkeypatch.setattr(
        cdc.perf, "_official_environment_lock_evidence", lambda _repo: {"lock": 1}
    )
    monkeypatch.setattr(cdc.perf, "_worker_environment_snapshot", _worker_environment)

    def run_cdc(_corpus, _queries, trace, **_kwargs):
        nonlocal measurement_finished
        measurement_finished = True
        return _validated_result(trace)

    monkeypatch.setattr(cdc, "run_omnifuse_cdc", run_cdc)
    monkeypatch.setattr(
        cdc,
        "capture_worker_identity",
        lambda run_id: (
            _worker_identity(run_id)
            if measurement_finished
            else pytest.fail("worker identity was captured during measurement")
        ),
    )

    cdc._run_worker(args)

    written = json.loads(result_file.read_text(encoding="utf-8"))
    assert written["status"] == "ok"
    assert written["trace"]["qrels_used_for_selection"] is False
    assert written["worker_identity"] == _worker_identity()
    with pytest.raises(cdc.ProvenanceError, match="write-once"):
        cdc._run_worker(args)
