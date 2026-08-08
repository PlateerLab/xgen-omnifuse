from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "persistence_bench.py"
SPEC = importlib.util.spec_from_file_location("omnifuse_persistence_bench", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
persistence_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(persistence_bench)


def test_input_preflight_protocol_preserves_direct_cases_and_accepts_trec():
    assert (
        persistence_bench._input_preflight_protocol("nfcorpus.json")
        == persistence_bench.perf.PROTOCOL_OFFICIAL_EXTERNAL_MEMORY
    )
    assert (
        persistence_bench._input_preflight_protocol("trec_covid.json")
        == persistence_bench.perf.PROTOCOL_SQLITE_NATIVE
    )


def test_accuracy_reports_synaptic_six_metric_contract():
    queries = [
        ("q1", "alpha", {"a", "b"}),
        ("q2", "gamma", {"c"}),
    ]
    rankings = {
        "q1": ["x", "a", "b"],
        "q2": ["c", "x"],
    }

    result = persistence_bench._accuracy(queries, rankings)

    assert set(result) == {
        "mrr_at_20",
        "mrr_at_10",
        "precision_at_10",
        "recall_at_10",
        "f1_at_10",
        "ndcg_at_10",
    }
    assert result["mrr_at_20"] == result["mrr_at_10"] == 0.75


def test_sync_measurement_verifies_rankings_and_sample_count():
    queries = [("q1", "alpha", {"a"}), ("q2", "beta", {"b"})]

    rankings, timing = persistence_bench._measure_sync(
        lambda text: [text[0], "a", "b"],
        queries,
        warmup=1,
        repeats=2,
    )

    assert rankings == {"q1": ["a", "b"], "q2": ["b", "a"]}
    assert timing["steady"]["samples"] == 4
    assert len(timing["round_seconds"]) == 2


def test_sync_measurement_rejects_nondeterministic_ranking():
    calls = 0

    def search(_text):
        nonlocal calls
        calls += 1
        return ["a"] if calls % 2 else ["b"]

    with pytest.raises(RuntimeError, match="ranking changed|first-query"):
        persistence_bench._measure_sync(
            search,
            [("q", "query", {"a"})],
            warmup=1,
            repeats=2,
        )


def test_artifact_manifest_binds_file_bytes(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "b").write_bytes(b"defg")

    artifact = persistence_bench._artifact(tmp_path)

    assert artifact["bytes"] == 7
    assert [row["path"] for row in artifact["files"]] == ["a", "b"]
    assert len(artifact["manifest_sha256"]) == 64


def test_omnifuse_worker_uses_disk_snapshot_and_preserves_rankings():
    corpus = [
        ("a", "Alpha", "alpha beta"),
        ("b", "Beta", "beta gamma"),
    ]
    queries = [("q", "alpha", {"a"})]

    result = persistence_bench._omnifuse_worker(
        corpus,
        queries,
        idf_pow=1.0,
        warmup=1,
        repeats=2,
    )

    assert result["configuration"]["idf_pow"] == 1.0
    assert result["accuracy"]["mrr_at_10"] == 1.0
    assert result["artifact_after_queries"]["files"][0]["path"] == "index.sqlite"
    assert result["artifact_after_create"] == result["artifact_after_queries"]
    assert result["process_memory"]["workload_peak_rss_mb"] > 0
    assert result["process_memory"]["post_create_rss_mb"] > 0
    assert result["process_memory"]["clean_open_rss_mb"] > 0
    assert result["process_memory"]["post_query_rss_mb"] > 0
