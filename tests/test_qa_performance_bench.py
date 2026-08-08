from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "qa_performance_bench.py"
SPEC = importlib.util.spec_from_file_location(
    "omnifuse_qa_performance_bench", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
qa_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa_bench)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_official_percentile_reproduces_upstream_index_rule() -> None:
    values = [float(value) for value in range(16)]

    assert qa_bench._official_p95(values) == 15.0
    assert qa_bench._latency_summary(values)["p95_ms"] == 15.0


def test_prepare_documents_reproduces_combined_fixture_selection(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write(
        data / "wikipedia_ko_tech.json",
        [{"title": f"wiki {index}", "content": "x" * 1600} for index in range(4)],
    )
    _write(
        data / "github_commits.json",
        [{"message": f"commit message {index}\nbody"} for index in range(4)],
    )
    _write(
        data / "github_issues.json",
        [{"title": f"issue {index}", "body": "issue body"} for index in range(4)],
    )

    documents, state = qa_bench._prepare_documents(tmp_path, data)

    assert len(documents) == state["documents"] == 12
    assert state["selected_records"] == {
        "wikipedia": 4,
        "commits": 4,
        "issues": 4,
    }
    assert len(documents[0]["content"]) == 1500
    assert {document["kind"] for document in documents} == {
        "concept",
        "artifact",
        "entity",
    }
    assert len(state["selected_payload_sha256"]) == 64


def test_sync_measurement_separates_official_and_steady_passes() -> None:
    calls = 0

    def search(query: str) -> list[str]:
        nonlocal calls
        calls += 1
        return [query]

    result = qa_bench._measure_sync(search, repeats=2)

    assert calls == len(qa_bench.QUERIES) * 3
    assert result["official_first_pass"]["samples"] == len(qa_bench.QUERIES)
    assert result["steady"]["samples"] == len(qa_bench.QUERIES) * 2
    assert result["rankings_deterministic"] is True
    assert len(result["all_ranking_hashes"]) == 3


def test_head_to_head_counts_every_common_metric() -> None:
    aggregates = {
        "omnifuse": {
            "metrics": {name: {"median": 1.0} for name in qa_bench.AGGREGATE_PATHS}
        },
        "synaptic": {
            "metrics": {name: {"median": 2.0} for name in qa_bench.AGGREGATE_PATHS}
        },
    }

    result = qa_bench._head_to_head(aggregates)

    assert result["verdict"] == {
        "omnifuse": len(qa_bench.AGGREGATE_PATHS),
        "synaptic": 0,
        "ties": 0,
    }
    assert all(metric["winner"] == "omnifuse" for metric in result["metrics"].values())
