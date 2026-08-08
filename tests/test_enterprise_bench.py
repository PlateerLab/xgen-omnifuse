import asyncio
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))

import enterprise_bench  # noqa: E402


def _scenario():
    return {
        "knowledge_sources": [
            {
                "id": "doc-a",
                "kind": "RULE",
                "title": "Deployment rule",
                "content": "Rollback after a failed deployment.",
                "tags": ["deploy"],
                "source": "runbook",
                "properties": {"version": "1"},
            },
            {
                "id": "doc-b",
                "kind": "CONCEPT",
                "title": "Inventory API",
                "content": "Inventory uses a cache.",
            },
        ],
        "knowledge_links": [
            {"source": "doc-a", "target": "doc-b", "kind": "DEPENDS_ON"}
        ],
        "agent_sessions": [
            {
                "agent_id": "agent",
                "description": "inventory deployment failed",
                "tool_calls": [
                    {
                        "tool": "deploy",
                        "params": {"service": "inventory"},
                        "result": "failed",
                        "success": False,
                        "duration_ms": 3,
                    }
                ],
                "decisions": [
                    {
                        "title": "rollback",
                        "rationale": "protect orders",
                        "alternatives": ["wait"],
                        "outcome": {
                            "title": "rolled back",
                            "content": "safe",
                            "success": False,
                        },
                    }
                ],
                "knowledge_accessed": ["doc-a", "doc-b"],
            }
        ],
        "evaluation_queries": [
            {
                "id": "q1",
                "query": "deployment rollback",
                "intent": "auto",
                "relevant_ids": ["doc-a"],
            }
        ],
    }


def test_omnifuse_materials_keep_docs_only_and_full_native_distinct():
    docs_nodes, docs_triples, docs_chunks, docs_feedback = (
        enterprise_bench._omnifuse_materials(_scenario(), full_native=False)
    )
    full_nodes, full_triples, full_chunks, full_feedback = (
        enterprise_bench._omnifuse_materials(_scenario(), full_native=True)
    )

    assert [node.id for node in docs_nodes] == ["doc-a", "doc-b"]
    assert [chunk.id for chunk in docs_chunks] == ["doc-a", "doc-b"]
    assert docs_triples == []
    assert docs_feedback == []
    assert full_nodes == docs_nodes
    assert full_chunks == docs_chunks
    assert [(triple.s, triple.p, triple.o) for triple in full_triples] == [
        ("doc-a", "depends_on", "doc-b")
    ]
    assert full_feedback == [("inventory deployment failed", ["doc-a", "doc-b"])]


def test_local_scorer_records_candidate_ranks_and_k_five_metrics():
    accumulator = enterprise_bench.BenchmarkResult()
    query = {
        "id": "q",
        "query": "needle",
        "intent": "auto",
        "relevant_ids": ["relevant", "missing"],
    }
    candidates = ["a", "b", "c", "d", "e", "relevant", "z"]

    row = enterprise_bench._add_scored_query(
        accumulator, query, candidates, 1.25, route="search"
    )

    assert row["first_relevant_rank"] == 6
    assert row["relevant_ranks"] == {"relevant": 6, "missing": None}
    assert row["reciprocal_rank"] == pytest.approx(1 / 6)
    assert row["recall@5"] == 0.0
    assert row["evaluated_top_k"] == candidates[:5]


class _Recorder:
    def __init__(self):
        self.calls = []
        self._counter = 0

    async def add(self, **kwargs):
        self._counter += 1
        self.calls.append(("add", kwargs))
        return SimpleNamespace(id=f"native-{self._counter}")

    async def link(self, source, target, **kwargs):
        self.calls.append(("link", source, target, kwargs))

    async def reinforce(self, node_ids, **kwargs):
        self.calls.append(("reinforce", node_ids, kwargs))


class _Tracker:
    def __init__(self):
        self.calls = []

    async def start_session(self, **kwargs):
        self.calls.append(("start_session", kwargs))
        return SimpleNamespace(id="session")

    async def log_tool_call(self, session_id, **kwargs):
        self.calls.append(("log_tool_call", session_id, kwargs))

    async def record_decision(self, session_id, **kwargs):
        self.calls.append(("record_decision", session_id, kwargs))
        return SimpleNamespace(id="decision")

    async def record_outcome(self, decision_id, **kwargs):
        self.calls.append(("record_outcome", decision_id, kwargs))

    async def end_session(self, session_id):
        self.calls.append(("end_session", session_id))


def _kinds():
    return SimpleNamespace(
        CONCEPT="concept",
        ENTITY="entity",
        LESSON="lesson",
        DECISION="decision",
        RULE="rule",
        ARTIFACT="artifact",
        RELATED="related",
        DEPENDS_ON="depends_on",
        LEARNED_FROM="learned_from",
        CAUSED="caused",
        PRODUCED="produced",
    )


def test_synaptic_population_reproduces_fixture_reinforcement_phases():
    graph = _Recorder()
    tracker = _Tracker()
    kinds = _kinds()

    id_map = asyncio.run(
        enterprise_bench._populate_synaptic(
            _scenario(),
            graph=graph,
            tracker=tracker,
            node_kind=kinds,
            edge_kind=kinds,
            full_native=True,
        )
    )

    assert id_map == {"doc-a": "native-1", "doc-b": "native-2"}
    assert ("link", "native-1", "native-2", {"kind": "depends_on"}) in graph.calls
    assert [call for call in graph.calls if call[0] == "reinforce"] == [
        ("reinforce", ["native-1", "native-2"], {"success": True}),
        ("reinforce", ["native-1", "native-2"], {"success": False}),
    ]
    assert [call[0] for call in tracker.calls] == [
        "start_session",
        "log_tool_call",
        "record_decision",
        "record_outcome",
        "end_session",
    ]


def test_synaptic_native_intent_routing_is_explicit():
    class Graph:
        def __init__(self):
            self.calls = []

        async def search(self, query, **kwargs):
            self.calls.append(("search", query, kwargs))
            return SimpleNamespace(nodes=[])

        async def agent_search(self, query, **kwargs):
            self.calls.append(("agent_search", query, kwargs))
            return SimpleNamespace(nodes=[])

    graph = Graph()
    _, auto_route = asyncio.run(
        enterprise_bench._synaptic_candidates(
            graph, {"query": "plain", "intent": "auto"}
        )
    )
    _, native_route = asyncio.run(
        enterprise_bench._synaptic_candidates(
            graph, {"query": "failure", "intent": "past_failures"}
        )
    )

    assert auto_route == "search"
    assert native_route == "agent_search"
    assert graph.calls == [
        ("search", "plain", {"limit": enterprise_bench.CANDIDATE_LIMIT}),
        (
            "agent_search",
            "failure",
            {"intent": "past_failures", "limit": enterprise_bench.CANDIDATE_LIMIT},
        ),
    ]


def _run_result(*, mrr, ndcg, recall, latency, build, candidates):
    return {
        "adapter": "test adapter",
        "build_time_ms": build,
        "aggregate": {
            "query_count": 1,
            "mrr": mrr,
            "ndcg@5": ndcg,
            "recall@5": recall,
            "mean_latency_ms": latency,
        },
        "queries": [
            {
                "query_id": "q1",
                "candidate_ids": candidates,
            }
        ],
    }


def test_repeated_summary_records_per_run_ranges_and_ranking_variability():
    runs = [
        _run_result(
            mrr=0.5,
            ndcg=0.7,
            recall=0.8,
            latency=2.0,
            build=4.0,
            candidates=["doc-a", "doc-b"],
        ),
        _run_result(
            mrr=0.25,
            ndcg=0.6,
            recall=0.8,
            latency=3.0,
            build=5.0,
            candidates=["doc-b", "doc-a"],
        ),
        _run_result(
            mrr=0.5,
            ndcg=0.7,
            recall=0.8,
            latency=4.0,
            build=6.0,
            candidates=["doc-a", "doc-b"],
        ),
    ]

    summary = enterprise_bench._summarize_repeated_runs(runs)
    mrr = summary["aggregate_metric_distributions"]["mrr"]

    assert mrr == {
        "per_run": [0.5, 0.25, 0.5],
        "median": 0.5,
        "min": 0.25,
        "max": 0.5,
        "range": 0.25,
        "deterministic": False,
    }
    assert summary["aggregate_metric_distributions"]["recall@5"]["deterministic"]
    assert summary["build_time_ms_distribution"]["median"] == 5.0
    assert not summary["accuracy_metrics_deterministic"]
    assert summary["candidate_rankings"] == {
        "deterministic": False,
        "query_sets_identical": True,
        "variable_query_ids": ["q1"],
    }
    assert summary["variability_observed"]


def test_repeated_result_keeps_complete_query_rows_per_run_as_diagnostics():
    run = _run_result(
        mrr=1.0,
        ndcg=1.0,
        recall=1.0,
        latency=1.0,
        build=2.0,
        candidates=["doc-a"],
    )

    result = enterprise_bench._repeated_system_result([run, run, run])

    assert result["run_count"] == 3
    assert [row["run_index"] for row in result["runs"]] == [1, 2, 3]
    assert all(row["queries"] == run["queries"] for row in result["runs"])
    assert "not a representative-query win/loss argument" in (
        enterprise_bench.__doc__ or ""
    )
    assert "Do not select a single query list" in result["query_results_caveat"]


def test_cli_defaults_to_three_repeats():
    args = enterprise_bench._parse_args(
        ["--synaptic-repo", "synaptic", "--out", "result.json"]
    )

    assert args.repeats == 3


def test_atomic_json_writer_replaces_destination_without_temp_files(tmp_path):
    destination = tmp_path / "nested" / "result.json"

    enterprise_bench._write_json_atomic(destination, {"schema_version": 2})
    enterprise_bench._write_json_atomic(destination, {"schema_version": 3})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"schema_version": 3}
    assert list(destination.parent.iterdir()) == [destination]


def test_repeats_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(enterprise_bench.run_benchmark(pathlib.Path("unused"), repeats=0))
