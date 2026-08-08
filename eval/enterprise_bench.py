"""Enterprise native-capability track for OmniFuse and synaptic-memory.

This is not a controlled equal-input accuracy comparison. Both systems receive the same
documents and query strings, but the adapters preserve product-native capabilities:
Synaptic receives dataset intent annotations while OmniFuse does not, and ``full_native``
uses different graph and memory semantics. Repeated aggregate results describe this fixed
scenario; per-query rows are diagnostics, not a representative-query win/loss argument.

    python eval/enterprise_bench.py --synaptic-repo PATH --repeats 3 --out result.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EVAL_DIR))

from metrics import BenchmarkResult  # noqa: E402
from omnifuse import Chunk, Feedback, Node, Triple, build_inmemory  # noqa: E402

K = 5
CANDIDATE_LIMIT = 10
MODES = ("docs_only", "full_native")

MEMORY_SEMANTICS_CAVEAT = (
    "full_native preserves different memory semantics: synaptic-memory materializes "
    "session, tool-call, decision, and outcome nodes and applies success/failure "
    "reinforcement, while OmniFuse indexes each session description as positive Feedback "
    "evidence for knowledge_accessed documents. Compare retrieval outcomes, not equivalent "
    "internal memory state."
)
DOCS_ONLY_INTENT_CAVEAT = (
    "docs_only is still asymmetric: dataset intent annotations select Synaptic search "
    "versus agent_search and are passed to agent_search, while OmniFuse receives only "
    "the raw query string. Treat it as a native-capability ablation, not equal-input "
    "head-to-head accuracy."
)
QUERY_RESULTS_CAVEAT = (
    "Each run contains the complete fixed query set for diagnostics. Do not select a "
    "single query list or individual query as representative win/loss evidence; compare "
    "the repeated aggregate metric distributions and report the adapter asymmetries."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repo_provenance(repo: Path) -> dict[str, object]:
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    return {
        "git_head_sha": head,
        "dirty": bool(status),
    }


def _load_scenario(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        scenario = json.load(fh)
    required = {
        "knowledge_sources",
        "knowledge_links",
        "agent_sessions",
        "evaluation_queries",
    }
    missing = required.difference(scenario)
    if missing:
        raise ValueError(f"enterprise scenario is missing keys: {sorted(missing)}")
    ids = [str(doc["id"]) for doc in scenario["knowledge_sources"]]
    if len(ids) != len(set(ids)):
        raise ValueError("enterprise scenario contains duplicate knowledge source ids")
    return scenario


def _omnifuse_materials(
    scenario: dict[str, Any], *, full_native: bool
) -> tuple[list[Node], list[Triple], list[Chunk], list[tuple[str, list[str]]]]:
    nodes = [
        Node(
            id=doc["id"],
            label=doc["title"],
            kind=str(doc.get("kind", "CONCEPT")).lower(),
        )
        for doc in scenario["knowledge_sources"]
    ]
    chunks = [
        Chunk(
            id=doc["id"],
            title=doc["title"],
            text=doc["content"],
            entities=[doc["id"]],
            meta={
                "tags": list(doc.get("tags", [])),
                "source": doc.get("source", ""),
                "properties": dict(doc.get("properties") or {}),
            },
        )
        for doc in scenario["knowledge_sources"]
    ]
    if not full_native:
        return nodes, [], chunks, []

    triples = [
        Triple(
            s=link["source"],
            p=str(link.get("kind", "RELATED")).lower(),
            o=link["target"],
        )
        for link in scenario["knowledge_links"]
    ]
    feedback_pairs = [
        (session["description"], list(session.get("knowledge_accessed", [])))
        for session in scenario["agent_sessions"]
        if session.get("description") and session.get("knowledge_accessed")
    ]
    return nodes, triples, chunks, feedback_pairs


def _first_relevant_rank(candidates: list[str], relevant: set[str]) -> int | None:
    return next(
        (index for index, doc_id in enumerate(candidates, 1) if doc_id in relevant),
        None,
    )


def _add_scored_query(
    accumulator: BenchmarkResult,
    query: dict[str, Any],
    candidates: list[str],
    latency_ms: float,
    *,
    route: str,
) -> dict[str, object]:
    relevant_ordered = [str(doc_id) for doc_id in query["relevant_ids"]]
    relevant = set(relevant_ordered)
    scored = accumulator.add(
        str(query["id"]),
        str(query["query"]),
        candidates,
        relevant,
        k=K,
        description=str(query.get("description", "")),
        search_time_ms=latency_ms,
    )
    ranks = {
        doc_id: (candidates.index(doc_id) + 1 if doc_id in candidates else None)
        for doc_id in relevant_ordered
    }
    return {
        "query_id": query["id"],
        "query": query["query"],
        "intent": query.get("intent", "auto"),
        "route": route,
        "relevant_ids": relevant_ordered,
        "candidate_ids": candidates[:CANDIDATE_LIMIT],
        "evaluated_top_k": candidates[:K],
        "relevant_ranks": ranks,
        "first_relevant_rank": _first_relevant_rank(candidates, relevant),
        "reciprocal_rank": scored["mrr"],
        f"ndcg@{K}": scored["ndcg@k"],
        f"recall@{K}": scored["recall@k"],
        "latency_ms": latency_ms,
    }


def _aggregate(accumulator: BenchmarkResult) -> dict[str, object]:
    summary = accumulator.summary()
    return {
        "query_count": int(summary["total_queries"]),
        "mrr": summary["mrr"],
        f"ndcg@{K}": summary["mean_ndcg@k"],
        f"recall@{K}": summary["mean_recall@k"],
        "mean_latency_ms": summary["mean_search_time_ms"],
    }


def _metric_distribution(values: list[float]) -> dict[str, object]:
    if not values:
        raise ValueError("metric distribution requires at least one run")
    minimum = min(values)
    maximum = max(values)
    return {
        "per_run": values,
        "median": median(values),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "deterministic": len(set(values)) == 1,
    }


def _summarize_repeated_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        raise ValueError("repeated-run summary requires at least one run")

    metric_names = ("mrr", f"ndcg@{K}", f"recall@{K}", "mean_latency_ms")
    aggregate_metrics = {
        metric: _metric_distribution(
            [float(run["aggregate"][metric]) for run in runs]  # type: ignore[index]
        )
        for metric in metric_names
    }
    build_time = _metric_distribution([float(run["build_time_ms"]) for run in runs])

    baseline_queries = {
        str(query["query_id"]): tuple(query["candidate_ids"])
        for query in runs[0]["queries"]  # type: ignore[index]
    }
    observed_query_ids = set(baseline_queries)
    variable_query_ids: set[str] = set()
    query_sets_identical = True
    for run in runs[1:]:
        current = {
            str(query["query_id"]): tuple(query["candidate_ids"])
            for query in run["queries"]  # type: ignore[index]
        }
        if set(current) != observed_query_ids:
            query_sets_identical = False
            variable_query_ids.update(observed_query_ids.symmetric_difference(current))
        for query_id in observed_query_ids.intersection(current):
            if current[query_id] != baseline_queries[query_id]:
                variable_query_ids.add(query_id)

    accuracy_names = ("mrr", f"ndcg@{K}", f"recall@{K}")
    return {
        "aggregate_metric_distributions": aggregate_metrics,
        "build_time_ms_distribution": build_time,
        "accuracy_metrics_deterministic": all(
            bool(aggregate_metrics[name]["deterministic"]) for name in accuracy_names
        ),
        "candidate_rankings": {
            "deterministic": query_sets_identical and not variable_query_ids,
            "query_sets_identical": query_sets_identical,
            "variable_query_ids": sorted(variable_query_ids),
        },
        "variability_observed": (
            any(not bool(row["deterministic"]) for row in aggregate_metrics.values())
            or not bool(build_time["deterministic"])
            or bool(variable_query_ids)
            or not query_sets_identical
        ),
    }


def _repeated_system_result(runs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run_count": len(runs),
        "summary": _summarize_repeated_runs(runs),
        "query_results_caveat": QUERY_RESULTS_CAVEAT,
        "runs": [dict(run, run_index=index) for index, run in enumerate(runs, 1)],
    }


def run_omnifuse_mode(scenario: dict[str, Any], *, mode: str) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    full_native = mode == "full_native"
    nodes, triples, chunks, feedback_pairs = _omnifuse_materials(
        scenario, full_native=full_native
    )
    feedback = Feedback() if full_native else None
    if feedback is not None:
        for description, accessed_ids in feedback_pairs:
            feedback.remember(description, accessed_ids)

    build_started = time.perf_counter()
    omnifuse = build_inmemory(
        nodes,
        triples,
        chunks,
        feedback=feedback,
        vector_k=CANDIDATE_LIMIT,
    )
    build_time_ms = (time.perf_counter() - build_started) * 1000.0

    accumulator = BenchmarkResult()
    query_results: list[dict[str, object]] = []
    for query in scenario["evaluation_queries"]:
        started = time.perf_counter()
        hits = omnifuse.retrieve(str(query["query"]), limit=CANDIDATE_LIMIT)
        latency_ms = (time.perf_counter() - started) * 1000.0
        candidates = [chunk.id for chunk, _score in hits]
        query_results.append(
            _add_scored_query(
                accumulator,
                query,
                candidates,
                latency_ms,
                route="retrieve",
            )
        )

    return {
        "adapter": (
            "documents as Node+Chunk; no triples or Feedback"
            if not full_native
            else "documents as Node+Chunk, knowledge_links as Triple, and session "
            "description -> knowledge_accessed as Feedback evidence"
        ),
        "intent_handling": "dataset intent annotations are not consumed by OmniFuse.retrieve",
        "build_time_ms": build_time_ms,
        "aggregate": _aggregate(accumulator),
        "queries": query_results,
    }


async def _populate_synaptic(
    scenario: dict[str, Any],
    *,
    graph: Any,
    tracker: Any,
    node_kind: Any,
    edge_kind: Any,
    full_native: bool,
) -> dict[str, str]:
    kind_map = {
        "CONCEPT": node_kind.CONCEPT,
        "ENTITY": node_kind.ENTITY,
        "LESSON": node_kind.LESSON,
        "DECISION": node_kind.DECISION,
        "RULE": node_kind.RULE,
        "ARTIFACT": node_kind.ARTIFACT,
    }
    edge_map = {
        "RELATED": edge_kind.RELATED,
        "DEPENDS_ON": edge_kind.DEPENDS_ON,
        "LEARNED_FROM": edge_kind.LEARNED_FROM,
        "CAUSED": edge_kind.CAUSED,
        "PRODUCED": edge_kind.PRODUCED,
    }
    id_map: dict[str, str] = {}
    for doc in scenario["knowledge_sources"]:
        node = await graph.add(
            title=doc["title"],
            content=doc["content"],
            kind=kind_map.get(doc.get("kind", "CONCEPT"), node_kind.CONCEPT),
            tags=doc.get("tags", []),
            source=doc.get("source", ""),
            properties=doc.get("properties"),
        )
        id_map[doc["id"]] = node.id

    if not full_native:
        return id_map

    for link in scenario["knowledge_links"]:
        source_id = id_map.get(link["source"])
        target_id = id_map.get(link["target"])
        if source_id and target_id:
            await graph.link(
                source_id,
                target_id,
                kind=edge_map.get(link.get("kind", "RELATED"), edge_kind.RELATED),
            )

    for session_data in scenario["agent_sessions"]:
        session = await tracker.start_session(
            agent_id=session_data["agent_id"],
            description=session_data["description"],
        )
        for tool_call in session_data["tool_calls"]:
            await tracker.log_tool_call(
                session.id,
                tool_name=tool_call["tool"],
                parameters=tool_call.get("params"),
                result=tool_call.get("result", ""),
                success=tool_call.get("success", True),
                duration_ms=tool_call.get("duration_ms", 0.0),
            )
        for decision_data in session_data.get("decisions", []):
            decision = await tracker.record_decision(
                session.id,
                title=decision_data["title"],
                rationale=decision_data["rationale"],
                alternatives=decision_data.get("alternatives"),
            )
            if "outcome" in decision_data:
                outcome = decision_data["outcome"]
                await tracker.record_outcome(
                    decision.id,
                    title=outcome["title"],
                    content=outcome["content"],
                    success=outcome["success"],
                )
        accessed_ids = [
            id_map[item]
            for item in session_data.get("knowledge_accessed", [])
            if item in id_map
        ]
        if accessed_ids:
            await graph.reinforce(accessed_ids, success=True)
        await tracker.end_session(session.id)

    for session_data in scenario["agent_sessions"]:
        for decision_data in session_data.get("decisions", []):
            outcome = decision_data.get("outcome")
            if outcome is not None and not outcome["success"]:
                accessed_ids = [
                    id_map[item]
                    for item in session_data.get("knowledge_accessed", [])
                    if item in id_map
                ]
                if accessed_ids:
                    await graph.reinforce(accessed_ids, success=False)
    return id_map


async def _synaptic_candidates(
    graph: Any, query: dict[str, Any]
) -> tuple[list[Any], str]:
    intent = str(query.get("intent", "auto"))
    if intent == "auto":
        result = await graph.search(str(query["query"]), limit=CANDIDATE_LIMIT)
        return list(result.nodes), "search"
    result = await graph.agent_search(
        str(query["query"]),
        intent=intent,
        limit=CANDIDATE_LIMIT,
    )
    return list(result.nodes), "agent_search"


async def run_synaptic_mode(
    scenario: dict[str, Any], *, mode: str, synaptic_repo: Path
) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    source_root = (synaptic_repo / "src").resolve()
    sys.path.insert(0, str(source_root))

    import synaptic
    from synaptic.activity import ActivityTracker
    from synaptic.backends.memory import MemoryBackend
    from synaptic.graph import SynapticGraph
    from synaptic.models import EdgeKind, NodeKind

    package_path = Path(synaptic.__file__).resolve()
    if source_root not in package_path.parents:
        raise RuntimeError(
            f"loaded synaptic from {package_path}, expected source below {source_root}"
        )

    backend = MemoryBackend()
    await backend.connect()
    try:
        graph = SynapticGraph(backend)
        tracker = ActivityTracker(graph)
        build_started = time.perf_counter()
        id_map = await _populate_synaptic(
            scenario,
            graph=graph,
            tracker=tracker,
            node_kind=NodeKind,
            edge_kind=EdgeKind,
            full_native=mode == "full_native",
        )
        build_time_ms = (time.perf_counter() - build_started) * 1000.0
        scenario_id_by_native_id = {
            native: scenario_id for scenario_id, native in id_map.items()
        }

        accumulator = BenchmarkResult()
        query_results: list[dict[str, object]] = []
        route_counts = {"search": 0, "agent_search": 0}
        for query in scenario["evaluation_queries"]:
            started = time.perf_counter()
            activated, route = await _synaptic_candidates(graph, query)
            latency_ms = (time.perf_counter() - started) * 1000.0
            route_counts[route] += 1
            candidates = [
                scenario_id_by_native_id.get(item.node.id, item.node.id)
                for item in activated[:CANDIDATE_LIMIT]
            ]
            query_results.append(
                _add_scored_query(
                    accumulator,
                    query,
                    candidates,
                    latency_ms,
                    route=route,
                )
            )
        return {
            "adapter": (
                "knowledge nodes only; links, ActivityTracker sessions, and reinforcement omitted"
                if mode == "docs_only"
                else "tests/benchmark/conftest.py semantics: knowledge nodes+links, complete "
                "ActivityTracker sessions, success reinforcement, then failed-outcome reinforcement"
            ),
            "native_intent_routing": {
                "policy": "intent=auto -> SynapticGraph.search; explicit intent -> SynapticGraph.agent_search",
                "counts": route_counts,
            },
            "build_time_ms": build_time_ms,
            "aggregate": _aggregate(accumulator),
            "queries": query_results,
        }
    finally:
        await backend.close()


async def run_benchmark(synaptic_repo: Path, *, repeats: int = 3) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    synaptic_repo = synaptic_repo.resolve()
    dataset_path = (
        synaptic_repo / "tests" / "benchmark" / "data" / "enterprise_scenario.json"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"enterprise scenario not found: {dataset_path}")
    local_scorer = EVAL_DIR / "metrics.py"
    synaptic_scorer = synaptic_repo / "tests" / "benchmark" / "metrics.py"
    local_scorer_sha = _sha256(local_scorer)
    synaptic_scorer_sha = _sha256(synaptic_scorer)
    if local_scorer_sha != synaptic_scorer_sha:
        raise RuntimeError(
            "scorer mismatch: eval/metrics.py is not byte-identical to the selected "
            "synaptic-memory tests/benchmark/metrics.py"
        )
    scenario = _load_scenario(dataset_path)

    modes: dict[str, object] = {}
    for mode in MODES:
        synaptic_runs: list[dict[str, object]] = []
        omnifuse_runs: list[dict[str, object]] = []
        for _run_index in range(repeats):
            synaptic_runs.append(
                await run_synaptic_mode(
                    scenario, mode=mode, synaptic_repo=synaptic_repo
                )
            )
            omnifuse_runs.append(run_omnifuse_mode(scenario, mode=mode))
        modes[mode] = {
            "semantics": (
                "native-capability docs-only ablation; links and sessions omitted, but "
                "Synaptic still consumes dataset intent routing metadata"
                if mode == "docs_only"
                else "each system's native/documented graph-and-memory adapter over the complete scenario"
            ),
            "asymmetry_caveat": (
                DOCS_ONLY_INTENT_CAVEAT
                if mode == "docs_only"
                else MEMORY_SEMANTICS_CAVEAT
            ),
            "synaptic_memory": _repeated_system_result(synaptic_runs),
            "omnifuse": _repeated_system_result(omnifuse_runs),
        }

    return {
        "schema_version": 2,
        "benchmark": "enterprise_scenario_native_capability_track",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": (
            "Native-capability track, not a controlled equal-input accuracy comparison. "
            "Aggregate metrics cover the complete fixed scenario query set."
        ),
        "evaluation_contract": {
            "k": K,
            "candidate_limit": CANDIDATE_LIMIT,
            "scorer": "eval/metrics.py::BenchmarkResult",
            "repeats": repeats,
            "repeat_assessment": {
                "recommended_minimum": 3,
                "meets_recommendation": repeats >= 3,
            },
            "repeat_recommendation": (
                "Use at least 3 fresh runs for reported results. A lower explicit value "
                "is diagnostic and insufficient to characterize variability."
            ),
            "latency": (
                "one direct wall-clock perf_counter measurement around each native search "
                "call in each fresh adapter run"
            ),
            "latency_caveat": (
                "These in-process latencies are diagnostic only. Mode and run order can "
                "warm lazy tokenizer/model state. Use perf_bench.py for isolated repeated "
                "latency and process-memory comparisons."
            ),
            "modes": list(MODES),
        },
        "docs_only_intent_caveat": DOCS_ONLY_INTENT_CAVEAT,
        "memory_semantics_caveat": MEMORY_SEMANTICS_CAVEAT,
        "query_results_caveat": QUERY_RESULTS_CAVEAT,
        "provenance": {
            "dataset_sha256": _sha256(dataset_path),
            "harness_sha256": _sha256(Path(__file__).resolve()),
            "scorer_sha256": local_scorer_sha,
            "synaptic_scorer_sha256": synaptic_scorer_sha,
            "scorers_byte_identical": True,
            "omnifuse_source": _repo_provenance(ROOT),
            "synaptic_source": _repo_provenance(synaptic_repo),
        },
        "dataset_counts": {
            "knowledge_sources": len(scenario["knowledge_sources"]),
            "knowledge_links": len(scenario["knowledge_links"]),
            "agent_sessions": len(scenario["agent_sessions"]),
            "evaluation_queries": len(scenario["evaluation_queries"]),
        },
        "modes": modes,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synaptic-repo",
        required=True,
        type=Path,
        help="synaptic-memory checkout containing tests/benchmark/data/enterprise_scenario.json",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="fresh runs per mode/system (default: 3; use >=3 for reported results)",
    )
    parser.add_argument("--out", required=True, type=Path, help="JSON result path")
    return parser.parse_args(argv)


def _write_json_atomic(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = asyncio.run(run_benchmark(args.synaptic_repo, repeats=args.repeats))
    _write_json_atomic(args.out, result)
    for mode in MODES:
        row = result["modes"][mode]
        syn = row["synaptic_memory"]["summary"]["aggregate_metric_distributions"]
        omni = row["omnifuse"]["summary"]["aggregate_metric_distributions"]
        print(
            f"{mode}: synaptic median MRR={syn['mrr']['median']:.4f} "
            f"(range {syn['mrr']['range']:.4f}), nDCG@{K}="
            f"{syn[f'ndcg@{K}']['median']:.4f}, R@{K}="
            f"{syn[f'recall@{K}']['median']:.4f}; OmniFuse median MRR="
            f"{omni['mrr']['median']:.4f} (range {omni['mrr']['range']:.4f}), "
            f"nDCG@{K}={omni[f'ndcg@{K}']['median']:.4f}, R@{K}="
            f"{omni[f'recall@{K}']['median']:.4f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
