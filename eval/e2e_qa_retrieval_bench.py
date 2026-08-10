"""Retrieval/evidence comparison for Synaptic's official HotPotQA E2E path.

The upstream test builds an auto-ontology/phrase graph, retrieves an evidence chain and
then calls an external answer LLM.  This runner reproduces the same 24 questions and the
retrieval-owned evidence budget without pretending that an unavailable LLM produced an
answer.  Gold-document retrieval and answer support in the assembled context are compared
in isolated, counterbalanced workers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import re
import site
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = (ROOT / "src").resolve()
sys.path.insert(0, str(EVAL_DIR))

import perf_bench as perf  # noqa: E402
from provenance import (  # noqa: E402
    ProvenanceError,
    assert_unchanged,
    canonical_json_sha256,
    capture_worker_identity,
    ensure_output_absent,
    file_fingerprint,
    new_worker_run_id,
    read_json_artifact,
    repository_fingerprint,
    run_with_launcher_pid,
    validate_worker_identity,
    worker_process_summary,
    write_json_once,
)

SCHEMA = "omnifuse.eval.e2e_qa_retrieval"
SCHEMA_VERSION = 2
WORKER_SCHEMA = "omnifuse.eval.e2e_qa_retrieval_worker"
WORKER_SCHEMA_VERSION = 1
PROVENANCE_LEVEL = "isolated-upstream-e2e-retrieval-ab-ba-write-once-v1"
SYSTEMS = ("omnifuse", "synaptic")
SYSTEM_LABELS = {"omnifuse": "OmniFuse", "synaptic": "synaptic-memory"}
SAMPLE_SEED = 42
MAX_QUESTIONS = 24
SEARCH_LIMIT = 10
EVIDENCE_STEPS = 8
MAX_CONTEXT_TOKENS = 2048
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptic-repo", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers-dir", type=Path)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--worker", choices=SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    return parser


def _load_data(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("HotPotQA E2E data must be a JSON object")
    for name in ("corpus", "queries", "qrels", "answers"):
        if not isinstance(value.get(name), dict) or not value[name]:
            raise ValueError(f"HotPotQA E2E data requires non-empty {name}")
    return value


def _sample_query_ids(data: Mapping[str, Any]) -> list[str]:
    """Reproduce the upstream source-order/seed-42 sample."""

    query_ids = list(data["queries"])
    if len(query_ids) <= MAX_QUESTIONS:
        return query_ids
    return random.Random(SAMPLE_SEED).sample(query_ids, MAX_QUESTIONS)


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _ranking_metrics(
    ranking: Sequence[str], gold_ids: Sequence[str], *, k: int = EVIDENCE_STEPS
) -> dict[str, float | int | bool]:
    ranked = _dedupe([str(value) for value in ranking])[:k]
    gold = set(str(value) for value in gold_ids)
    if not gold:
        raise ValueError("ranking metrics require at least one gold document")
    hit_ranks = [rank for rank, value in enumerate(ranked, 1) if value in gold]
    hits = len(hit_ranks)
    precision = hits / k
    recall = hits / len(gold)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hit_ranks)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return {
        "retrieved_documents": len(ranked),
        "gold_documents": len(gold),
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": hits > 0,
        "all_gold": hits == len(gold),
        "reciprocal_rank": 0.0 if not hit_ranks else 1.0 / hit_ranks[0],
        "ndcg": 0.0 if ideal == 0.0 else dcg / ideal,
    }


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(value)]


def _context_support(context: str, answer: str) -> dict[str, float | int | bool]:
    context_tokens = _tokens(context)
    answer_tokens = _tokens(answer)
    normalized_context = " ".join(context_tokens)
    normalized_answer = " ".join(answer_tokens)
    answer_set = set(answer_tokens)
    context_set = set(context_tokens)
    token_recall = (
        0.0 if not answer_set else len(answer_set & context_set) / len(answer_set)
    )
    return {
        "context_characters": len(context),
        "context_tokens": len(context.split()),
        "answer_tokens": len(answer_tokens),
        "answer_exact": bool(
            normalized_answer and normalized_answer in normalized_context
        ),
        "answer_token_recall": token_recall,
    }


def _truncate_context(value: str, *, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    words = value.split()
    if len(words) <= max_tokens:
        return value
    return " ".join(words[:max_tokens])


def _process_memory() -> dict[str, float | str | None]:
    current, lifetime_peak, kind = perf._process_memory_bytes()
    return {
        "current_rss_mb": None if current is None else current / 1_000_000.0,
        "lifetime_peak_rss_mb": (
            None if lifetime_peak is None else lifetime_peak / 1_000_000.0
        ),
        "measurement_kind": kind,
    }


def _module_file(value: Any, *, source_root: Path, name: str) -> dict[str, Any]:
    module = sys.modules.get(value.__module__)
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise RuntimeError(f"cannot resolve source for {name}")
    path = Path(raw_path).resolve()
    try:
        display = str(path.relative_to(source_root.parent))
    except ValueError as exc:
        raise RuntimeError(f"loaded {name} from {path}, outside {source_root}") from exc
    return {
        **file_fingerprint(path, display_path=display),
        "resolved_path": str(path),
    }


def _data_state(path: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    query_ids = _sample_query_ids(data)
    selected = [
        {
            "query_id": query_id,
            "query": data["queries"][query_id],
            "answer": data["answers"][query_id],
            "qrels": sorted(data["qrels"][query_id]),
        }
        for query_id in query_ids
    ]
    return {
        **file_fingerprint(path, display_path=str(path)),
        "corpus_documents": len(data["corpus"]),
        "total_queries": len(data["queries"]),
        "sampled_queries": len(query_ids),
        "sample_seed": SAMPLE_SEED,
        "sample_query_ids": query_ids,
        "sample_payload_sha256": canonical_json_sha256(selected),
    }


def _map_selected_texts(
    hits: Sequence[tuple[Any, float]], selected_texts: Sequence[str]
) -> list[str]:
    used: set[int] = set()
    result: list[str] = []
    for text in selected_texts:
        for index, (chunk, _score) in enumerate(hits):
            if index not in used and chunk.text == text:
                used.add(index)
                result.append(str(chunk.id))
                break
    return result


def _run_omnifuse(
    data: Mapping[str, Any], *, include_context: bool = False
) -> dict[str, Any]:
    sys.path.insert(0, str(SOURCE_ROOT))
    import omnifuse as package
    from omnifuse import Chunk, build_inmemory
    from omnifuse.fusion import mmr
    from omnifuse.linking import derive_title_links
    from omnifuse.oneshot import OmniFuse

    bindings = {
        "package": _module_file(
            package.Chunk, source_root=SOURCE_ROOT, name="omnifuse"
        ),
        "build_inmemory": _module_file(
            build_inmemory, source_root=SOURCE_ROOT, name="omnifuse.build_inmemory"
        ),
        "retrieve": _module_file(
            OmniFuse.retrieve,
            source_root=SOURCE_ROOT,
            name="omnifuse.OmniFuse.retrieve",
        ),
        "mmr": _module_file(mmr, source_root=SOURCE_ROOT, name="omnifuse.mmr"),
        "title_linking": _module_file(
            derive_title_links,
            source_root=SOURCE_ROOT,
            name="omnifuse.derive_title_links",
        ),
    }
    chunks = [
        Chunk(
            id=str(document_id),
            title=str(document.get("title", "")),
            text=str(document.get("text", "")),
            meta={"source": "hotpotqa-e2e"},
        )
        for document_id, document in data["corpus"].items()
    ]
    before = _process_memory()
    started = time.perf_counter_ns()
    graph = build_inmemory([], [], chunks, vector_k=SEARCH_LIMIT, auto_link_titles=True)
    build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    post_build = _process_memory()
    rows: list[dict[str, Any]] = []
    for query_id in _sample_query_ids(data):
        query = str(data["queries"][query_id])
        started = time.perf_counter_ns()
        hits = graph.retrieve(query, limit=SEARCH_LIMIT)
        selected_texts = mmr(
            [(chunk.text, score) for chunk, score in hits],
            lam=graph.mmr_lambda,
            k=EVIDENCE_STEPS,
        )
        context = _truncate_context("\n\n---\n\n".join(selected_texts))
        retrieval_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        ranking = [str(chunk.id) for chunk, _score in hits[:EVIDENCE_STEPS]]
        evidence_ranking = _map_selected_texts(hits, selected_texts)
        row = {
            "query_id": query_id,
            "query": query,
            "answer": str(data["answers"][query_id]),
            "gold_document_ids": sorted(data["qrels"][query_id]),
            "retrieved_document_ids": ranking,
            "evidence_document_ids": evidence_ranking,
            "retrieval_ms": retrieval_ms,
            "metrics": _ranking_metrics(
                evidence_ranking, list(data["qrels"][query_id])
            ),
            "context_support": _context_support(
                context, str(data["answers"][query_id])
            ),
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        }
        if include_context:
            row["context"] = context
        rows.append(row)
    post_query = _process_memory()
    graph.close()
    return {
        "system": SYSTEM_LABELS["omnifuse"],
        "source_bindings": bindings,
        "build_ms": build_ms,
        "process_memory": {
            "before_build": before,
            "post_build": post_build,
            "post_query": post_query,
        },
        "graph_counts": {
            "documents": len(chunks),
            "nodes": len(graph.graph.nodes),
            "edges": len(graph.graph.triples),
            "generated_phrase_nodes": 0,
            "title_reference_edges": len(graph.graph.triples),
        },
        "questions": rows,
    }


async def _run_synaptic(
    repo: Path, data: Mapping[str, Any], *, include_context: bool = False
) -> dict[str, Any]:
    source_root = (repo / "src").resolve()
    sys.path.insert(0, str(source_root))
    import synaptic as package
    from synaptic.backends.memory import MemoryBackend
    from synaptic.extensions.classifier_rules import RuleBasedClassifier
    from synaptic.extensions.phrase_extractor import PhraseExtractor
    from synaptic.extensions.relation_detector import RuleBasedRelationDetector
    from synaptic.graph import SynapticGraph

    bindings = {
        "package": _module_file(package.Node, source_root=source_root, name="synaptic"),
        "memory_backend": _module_file(
            MemoryBackend, source_root=source_root, name="synaptic.MemoryBackend"
        ),
        "graph": _module_file(
            SynapticGraph, source_root=source_root, name="synaptic.SynapticGraph"
        ),
        "phrase_extractor": _module_file(
            PhraseExtractor, source_root=source_root, name="synaptic.PhraseExtractor"
        ),
    }
    before = _process_memory()
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(
        backend,
        classifier=RuleBasedClassifier(),
        relation_detector=RuleBasedRelationDetector(),
        phrase_extractor=PhraseExtractor(max_phrases_per_node=5),
    )
    reverse_ids: dict[str, str] = {}
    started = time.perf_counter_ns()
    for document_id, document in data["corpus"].items():
        node = await graph.add(
            title=str(document.get("title", "")),
            content=str(document.get("text", "")),
        )
        reverse_ids[str(node.id)] = str(document_id)
    build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    post_build = _process_memory()
    rows: list[dict[str, Any]] = []
    for query_id in _sample_query_ids(data):
        query = str(data["queries"][query_id])
        started = time.perf_counter_ns()
        evidence = await graph.build_evidence(
            query,
            limit=SEARCH_LIMIT,
            max_steps=EVIDENCE_STEPS,
            max_tokens=MAX_CONTEXT_TOKENS,
        )
        retrieval_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        evidence_ranking = _dedupe(
            [
                reverse_ids[str(step.node.id)]
                for step in evidence.steps
                if str(step.node.id) in reverse_ids
            ]
        )
        context = evidence.compressed_context
        row = {
            "query_id": query_id,
            "query": query,
            "answer": str(data["answers"][query_id]),
            "gold_document_ids": sorted(data["qrels"][query_id]),
            "retrieved_document_ids": evidence_ranking,
            "evidence_document_ids": evidence_ranking,
            "retrieval_ms": retrieval_ms,
            "metrics": _ranking_metrics(
                evidence_ranking, list(data["qrels"][query_id])
            ),
            "context_support": _context_support(
                context, str(data["answers"][query_id])
            ),
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "evidence_steps": len(evidence.steps),
            "evidence_tokens_approx": evidence.total_tokens_approx,
        }
        if include_context:
            row["context"] = context
        rows.append(row)
    post_query = _process_memory()
    node_count = len(backend._nodes)
    edge_count = len(backend._edges)
    await backend.close()
    return {
        "system": SYSTEM_LABELS["synaptic"],
        "source_bindings": bindings,
        "build_ms": build_ms,
        "process_memory": {
            "before_build": before,
            "post_build": post_build,
            "post_query": post_query,
        },
        "graph_counts": {
            "documents": len(reverse_ids),
            "nodes": node_count,
            "edges": edge_count,
            "generated_phrase_nodes": node_count - len(reverse_ids),
        },
        "questions": rows,
    }


def _worker_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "isolated": bool(sys.flags.isolated),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "no_user_site": bool(sys.flags.no_user_site),
        "safe_path": bool(getattr(sys.flags, "safe_path", False)),
        "utf8_mode": bool(sys.flags.utf8_mode),
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "pythonhome": os.environ.get("PYTHONHOME"),
        "pythonusersite": os.environ.get("PYTHONUSERBASE"),
        "python_no_user_site_env": os.environ.get("PYTHONNOUSERSITE"),
    }


def _isolated_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_worker(args: argparse.Namespace) -> int:
    if args.result_file is None or args.worker_run_id is None:
        raise ValueError("worker mode requires --result-file and --worker-run-id")
    ensure_output_absent(args.result_file.resolve())
    repo = args.synaptic_repo.resolve()
    data_path = _resolve_data_path(args)
    data = _load_data(data_path)
    state = _data_state(data_path, data)
    result = (
        _run_omnifuse(data)
        if args.worker == "omnifuse"
        else asyncio.run(_run_synaptic(repo, data))
    )
    assert_unchanged(
        "E2E QA retrieval worker input",
        state,
        _data_state(data_path, _load_data(data_path)),
    )
    write_json_once(
        args.result_file.resolve(),
        {
            "schema": WORKER_SCHEMA,
            "schema_version": WORKER_SCHEMA_VERSION,
            "status": "ok",
            "system": args.worker,
            "configuration": {
                "sample_seed": SAMPLE_SEED,
                "max_questions": MAX_QUESTIONS,
                "search_limit": SEARCH_LIMIT,
                "evidence_steps": EVIDENCE_STEPS,
                "max_context_tokens": MAX_CONTEXT_TOKENS,
            },
            "data": state,
            "worker_identity": capture_worker_identity(args.worker_run_id),
            "environment": _worker_environment(),
            "result": result,
        },
    )
    return 0


def _resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data is not None:
        return args.data.resolve()
    return (
        args.synaptic_repo.resolve()
        / "tests"
        / "benchmark"
        / "data"
        / "hotpotqa_24.json"
    )


def _worker_command(
    args: argparse.Namespace,
    *,
    system: str,
    result_file: Path,
    run_id: str,
) -> list[str]:
    command = [
        str(args.python.resolve()),
        "-I",
        "-X",
        "utf8",
        "-B",
        str(SCRIPT_PATH),
        "--synaptic-repo",
        str(args.synaptic_repo.resolve()),
        "--data",
        str(_resolve_data_path(args)),
        "--worker",
        system,
        "--result-file",
        str(result_file),
        "--worker-run-id",
        run_id,
    ]
    return command


def _expected_configuration() -> dict[str, int]:
    return {
        "sample_seed": SAMPLE_SEED,
        "max_questions": MAX_QUESTIONS,
        "search_limit": SEARCH_LIMIT,
        "evidence_steps": EVIDENCE_STEPS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }


def _validate_worker(
    payload: Mapping[str, Any],
    *,
    system: str,
    expected_data: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("schema_version") != WORKER_SCHEMA_VERSION
        or payload.get("status") != "ok"
        or payload.get("system") != system
    ):
        raise ProvenanceError(f"invalid {system} E2E retrieval worker contract")
    assert_unchanged(f"{system} E2E data", expected_data, payload.get("data"))
    assert_unchanged(
        f"{system} E2E configuration",
        _expected_configuration(),
        payload.get("configuration"),
    )
    identity = validate_worker_identity(
        payload.get("worker_identity"),
        expected_run_id=run_id,
        label=f"{system} E2E retrieval worker",
    )
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ProvenanceError(f"{system} worker omitted environment")
    for flag in ("isolated", "ignore_environment", "no_user_site", "safe_path"):
        if not environment.get(flag):
            raise ProvenanceError(f"{system} worker did not enable {flag}")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ProvenanceError(f"{system} worker omitted result")
    questions = result.get("questions")
    if not isinstance(questions, list) or len(questions) != MAX_QUESTIONS:
        raise ProvenanceError(f"{system} worker question count mismatch")
    if [row.get("query_id") for row in questions] != expected_data["sample_query_ids"]:
        raise ProvenanceError(f"{system} worker query order mismatch")
    return {
        "worker_identity": identity,
        "environment": dict(environment),
        "result": dict(result),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires samples")
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("summary requires samples")
    return {
        "samples": len(values),
        "minimum": min(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "average": statistics.fmean(values),
        "maximum": max(values),
    }


def _trial_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    questions = result["questions"]
    quality_fields = (
        "precision",
        "recall",
        "f1",
        "reciprocal_rank",
        "ndcg",
    )
    quality = {
        field: statistics.fmean(float(row["metrics"][field]) for row in questions)
        for field in quality_fields
    }
    quality["hit_rate"] = statistics.fmean(
        float(bool(row["metrics"]["hit"])) for row in questions
    )
    quality["all_gold_rate"] = statistics.fmean(
        float(bool(row["metrics"]["all_gold"])) for row in questions
    )
    quality["answer_exact_rate"] = statistics.fmean(
        float(bool(row["context_support"]["answer_exact"])) for row in questions
    )
    quality["answer_token_recall"] = statistics.fmean(
        float(row["context_support"]["answer_token_recall"]) for row in questions
    )
    memory = result["process_memory"]
    before = memory["before_build"]["current_rss_mb"]
    post_query = memory["post_query"]["current_rss_mb"]
    rss_delta = (
        None
        if before is None or post_query is None
        else max(0.0, float(post_query) - float(before))
    )
    return {
        "quality": quality,
        "build_ms": float(result["build_ms"]),
        "retrieval_ms": _summary([float(row["retrieval_ms"]) for row in questions]),
        "rss_delta_mb": rss_delta,
        "post_query_rss_mb": post_query,
        "ranking_sha256": canonical_json_sha256(
            [[row["query_id"], row["evidence_document_ids"]] for row in questions]
        ),
        "context_sha256": canonical_json_sha256(
            [[row["query_id"], row["context_sha256"]] for row in questions]
        ),
        "graph_counts": result["graph_counts"],
    }


def _aggregate(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [_trial_summary(trial["result"]) for trial in trials]
    quality_names = tuple(rows[0]["quality"])
    metrics = {
        name: _summary([float(row["quality"][name]) for row in rows])
        for name in quality_names
    }
    metrics.update(
        {
            "build_ms": _summary([float(row["build_ms"]) for row in rows]),
            "mean_retrieval_ms": _summary(
                [float(row["retrieval_ms"]["average"]) for row in rows]
            ),
            "p95_retrieval_ms": _summary(
                [float(row["retrieval_ms"]["p95"]) for row in rows]
            ),
        }
    )
    rss = [
        float(row["rss_delta_mb"]) for row in rows if row["rss_delta_mb"] is not None
    ]
    metrics["rss_delta_mb"] = None if not rss else _summary(rss)
    ranking_hashes = [row["ranking_sha256"] for row in rows]
    context_hashes = [row["context_sha256"] for row in rows]
    return {
        "trials": len(rows),
        "metrics": metrics,
        "rankings_deterministic": len(set(ranking_hashes)) == 1,
        "contexts_deterministic": len(set(context_hashes)) == 1,
        "ranking_hashes": ranking_hashes,
        "context_hashes": context_hashes,
        "graph_counts": [row["graph_counts"] for row in rows],
    }


def _per_question_head_to_head(
    trials: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    question_rows = {
        system: {
            row["query_id"]: row for row in system_trials[0]["result"]["questions"]
        }
        for system, system_trials in trials.items()
    }
    omni = question_rows["omnifuse"]
    synaptic = question_rows["synaptic"]
    if list(omni) != list(synaptic):
        raise ValueError("E2E QA per-question system cohorts differ")

    metric_sources = {
        "precision": "metrics",
        "recall": "metrics",
        "f1": "metrics",
        "hit": "metrics",
        "all_gold": "metrics",
        "reciprocal_rank": "metrics",
        "ndcg": "metrics",
        "answer_exact": "context_support",
        "answer_token_recall": "context_support",
    }
    quality_counts = {
        field: {"omnifuse": 0, "synaptic": 0, "tie": 0} for field in metric_sources
    }
    retrieval_counts = {"omnifuse": 0, "synaptic": 0, "tie": 0}
    quality_losses: list[dict[str, Any]] = []
    retrieval_losses: list[str] = []

    for query_id, omni_row in omni.items():
        synaptic_row = synaptic[query_id]
        lost_quality: list[str] = []
        for field, source in metric_sources.items():
            omni_value = float(omni_row[source][field])
            synaptic_value = float(synaptic_row[source][field])
            if math.isclose(omni_value, synaptic_value, rel_tol=1e-12, abs_tol=1e-12):
                winner = "tie"
            elif omni_value > synaptic_value:
                winner = "omnifuse"
            else:
                winner = "synaptic"
                lost_quality.append(field)
            quality_counts[field][winner] += 1
        if lost_quality:
            quality_losses.append({"query_id": query_id, "metrics": lost_quality})

        omni_ms = float(omni_row["retrieval_ms"])
        synaptic_ms = float(synaptic_row["retrieval_ms"])
        if math.isclose(omni_ms, synaptic_ms, rel_tol=1e-12, abs_tol=1e-12):
            winner = "tie"
        elif omni_ms < synaptic_ms:
            winner = "omnifuse"
        else:
            winner = "synaptic"
            retrieval_losses.append(query_id)
        retrieval_counts[winner] += 1

    return {
        "questions": len(omni),
        "quality": {
            "metrics": quality_counts,
            "questions_with_any_omnifuse_loss": len(quality_losses),
            "losses": quality_losses,
        },
        "retrieval_ms": {
            "counts": retrieval_counts,
            "questions_with_omnifuse_loss": len(retrieval_losses),
            "loss_query_ids": retrieval_losses,
        },
    }


def _head_to_head(
    aggregates: Mapping[str, Any],
    trials: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    quality = (
        "recall",
        "hit_rate",
        "all_gold_rate",
        "reciprocal_rank",
        "ndcg",
        "answer_exact_rate",
        "answer_token_recall",
    )
    efficiency = ("build_ms", "mean_retrieval_ms", "p95_retrieval_ms", "rss_delta_mb")
    rows: list[dict[str, Any]] = []
    for metric in quality + efficiency:
        omni_record = aggregates["omnifuse"]["metrics"].get(metric)
        syn_record = aggregates["synaptic"]["metrics"].get(metric)
        if omni_record is None or syn_record is None:
            continue
        omni = float(omni_record["p50"])
        syn = float(syn_record["p50"])
        higher = metric in quality
        if math.isclose(omni, syn, rel_tol=1e-12, abs_tol=1e-12):
            winner = "tie"
        elif (higher and omni > syn) or (not higher and omni < syn):
            winner = "omnifuse"
        else:
            winner = "synaptic"
        rows.append(
            {
                "metric": metric,
                "direction": "higher" if higher else "lower",
                "omnifuse": omni,
                "synaptic": syn,
                "winner": winner,
            }
        )
    result = {
        "metrics": rows,
        "verdict": {
            "omnifuse": sum(row["winner"] == "omnifuse" for row in rows),
            "synaptic": sum(row["winner"] == "synaptic" for row in rows),
            "ties": sum(row["winner"] == "tie" for row in rows),
            "common_metrics": len(rows),
        },
    }
    if trials is not None:
        result["per_question"] = _per_question_head_to_head(trials)
    return result


def _source_state(repo: Path) -> dict[str, Any]:
    return {
        "benchmark": file_fingerprint(
            SCRIPT_PATH, display_path="eval/e2e_qa_retrieval_bench.py"
        ),
        "process_memory_support": file_fingerprint(
            EVAL_DIR / "perf_bench.py", display_path="eval/perf_bench.py"
        ),
        "upstream_e2e_test": file_fingerprint(
            repo / "tests" / "benchmark" / "test_e2e_qa.py",
            display_path="tests/benchmark/test_e2e_qa.py",
        ),
        "upstream_graph": file_fingerprint(
            repo / "src" / "synaptic" / "graph.py",
            display_path="src/synaptic/graph.py",
        ),
        "upstream_evidence": file_fingerprint(
            repo / "src" / "synaptic" / "evidence.py",
            display_path="src/synaptic/evidence.py",
        ),
        "omnifuse_facade": file_fingerprint(
            ROOT / "src" / "omnifuse" / "facade.py",
            display_path="src/omnifuse/facade.py",
        ),
        "omnifuse_fusion": file_fingerprint(
            ROOT / "src" / "omnifuse" / "fusion.py",
            display_path="src/omnifuse/fusion.py",
        ),
        "omnifuse_compact_postings": file_fingerprint(
            ROOT / "src" / "omnifuse" / "_compact_postings.py",
            display_path="src/omnifuse/_compact_postings.py",
        ),
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.out is None:
        raise ValueError("controller mode requires --out")
    if args.trials < 2 or args.trials % 2:
        raise ValueError("--trials must be an even value of at least 2")
    output = args.out.resolve()
    ensure_output_absent(output)
    repo = args.synaptic_repo.resolve()
    if not (repo / "src" / "synaptic").is_dir():
        raise FileNotFoundError(f"Synaptic source package not found below {repo}")
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable not found: {python}")
    data_path = _resolve_data_path(args)
    data = _load_data(data_path)
    state = _data_state(data_path, data)
    if state["sampled_queries"] != MAX_QUESTIONS:
        raise ValueError(f"official E2E cohort requires {MAX_QUESTIONS} questions")
    return {
        "output": output,
        "repo": repo,
        "python": file_fingerprint(python),
        "data_path": data_path,
        "data": state,
        "repositories": {
            "omnifuse": repository_fingerprint(ROOT),
            "synaptic_memory": repository_fingerprint(repo),
        },
        "sources": _source_state(repo),
    }


def _controller(args: argparse.Namespace) -> int:
    state = _preflight(args)
    output: Path = state["output"]
    workers = (
        args.workers_dir.resolve()
        if args.workers_dir is not None
        else output.parent / f".{output.stem}.workers"
    )
    if workers.exists():
        raise ProvenanceError(f"refusing to reuse worker directory {workers}")
    workers.mkdir(parents=True)
    schedule = [
        ("omnifuse", "synaptic") if trial % 2 else ("synaptic", "omnifuse")
        for trial in range(1, args.trials + 1)
    ]
    rows: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    processes: list[dict[str, Any]] = []
    for trial_number, order in enumerate(schedule, 1):
        for position, system in enumerate(order, 1):
            result_file = workers / f"trial-{trial_number:02d}-{position}-{system}.json"
            run_id = new_worker_run_id()
            completed, launcher_pid = run_with_launcher_pid(
                _worker_command(
                    args,
                    system=system,
                    result_file=result_file,
                    run_id=run_id,
                ),
                cwd=ROOT,
                env=_isolated_environment(),
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"{system} E2E retrieval worker failed with "
                    f"{completed.returncode}: {detail}"
                )
            payload, artifact = read_json_artifact(result_file)
            validated = _validate_worker(
                payload,
                system=system,
                expected_data=state["data"],
                run_id=run_id,
            )
            rows[system].append(validated)
            identity = validated["worker_identity"]
            processes.append(
                {
                    "trial_number": trial_number,
                    "order_position": position,
                    "system": system,
                    "worker_run_id": identity["worker_run_id"],
                    "launcher_pid": launcher_pid,
                    "worker_pid": identity["worker_pid"],
                    "same_process_id": launcher_pid == identity["worker_pid"],
                    "returncode": completed.returncode,
                    "result_artifact": artifact,
                    "stdout_bytes": len((completed.stdout or "").encode()),
                    "stderr_bytes": len((completed.stderr or "").encode()),
                }
            )
    assert_unchanged(
        "E2E QA data postflight",
        state["data"],
        _data_state(state["data_path"], _load_data(state["data_path"])),
    )
    assert_unchanged(
        "repository fingerprints postflight",
        state["repositories"],
        {
            "omnifuse": repository_fingerprint(ROOT),
            "synaptic_memory": repository_fingerprint(state["repo"]),
        },
    )
    assert_unchanged(
        "benchmark sources postflight", state["sources"], _source_state(state["repo"])
    )
    assert_unchanged(
        "Python executable postflight",
        state["python"],
        file_fingerprint(args.python.resolve()),
    )
    aggregates = {system: _aggregate(rows[system]) for system in SYSTEMS}
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_level": PROVENANCE_LEVEL,
        "repositories": state["repositories"],
        "sources": state["sources"],
        "python": state["python"],
        "data": state["data"],
        "contract": {
            "upstream_test": (
                "tests/benchmark/test_e2e_qa.py::TestE2EHotPotQA.test_hotpotqa_e2e"
            ),
            "sample_seed": SAMPLE_SEED,
            "questions": MAX_QUESTIONS,
            "search_limit": SEARCH_LIMIT,
            "evidence_steps": EVIDENCE_STEPS,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "synaptic_path": (
                "RuleBasedClassifier + RuleBasedRelationDetector + "
                "PhraseExtractor(5) + build_evidence"
            ),
            "omnifuse_path": (
                "native title-link graph retrieve + product MMR evidence selection"
            ),
            "answer_generation_excluded": (
                "The exact upstream test fails when localhost Ollama is unavailable. "
                "No answer or correctness value is fabricated; answer support only "
                "checks whether the gold answer is present in retrieved context."
            ),
            "build_caveat": (
                "OmniFuse lexical materialization is charged to the first retrieval; "
                "there is no warm-up."
            ),
            "trials": args.trials,
            "counterbalanced_ab_ba": True,
        },
        "worker_processes": processes,
        "worker_process_summary": worker_process_summary(
            processes, expected_count=args.trials * len(SYSTEMS)
        ),
        "results": {
            system: {"trials": rows[system], "aggregate": aggregates[system]}
            for system in SYSTEMS
        },
        "head_to_head": _head_to_head(aggregates, rows),
        "postflight": {
            "data_unchanged": True,
            "repositories_unchanged": True,
            "sources_unchanged": True,
            "python_unchanged": True,
        },
    }
    write_json_once(output, report)
    print(
        "E2E retrieval benchmark: "
        f"OmniFuse recall={aggregates['omnifuse']['metrics']['recall']['p50']:.6f}; "
        f"Synaptic={aggregates['synaptic']['metrics']['recall']['p50']:.6f}"
    )
    print(f"wrote {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.worker is not None:
            return _run_worker(args)
        return _controller(args)
    except (OSError, ValueError, RuntimeError, ProvenanceError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
