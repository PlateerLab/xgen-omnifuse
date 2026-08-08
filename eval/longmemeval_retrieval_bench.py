"""LongMemEval-S retrieval comparison against Synaptic's official test path.

This benchmark deliberately stops at retrieval.  Synaptic's upstream
``test_longmemeval.py`` sends the retrieved context to an external LLM, so
answer correctness is not a property of either memory implementation alone.
The comparable upstream metric is mean gold-session recall from the plain
``graph.search(question, limit=20)`` result.

Both systems receive the same balanced sample and the same turn-pair records.
Each question starts from a fresh in-memory index, matching the upstream test.
Workers are isolated processes and reports are immutable, fingerprinted JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import platform
import random
import site
import statistics
import sys
import time
from collections import Counter
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

SCHEMA = "omnifuse.eval.longmemeval_retrieval"
SCHEMA_VERSION = 1
WORKER_SCHEMA = "omnifuse.eval.longmemeval_retrieval_worker"
WORKER_SCHEMA_VERSION = 1
PROVENANCE_LEVEL = "isolated-balanced-upstream-path-preflight-postflight-write-once-v1"
SYSTEMS = ("omnifuse", "synaptic")
SYSTEM_LABELS = {"omnifuse": "OmniFuse", "synaptic": "synaptic-memory"}
DEFAULT_MAX_QUESTIONS = 50
DEFAULT_LIMIT = 20
SAMPLE_SEED = 42
REQUIRED_INSTANCE_FIELDS = frozenset(
    {
        "question_id",
        "question_type",
        "question",
        "haystack_sessions",
        "haystack_session_ids",
        "haystack_dates",
        "answer_session_ids",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptic-repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers-dir", type=Path)
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--worker", choices=SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    parser.add_argument("--sample-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-total-questions", type=int, help=argparse.SUPPRESS)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_data(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list) or not value:
        raise ValueError("LongMemEval data must be a non-empty JSON array")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"LongMemEval row {index} is not an object")
        missing = REQUIRED_INSTANCE_FIELDS - row.keys()
        if missing:
            raise ValueError(f"LongMemEval row {index} is missing {sorted(missing)}")
        sessions = row["haystack_sessions"]
        session_ids = row["haystack_session_ids"]
        session_dates = row["haystack_dates"]
        if not (
            isinstance(sessions, list)
            and isinstance(session_ids, list)
            and isinstance(session_dates, list)
            and len(sessions) == len(session_ids) == len(session_dates)
        ):
            raise ValueError(f"LongMemEval row {index} has misaligned sessions")
        rows.append(row)
    return rows


def _balanced_sample(
    data: Sequence[dict[str, Any]],
    *,
    max_questions: int,
    seed: int = SAMPLE_SEED,
) -> list[dict[str, Any]]:
    """Reproduce Synaptic's insertion-order grouping and seeded shuffle."""

    if max_questions < 1:
        raise ValueError("--max-questions must be at least 1")
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in data:
        by_type.setdefault(str(row["question_type"]), []).append(row)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    per_type = max(1, max_questions // len(by_type))
    for items in by_type.values():
        candidates = list(items)
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_type])
    return sampled[:max_questions]


def _turn_pair_records(instance: Mapping[str, Any]) -> list[dict[str, str | int]]:
    """Reproduce upstream ``_index_sessions`` turn-pair construction."""

    records: list[dict[str, str | int]] = []
    for session, raw_sid, raw_date in zip(
        instance["haystack_sessions"],
        instance["haystack_session_ids"],
        instance["haystack_dates"],
    ):
        sid = str(raw_sid)
        date = str(raw_date)
        index = 0
        pair_index = 0
        while index < len(session):
            turn = session[index]
            role = turn.get("role", "user")
            content = str(turn.get("content", ""))
            if role == "user":
                user_text = content
                assistant_text = ""
                if (
                    index + 1 < len(session)
                    and session[index + 1].get("role") == "assistant"
                ):
                    assistant_text = str(session[index + 1].get("content", ""))
                    index += 2
                else:
                    index += 1
                pair_text = f"[User] {user_text}"
                if assistant_text:
                    pair_text += f"\n[Assistant] {assistant_text}"
                records.append(
                    {
                        "id": f"{sid}:{pair_index}",
                        "session_id": sid,
                        "date": date,
                        "turn": pair_index,
                        "title": user_text[:80],
                        "content": pair_text[:2000],
                        "source": f"longmemeval:{sid}:{pair_index}",
                    }
                )
                pair_index += 1
                continue
            if content.strip():
                records.append(
                    {
                        "id": f"{sid}:{pair_index}",
                        "session_id": sid,
                        "date": date,
                        "turn": pair_index,
                        "title": content[:80],
                        "content": f"[Assistant] {content[:2000]}",
                        "source": f"longmemeval:{sid}:{pair_index}",
                    }
                )
                pair_index += 1
            index += 1
    return records


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _retrieval_metrics(
    retrieved_session_ids: Sequence[str],
    answer_session_ids: Sequence[str],
) -> dict[str, float | int | bool | None]:
    retrieved = _dedupe([str(value) for value in retrieved_session_ids])
    gold = set(str(value) for value in answer_session_ids)
    if not gold:
        return {
            "eligible": False,
            "gold_sessions": 0,
            "retrieved_unique_sessions": len(retrieved),
            "hits": 0,
            "session_recall": None,
            "session_hit": None,
            "reciprocal_rank": None,
            "ndcg": None,
        }
    hits = len(gold.intersection(retrieved))
    first_rank = next(
        (rank for rank, session_id in enumerate(retrieved, 1) if session_id in gold),
        None,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, session_id in enumerate(retrieved, 1)
        if session_id in gold
    )
    ideal_hits = min(len(gold), len(retrieved))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "eligible": True,
        "gold_sessions": len(gold),
        "retrieved_unique_sessions": len(retrieved),
        "hits": hits,
        "session_recall": hits / len(gold),
        "session_hit": hits > 0,
        "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
        "ndcg": 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg,
    }


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


def _instance_header(
    instance: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "question_id": str(instance["question_id"]),
        "question_type": str(instance["question_type"]),
        "question": str(instance["question"]),
        "answer_session_ids": [str(value) for value in instance["answer_session_ids"]],
        "sessions": len(instance["haystack_sessions"]),
        "turn_pair_records": len(records),
        "turn_pair_sha256": canonical_json_sha256(records),
    }


def _run_omnifuse(sampled: Sequence[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    sys.path.insert(0, str(SOURCE_ROOT))
    import omnifuse as package
    from omnifuse import Chunk, build_inmemory
    from omnifuse.oneshot import OmniFuse

    source_bindings = {
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
    }
    baseline = _process_memory()
    rows: list[dict[str, Any]] = []
    for instance in sampled:
        records = _turn_pair_records(instance)
        before = _process_memory()
        started = time.perf_counter_ns()
        chunks = [
            Chunk(
                id=str(record["id"]),
                title=str(record["title"]),
                text=str(record["content"]),
                meta={
                    "session_id": str(record["session_id"]),
                    "date": str(record["date"]),
                    "turn": int(record["turn"]),
                    "source": str(record["source"]),
                },
            )
            for record in records
        ]
        graph = build_inmemory([], [], chunks, vector_k=limit)
        build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        post_build = _process_memory()
        started = time.perf_counter_ns()
        hits = graph.retrieve(str(instance["question"]), limit=limit)
        retrieval_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        retrieved = [str(chunk.meta["session_id"]) for chunk, _score in hits]
        post_query = _process_memory()
        ranking = _dedupe(retrieved)
        rows.append(
            {
                **_instance_header(instance, records),
                "build_ms": build_ms,
                "retrieval_ms": retrieval_ms,
                "retrieved_session_ids": ranking,
                "retrieved_chunks": len(hits),
                "ranking_sha256": canonical_json_sha256(ranking),
                "metrics": _retrieval_metrics(ranking, instance["answer_session_ids"]),
                "process_memory": {
                    "before_build": before,
                    "post_build": post_build,
                    "post_query": post_query,
                },
            }
        )
        graph.close()
        del graph, chunks, hits
        gc.collect()
    return {
        "system": SYSTEM_LABELS["omnifuse"],
        "source_bindings": source_bindings,
        "baseline_process_memory": baseline,
        "questions": rows,
    }


async def _run_synaptic(
    repo: Path,
    sampled: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    source_root = (repo / "src").resolve()
    sys.path.insert(0, str(source_root))
    import synaptic as package
    from synaptic.backends.memory import MemoryBackend
    from synaptic.graph import SynapticGraph
    from synaptic.models import NodeKind

    source_bindings = {
        "package": _module_file(package.Node, source_root=source_root, name="synaptic"),
        "memory_backend": _module_file(
            MemoryBackend, source_root=source_root, name="synaptic.MemoryBackend"
        ),
        "graph": _module_file(
            SynapticGraph, source_root=source_root, name="synaptic.SynapticGraph"
        ),
    }
    baseline = _process_memory()
    rows: list[dict[str, Any]] = []
    for instance in sampled:
        records = _turn_pair_records(instance)
        before = _process_memory()
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)
        started = time.perf_counter_ns()
        for record in records:
            await graph.add(
                title=str(record["title"]),
                content=str(record["content"]),
                kind=NodeKind.CONCEPT,
                tags=[
                    f"session:{record['session_id']}",
                    f"date:{record['date']}",
                    f"turn:{record['turn']}",
                ],
                source=str(record["source"]),
            )
        build_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        post_build = _process_memory()
        started = time.perf_counter_ns()
        result = await graph.search(str(instance["question"]), limit=limit)
        retrieval_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        retrieved: list[str] = []
        for activated in result.nodes:
            for tag in activated.node.tags:
                if tag.startswith("session:"):
                    retrieved.append(tag.removeprefix("session:"))
                    break
        post_query = _process_memory()
        ranking = _dedupe(retrieved)
        rows.append(
            {
                **_instance_header(instance, records),
                "build_ms": build_ms,
                "retrieval_ms": retrieval_ms,
                "retrieved_session_ids": ranking,
                "retrieved_chunks": len(result.nodes),
                "ranking_sha256": canonical_json_sha256(ranking),
                "metrics": _retrieval_metrics(ranking, instance["answer_session_ids"]),
                "process_memory": {
                    "before_build": before,
                    "post_build": post_build,
                    "post_query": post_query,
                },
            }
        )
        await backend.close()
        del graph, backend, result
        gc.collect()
    return {
        "system": SYSTEM_LABELS["synaptic"],
        "source_bindings": source_bindings,
        "baseline_process_memory": baseline,
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


def _data_state(
    path: Path,
    sampled: Sequence[Mapping[str, Any]],
    *,
    total_records: int,
) -> dict[str, Any]:
    counts = Counter(str(row["question_type"]) for row in sampled)
    return {
        **file_fingerprint(path, display_path=str(path)),
        "total_questions": total_records,
        "sampled_questions": len(sampled),
        "sample_seed": SAMPLE_SEED,
        "sample_question_ids": [str(row["question_id"]) for row in sampled],
        "sample_question_ids_sha256": canonical_json_sha256(
            [str(row["question_id"]) for row in sampled]
        ),
        "sample_question_types": dict(sorted(counts.items())),
        "sample_gold_eligible": sum(bool(row["answer_session_ids"]) for row in sampled),
        "sample_sessions": sum(len(row["haystack_sessions"]) for row in sampled),
        "sample_turns": sum(
            len(session) for row in sampled for session in row["haystack_sessions"]
        ),
    }


def _file_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("path", "sha256", "bytes")}


def _source_state(repo: Path) -> dict[str, Any]:
    return {
        "benchmark": file_fingerprint(
            SCRIPT_PATH, display_path="eval/longmemeval_retrieval_bench.py"
        ),
        "process_memory_support": file_fingerprint(
            EVAL_DIR / "perf_bench.py", display_path="eval/perf_bench.py"
        ),
        "upstream_longmemeval_test": file_fingerprint(
            repo / "tests" / "benchmark" / "test_longmemeval.py",
            display_path="tests/benchmark/test_longmemeval.py",
        ),
        "upstream_graph": file_fingerprint(
            repo / "src" / "synaptic" / "graph.py",
            display_path="src/synaptic/graph.py",
        ),
        "upstream_memory_backend": file_fingerprint(
            repo / "src" / "synaptic" / "backends" / "memory.py",
            display_path="src/synaptic/backends/memory.py",
        ),
        "omnifuse_facade": file_fingerprint(
            ROOT / "src" / "omnifuse" / "facade.py",
            display_path="src/omnifuse/facade.py",
        ),
        "omnifuse_memory_backend": file_fingerprint(
            ROOT / "src" / "omnifuse" / "backends" / "memory.py",
            display_path="src/omnifuse/backends/memory.py",
        ),
        "omnifuse_compact_postings": file_fingerprint(
            ROOT / "src" / "omnifuse" / "_compact_postings.py",
            display_path="src/omnifuse/_compact_postings.py",
        ),
    }


def _run_worker(args: argparse.Namespace) -> int:
    if (
        args.result_file is None
        or args.worker_run_id is None
        or args.sample_file is None
        or args.expected_total_questions is None
    ):
        raise ValueError(
            "worker mode requires --result-file, --worker-run-id, "
            "--sample-file, and --expected-total-questions"
        )
    ensure_output_absent(args.result_file.resolve())
    data_path = args.data.resolve()
    sample_path = args.sample_file.resolve()
    sampled = _load_data(sample_path)
    if len(sampled) > args.max_questions:
        raise ValueError("worker sample exceeds --max-questions")
    state = _data_state(
        data_path,
        sampled,
        total_records=args.expected_total_questions,
    )
    sample_artifact = file_fingerprint(sample_path, display_path=str(sample_path))
    gc.collect()
    if args.worker == "omnifuse":
        result = _run_omnifuse(sampled, limit=args.limit)
    else:
        result = asyncio.run(
            _run_synaptic(args.synaptic_repo.resolve(), sampled, limit=args.limit)
        )
    post_sample = _load_data(sample_path)
    post_state = _data_state(
        data_path,
        post_sample,
        total_records=args.expected_total_questions,
    )
    assert_unchanged("LongMemEval worker input", state, post_state)
    assert_unchanged(
        "LongMemEval worker sample artifact",
        sample_artifact,
        file_fingerprint(sample_path, display_path=str(sample_path)),
    )
    write_json_once(
        args.result_file.resolve(),
        {
            "schema": WORKER_SCHEMA,
            "schema_version": WORKER_SCHEMA_VERSION,
            "status": "ok",
            "system": args.worker,
            "configuration": {
                "max_questions": args.max_questions,
                "actual_questions": len(sampled),
                "limit": args.limit,
                "sample_seed": SAMPLE_SEED,
            },
            "data": state,
            "sample_artifact": sample_artifact,
            "worker_identity": capture_worker_identity(args.worker_run_id),
            "environment": _worker_environment(),
            "result": result,
        },
    )
    return 0


def _worker_command(
    args: argparse.Namespace,
    *,
    system: str,
    result_file: Path,
    run_id: str,
    sample_file: Path,
    expected_total_questions: int,
) -> list[str]:
    return [
        str(args.python.resolve()),
        "-I",
        "-X",
        "utf8",
        "-B",
        str(SCRIPT_PATH),
        "--synaptic-repo",
        str(args.synaptic_repo.resolve()),
        "--data",
        str(args.data.resolve()),
        "--max-questions",
        str(args.max_questions),
        "--limit",
        str(args.limit),
        "--sample-file",
        str(sample_file),
        "--expected-total-questions",
        str(expected_total_questions),
        "--worker",
        system,
        "--result-file",
        str(result_file),
        "--worker-run-id",
        run_id,
    ]


def _validate_worker(
    payload: Mapping[str, Any],
    *,
    system: str,
    expected_data: Mapping[str, Any],
    expected_sample_artifact: Mapping[str, Any],
    run_id: str,
    max_questions: int,
    limit: int,
) -> dict[str, Any]:
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("schema_version") != WORKER_SCHEMA_VERSION
        or payload.get("status") != "ok"
        or payload.get("system") != system
    ):
        raise ProvenanceError(f"invalid {system} LongMemEval worker contract")
    assert_unchanged(f"{system} LongMemEval data", expected_data, payload.get("data"))
    assert_unchanged(
        f"{system} LongMemEval sample artifact",
        expected_sample_artifact,
        payload.get("sample_artifact"),
    )
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ProvenanceError(f"{system} worker omitted configuration")
    if (
        configuration.get("max_questions") != max_questions
        or configuration.get("limit") != limit
        or configuration.get("sample_seed") != SAMPLE_SEED
    ):
        raise ProvenanceError(f"{system} worker configuration mismatch")
    identity = validate_worker_identity(
        payload.get("worker_identity"),
        expected_run_id=run_id,
        label=f"{system} LongMemEval worker",
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
    expected_count = int(configuration["actual_questions"])
    if not isinstance(questions, list) or len(questions) != expected_count:
        raise ProvenanceError(f"{system} worker question count mismatch")
    expected_ids = expected_data["sample_question_ids"]
    if [row.get("question_id") for row in questions] != expected_ids:
        raise ProvenanceError(f"{system} worker question order mismatch")
    return {
        "worker_identity": identity,
        "environment": dict(environment),
        "sample_artifact": dict(expected_sample_artifact),
        "result": dict(result),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
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


def _rss_delta(row: Mapping[str, Any]) -> float | None:
    memory = row["process_memory"]
    before = memory["before_build"]["current_rss_mb"]
    after = memory["post_query"]["current_rss_mb"]
    if before is None or after is None:
        return None
    return max(0.0, float(after) - float(before))


def _aggregate_trial(result: Mapping[str, Any]) -> dict[str, Any]:
    questions = result["questions"]
    eligible = [row for row in questions if row["metrics"]["eligible"]]
    if not eligible:
        raise ValueError("LongMemEval sample contains no gold-session questions")
    rss = [value for row in questions if (value := _rss_delta(row)) is not None]
    quality_fields = ("session_recall", "reciprocal_rank", "ndcg")
    quality = {
        field: statistics.fmean(float(row["metrics"][field]) for row in eligible)
        for field in quality_fields
    }
    quality["session_hit_rate"] = statistics.fmean(
        float(bool(row["metrics"]["session_hit"])) for row in eligible
    )
    return {
        "questions": len(questions),
        "eligible_questions": len(eligible),
        "quality": quality,
        "build_ms": {
            **_summary([float(row["build_ms"]) for row in questions]),
            "total": sum(float(row["build_ms"]) for row in questions),
        },
        "retrieval_ms": _summary([float(row["retrieval_ms"]) for row in questions]),
        "rss_delta_mb": None if not rss else _summary(rss),
        "rankings_sha256": canonical_json_sha256(
            [[row["question_id"], row["retrieved_session_ids"]] for row in questions]
        ),
        "turn_pairs": sum(int(row["turn_pair_records"]) for row in questions),
    }


def _aggregate_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [_aggregate_trial(trial["result"]) for trial in trials]
    ranking_hashes = [row["rankings_sha256"] for row in rows]
    metrics = {
        "mean_session_recall": _summary(
            [float(row["quality"]["session_recall"]) for row in rows]
        ),
        "session_hit_rate": _summary(
            [float(row["quality"]["session_hit_rate"]) for row in rows]
        ),
        "mrr": _summary([float(row["quality"]["reciprocal_rank"]) for row in rows]),
        "ndcg": _summary([float(row["quality"]["ndcg"]) for row in rows]),
        "total_build_ms": _summary([float(row["build_ms"]["total"]) for row in rows]),
        "average_retrieval_ms": _summary(
            [float(row["retrieval_ms"]["average"]) for row in rows]
        ),
        "p95_retrieval_ms": _summary(
            [float(row["retrieval_ms"]["p95"]) for row in rows]
        ),
    }
    rss_values = [
        float(row["rss_delta_mb"]["maximum"])
        for row in rows
        if row["rss_delta_mb"] is not None
    ]
    metrics["maximum_rss_delta_mb"] = None if not rss_values else _summary(rss_values)
    return {
        "trials": len(rows),
        "questions_per_trial": rows[0]["questions"],
        "eligible_questions_per_trial": rows[0]["eligible_questions"],
        "turn_pairs_per_trial": rows[0]["turn_pairs"],
        "deterministic_rankings": len(set(ranking_hashes)) == 1,
        "ranking_hashes": ranking_hashes,
        "metrics": metrics,
    }


def _head_to_head(aggregates: Mapping[str, Any]) -> dict[str, Any]:
    omni = aggregates["omnifuse"]["metrics"]
    syn = aggregates["synaptic"]["metrics"]
    rows: list[dict[str, Any]] = []
    quality = (
        "mean_session_recall",
        "session_hit_rate",
        "mrr",
        "ndcg",
    )
    efficiency = (
        "total_build_ms",
        "average_retrieval_ms",
        "p95_retrieval_ms",
        "maximum_rss_delta_mb",
    )
    for metric in quality + efficiency:
        omni_record = omni.get(metric)
        syn_record = syn.get(metric)
        if omni_record is None or syn_record is None:
            rows.append(
                {
                    "metric": metric,
                    "direction": "higher" if metric in quality else "lower",
                    "omnifuse": None,
                    "synaptic": None,
                    "winner": "unavailable",
                }
            )
            continue
        omni_value = float(omni_record["p50"])
        syn_value = float(syn_record["p50"])
        if math.isclose(omni_value, syn_value, rel_tol=1e-12, abs_tol=1e-12):
            winner = "tie"
        elif (metric in quality and omni_value > syn_value) or (
            metric in efficiency and omni_value < syn_value
        ):
            winner = "omnifuse"
        else:
            winner = "synaptic"
        rows.append(
            {
                "metric": metric,
                "direction": "higher" if metric in quality else "lower",
                "omnifuse": omni_value,
                "synaptic": syn_value,
                "winner": winner,
            }
        )
    comparable = [row for row in rows if row["winner"] != "unavailable"]
    quality_rows = [row for row in comparable if row["metric"] in quality]
    efficiency_rows = [row for row in comparable if row["metric"] in efficiency]
    return {
        "metrics": rows,
        "verdict": {
            "omnifuse_wins_or_ties_all_quality": all(
                row["winner"] in {"omnifuse", "tie"} for row in quality_rows
            ),
            "omnifuse_wins_or_ties_all_efficiency": all(
                row["winner"] in {"omnifuse", "tie"} for row in efficiency_rows
            ),
            "omnifuse_strict_wins": sum(
                row["winner"] == "omnifuse" for row in comparable
            ),
            "synaptic_strict_wins": sum(
                row["winner"] == "synaptic" for row in comparable
            ),
            "ties": sum(row["winner"] == "tie" for row in comparable),
            "comparable_metrics": len(comparable),
        },
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.out is None:
        raise ValueError("controller mode requires --out")
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    output = args.out.resolve()
    ensure_output_absent(output)
    repo = args.synaptic_repo.resolve()
    if not (repo / "src" / "synaptic").is_dir():
        raise FileNotFoundError(f"Synaptic source package not found below {repo}")
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable not found: {python}")
    data_path = args.data.resolve()
    data = _load_data(data_path)
    sampled = _balanced_sample(data, max_questions=args.max_questions)
    data_state = _data_state(data_path, sampled, total_records=len(data))
    return {
        "output": output,
        "repo": repo,
        "python": file_fingerprint(python),
        "data": data_state,
        "repositories": {
            "omnifuse": repository_fingerprint(ROOT),
            "synaptic_memory": repository_fingerprint(repo),
        },
        "sources": _source_state(repo),
        "sampled": sampled,
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
    sample_file = workers / "balanced_sample.json"
    write_json_once(sample_file, state["sampled"])
    sample_artifact = file_fingerprint(sample_file, display_path=str(sample_file))
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
                    sample_file=sample_file,
                    expected_total_questions=state["data"]["total_questions"],
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
                    f"{system} LongMemEval worker failed with "
                    f"{completed.returncode}: {detail}"
                )
            payload, artifact = read_json_artifact(result_file)
            validated = _validate_worker(
                payload,
                system=system,
                expected_data=state["data"],
                expected_sample_artifact=sample_artifact,
                run_id=run_id,
                max_questions=args.max_questions,
                limit=args.limit,
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
                    "stdout_bytes": len((completed.stdout or "").encode("utf-8")),
                    "stderr_bytes": len((completed.stderr or "").encode("utf-8")),
                }
            )
    current_data_artifact = file_fingerprint(
        args.data.resolve(), display_path=str(args.data.resolve())
    )
    assert_unchanged(
        "LongMemEval data artifact postflight",
        _file_identity(state["data"]),
        _file_identity(current_data_artifact),
    )
    assert_unchanged(
        "LongMemEval sample artifact postflight",
        sample_artifact,
        file_fingerprint(sample_file, display_path=str(sample_file)),
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
    aggregates = {system: _aggregate_trials(rows[system]) for system in SYSTEMS}
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
        "sample_artifact": sample_artifact,
        "contract": {
            "upstream_test": (
                "tests/benchmark/test_longmemeval.py::"
                "TestLongMemEval.test_longmemeval_s"
            ),
            "upstream_indexing": (
                "tests/benchmark/test_longmemeval.py::_index_sessions"
            ),
            "upstream_retrieval_metric": "mean gold-session recall",
            "retrieval_call": "graph.search(question, limit=20)",
            "turn_pair_granularity": True,
            "fresh_index_per_question": True,
            "balanced_sample": True,
            "sample_seed": SAMPLE_SEED,
            "requested_max_questions": args.max_questions,
            "actual_questions": state["data"]["sampled_questions"],
            "limit": args.limit,
            "trials": args.trials,
            "counterbalanced_ab_ba_when_multiple_trials": True,
            "answer_generation_excluded": (
                "The upstream answer stage depends on an external LLM. This report "
                "compares memory retrieval only and makes no QA-accuracy claim."
            ),
            "additional_rank_metrics": (
                "session hit rate, MRR, and binary nDCG over deduplicated session "
                "rankings are diagnostic additions; session recall is the upstream metric."
            ),
        },
        "worker_processes": processes,
        "worker_process_summary": worker_process_summary(
            processes, expected_count=args.trials * len(SYSTEMS)
        ),
        "results": {
            system: {
                "trials": rows[system],
                "aggregate": aggregates[system],
            }
            for system in SYSTEMS
        },
        "head_to_head": _head_to_head(aggregates),
        "postflight": {
            "data_unchanged": True,
            "repositories_unchanged": True,
            "sources_unchanged": True,
            "python_unchanged": True,
        },
    }
    write_json_once(output, report)
    omni_recall = aggregates["omnifuse"]["metrics"]["mean_session_recall"]["p50"]
    syn_recall = aggregates["synaptic"]["metrics"]["mean_session_recall"]["p50"]
    print(
        "LongMemEval retrieval benchmark: "
        f"OmniFuse session recall={omni_recall:.6f}; "
        f"Synaptic={syn_recall:.6f}"
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
