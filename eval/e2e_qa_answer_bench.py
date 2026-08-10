"""Fair local-LLM QA comparison on Synaptic's official HotPotQA cohort.

The upstream benchmark owns the prompt, Ollama payload, 24-question cohort,
and simple correctness metric.  This runner changes only the retrieval context:
Synaptic uses its official evidence chain and OmniFuse uses its native retrieval
plus product MMR.  Calls are interleaved AB/BA and checkpointed outside the
repository so a long local-model run can resume without repeating completions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(EVAL_DIR))

from e2e_qa_retrieval_bench import (  # noqa: E402
    MAX_QUESTIONS,
    _data_state,
    _load_data,
    _run_omnifuse,
    _run_synaptic,
    _sample_query_ids,
)
from provenance import (  # noqa: E402
    ProvenanceError,
    assert_unchanged,
    ensure_output_absent,
    file_fingerprint,
    read_json_artifact,
    repository_fingerprint,
    write_json_once,
)

RETRIEVAL_RUNNER = SCRIPT_PATH.with_name("e2e_qa_retrieval_bench.py")

SCHEMA = "omnifuse.e2e_qa_answer_comparison"
SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA = "omnifuse.e2e_qa_answer_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA = "omnifuse.e2e_qa_answer_work_manifest"
MANIFEST_SCHEMA_VERSION = 1
SYSTEMS = ("omnifuse", "synaptic")
SYSTEM_LABELS = {"omnifuse": "OmniFuse", "synaptic": "Synaptic Memory"}
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the provided context. "
    "If the context doesn't contain enough information, say 'I don't know'. "
    "Keep the answer concise (1-2 sentences). Give the direct answer."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synaptic-repo", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


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


def _require_loopback_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("E2E answer benchmark requires a loopback HTTP Ollama URL")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a path, params, query, or fragment")
    return value.rstrip("/")


def _request_json(
    *,
    url: str,
    timeout_seconds: float,
    payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if status != 200:
        raise RuntimeError(f"Ollama returned unexpected status {status}")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama response is not a JSON object")
    return parsed, elapsed_ms


def _ollama_state(
    *, base_url: str, model: str, timeout_seconds: float
) -> dict[str, Any]:
    version, _ = _request_json(
        url=f"{base_url}/api/version", timeout_seconds=timeout_seconds
    )
    tags, _ = _request_json(url=f"{base_url}/api/tags", timeout_seconds=timeout_seconds)
    models = tags.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama /api/tags omitted models")
    record = next(
        (
            item
            for item in models
            if isinstance(item, Mapping)
            and (item.get("name") == model or item.get("model") == model)
        ),
        None,
    )
    if record is None:
        raise RuntimeError(f"Ollama model is not local: {model}")
    digest = record.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"Ollama model omitted a full digest: {model}")
    return {
        "base_url": base_url,
        "version": str(version.get("version", "")),
        "model": model,
        "digest": digest,
        "size": int(record.get("size", 0)),
    }


def _normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _simple_correctness(answer: str, ground_truth: str) -> float:
    """Exact copy of the official three-stage non-judge score."""
    normalized_answer = _normalize(answer)
    normalized_truth = _normalize(ground_truth)
    if not normalized_truth:
        return 0.0
    if normalized_truth == normalized_answer or normalized_truth in normalized_answer:
        return 1.0
    truth_tokens = {token for token in normalized_truth.split() if len(token) >= 2}
    prediction_tokens = {
        token for token in normalized_answer.split() if len(token) >= 2
    }
    if not truth_tokens or not prediction_tokens:
        return 0.0
    common = truth_tokens & prediction_tokens
    recall = len(common) / len(truth_tokens)
    if recall >= 1.0:
        return 0.9
    if not common:
        return 0.0
    precision = len(common) / len(prediction_tokens)
    return 2 * precision * recall / (precision + recall)


def _prompt(question: str, context: str) -> tuple[str, str]:
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return SYSTEM_PROMPT, user


def _payload(*, model: str, question: str, context: str) -> dict[str, Any]:
    system, user = _prompt(question, context)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
    }


def _warmup_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "stream": False,
        "think": False,
        "options": {"num_predict": 1},
        "keep_alive": "10m",
    }


def _warm_model(*, base_url: str, model: str, timeout_seconds: float) -> dict[str, Any]:
    response, elapsed_ms = _request_json(
        url=f"{base_url}/api/chat",
        timeout_seconds=max(timeout_seconds, 600.0),
        payload=_warmup_payload(model),
    )
    if response.get("done") is not True:
        raise RuntimeError("Ollama neutral warm-up did not complete")
    return {
        "policy": "one unrelated one-token completion before the AB/BA schedule",
        "elapsed_ms": elapsed_ms,
        "load_duration": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_contract(
    *, system: str, model: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    context = str(row["context"])
    question = str(row["query"])
    payload = _payload(model=model, question=question, context=context)
    return {
        "system": system,
        "query_id": str(row["query_id"]),
        "question_sha256": _sha256_text(question),
        "context_sha256": _sha256_text(context),
        "payload_sha256": hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }


def _checkpoint_path(work_dir: Path, *, index: int, system: str) -> Path:
    return work_dir / f"question-{index + 1:02d}-{system}.json"


def _generate(
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
    system: str,
    row: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    contract = _query_contract(system=system, model=model, row=row)
    if checkpoint.exists():
        payload, _artifact = read_json_artifact(checkpoint)
        if (
            payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ProvenanceError(f"invalid answer checkpoint {checkpoint}")
        assert_unchanged(
            f"answer checkpoint contract {checkpoint.name}",
            contract,
            payload.get("contract"),
        )
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise ProvenanceError(f"checkpoint omitted result {checkpoint}")
        return dict(result)

    response, elapsed_ms = _request_json(
        url=f"{base_url}/api/chat",
        timeout_seconds=timeout_seconds,
        payload=_payload(
            model=model,
            question=str(row["query"]),
            context=str(row["context"]),
        ),
    )
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("Ollama chat response omitted message")
    answer = str(message.get("content", "")).strip()
    result = {
        "system": system,
        "query_id": str(row["query_id"]),
        "question": str(row["query"]),
        "ground_truth": str(row["answer"]),
        "answer": answer,
        "correctness": _simple_correctness(answer, str(row["answer"])),
        "generation_ms": elapsed_ms,
        "retrieval_ms": float(row["retrieval_ms"]),
        "context_sha256": _sha256_text(str(row["context"])),
        "context_characters": len(str(row["context"])),
        "ollama": {
            key: response.get(key)
            for key in (
                "model",
                "created_at",
                "done",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
        },
    }
    write_json_once(
        checkpoint,
        {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "contract": contract,
            "result": result,
        },
    )
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return float(ordered[index])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correctness = [float(row["correctness"]) for row in rows]
    positive = [score for score in correctness if score > 0]
    generation = [float(row["generation_ms"]) for row in rows]
    retrieval = [float(row["retrieval_ms"]) for row in rows]
    prompt_counts = [float(row["ollama"]["prompt_eval_count"] or 0) for row in rows]
    eval_counts = [float(row["ollama"]["eval_count"] or 0) for row in rows]
    return {
        "questions": len(rows),
        "cohort_mean_correctness": _mean(correctness),
        "official_mean_correctness_nonzero_only": _mean(positive),
        "accuracy_at_0_5": _mean([score >= 0.5 for score in correctness]),
        "exact_score_rate": _mean([score == 1.0 for score in correctness]),
        "positive_score_rate": _mean([score > 0 for score in correctness]),
        "mean_generation_ms": _mean(generation),
        "p95_generation_ms": _percentile(generation, 0.95),
        "total_generation_ms": sum(generation),
        "mean_retrieval_ms": _mean(retrieval),
        "mean_prompt_tokens": _mean(prompt_counts),
        "mean_output_tokens": _mean(eval_counts),
    }


def _per_question_head_to_head(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    question_rows = {
        system: {row["query_id"]: row for row in rows}
        for system, rows in results.items()
    }
    omni = question_rows["omnifuse"]
    synaptic = question_rows["synaptic"]
    if list(omni) != list(synaptic):
        raise ValueError("E2E answer per-question system cohorts differ")

    fields = {
        "correctness": ("higher", lambda row: row["correctness"]),
        "generation_ms": ("lower", lambda row: row["generation_ms"]),
        "retrieval_ms": ("lower", lambda row: row["retrieval_ms"]),
        "prompt_tokens": (
            "lower",
            lambda row: row["ollama"]["prompt_eval_count"] or 0,
        ),
        "output_tokens": ("lower", lambda row: row["ollama"]["eval_count"] or 0),
    }
    counts = {field: {"omnifuse": 0, "synaptic": 0, "tie": 0} for field in fields}
    losses = {field: [] for field in fields}
    for query_id, omni_row in omni.items():
        synaptic_row = synaptic[query_id]
        for field, (direction, extract) in fields.items():
            omni_value = float(extract(omni_row))
            synaptic_value = float(extract(synaptic_row))
            if math.isclose(omni_value, synaptic_value, rel_tol=1e-12, abs_tol=1e-12):
                winner = "tie"
            elif (direction == "higher" and omni_value > synaptic_value) or (
                direction == "lower" and omni_value < synaptic_value
            ):
                winner = "omnifuse"
            else:
                winner = "synaptic"
                losses[field].append(query_id)
            counts[field][winner] += 1
    return {
        "questions": len(omni),
        "metrics": counts,
        "loss_query_ids": losses,
        "questions_with_omnifuse_correctness_loss": len(losses["correctness"]),
    }


def _head_to_head(
    aggregates: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    directions = {
        "cohort_mean_correctness": "higher",
        "official_mean_correctness_nonzero_only": "higher",
        "accuracy_at_0_5": "higher",
        "exact_score_rate": "higher",
        "positive_score_rate": "higher",
        "mean_generation_ms": "lower",
        "p95_generation_ms": "lower",
        "total_generation_ms": "lower",
        "mean_retrieval_ms": "lower",
        "mean_prompt_tokens": "lower",
    }
    rows: list[dict[str, Any]] = []
    for metric, direction in directions.items():
        omni = float(aggregates["omnifuse"][metric])
        synaptic = float(aggregates["synaptic"][metric])
        if math.isclose(omni, synaptic, rel_tol=1e-12, abs_tol=1e-12):
            winner = "tie"
        elif (direction == "higher" and omni > synaptic) or (
            direction == "lower" and omni < synaptic
        ):
            winner = "omnifuse"
        else:
            winner = "synaptic"
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "omnifuse": omni,
                "synaptic": synaptic,
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
    if results is not None:
        result["per_question"] = _per_question_head_to_head(results)
    return result


def _source_state(repo: Path) -> dict[str, Any]:
    return {
        "benchmark": file_fingerprint(
            SCRIPT_PATH, display_path="eval/e2e_qa_answer_bench.py"
        ),
        "retrieval_benchmark": file_fingerprint(
            RETRIEVAL_RUNNER, display_path="eval/e2e_qa_retrieval_bench.py"
        ),
        "upstream_e2e_test": file_fingerprint(
            repo / "tests" / "benchmark" / "test_e2e_qa.py",
            display_path="tests/benchmark/test_e2e_qa.py",
        ),
    }


def _stable_manifest(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "repositories": state["repositories"],
        "sources": state["sources"],
        "data": state["data"],
        "ollama": state["ollama"],
        "contract": state["contract"],
    }


def _prepare_work_dir(work_dir: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = work_dir / "manifest.json"
    if work_dir.exists():
        if not work_dir.is_dir() or not manifest_path.is_file():
            raise ProvenanceError(f"refusing unrecognized work directory {work_dir}")
        existing, _artifact = read_json_artifact(manifest_path)
        assert_unchanged("answer work manifest", manifest, existing)
        return
    work_dir.mkdir(parents=True)
    write_json_once(manifest_path, manifest)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    output = args.out.resolve()
    ensure_output_absent(output)
    repo = args.synaptic_repo.resolve()
    if not (repo / "src" / "synaptic").is_dir():
        raise FileNotFoundError(f"Synaptic source package not found below {repo}")
    data_path = _resolve_data_path(args)
    data = _load_data(data_path)
    data_state = _data_state(data_path, data)
    if data_state["sampled_queries"] != MAX_QUESTIONS:
        raise ValueError(f"official E2E cohort requires {MAX_QUESTIONS} questions")
    base_url = _require_loopback_base_url(args.base_url)
    contract = {
        "upstream_test": (
            "tests/benchmark/test_e2e_qa.py::TestE2EHotPotQA.test_hotpotqa_e2e"
        ),
        "questions": MAX_QUESTIONS,
        "model": args.model,
        "payload": "official /api/chat messages + stream:false + think:false",
        "correctness": "official _evaluate_correctness_simple",
        "system_order": "alternating per-question AB/BA",
        "timeout_seconds": args.timeout_seconds,
        "timeout_caveat": "transport allowance only; prompt and model payload are unchanged",
        "neutral_model_warmup": (
            "one unrelated one-token completion before the measured AB/BA schedule"
        ),
    }
    return {
        "output": output,
        "work_dir": args.work_dir.resolve(),
        "repo": repo,
        "data_path": data_path,
        "loaded_data": data,
        "data": data_state,
        "repositories": {
            "omnifuse": repository_fingerprint(ROOT),
            "synaptic_memory": repository_fingerprint(repo),
        },
        "sources": _source_state(repo),
        "ollama": _ollama_state(
            base_url=base_url,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        ),
        "base_url": base_url,
        "contract": contract,
    }


def _controller(args: argparse.Namespace) -> int:
    state = _preflight(args)
    manifest = _stable_manifest(state)
    _prepare_work_dir(state["work_dir"], manifest)

    omnifuse = _run_omnifuse(state["loaded_data"], include_context=True)
    synaptic = asyncio.run(
        _run_synaptic(state["repo"], state["loaded_data"], include_context=True)
    )
    context_rows = {
        "omnifuse": {row["query_id"]: row for row in omnifuse["questions"]},
        "synaptic": {row["query_id"]: row for row in synaptic["questions"]},
    }
    query_ids = _sample_query_ids(state["loaded_data"])
    if any(set(rows) != set(query_ids) for rows in context_rows.values()):
        raise ProvenanceError("retrieval contexts do not cover the official E2E cohort")

    model_warmup = _warm_model(
        base_url=state["base_url"],
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    results: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    schedule: list[dict[str, Any]] = []
    for index, query_id in enumerate(query_ids):
        order = SYSTEMS if index % 2 == 0 else tuple(reversed(SYSTEMS))
        schedule.append(
            {"question_index": index + 1, "query_id": query_id, "order": list(order)}
        )
        for system in order:
            row = context_rows[system][query_id]
            result = _generate(
                base_url=state["base_url"],
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                system=system,
                row=row,
                checkpoint=_checkpoint_path(
                    state["work_dir"], index=index, system=system
                ),
            )
            results[system].append(result)
            print(
                f"[{index + 1:02d}/{len(query_ids)}] {system}: "
                f"correctness={result['correctness']:.6f} "
                f"generation_ms={result['generation_ms']:.1f}",
                flush=True,
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
        "Ollama model postflight",
        state["ollama"],
        _ollama_state(
            base_url=state["base_url"],
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    aggregates = {system: _aggregate(results[system]) for system in SYSTEMS}
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": state["repositories"],
        "sources": state["sources"],
        "data": state["data"],
        "ollama": state["ollama"],
        "contract": state["contract"],
        "retrieval": {
            "omnifuse": {
                "source_bindings": omnifuse["source_bindings"],
                "build_ms": omnifuse["build_ms"],
                "graph_counts": omnifuse["graph_counts"],
            },
            "synaptic": {
                "source_bindings": synaptic["source_bindings"],
                "build_ms": synaptic["build_ms"],
                "graph_counts": synaptic["graph_counts"],
            },
        },
        "schedule": schedule,
        "model_warmup": model_warmup,
        "results": {
            system: {"questions": results[system], "aggregate": aggregates[system]}
            for system in SYSTEMS
        },
        "head_to_head": _head_to_head(aggregates, results),
        "postflight": {
            "data_unchanged": True,
            "repositories_unchanged": True,
            "sources_unchanged": True,
            "ollama_model_unchanged": True,
        },
    }
    write_json_once(state["output"], report)
    verdict = report["head_to_head"]["verdict"]
    print(
        "E2E answer benchmark: "
        f"OmniFuse={verdict['omnifuse']} Synaptic={verdict['synaptic']} "
        f"ties={verdict['ties']}",
        flush=True,
    )
    print(f"wrote {state['output']}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _controller(args)
    except (OSError, ValueError, RuntimeError, ProvenanceError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
