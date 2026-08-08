"""OmniFuse retrieval benchmark on the Korean finreg fixtures.

The command rebuilds the zero-infrastructure in-memory OmniFuse index from the
tracked corpus on every invocation. Retrieval is single-shot and uses the same
``eval/metrics.py`` scorer as synaptic-memory.
"""

from __future__ import annotations

import argparse
import inspect
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(EVAL_DIR))

import _common as common_module  # noqa: E402
import metrics as metrics_module  # noqa: E402
import omnifuse as omnifuse_package  # noqa: E402
from _common import (  # noqa: E402
    K,
    QUERIES,
    RAW,
    build_reference_triples,
    load_corpus,
    load_queries,
    score_mrr,
    score_strict,
    to_chunks_nodes,
)
from omnifuse import build_inmemory  # noqa: E402
from provenance import (  # noqa: E402
    ProvenanceError,
    assert_unchanged,
    ensure_output_absent,
    file_fingerprint,
    load_doctor_manifest,
    repository_fingerprint,
    verify_doctor_manifest,
    verify_doctor_runtime,
    write_json_once,
)


CANDIDATE_LIMIT = K * 2
SINGLE_QUERY_PATH = QUERIES / "finreg.json"
MULTIHOP_QUERY_PATH = QUERIES / "finreg_multihop.json"
SCORER_PATH = EVAL_DIR / "metrics.py"
COMMON_HARNESS_PATH = EVAL_DIR / "_common.py"
PROVENANCE_PATH = EVAL_DIR / "provenance.py"
CORPUS_RELATIVE_PATH = "eval/data/finreg/raw.jsonl"
SINGLE_QUERY_RELATIVE_PATH = "eval/data/queries/finreg.json"
MULTIHOP_QUERY_RELATIVE_PATH = "eval/data/queries/finreg_multihop.json"
DEDUPE_POLICY = "first occurrence of each non-empty document id, preserving rank order"
PROVENANCE_LEVEL = "strict-doctor-exact-scorer-preflight-postflight-write-once-v2"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="disable graph-companion fusion (BM25F only)",
    )
    parser.add_argument("--out", type=Path, help="write the JSON report atomically")
    parser.add_argument(
        "--doctor-manifest",
        type=Path,
        help="strict eval/bench.py doctor JSON; required with --out",
    )
    return parser


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, object]:
    return file_fingerprint(path, display_path=display_path)


def _runtime_environment() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "dont_write_bytecode": sys.dont_write_bytecode,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    write_json_once(path, payload)


def _input_records() -> dict[str, dict[str, object]]:
    return {
        "corpus": _file_record(RAW, display_path=CORPUS_RELATIVE_PATH),
        "single_hop_queries": _file_record(
            SINGLE_QUERY_PATH, display_path=SINGLE_QUERY_RELATIVE_PATH
        ),
        "multi_hop_queries": _file_record(
            MULTIHOP_QUERY_PATH, display_path=MULTIHOP_QUERY_RELATIVE_PATH
        ),
    }


def _verified_source_file(
    raw_path: str | None,
    *,
    expected_root: Path,
    label: str,
    display_path: str,
) -> dict[str, object]:
    if not raw_path:
        raise ProvenanceError(f"{label} has no inspectable source file")
    resolved = Path(raw_path).resolve()
    try:
        resolved.relative_to(expected_root.resolve())
    except ValueError as exc:
        raise ProvenanceError(
            f"loaded {label} from {resolved}; expected source below "
            f"{expected_root.resolve()}"
        ) from exc
    return _file_record(resolved, display_path=display_path)


def _verified_exact_source_file(
    raw_path: str | None,
    *,
    expected_path: Path,
    label: str,
    display_path: str,
) -> dict[str, object]:
    if not raw_path:
        raise ProvenanceError(f"{label} has no inspectable source file")
    resolved = Path(raw_path).resolve()
    expected = expected_path.resolve()
    if resolved != expected:
        raise ProvenanceError(
            f"loaded {label} from {resolved}; expected exact source {expected}"
        )
    return _file_record(resolved, display_path=display_path)


def _omnifuse_import_sources() -> dict[str, object]:
    package_path = getattr(omnifuse_package, "__file__", None)
    build_path = inspect.getsourcefile(build_inmemory)
    common_path = getattr(common_module, "__file__", None)
    if not build_path:
        raise ProvenanceError("omnifuse.build_inmemory has no inspectable source file")
    try:
        build_display_path = (
            Path(build_path).resolve().relative_to(REPOSITORY_ROOT).as_posix()
        )
    except ValueError as exc:
        raise ProvenanceError(
            "omnifuse.build_inmemory was loaded from outside this checkout"
        ) from exc
    return {
        "package": _verified_source_file(
            package_path,
            expected_root=SOURCE_ROOT,
            label="omnifuse",
            display_path="src/omnifuse/__init__.py",
        ),
        "build_inmemory": _verified_source_file(
            build_path,
            expected_root=SOURCE_ROOT,
            label="omnifuse.build_inmemory",
            display_path=build_display_path,
        ),
        "shared_finreg_module": _verified_source_file(
            common_path,
            expected_root=EVAL_DIR,
            label="eval._common",
            display_path="eval/_common.py",
        ),
    }


def _active_scorer_records() -> dict[str, object]:
    if sys.modules.get("metrics") is not metrics_module:
        raise ProvenanceError(
            "sys.modules['metrics'] is not the exact bound eval.metrics module"
        )
    benchmark_result = getattr(metrics_module, "BenchmarkResult", None)
    if benchmark_result is None:
        raise ProvenanceError("loaded metrics module has no BenchmarkResult")
    if common_module.score_mrr is not score_mrr:
        raise ProvenanceError("active score_mrr is not eval._common.score_mrr")
    if common_module.score_strict is not score_strict:
        raise ProvenanceError("active score_strict is not eval._common.score_strict")

    module = _verified_exact_source_file(
        getattr(metrics_module, "__file__", None),
        expected_path=SCORER_PATH,
        label="metrics module",
        display_path="eval/metrics.py",
    )
    benchmark_result_record = _verified_exact_source_file(
        inspect.getsourcefile(benchmark_result),
        expected_path=SCORER_PATH,
        label="metrics.BenchmarkResult",
        display_path="eval/metrics.py",
    )
    score_mrr_record = _verified_exact_source_file(
        inspect.getsourcefile(score_mrr),
        expected_path=COMMON_HARNESS_PATH,
        label="eval._common.score_mrr",
        display_path="eval/_common.py",
    )
    score_strict_record = _verified_exact_source_file(
        inspect.getsourcefile(score_strict),
        expected_path=COMMON_HARNESS_PATH,
        label="eval._common.score_strict",
        display_path="eval/_common.py",
    )
    return {
        "module": module,
        "benchmark_result": benchmark_result_record,
        "score_mrr": score_mrr_record,
        "score_strict": score_strict_record,
        "exact_path_bound": True,
        "sys_modules_identity_bound": True,
        "common_function_identity_bound": True,
    }


def _scorer_records(synaptic_repo: Path) -> dict[str, object]:
    active = _active_scorer_records()
    local = active["module"]
    competitor_path = synaptic_repo.resolve() / "tests" / "benchmark" / "metrics.py"
    competitor = _file_record(
        competitor_path, display_path="tests/benchmark/metrics.py"
    )
    if local["sha256"] != competitor["sha256"] or local["bytes"] != competitor["bytes"]:
        raise ProvenanceError(
            "eval/metrics.py is not byte-identical to the selected "
            "synaptic-memory tests/benchmark/metrics.py"
        )
    return {
        "active": local,
        "active_bindings": active,
        "synaptic_checkout_copy": competitor,
        "byte_identical": True,
    }


def _benchmark_sources(synaptic_repo: Path) -> dict[str, object]:
    return {
        "entrypoint": _file_record(SCRIPT_PATH, display_path="eval/finreg_bench.py"),
        "provenance_helper": _file_record(
            PROVENANCE_PATH, display_path="eval/provenance.py"
        ),
        "shared_finreg_logic": _file_record(
            COMMON_HARNESS_PATH, display_path="eval/_common.py"
        ),
        "scorer": _scorer_records(synaptic_repo),
        "imported_omnifuse": _omnifuse_import_sources(),
    }


def _doctor_input_specs(
    inputs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    corpus = inputs["corpus"]
    single = inputs["single_hop_queries"]
    multi = inputs["multi_hop_queries"]

    def spec(
        name: str,
        target_id: str,
        role: str,
        record: dict[str, object],
    ) -> dict[str, object]:
        return {
            "name": name,
            "target_id": target_id,
            "role": role,
            "path": record["path"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }

    return [
        spec("finreg_single_corpus", "finreg_single", "corpus", corpus),
        spec("finreg_single_queries", "finreg_single", "queries", single),
        spec("finreg_multi_corpus", "finreg_multi", "corpus", corpus),
        spec("finreg_multi_queries", "finreg_multi", "queries", multi),
    ]


def _group_doctor_links(
    links: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        "finreg_single": {
            "corpus": links["finreg_single_corpus"],
            "queries": links["finreg_single_queries"],
        },
        "finreg_multi": {
            "corpus": links["finreg_multi_corpus"],
            "queries": links["finreg_multi_queries"],
        },
    }


def _doctor_synaptic_repo(doctor: dict[str, Any]) -> Path:
    snapshot = doctor.get("snapshot")
    repositories = snapshot.get("repositories") if isinstance(snapshot, dict) else None
    synaptic = (
        repositories.get("synaptic_memory") if isinstance(repositories, dict) else None
    )
    raw_path = synaptic.get("path") if isinstance(synaptic, dict) else None
    if not isinstance(raw_path, str) or not raw_path:
        raise ProvenanceError(
            "doctor manifest has no selected synaptic-memory checkout path"
        )
    repo = Path(raw_path).resolve()
    if not repo.is_dir():
        raise ProvenanceError(f"doctor synaptic-memory checkout not found: {repo}")
    return repo


def _machine_preflight(*, output: Path, doctor_manifest: Path) -> dict[str, Any]:
    ensure_output_absent(output)
    inputs = _input_records()
    doctor, raw_links = load_doctor_manifest(
        doctor_manifest, _doctor_input_specs(inputs)
    )
    synaptic_repo = _doctor_synaptic_repo(doctor)
    repositories = {
        "omnifuse": repository_fingerprint(REPOSITORY_ROOT),
        "synaptic_memory_doctor_reference": repository_fingerprint(synaptic_repo),
    }
    sources = _benchmark_sources(synaptic_repo)
    verify_doctor_runtime(
        doctor,
        omnifuse_repository=repositories["omnifuse"],
        synaptic_repository=repositories["synaptic_memory_doctor_reference"],
        omnifuse_scorer=sources["scorer"]["active"],
        synaptic_scorer=sources["scorer"]["synaptic_checkout_copy"],
    )
    assert_unchanged("finreg inputs during preflight", inputs, _input_records())
    return {
        "inputs": inputs,
        "repositories": repositories,
        "sources": sources,
        "doctor_manifest": doctor,
        "doctor_links": _group_doctor_links(raw_links),
        "synaptic_repo": synaptic_repo,
    }


def _verify_machine_postflight(state: dict[str, Any]) -> dict[str, Any]:
    synaptic_repo: Path = state["synaptic_repo"]
    after = {
        "inputs": _input_records(),
        "repositories": {
            "omnifuse": repository_fingerprint(REPOSITORY_ROOT),
            "synaptic_memory_doctor_reference": repository_fingerprint(synaptic_repo),
        },
        "sources": _benchmark_sources(synaptic_repo),
    }
    assert_unchanged("finreg inputs", state["inputs"], after["inputs"])
    assert_unchanged(
        "repository fingerprints", state["repositories"], after["repositories"]
    )
    assert_unchanged(
        "benchmark source fingerprints", state["sources"], after["sources"]
    )
    verify_doctor_manifest(state["doctor_manifest"])
    scorer = after["sources"]["scorer"]
    verify_doctor_runtime(
        state["doctor_manifest"],
        omnifuse_repository=after["repositories"]["omnifuse"],
        synaptic_repository=after["repositories"]["synaptic_memory_doctor_reference"],
        omnifuse_scorer=scorer["active"],
        synaptic_scorer=scorer["synaptic_checkout_copy"],
    )
    return {
        **after,
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


def _run_omnifuse(*, graph_fusion: bool) -> dict[str, Any]:
    docs = load_corpus()
    nodes, chunks = to_chunks_nodes(docs)
    triples = build_reference_triples(docs)
    single_hop = load_queries(SINGLE_QUERY_PATH.name)
    multi_hop = load_queries(MULTIHOP_QUERY_PATH.name)

    total_started = time.perf_counter()
    build_started = time.perf_counter()
    omnifuse = build_inmemory(
        nodes,
        triples,
        chunks,
        graph_fusion=graph_fusion,
        vector_k=CANDIDATE_LIMIT,
    )
    build_seconds = time.perf_counter() - build_started

    def retrieve(query: str) -> list[str]:
        return [
            chunk.id
            for chunk, _score in omnifuse.retrieve(query, limit=CANDIDATE_LIMIT)
        ]

    score_started = time.perf_counter()
    single_score = score_mrr(retrieve, single_hop, k=K)
    multi_score = score_strict(retrieve, multi_hop, k=K)
    score_seconds = time.perf_counter() - score_started
    return {
        "corpus": {
            "documents": len(docs),
            "reference_edges": len(triples),
        },
        "queries": {
            "single_hop": len(single_hop),
            "multi_hop": len(multi_hop),
        },
        "scores": {
            "single_hop": single_score,
            "multi_hop": multi_score,
        },
        "timing_seconds": {
            "rebuild": build_seconds,
            "scoring_all_queries": score_seconds,
            "total": time.perf_counter() - total_started,
        },
    }


def _evaluation_contract(graph_fusion: bool) -> dict[str, object]:
    return {
        "k": K,
        "candidate_limit": CANDIDATE_LIMIT,
        "same_candidate_limit_for_all_query_tracks": True,
        "dedupe": DEDUPE_POLICY,
        "score_after_dedupe": True,
        "retrieval": "single-shot; no LLM, embedder, or reranker",
        "graph_fusion": graph_fusion,
    }


def _build_report(
    *,
    graph_fusion: bool,
    result: dict[str, Any],
    state: dict[str, Any],
    postflight: dict[str, Any],
) -> dict[str, object]:
    return {
        "schema": "omnifuse.eval.finreg",
        "schema_version": 2,
        "provenance_level": PROVENANCE_LEVEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "environment": _runtime_environment(),
        "inputs": {
            "before": state["inputs"],
            "after": postflight["inputs"],
        },
        "repositories": {
            "before": state["repositories"],
            "after": postflight["repositories"],
        },
        "provenance": {
            "level": PROVENANCE_LEVEL,
            "benchmark_sources": {
                "before": state["sources"],
                "after": postflight["sources"],
            },
            "doctor_manifest": state["doctor_manifest"],
            "doctor_targets": state["doctor_links"],
        },
        "integrity": postflight["integrity"],
        "evaluation_contract": _evaluation_contract(graph_fusion),
        "index_condition": {
            "system": "omnifuse",
            "state": "rebuilt_in_process_from_tracked_corpus",
            "rebuild_included_in_total_timing": True,
            "backend": "zero-infrastructure in-memory",
        },
        "protocol_scope": {
            "kind": "omnifuse_local_only",
            "synaptic_memory_executed": False,
            "doctor_synaptic_checkout_role": "input_and_scorer_reference_only",
            "cross_artifact_score_comparison_valid": False,
        },
        "result": result,
    }


def _print_report(report: dict[str, object]) -> None:
    result: dict[str, Any] = report["result"]  # type: ignore[assignment]
    contract: dict[str, Any] = report["evaluation_contract"]  # type: ignore[assignment]
    corpus = result["corpus"]
    queries = result["queries"]
    single = result["scores"]["single_hop"]
    multi = result["scores"]["multi_hop"]
    print(
        f"corpus: {corpus['documents']} articles, {corpus['reference_edges']} "
        "REFERENCES edges"
    )
    print(
        f"queries: {queries['single_hop']} single-hop, {queries['multi_hop']} "
        f"multi-hop (k={contract['k']}, candidates={contract['candidate_limit']}, "
        f"graph_fusion={contract['graph_fusion']})\n"
    )
    print(
        f"single-hop  MRR@{K}={single['mrr']:.4f}  "
        f"nDCG@{K}={single['mean_ndcg@k']:.4f}  "
        f"hit@{K}={single['hits']}/{single['n']}"
    )
    print(
        f"multi-hop   strict-solved={multi['strict']}/{multi['n']}  "
        f"R@{K}={multi['mean_recall@k']:.4f}"
    )
    print(f"\n[{result['timing_seconds']['total']:.1f}s, including rebuild]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.out is not None and args.doctor_manifest is None:
        parser.error("--doctor-manifest is required when --out is used")

    graph_fusion = not args.no_graph
    state: dict[str, Any] | None = None
    postflight: dict[str, Any] | None = None
    try:
        if args.out is not None:
            state = _machine_preflight(
                output=args.out,
                doctor_manifest=args.doctor_manifest,
            )
        else:
            _active_scorer_records()
        result = _run_omnifuse(graph_fusion=graph_fusion)
        if state is not None:
            postflight = _verify_machine_postflight(state)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    display_report = {
        "result": result,
        "evaluation_contract": _evaluation_contract(graph_fusion),
    }
    _print_report(display_report)
    if args.out is not None:
        assert state is not None and postflight is not None
        report = _build_report(
            graph_fusion=graph_fusion,
            result=result,
            state=state,
            postflight=postflight,
        )
        try:
            _atomic_write_json(args.out, report)
        except ProvenanceError as exc:
            parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
