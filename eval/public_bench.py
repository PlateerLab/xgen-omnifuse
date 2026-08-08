"""Compare OmniFuse with synaptic-memory on its eight tracked public datasets.

This harness intentionally excludes downloader-generated extended datasets.  Every
declared input must exist in, and be tracked by, the selected synaptic-memory Git
checkout.  Both systems retrieve 20 candidates, de-duplicate by document id while
preserving rank, and score the first 10 with the same BenchmarkResult scorer.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = (REPOSITORY_ROOT / "src").resolve()
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from metrics import BenchmarkResult  # noqa: E402
import omnifuse as omnifuse_package  # noqa: E402
from omnifuse import build_inmemory  # noqa: E402
from provenance import (  # noqa: E402
    ProvenanceError,
    assert_unchanged,
    ensure_output_absent,
    file_fingerprint,
    load_doctor_manifest,
    repository_fingerprint,
    sha256_file,
    verify_doctor_manifest,
    verify_doctor_runtime,
    write_json_once,
)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    filename: str


DATASETS = (
    DatasetSpec("hotpotqa_24", "HotPotQA-24", "hotpotqa_24.json"),
    DatasetSpec("hotpotqa_200", "HotPotQA-200", "hotpotqa.json"),
    DatasetSpec("allganize_rag_ko", "Allganize RAG-ko", "allganize_rag_ko.json"),
    DatasetSpec("allganize_rag_eval", "Allganize RAG-Eval", "allganize_rag_eval.json"),
    DatasetSpec("publichealthqa_ko", "PublicHealthQA", "publichealthqa_ko.json"),
    DatasetSpec("autorag_retrieval", "AutoRAG", "autorag_retrieval.json"),
    DatasetSpec("klue_mrc", "KLUE-MRC", "klue_mrc.json"),
    DatasetSpec("ko_strategyqa", "Ko-StrategyQA", "ko_strategyqa.json"),
)
K = 10
CANDIDATE_LIMIT = K * 2
DEDUPE_POLICY = "first occurrence of non-empty document id, preserving rank"
SCORER_PATH = EVAL_DIR / "metrics.py"
SYNAPTIC_DRIVER_RELATIVE = Path("eval/run_all.py")
SYNAPTIC_SCORER_RELATIVE = Path("tests/benchmark/metrics.py")
NATIVE_LIFECYCLE_CAVEAT = (
    "The synaptic arm delegates ingestion, search, backend close, and temporary-artifact "
    "cleanup to eval/run_all.py::run_public_dataset. The selected upstream runner closes "
    "and removes its temporary database on the successful path, but does not expose the "
    "backend to this harness; this harness therefore cannot close it itself if upstream "
    "raises before cleanup."
)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, object]:
    try:
        return file_fingerprint(path, display_path=display_path)
    except ProvenanceError as exc:
        raise FileNotFoundError(str(exc)) from exc


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise RuntimeError(f"could not inspect Git checkout {repo}: {exc}") from exc


def _git_state(repo: Path) -> dict[str, object]:
    try:
        return repository_fingerprint(repo)
    except ProvenanceError as exc:
        raise RuntimeError(str(exc)) from exc


def _require_tracked(repo: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"dataset is outside the selected checkout: {path}") from exc
    result = _run_git(repo, "ls-files", "--error-unmatch", "--", relative)
    if result.returncode:
        raise ValueError(
            f"public benchmark input is not tracked by the selected checkout: {relative}"
        )
    return relative


def _source_snapshot(source_root: Path) -> dict[str, str]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"synaptic source tree not found: {source_root}")
    snapshot: dict[str, str] = {}
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            snapshot[path.relative_to(source_root).as_posix()] = _sha256(path)
    if not snapshot:
        raise RuntimeError(f"synaptic source tree has no files: {source_root}")
    return snapshot


def _snapshot_record(snapshot: dict[str, str]) -> dict[str, object]:
    payload = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "files": len(snapshot),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _snapshot_diff(
    before: dict[str, str], after: dict[str, str]
) -> dict[str, list[str]]:
    before_names = set(before)
    after_names = set(after)
    return {
        "added": sorted(after_names - before_names),
        "removed": sorted(before_names - after_names),
        "modified": sorted(
            name for name in before_names & after_names if before[name] != after[name]
        ),
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    write_json_once(path, payload)


def parse_public(
    data: dict[str, Any],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, set[str]]]]:
    raw = data.get("corpus", data.get("documents", []))
    corpus: list[tuple[str, str, str]] = []
    if isinstance(raw, dict):
        for document_id, document in raw.items():
            if isinstance(document, dict):
                corpus.append(
                    (
                        str(document_id),
                        str(document.get("title", "")),
                        str(document.get("text", document.get("content", ""))),
                    )
                )
            elif isinstance(document, str):
                corpus.append((str(document_id), "", document))
    elif isinstance(raw, list):
        for document in raw:
            if isinstance(document, dict):
                document_id = str(
                    document.get("doc_id", document.get("_id", document.get("id", "")))
                )
                corpus.append(
                    (
                        document_id,
                        str(document.get("title", "")),
                        str(document.get("text", document.get("content", ""))),
                    )
                )

    queries = data.get("queries", [])
    qrels = data.get("relevant_docs", data.get("qrels", {}))
    query_list: list[tuple[str, str, set[str]]] = []
    if isinstance(queries, dict):
        if not isinstance(qrels, dict):
            raise ValueError("mapping-style queries require mapping-style qrels")
        for query_id, text in queries.items():
            raw_relevant = qrels.get(query_id, {})
            relevant = (
                set(map(str, raw_relevant))
                if isinstance(raw_relevant, (dict, list))
                else set()
            )
            if relevant and text:
                query_list.append((str(query_id), str(text), relevant))
    elif isinstance(queries, list):
        for query in queries:
            if not isinstance(query, dict):
                continue
            query_id = str(
                query.get("qid", query.get("query_id", query.get("_id", "")))
            )
            text = str(query.get("query", query.get("question", "")))
            raw_relevant = query.get(
                "relevant_docs",
                query.get("answer_ids", query.get("positive_doc_ids", [])),
            )
            relevant = (
                set(map(str, raw_relevant))
                if isinstance(raw_relevant, (dict, list))
                else set()
            )
            if relevant and text:
                query_list.append((query_id, text, relevant))

    if not corpus:
        raise ValueError("dataset has no parser-compatible documents")
    if not query_list:
        raise ValueError("dataset has no parser-compatible scored queries")
    return corpus, query_list


def _load_public(
    path: Path,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, set[str]]]]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("public dataset must be a JSON object")
    return parse_public(data)


def _score_omnifuse(
    corpus: Sequence[tuple[str, str, str]],
    queries: Sequence[tuple[str, str, set[str]]],
) -> dict[str, float]:
    chunks = [
        {"id": document_id, "title": title, "text": text}
        for document_id, title, text in corpus
    ]
    omnifuse = build_inmemory([], [], chunks)
    benchmark = BenchmarkResult()
    for query_id, text, relevant in queries:
        ranked: list[str] = []
        seen: set[str] = set()
        for chunk, _score in omnifuse.retrieve(text, limit=CANDIDATE_LIMIT):
            document_id = str(chunk.id)
            if document_id and document_id not in seen:
                seen.add(document_id)
                ranked.append(document_id)
        benchmark.add(
            query_id=query_id,
            query=text,
            retrieved=ranked[:K],
            relevant=relevant,
            k=K,
        )
    summary = benchmark.summary()
    if not summary:
        raise RuntimeError("OmniFuse scorer produced no rows")
    return {
        "mrr_at_10": float(summary["mrr"]),
        "precision_at_10": float(summary["mean_precision@k"]),
        "recall_at_10": float(summary["mean_recall@k"]),
        "ndcg_at_10": float(summary["mean_ndcg@k"]),
    }


def omni_mrr(path: Path) -> tuple[float, int]:
    """Compatibility wrapper used by the IDF sweep harness."""
    corpus, queries = _load_public(path)
    return _score_omnifuse(corpus, queries)["mrr_at_10"], len(corpus)


def _is_below(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _verify_module_path(module: Any, expected: Path, name: str) -> Path:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise RuntimeError(f"{name} has no inspectable source path")
    actual = Path(raw_path).resolve()
    if actual != expected.resolve():
        raise RuntimeError(
            f"loaded {name} from {actual}; expected {expected.resolve()}"
        )
    return actual


def _prepend_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != resolved]
    sys.path.insert(0, resolved)
    importlib.invalidate_caches()


def _load_synaptic_runner(repo: Path) -> tuple[Any, Any, dict[str, str]]:
    repo = repo.resolve()
    source_root = (repo / "src").resolve()
    package_init = source_root / "synaptic" / "__init__.py"
    driver_path = (repo / SYNAPTIC_DRIVER_RELATIVE).resolve()
    if not package_init.is_file():
        raise FileNotFoundError(f"synaptic source package not found: {package_init}")
    if not driver_path.is_file():
        raise FileNotFoundError(
            f"synaptic native benchmark driver not found: {driver_path}"
        )

    _prepend_import_path(repo / "tests" / "benchmark")
    _prepend_import_path(repo)
    _prepend_import_path(source_root)
    package = importlib.import_module("synaptic")
    package_path = getattr(package, "__file__", None)
    if not package_path or not _is_below(Path(package_path), source_root):
        raise RuntimeError(
            f"loaded synaptic from {package_path}; expected source below {source_root}"
        )
    runner_module = importlib.import_module("eval.run_all")
    actual_driver = _verify_module_path(runner_module, driver_path, "eval.run_all")
    native_scorer_path = inspect.getsourcefile(runner_module.BenchmarkResult)
    expected_scorer = (repo / SYNAPTIC_SCORER_RELATIVE).resolve()
    if (
        native_scorer_path is None
        or Path(native_scorer_path).resolve() != expected_scorer
    ):
        raise RuntimeError(
            "synaptic native runner loaded BenchmarkResult from "
            f"{native_scorer_path}; expected {expected_scorer}"
        )
    return (
        runner_module.DatasetConfig,
        runner_module.run_public_dataset,
        {
            "package": str(Path(package_path).resolve()),
            "native_driver": str(actual_driver),
            "native_scorer": str(expected_scorer),
        },
    )


def preflight_synaptic_runner(repo: Path) -> dict[str, str]:
    """Fail before expensive local scoring if the selected native runner cannot load."""
    _dataset_config, _runner, module_paths = _load_synaptic_runner(repo)
    return module_paths


async def synaptic_metrics(repo: Path, path: Path, name: str) -> dict[str, object]:
    dataset_config, run_public_dataset, module_paths = _load_synaptic_runner(repo)
    result = await run_public_dataset(
        dataset_config(name=name, path=path, k=K, quick=True),
        embedder=None,
        reranker=None,
    )
    if result.error:
        raise RuntimeError(f"synaptic native runner returned an error: {result.error}")
    return {
        "mrr_at_10": float(result.mrr),
        "precision_at_10": float(result.p_at_k),
        "recall_at_10": float(result.r_at_k),
        "ndcg_at_10": float(result.ndcg),
        "reported_corpus_size": int(result.corpus_size),
        "module_paths": module_paths,
    }


async def synaptic_mrr(repo: Path, path: Path, name: str) -> float:
    """Compatibility wrapper used by the IDF sweep harness."""
    return float((await synaptic_metrics(repo, path, name))["mrr_at_10"])


def _dataset_input(repo: Path, path: Path) -> tuple[dict[str, object], list, list]:
    relative = _require_tracked(repo, path)
    corpus, queries = _load_public(path)
    return (
        {
            **_file_record(path, display_path=relative),
            "git_tracked": True,
            "documents": len(corpus),
            "scored_queries": len(queries),
            "relevance_judgments": sum(len(relevant) for _, _, relevant in queries),
        },
        corpus,
        queries,
    )


def _doctor_provenance(
    doctor_path: Path | None,
    inputs: dict[str, tuple[dict[str, object], list, list]],
) -> tuple[dict[str, object] | None, dict[str, dict[str, object]]]:
    if doctor_path is None:
        return None, {}
    manifest_inputs = []
    for spec in DATASETS:
        input_record = inputs[spec.id][0]
        manifest_inputs.append(
            {
                "name": spec.name,
                "target_id": spec.id,
                "path": str(input_record["path"]),
                "sha256": str(input_record["sha256"]),
                "bytes": input_record["bytes"],
            }
        )
    doctor, links = load_doctor_manifest(doctor_path, manifest_inputs)
    return doctor, links


def _scorer_provenance(repo: Path) -> dict[str, object]:
    local = _file_record(SCORER_PATH, display_path="eval/metrics.py")
    upstream = _file_record(
        repo / SYNAPTIC_SCORER_RELATIVE,
        display_path=SYNAPTIC_SCORER_RELATIVE.as_posix(),
    )
    identical = local["sha256"] == upstream["sha256"]
    if not identical:
        raise RuntimeError(
            "eval/metrics.py is not byte-identical to the selected synaptic-memory "
            "tests/benchmark/metrics.py"
        )
    return {"active": local, "synaptic_checkout_copy": upstream, "byte_identical": True}


def _omnifuse_import_provenance() -> dict[str, object]:
    package_path = Path(inspect.getfile(omnifuse_package)).resolve()
    build_source = Path(inspect.getsourcefile(build_inmemory) or "").resolve()
    expected_package = (SOURCE_ROOT / "omnifuse" / "__init__.py").resolve()
    expected_build_source = (SOURCE_ROOT / "omnifuse" / "facade.py").resolve()
    if package_path != expected_package:
        raise RuntimeError(
            f"omnifuse package loaded from {package_path}, expected {expected_package}"
        )
    if build_source != expected_build_source:
        raise RuntimeError(
            f"omnifuse.build_inmemory loaded from {build_source}, "
            f"expected {expected_build_source}"
        )
    return {
        "package": _file_record(package_path, display_path="src/omnifuse/__init__.py"),
        "build_inmemory": _file_record(
            build_source, display_path="src/omnifuse/facade.py"
        ),
    }


def _benchmark_sources(repo: Path) -> dict[str, object]:
    return {
        "harness": _file_record(SCRIPT_PATH, display_path="eval/public_bench.py"),
        "provenance_helper": _file_record(
            EVAL_DIR / "provenance.py", display_path="eval/provenance.py"
        ),
        "scorer": _scorer_provenance(repo),
        "omnifuse_imports": _omnifuse_import_provenance(),
        "synaptic_native_driver": _file_record(
            repo / SYNAPTIC_DRIVER_RELATIVE,
            display_path=SYNAPTIC_DRIVER_RELATIVE.as_posix(),
        ),
    }


def _winner(omnifuse_mrr: float, synaptic_mrr_value: float) -> str:
    if omnifuse_mrr > synaptic_mrr_value:
        return "omnifuse"
    if synaptic_mrr_value > omnifuse_mrr:
        return "synaptic_memory"
    return "tie"


def execute_benchmark(
    synaptic_repo: Path, doctor_manifest: Path | None = None
) -> dict[str, object]:
    synaptic_repo = synaptic_repo.resolve()
    source_root = (synaptic_repo / "src").resolve()
    source_before = _source_snapshot(source_root)
    repositories_before = {
        "omnifuse": _git_state(REPOSITORY_ROOT),
        "synaptic_memory": _git_state(synaptic_repo),
    }
    sources_before = _benchmark_sources(synaptic_repo)
    inputs: dict[str, tuple[dict[str, object], list, list]] = {}
    input_errors: list[str] = []
    for spec in DATASETS:
        path = synaptic_repo / "tests" / "benchmark" / "data" / spec.filename
        try:
            inputs[spec.id] = _dataset_input(synaptic_repo, path)
        except (OSError, ValueError, RuntimeError) as exc:
            input_errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")
    if input_errors:
        raise RuntimeError(
            "required tracked-public input preflight failed: " + "; ".join(input_errors)
        )
    if set(inputs) != {spec.id for spec in DATASETS}:
        raise RuntimeError("tracked-public dataset scope was not resolved exactly")
    doctor, doctor_links = _doctor_provenance(doctor_manifest, inputs)
    if doctor is not None:
        scorer_provenance = sources_before["scorer"]
        verify_doctor_runtime(
            doctor,
            omnifuse_repository=repositories_before["omnifuse"],
            synaptic_repository=repositories_before["synaptic_memory"],
            omnifuse_scorer=scorer_provenance["active"],
            synaptic_scorer=scorer_provenance["synaptic_checkout_copy"],
        )

    rows: list[dict[str, object]] = []
    suite_errors: list[str] = []
    _load_synaptic_runner(synaptic_repo)

    for spec in DATASETS:
        path = synaptic_repo / "tests" / "benchmark" / "data" / spec.filename
        input_record, corpus, queries = inputs[spec.id]
        row: dict[str, object] = {
            "id": spec.id,
            "name": spec.name,
            "status": "error",
            "input": input_record,
            "doctor_target_id": doctor_links.get(spec.name, {}).get("target_id"),
            "systems": None,
            "winner": None,
            "error": None,
        }
        try:
            omnifuse = _score_omnifuse(corpus, queries)
            synaptic = asyncio.run(synaptic_metrics(synaptic_repo, path, spec.name))
            if synaptic["reported_corpus_size"] != len(corpus):
                raise RuntimeError(
                    "synaptic native runner corpus count differs from the shared parser: "
                    f"{synaptic['reported_corpus_size']} != {len(corpus)}"
                )
            row.update(
                status="ok",
                systems={"omnifuse": omnifuse, "synaptic_memory": synaptic},
                winner=_winner(
                    float(omnifuse["mrr_at_10"]), float(synaptic["mrr_at_10"])
                ),
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    source_after = _source_snapshot(source_root)
    source_diff = _snapshot_diff(source_before, source_after)
    source_unchanged = not any(source_diff.values())
    if not source_unchanged:
        suite_errors.append(
            "synaptic source tree changed while the benchmark was running"
        )

    repositories_after = {
        "omnifuse": _git_state(REPOSITORY_ROOT),
        "synaptic_memory": _git_state(synaptic_repo),
    }
    repository_states_unchanged = repositories_before == repositories_after
    if not repository_states_unchanged:
        suite_errors.append(
            "repository Git state changed while the benchmark was running"
        )

    sources_after: dict[str, object] | None = None
    try:
        sources_after = _benchmark_sources(synaptic_repo)
        assert_unchanged(
            "benchmark source fingerprints",
            sources_before,
            sources_after,
        )
    except ProvenanceError as exc:
        suite_errors.append(str(exc))
    for spec in DATASETS:
        try:
            current_input = _dataset_input(
                synaptic_repo,
                synaptic_repo / "tests" / "benchmark" / "data" / spec.filename,
            )[0]
            assert_unchanged(
                f"dataset input {spec.name}", inputs[spec.id][0], current_input
            )
        except (OSError, ValueError, RuntimeError) as exc:
            suite_errors.append(str(exc))
    if doctor is not None:
        try:
            verify_doctor_manifest(doctor)
        except ProvenanceError as exc:
            suite_errors.append(str(exc))

    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    if not rows:
        suite_errors.append("benchmark produced no dataset rows")
    if failed:
        suite_errors.append(
            f"{len(failed)} of {len(DATASETS)} required datasets failed"
        )

    wins = {"omnifuse": 0, "synaptic_memory": 0, "tie": 0}
    for row in successful:
        winner = str(row["winner"])
        wins[winner] += 1

    return {
        "schema": "omnifuse.eval.tracked_public_comparison",
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not suite_errors else "error",
        "errors": suite_errors,
        "scope": {
            "group": "tracked_public",
            "required_dataset_count": len(DATASETS),
            "download_generated_extended_datasets_included": False,
            "selection": "the eight public JSON files tracked by the selected checkout",
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "dont_write_bytecode": sys.dont_write_bytecode,
        },
        "repositories": repositories_before,
        "repositories_after": repositories_after,
        "provenance": sources_before,
        "provenance_after": sources_after,
        "doctor_manifest": doctor,
        "evaluation_contract": {
            "metric": "MRR@10",
            "k": K,
            "candidate_limit": CANDIDATE_LIMIT,
            "same_k_for_both_systems": True,
            "same_candidate_limit_for_both_systems": True,
            "dedupe": DEDUPE_POLICY,
            "score_after_dedupe": True,
            "retrieval": "single-shot; no LLM, embedder, or reranker",
            "native_runner_lifecycle_caveat": NATIVE_LIFECYCLE_CAVEAT,
        },
        "source_integrity": {
            "synaptic_source_root": str(source_root),
            "before": _snapshot_record(source_before),
            "after": _snapshot_record(source_after),
            "diff": source_diff,
            "unchanged": source_unchanged,
            "repository_states_unchanged": repository_states_unchanged,
        },
        "datasets": rows,
        "summary": {
            "required": len(DATASETS),
            "completed": len(successful),
            "failed": len(failed),
            "wins": wins,
        },
    }


def _print_report(report: dict[str, object]) -> None:
    print(f"{'dataset':22}{'synaptic':>12}{'OmniFuse':>12}  winner/status")
    print("-" * 68)
    for row in report["datasets"]:  # type: ignore[union-attr]
        if row["status"] != "ok":
            print(f"{row['name']:22}{'error':>12}{'error':>12}  {row['error']}")
            continue
        systems = row["systems"]
        synaptic = systems["synaptic_memory"]["mrr_at_10"]
        omnifuse = systems["omnifuse"]["mrr_at_10"]
        print(f"{row['name']:22}{synaptic:>12.4f}{omnifuse:>12.4f}  {row['winner']}")
    print("-" * 68)
    summary = report["summary"]  # type: ignore[assignment]
    print(
        f"completed {summary['completed']}/{summary['required']}; "
        f"wins={summary['wins']}; status={report['status']}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synaptic-repo",
        required=True,
        type=Path,
        help="synaptic-memory Git checkout whose tracked public data and src are used",
    )
    parser.add_argument(
        "--out", type=Path, help="write a new immutable exact-float JSON report"
    )
    parser.add_argument(
        "--doctor-manifest",
        type=Path,
        help="strict eval/bench.py doctor JSON; required with --out",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.out is not None and args.doctor_manifest is None:
        parser.error("--doctor-manifest is required when --out is used")
    synaptic_repo = args.synaptic_repo.resolve()
    if not synaptic_repo.is_dir():
        parser.error(f"synaptic checkout not found: {synaptic_repo}")

    try:
        if args.out is not None:
            ensure_output_absent(args.out)
        report = execute_benchmark(synaptic_repo, doctor_manifest=args.doctor_manifest)
    except (OSError, ValueError, ImportError, AttributeError, RuntimeError) as exc:
        parser.error(str(exc))
    _print_report(report)
    if args.out is not None:
        try:
            _atomic_write_json(args.out, report)
        except ProvenanceError as exc:
            parser.error(str(exc))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
