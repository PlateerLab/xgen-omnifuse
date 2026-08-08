"""Adversarial ``idf_pow`` ablation with a same-pass synaptic re-run.

The 17 public-IR targets use the exact corpus, queries, qrels, scorer, and
candidate cutoff for synaptic-memory and OmniFuse.  OmniFuse is evaluated at
the control, claimed band edges, shipped midpoint, and former default:
1.0/1.1/1.2/1.3/1.5.

    python eval/idf_pow_bench.py --synaptic-repo /path/to/synaptic-memory
    python eval/idf_pow_bench.py --synaptic-repo PATH --doctor-manifest DOCTOR.json --out RESULT.json
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import platform
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
for import_root in (REPO_ROOT / "src", EVAL_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import public_bench as public_bench_driver  # noqa: E402
from metrics import BenchmarkResult  # noqa: E402
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


# Eight tracked public sets plus nine upstream-declared/download-only sets.
DATASETS = [
    ("HotPotQA-24", "hotpotqa_24.json"),
    ("HotPotQA-200", "hotpotqa.json"),
    ("Allganize RAG-ko", "allganize_rag_ko.json"),
    ("Allganize RAG-Eval", "allganize_rag_eval.json"),
    ("PublicHealthQA", "publichealthqa_ko.json"),
    ("AutoRAG", "autorag_retrieval.json"),
    ("KLUE-MRC", "klue_mrc.json"),
    ("Ko-StrategyQA", "ko_strategyqa.json"),
    ("2Wiki-dev", "2wiki_dev.json"),
    ("MuSiQue-dev", "musique_dev.json"),
    ("TREC-COVID", "trec_covid.json"),
    ("SciFact", "scifact.json"),
    ("XPQA-ko", "xpqa_ko.json"),
    ("NFCorpus", "nfcorpus.json"),
    ("MIRACL-ko", "miracl_retrieval_ko.json"),
    ("FiQA", "fiqa.json"),
    ("MultiLongDoc-ko", "multilongdoc_ko.json"),
]
DOCTOR_TARGET_IDS = {
    "hotpotqa_24.json": "hotpotqa_24",
    "hotpotqa.json": "hotpotqa_200",
    "allganize_rag_ko.json": "allganize_rag_ko",
    "allganize_rag_eval.json": "allganize_rag_eval",
    "publichealthqa_ko.json": "publichealthqa_ko",
    "autorag_retrieval.json": "autorag_retrieval",
    "klue_mrc.json": "klue_mrc",
    "ko_strategyqa.json": "ko_strategyqa",
    "2wiki_dev.json": "2wiki_dev",
    "musique_dev.json": "musique_dev",
    "trec_covid.json": "trec_covid",
    "scifact.json": "scifact",
    "xpqa_ko.json": "xpqa_ko",
    "nfcorpus.json": "nfcorpus",
    "miracl_retrieval_ko.json": "miracl_retrieval_ko",
    "fiqa.json": "fiqa",
    "multilongdoc_ko.json": "multilongdoc_ko",
}
K = 10
ARMS = (1.0, 1.1, 1.2, 1.3, 1.5)
SHIPPED_ARM = 1.2

OmniRunner = Callable[[Path, float], float]
SynapticRunner = Callable[[Path, Path, str], Awaitable[float]]


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce a reproducible result."""


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _file_fingerprint(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    try:
        return file_fingerprint(path, display_path=display_path)
    except ProvenanceError as exc:
        raise BenchmarkError(str(exc)) from exc


def _repository_state(repo: Path) -> dict[str, Any]:
    try:
        return repository_fingerprint(repo)
    except ProvenanceError as exc:
        raise BenchmarkError(str(exc)) from exc


def _source_provenance(synaptic_repo: Path) -> dict[str, Any]:
    harness = Path(__file__).resolve()
    driver = Path(
        inspect.getsourcefile(public_bench_driver.synaptic_mrr) or ""
    ).resolve()
    scorer = Path(inspect.getsourcefile(BenchmarkResult) or "").resolve()
    expected_driver = (EVAL_ROOT / "public_bench.py").resolve()
    expected_scorer = (EVAL_ROOT / "metrics.py").resolve()
    build_source = Path(inspect.getsourcefile(build_inmemory) or "").resolve()
    expected_build_source = (REPO_ROOT / "src" / "omnifuse" / "facade.py").resolve()
    if driver != expected_driver:
        raise BenchmarkError(
            f"public_bench.synaptic_mrr loaded from {driver}, expected {expected_driver}"
        )
    if scorer != expected_scorer:
        raise BenchmarkError(
            f"BenchmarkResult loaded from {scorer}, expected {expected_scorer}"
        )
    if build_source != expected_build_source:
        raise BenchmarkError(
            f"omnifuse.build_inmemory loaded from {build_source}, "
            f"expected {expected_build_source}"
        )
    try:
        omnifuse_imports = public_bench_driver._omnifuse_import_provenance()
    except RuntimeError as exc:
        raise BenchmarkError(str(exc)) from exc
    local_scorer = _file_fingerprint(scorer, display_path="eval/metrics.py")
    upstream_scorer = _file_fingerprint(
        synaptic_repo / public_bench_driver.SYNAPTIC_SCORER_RELATIVE,
        display_path=public_bench_driver.SYNAPTIC_SCORER_RELATIVE.as_posix(),
    )
    if local_scorer["sha256"] != upstream_scorer["sha256"]:
        raise BenchmarkError(
            "eval/metrics.py is not byte-identical to the selected synaptic-memory "
            "tests/benchmark/metrics.py"
        )
    return {
        "harness": _file_fingerprint(harness, display_path="eval/idf_pow_bench.py"),
        "provenance_helper": _file_fingerprint(
            EVAL_ROOT / "provenance.py", display_path="eval/provenance.py"
        ),
        "synaptic_driver_wrapper": _file_fingerprint(
            driver, display_path="eval/public_bench.py"
        ),
        "synaptic_native_driver": _file_fingerprint(
            synaptic_repo / public_bench_driver.SYNAPTIC_DRIVER_RELATIVE,
            display_path=public_bench_driver.SYNAPTIC_DRIVER_RELATIVE.as_posix(),
        ),
        "omnifuse_imports": omnifuse_imports,
        "scorer": {
            "active": local_scorer,
            "synaptic_checkout_copy": upstream_scorer,
            "byte_identical": True,
        },
    }


def _assert_unchanged(label: str, before: Any, after: Any) -> None:
    try:
        assert_unchanged(label, before, after)
    except ProvenanceError as exc:
        raise BenchmarkError(str(exc)) from exc


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_once(path, payload)


def _dataset_inputs(
    synaptic_repo: Path, requested: set[str] | None, emit: Callable[[str], None]
) -> list[dict[str, Any]]:
    known = {name for name, _ in DATASETS}
    if requested is not None:
        unknown = sorted(requested - known)
        if unknown:
            raise BenchmarkError(f"unknown dataset name(s): {', '.join(unknown)}")

    data_root = synaptic_repo / "tests" / "benchmark" / "data"
    inputs: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, filename in DATASETS:
        if requested is not None and name not in requested:
            continue
        path = data_root / filename
        if not path.is_file():
            emit(f"{name:20}{'(missing)':>10}")
            missing.append(name)
            continue
        inputs.append(
            {
                "name": name,
                "filename": filename,
                "path": path.resolve(),
                "fingerprint": _file_fingerprint(
                    path, display_path=f"tests/benchmark/data/{filename}"
                ),
            }
        )
    if missing:
        raise BenchmarkError(
            "required dataset file(s) missing for selected scope: " + ", ".join(missing)
        )
    expected = known if requested is None else requested
    if {item["name"] for item in inputs} != expected:
        raise BenchmarkError("selected dataset scope was not resolved exactly")
    return inputs


def _doctor_provenance(
    doctor_path: Path | None, inputs: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    if doctor_path is None:
        return None, {}

    manifest_inputs = [
        {
            "name": dataset["name"],
            "target_id": DOCTOR_TARGET_IDS[dataset["filename"]],
            "path": dataset["fingerprint"]["path"],
            "sha256": dataset["fingerprint"]["sha256"],
            "bytes": dataset["fingerprint"]["bytes"],
        }
        for dataset in inputs
    ]
    try:
        return load_doctor_manifest(doctor_path, manifest_inputs)
    except ProvenanceError as exc:
        raise BenchmarkError(str(exc)) from exc


def _load_public(
    path: Path,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, set[str]]]]:
    return public_bench_driver.parse_public(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _omni_mrr_loaded(
    corpus: list[tuple[str, str, str]],
    queries: list[tuple[str, str, set[str]]],
    idf_pow: float,
) -> float:
    chunks = [
        {"id": doc_id, "title": title, "text": text} for doc_id, title, text in corpus
    ]
    graph = build_inmemory([], [], chunks, vector_kwargs={"idf_pow": idf_pow})
    assert graph.vector._bm25.idf, "empty index"
    benchmark = BenchmarkResult()
    for query_id, text, relevant in queries:
        ranked: list[str] = []
        seen: set[str] = set()
        for chunk, _ in graph.retrieve(text, limit=K * 2):
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
    return benchmark.summary()["mrr"]


def omni_mrr(path: Path, idf_pow: float) -> float:
    corpus, queries = _load_public(path)
    return _omni_mrr_loaded(corpus, queries, idf_pow)


def run_benchmark(
    synaptic_repo: Path,
    inputs: Sequence[dict[str, Any]],
    *,
    doctor_links: dict[str, dict[str, Any]] | None = None,
    omni_runner: OmniRunner = omni_mrr,
    synaptic_runner: SynapticRunner = public_bench_driver.synaptic_mrr,
    emit: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for dataset in inputs:
        name = dataset["name"]
        path = dataset["path"]
        if omni_runner is omni_mrr:
            corpus, queries = _load_public(path)
            arms = {arm: float(_omni_mrr_loaded(corpus, queries, arm)) for arm in ARMS}
        else:
            arms = {arm: float(omni_runner(path, arm)) for arm in ARMS}
        synaptic = float(asyncio.run(synaptic_runner(synaptic_repo, path, name)))
        winners = []
        for arm in ARMS:
            if arms[arm] > synaptic:
                outcome = "wins"
            elif arms[arm] < synaptic:
                outcome = "LOSES"
            else:
                outcome = "ties"
            winners.append(f"p={arm} {outcome}")
        row: dict[str, Any] = {
            "input": dataset["fingerprint"],
            "synaptic": synaptic,
            **{f"omnifuse_idf_pow_{arm}": arms[arm] for arm in ARMS},
            "delta_1.5_minus_1.0": arms[1.5] - arms[1.0],
            "delta_shipped_minus_synaptic": arms[SHIPPED_ARM] - synaptic,
            **{f"p{arm:.1f}_beats_synaptic": arms[arm] > synaptic for arm in ARMS},
            "shipped_1.2_beats_synaptic": arms[SHIPPED_ARM] > synaptic,
        }
        if doctor_links:
            row["doctor_target_id"] = doctor_links[name]["target_id"]
        rows[name] = row
        arm_values = "".join(f"{arms[arm]:>12.4f}" for arm in ARMS)
        emit(
            f"{name:20}{synaptic:>10.4f}{arm_values}"
            f"{arms[SHIPPED_ARM] - synaptic:>+10.4f}  {', '.join(winners)}"
        )
    if not rows:
        raise BenchmarkError("no requested dataset was available")
    return rows


def execute_benchmark(
    *,
    synaptic_repo: Path,
    requested: set[str] | None = None,
    doctor_manifest: Path | None = None,
    omni_runner: OmniRunner = omni_mrr,
    synaptic_runner: SynapticRunner = public_bench_driver.synaptic_mrr,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    synaptic_repo = synaptic_repo.resolve()
    repositories = {
        "omnifuse": _repository_state(REPO_ROOT),
        "synaptic_memory": _repository_state(synaptic_repo),
    }
    sources = _source_provenance(synaptic_repo)
    if synaptic_runner is public_bench_driver.synaptic_mrr:
        try:
            public_bench_driver.preflight_synaptic_runner(synaptic_repo)
        except (FileNotFoundError, ImportError, AttributeError, RuntimeError) as exc:
            raise BenchmarkError(
                f"synaptic native runner preflight failed: {exc}"
            ) from exc
    inputs = _dataset_inputs(synaptic_repo, requested, emit)
    doctor, doctor_links = _doctor_provenance(doctor_manifest, inputs)
    if doctor is not None:
        verify_doctor_runtime(
            doctor,
            omnifuse_repository=repositories["omnifuse"],
            synaptic_repository=repositories["synaptic_memory"],
            omnifuse_scorer=sources["scorer"]["active"],
            synaptic_scorer=sources["scorer"]["synaptic_checkout_copy"],
        )

    emit("scored by eval/metrics.py (synaptic's own), MRR@10, k=10; same-pass rerun.\n")
    arm_header = "".join(f"omni p={arm:.1f}".rjust(12) for arm in ARMS)
    emit(f"{'dataset':20}{'synaptic':>10}{arm_header}{'ship-syn':>10}  who wins")
    emit("-" * (42 + 12 * len(ARMS)))
    started = time.monotonic()
    rows = run_benchmark(
        synaptic_repo,
        inputs,
        doctor_links=doctor_links,
        omni_runner=omni_runner,
        synaptic_runner=synaptic_runner,
        emit=emit,
    )
    elapsed = time.monotonic() - started

    repositories_after = {
        "omnifuse": _repository_state(REPO_ROOT),
        "synaptic_memory": _repository_state(synaptic_repo),
    }
    sources_after = _source_provenance(synaptic_repo)
    _assert_unchanged("repositories", repositories, repositories_after)
    _assert_unchanged("benchmark sources", sources, sources_after)
    for dataset in inputs:
        current = _file_fingerprint(
            dataset["path"], display_path=dataset["fingerprint"]["path"]
        )
        _assert_unchanged(f"dataset {dataset['name']}", dataset["fingerprint"], current)
    if doctor is not None:
        try:
            verify_doctor_manifest(doctor)
        except ProvenanceError as exc:
            raise BenchmarkError(str(exc)) from exc

    net = sum(row["delta_1.5_minus_1.0"] for row in rows.values())
    win_counts = {
        arm: sum(row[f"p{arm:.1f}_beats_synaptic"] for row in rows.values())
        for arm in ARMS
    }
    emit("-" * (42 + 12 * len(ARMS)))
    emit(
        f"net(p=1.5 - p=1.0) over {len(rows)} datasets: {net:+.4f}  "
        f"(mean {net / len(rows):+.5f})"
    )
    emit(
        "beats synaptic:  "
        + "   ".join(
            f"p={arm:.1f}{' (shipped)' if arm == SHIPPED_ARM else ''} -> "
            f"{win_counts[arm]}/{len(rows)}"
            for arm in ARMS
        )
    )
    emit(f"[{elapsed:.0f}s]")

    return {
        "schema": "omnifuse.eval.idf_pow",
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "idf_pow ablation with synaptic re-run in the same pass",
        "scorer": "eval/metrics.py (synaptic's own), MRR@10",
        "synaptic_driver": (
            "synaptic's own eval.run_all.run_public_dataset, embedder=None, reranker=None"
        ),
        "k": K,
        "arms": list(ARMS),
        "shipped_arm": SHIPPED_ARM,
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "repositories": repositories,
        "repositories_after": repositories_after,
        "sources": sources,
        "sources_after": sources_after,
        "doctor_manifest": doctor,
        "datasets": rows,
        "net_1.5_minus_1.0": net,
        "beats_synaptic": {
            f"idf_pow={arm:.1f}{' (shipped)' if arm == SHIPPED_ARM else ''}": (
                f"{win_counts[arm]}/{len(rows)}"
            )
            for arm in ARMS
        },
        "elapsed_seconds": elapsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synaptic-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write a new immutable JSON result")
    parser.add_argument(
        "--datasets", help="comma-separated exact dataset names (default: all 17)"
    )
    parser.add_argument(
        "--doctor-manifest",
        type=Path,
        help="strict eval/bench.py doctor JSON; required with --out",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.out is not None and args.doctor_manifest is None:
        parser.error("--doctor-manifest is required when --out is used")
    requested = None
    if args.datasets is not None:
        requested = {name.strip() for name in args.datasets.split(",") if name.strip()}
    try:
        if args.out is not None:
            ensure_output_absent(args.out)
        report = execute_benchmark(
            synaptic_repo=args.synaptic_repo,
            requested=requested,
            doctor_manifest=args.doctor_manifest,
        )
        if args.out is not None:
            _atomic_write_json(args.out, report)
    except (BenchmarkError, ProvenanceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
