"""Reproducibility doctor for the OmniFuse/Synaptic benchmark inputs.

This command does not run retrieval.  It records whether every declared target is
available and fingerprints the exact data, scorer, repositories, and runtime that
would be used by a later benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from provenance import (  # noqa: E402
    DOCTOR_SCHEMA,
    DOCTOR_SCHEMA_VERSION,
    STRICT_PUBLIC_TARGET_CONTRACTS,
    STRICT_PUBLIC_TARGET_IDS,
    ProvenanceError,
    ensure_output_absent,
    repository_fingerprint,
    write_json_once,
)


STATUSES = ("ok", "skipped_missing_external", "skipped_private", "error")


@dataclass(frozen=True)
class TargetSpec:
    id: str
    name: str
    group: str
    source: str
    files: tuple[tuple[str, str], ...]
    missing_status: str
    strict_public: bool


TARGETS = (
    TargetSpec(
        "finreg_single",
        "finreg single-hop",
        "finreg",
        "omnifuse",
        (
            ("corpus", "eval/data/finreg/raw.jsonl"),
            ("queries", "eval/data/queries/finreg.json"),
        ),
        "error",
        True,
    ),
    TargetSpec(
        "finreg_multi",
        "finreg multi-hop",
        "finreg",
        "omnifuse",
        (
            ("corpus", "eval/data/finreg/raw.jsonl"),
            ("queries", "eval/data/queries/finreg_multihop.json"),
        ),
        "error",
        True,
    ),
    TargetSpec(
        "hotpotqa_24",
        "HotPotQA-24",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/hotpotqa_24.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "hotpotqa_200",
        "HotPotQA-200",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/hotpotqa.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "allganize_rag_ko",
        "Allganize RAG-ko",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/allganize_rag_ko.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "allganize_rag_eval",
        "Allganize RAG-Eval",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/allganize_rag_eval.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "publichealthqa_ko",
        "PublicHealthQA",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/publichealthqa_ko.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "autorag_retrieval",
        "AutoRAG",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/autorag_retrieval.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "klue_mrc",
        "KLUE-MRC",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/klue_mrc.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "ko_strategyqa",
        "Ko-StrategyQA",
        "tracked_public",
        "synaptic",
        (("dataset", "tests/benchmark/data/ko_strategyqa.json"),),
        "error",
        True,
    ),
    TargetSpec(
        "2wiki_dev",
        "2Wiki-dev",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/2wiki_dev.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "musique_dev",
        "MuSiQue-dev",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/musique_dev.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "trec_covid",
        "TREC-COVID",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/trec_covid.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "scifact",
        "SciFact",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/scifact.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "xpqa_ko",
        "XPQA-ko",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/xpqa_ko.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "nfcorpus",
        "NFCorpus",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/nfcorpus.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "miracl_retrieval_ko",
        "MIRACL-retrieval-ko",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/miracl_retrieval_ko.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "fiqa",
        "FiQA",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/fiqa.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "multilongdoc_ko",
        "MultiLongDoc-ko",
        "download_only_extended",
        "synaptic",
        (("dataset", "tests/benchmark/data/multilongdoc_ko.json"),),
        "skipped_missing_external",
        True,
    ),
    TargetSpec(
        "enterprise_scenario",
        "enterprise scenario",
        "enterprise",
        "synaptic",
        (("dataset", "tests/benchmark/data/enterprise_scenario.json"),),
        "error",
        False,
    ),
    TargetSpec(
        "qa_combined",
        "QA combined performance fixture",
        "qa",
        "synaptic",
        (
            ("wikipedia", "tests/qa/data/wikipedia_ko_tech.json"),
            ("commits", "tests/qa/data/github_commits.json"),
            ("issues", "tests/qa/data/github_issues.json"),
        ),
        "error",
        False,
    ),
    TargetSpec(
        "kra_golden",
        "KRA private golden",
        "private",
        "private",
        (("dataset", ""),),
        "skipped_private",
        False,
    ),
)

if (
    frozenset(spec.id for spec in TARGETS if spec.strict_public)
    != STRICT_PUBLIC_TARGET_IDS
):
    raise RuntimeError(
        "doctor target declarations do not match the strict-public contract"
    )
if {
    spec.id: (spec.source, spec.files) for spec in TARGETS if spec.strict_public
} != STRICT_PUBLIC_TARGET_CONTRACTS:
    raise RuntimeError(
        "doctor target file declarations do not match the strict-public contract"
    )


def _nonempty(value: object) -> bool:
    return isinstance(value, (dict, list)) and bool(value)


def _validate_json_input(path: Path, *, group: str, role: str) -> dict[str, object]:
    try:
        if path.suffix == ".jsonl":
            rows = 0
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or not value.get(
                        "doc_id", value.get("id")
                    ):
                        raise ValueError(
                            f"line {line_number} is not a document object with an id"
                        )
                    rows += 1
            if not rows:
                raise ValueError("JSONL contains no documents")
            return {"status": "ok", "format": "jsonl", "documents": rows}

        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if group == "qa":
            if not isinstance(value, list) or not value:
                raise ValueError("QA source must be a non-empty JSON array")
            if not all(isinstance(row, dict) for row in value):
                raise ValueError("QA source rows must be JSON objects")
            return {
                "status": "ok",
                "format": "qa-source-json",
                "records": len(value),
            }
        if not isinstance(value, dict) or not value:
            raise ValueError("top-level JSON must be a non-empty object")

        if group in {"tracked_public", "download_only_extended"}:
            corpus = value.get("corpus", value.get("documents"))
            queries = value.get("queries")
            qrels = value.get("relevant_docs", value.get("qrels"))
            if not _nonempty(corpus) or not _nonempty(queries):
                raise ValueError(
                    "public dataset requires non-empty corpus/documents and queries"
                )

            query_count = 0
            if isinstance(queries, dict):
                if not isinstance(qrels, dict) or not qrels:
                    raise ValueError(
                        "mapping-style queries require non-empty qrels/relevant_docs"
                    )
                for query_id, text in queries.items():
                    relevant = qrels.get(query_id, {})
                    if text and _nonempty(relevant):
                        query_count += 1
            else:
                for query in queries:
                    if not isinstance(query, dict):
                        continue
                    text = query.get("query", query.get("question"))
                    relevant = query.get(
                        "relevant_docs",
                        query.get("answer_ids", query.get("positive_doc_ids")),
                    )
                    if text and _nonempty(relevant):
                        query_count += 1
            if not query_count:
                raise ValueError(
                    "public dataset has no parser-compatible query with relevance labels"
                )
            return {
                "status": "ok",
                "format": "public-ir-json",
                "documents": len(corpus),
                "queries": query_count,
            }

        if group == "enterprise":
            knowledge = value.get("knowledge_sources")
            queries = value.get("evaluation_queries")
            if not _nonempty(knowledge) or not _nonempty(queries):
                raise ValueError(
                    "enterprise dataset requires non-empty knowledge_sources and evaluation_queries"
                )
            return {
                "status": "ok",
                "format": "enterprise-scenario-json",
                "documents": len(knowledge),
                "queries": len(queries),
            }

        if role == "queries":
            queries = value.get("queries")
            if not _nonempty(queries):
                raise ValueError("query file requires a non-empty queries collection")
            return {"status": "ok", "format": "query-json", "queries": len(queries)}

        return {"status": "ok", "format": "json"}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return {
            "status": "error",
            "format": "jsonl" if path.suffix == ".jsonl" else "json",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _file_fingerprint(
    path: Path | None,
    display_path: str | None,
    *,
    group: str | None = None,
    role: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "path": display_path,
        "status": "missing",
        "sha256": None,
        "bytes": None,
    }
    if path is None or not path.is_file():
        return record

    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record.update(status="ok", sha256=digest.hexdigest(), bytes=size)
    if group is not None and role is not None:
        validation = _validate_json_input(path, group=group, role=role)
        record["validation"] = validation
        if validation["status"] != "ok":
            record["status"] = "error"
            record["error"] = validation["error"]
    return record


def _git_repository(path: Path | None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.resolve()) if path is not None else None,
        "status": "missing",
        "sha": None,
        "dirty": None,
    }
    if path is None or not path.is_dir():
        return record
    try:
        fingerprint = repository_fingerprint(path)
    except ProvenanceError as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        return record
    record.update(fingerprint, status="ok")
    return record


def _target_fingerprint(
    files: list[dict[str, object]],
) -> tuple[str | None, int | None, str | None]:
    if any(item["status"] != "ok" for item in files):
        return None, None, None
    if len(files) == 1:
        return str(files[0]["sha256"]), int(files[0]["bytes"]), "file"

    content = [
        {
            "role": item["role"],
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
        }
        for item in files
    ]
    encoded = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        hashlib.sha256(encoded).hexdigest(),
        sum(int(item["bytes"]) for item in files),
        "file-manifest-v1",
    )


def _inspect_target(
    spec: TargetSpec,
    *,
    omnifuse_repo: Path,
    synaptic_repo: Path | None,
    kra_golden: Path | None,
) -> dict[str, object]:
    roots = {"omnifuse": omnifuse_repo, "synaptic": synaptic_repo, "private": None}
    root = roots[spec.source]
    files: list[dict[str, object]] = []
    for role, relative in spec.files:
        if spec.source == "private":
            path = kra_golden
            display_path = str(kra_golden.resolve()) if kra_golden is not None else None
        else:
            path = root / relative if root is not None else None
            display_path = relative
        item = _file_fingerprint(path, display_path, group=spec.group, role=role)
        item["role"] = role
        files.append(item)

    if any(item["status"] == "error" for item in files):
        status = "error"
    elif all(item["status"] == "ok" for item in files):
        status = "ok"
    elif spec.source == "synaptic" and (
        synaptic_repo is None or not synaptic_repo.is_dir()
    ):
        status = "skipped_missing_external"
    else:
        status = spec.missing_status

    sha256, size, sha256_kind = _target_fingerprint(files)
    return {
        "id": spec.id,
        "name": spec.name,
        "group": spec.group,
        "source": spec.source,
        "strict_public": spec.strict_public,
        "status": status,
        "sha256": sha256,
        "sha256_kind": sha256_kind,
        "bytes": size,
        "files": files,
    }


def build_manifest(
    *,
    omnifuse_repo: Path,
    synaptic_repo: Path | None,
    kra_golden: Path | None,
    strict_public: bool,
    allow_dirty: bool = False,
) -> dict[str, object]:
    omnifuse_repo = omnifuse_repo.resolve()
    synaptic_repo = synaptic_repo.resolve() if synaptic_repo is not None else None
    kra_golden = kra_golden.resolve() if kra_golden is not None else None

    targets = [
        _inspect_target(
            spec,
            omnifuse_repo=omnifuse_repo,
            synaptic_repo=synaptic_repo,
            kra_golden=kra_golden,
        )
        for spec in TARGETS
    ]

    omni_scorer = _file_fingerprint(
        omnifuse_repo / "eval/metrics.py", "eval/metrics.py"
    )
    syn_scorer = _file_fingerprint(
        synaptic_repo / "tests/benchmark/metrics.py"
        if synaptic_repo is not None
        else None,
        "tests/benchmark/metrics.py",
    )
    scorers_equal = (
        omni_scorer["sha256"] == syn_scorer["sha256"]
        if omni_scorer["status"] == syn_scorer["status"] == "ok"
        else None
    )

    counts = {
        status: sum(target["status"] == status for target in targets)
        for status in STATUSES
    }
    public_targets = [target for target in targets if target["strict_public"]]
    incomplete_public = [
        str(target["id"]) for target in public_targets if target["status"] != "ok"
    ]
    repositories = {
        "omnifuse": _git_repository(omnifuse_repo),
        "synaptic_memory": _git_repository(synaptic_repo),
    }
    blockers: list[dict[str, object]] = [
        {"kind": "target", "id": target["id"], "status": target["status"]}
        for target in public_targets
        if target["status"] != "ok"
    ]
    if scorers_equal is not True:
        blockers.append({"kind": "scorer", "equal": scorers_equal})
    for name, repository in repositories.items():
        if repository["status"] != "ok":
            blockers.append(
                {
                    "kind": "repository",
                    "repository": name,
                    "status": repository["status"],
                }
            )
        elif repository["dirty"] and not allow_dirty:
            blockers.append(
                {"kind": "repository", "repository": name, "status": "dirty"}
            )

    return {
        "schema": DOCTOR_SCHEMA,
        "schema_version": DOCTOR_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
        "repositories": {
            **repositories,
            "captured_before_output_write": True,
            "allow_dirty": allow_dirty,
        },
        "scorer": {
            "omnifuse": omni_scorer,
            "synaptic_memory": syn_scorer,
            "equal": scorers_equal,
        },
        "targets": targets,
        "summary": {
            "total_targets": len(targets),
            "ok_targets": counts["ok"],
            "status_counts": counts,
            "incomplete_targets": [
                str(target["id"]) for target in targets if target["status"] != "ok"
            ],
            "public": {
                "total_targets": len(public_targets),
                "ok_targets": sum(
                    target["status"] == "ok" for target in public_targets
                ),
                "incomplete_targets": incomplete_public,
            },
        },
        "strict_public": {
            "enabled": strict_public,
            "would_pass": not blockers,
            "passed": not blockers if strict_public else None,
            "blockers": blockers,
        },
    }


def atomic_write_json(path: Path, value: object) -> None:
    write_json_once(path, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniFuse benchmark orchestration")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="fingerprint benchmark inputs without running retrieval"
    )
    doctor.add_argument(
        "--synaptic-repo", type=Path, default=os.environ.get("SYNAPTIC_REPO")
    )
    doctor.add_argument(
        "--kra-golden", type=Path, default=os.environ.get("OMNIFUSE_KRA_GOLDEN")
    )
    doctor.add_argument(
        "--out", type=Path, required=True, help="write a new immutable doctor JSON"
    )
    doctor.add_argument(
        "--strict-public",
        action="store_true",
        help="exit 2 unless all 19 public targets validate, scorers match, and repositories are clean",
    )
    doctor.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record dirty repositories but do not make them a strict-public blocker",
    )
    doctor.add_argument(
        "--omnifuse-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        ensure_output_absent(args.out)
    except ProvenanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    manifest = build_manifest(
        omnifuse_repo=args.omnifuse_repo,
        synaptic_repo=args.synaptic_repo,
        kra_golden=args.kra_golden,
        strict_public=args.strict_public,
        allow_dirty=args.allow_dirty,
    )
    try:
        atomic_write_json(args.out, manifest)
    except ProvenanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = manifest["summary"]
    public = summary["public"]
    print(
        f"doctor: {summary['ok_targets']}/{summary['total_targets']} targets ready; "
        f"public {public['ok_targets']}/{public['total_targets']}; wrote {args.out.resolve()}"
    )
    if args.strict_public and not manifest["strict_public"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
