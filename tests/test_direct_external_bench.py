import asyncio
import json
import math
import pathlib
import random
import subprocess
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))

import direct_external_bench as direct  # noqa: E402
import provenance  # noqa: E402


WORKER_RUN_ID = "00000000000040008000000000000001"


def _worker_identity(run_id: str = WORKER_RUN_ID, pid: int = 123) -> dict:
    return {
        "schema": provenance.WORKER_IDENTITY_SCHEMA,
        "schema_version": provenance.WORKER_IDENTITY_SCHEMA_VERSION,
        "worker_run_id": run_id,
        "worker_pid": pid,
        "capture_phase": provenance.WORKER_IDENTITY_CAPTURE_PHASE,
    }


def _summary(total: int = 1, *, mrr: float = 1.0) -> dict[str, float]:
    return {
        "mean_precision@k": 1.0,
        "mean_recall@k": 1.0,
        "mean_f1@k": 1.0,
        "mrr": mrr,
        "mean_ndcg@k": 1.0,
        "mean_search_time_ms": 0.25,
        "total_queries": total,
    }


class _Benchmark:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.queries = rows or []

    def add(
        self,
        query_id,
        query,
        retrieved,
        relevant,
        *,
        k,
        description,
        search_time_ms,
    ) -> None:
        reciprocal_rank = next(
            (
                1.0 / index
                for index, document_id in enumerate(retrieved, 1)
                if document_id in relevant
            ),
            0.0,
        )
        top_k = list(retrieved[:k])
        hits = len(set(top_k) & relevant)
        precision = hits / len(top_k) if top_k else 0.0
        recall = hits / len(relevant) if relevant else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        self.queries.append(
            {
                "query_id": query_id,
                "query": query,
                "retrieved_top_k": top_k,
                "relevant": sorted(relevant),
                "precision@k": precision,
                "recall@k": recall,
                "f1@k": f1,
                "mrr": reciprocal_rank,
                "ndcg@k": 1.0 if hits else 0.0,
                "search_time_ms": search_time_ms,
                "description": description,
            },
        )

    def summary(self) -> dict[str, float]:
        count = len(self.queries)
        if not count:
            return {}
        return {
            "mean_precision@k": sum(row["precision@k"] for row in self.queries) / count,
            "mean_recall@k": sum(row["recall@k"] for row in self.queries) / count,
            "mean_f1@k": sum(row["f1@k"] for row in self.queries) / count,
            "mrr": sum(row["mrr"] for row in self.queries) / count,
            "mean_ndcg@k": sum(row["ndcg@k"] for row in self.queries) / count,
            "mean_search_time_ms": sum(row["search_time_ms"] for row in self.queries)
            / count,
            "total_queries": count,
        }


def _process_record(
    repo: pathlib.Path, python: pathlib.Path, *, phase: str
) -> dict[str, object]:
    roots = direct._worker_pythonpath_entries(repo)
    normalized = list(roots)
    if phase == "runtime":
        source_root = str((repo / "src").resolve())
        normalized = [source_root, *(entry for entry in roots if entry != source_root)]
    return {
        "phase": phase,
        "python_version": "3.12.10",
        "python_executable": str(python.resolve()),
        "python_prefix": str(python.resolve().parent.parent),
        "python_base_prefix": str(python.resolve().parent.parent),
        "platform": "test",
        "variables": dict(direct.REQUIRED_WORKER_ENVIRONMENT),
        "pythonpath_entries": roots,
        "flags": {
            "utf8_mode": 1,
            "no_user_site": 1,
            "dont_write_bytecode": True,
        },
        "sys_path": normalized,
        "normalized_sys_path": normalized,
        "user_site_enabled": False,
        "user_site_paths": [],
        "user_site_present_on_sys_path": False,
    }


def _query_row(
    query_id: str,
    retrieved: list[str],
    relevant: list[str],
    search_time_ms: float,
) -> dict[str, object]:
    relevant_set = set(relevant)

    def reciprocal_rank(k: int) -> float:
        return next(
            (
                1.0 / index
                for index, document_id in enumerate(retrieved[:k], 1)
                if document_id in relevant_set
            ),
            0.0,
        )

    return {
        "query_id": query_id,
        "retrieved_top_10": retrieved[: direct.K],
        "retrieved_top_20": retrieved,
        "relevant": sorted(relevant_set),
        "reciprocal_rank_at_20": reciprocal_rank(direct.CANDIDATE_LIMIT),
        "reciprocal_rank_at_10": reciprocal_rank(direct.K),
        "search_time_ms": search_time_ms,
    }


def _fixture_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    def values(row: dict[str, object]) -> tuple[float, float, float, float]:
        retrieved = list(row["retrieved_top_10"])
        relevant = set(row["relevant"])
        hits = sum(document_id in relevant for document_id in retrieved)
        precision = hits / len(retrieved) if retrieved else 0.0
        recall = hits / len(relevant) if relevant else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        actual_dcg = sum(
            (1.0 if document_id in relevant else 0.0) / math.log2(index + 2)
            for index, document_id in enumerate(retrieved)
        )
        ideal_dcg = sum(
            1.0 / math.log2(index + 2) for index in range(min(len(relevant), direct.K))
        )
        ndcg = actual_dcg / ideal_dcg if ideal_dcg else 0.0
        return precision, recall, f1, ndcg

    computed = [values(row) for row in rows]
    count = len(rows)
    return {
        "mrr_at_20": sum(float(row["reciprocal_rank_at_20"]) for row in rows) / count,
        "mrr_at_10": sum(float(row["reciprocal_rank_at_10"]) for row in rows) / count,
        "precision_at_10": sum(value[0] for value in computed) / count,
        "recall_at_10": sum(value[1] for value in computed) / count,
        "f1_at_10": sum(value[2] for value in computed) / count,
        "ndcg_at_10": sum(value[3] for value in computed) / count,
    }


def _system_result_fixture(
    rows: list[dict[str, object]], *, ingest_seconds: float
) -> dict[str, object]:
    return {
        "metrics": _fixture_metrics(rows),
        "evaluated_queries": len(rows),
        "ingest_seconds_observed": ingest_seconds,
        "mean_search_time_ms_observed": sum(
            float(row["search_time_ms"]) for row in rows
        )
        / len(rows),
        "query_ids_ordered_sha256": provenance.canonical_json_sha256(
            [row["query_id"] for row in rows]
        ),
        "queries": rows,
    }


def _direct_result_fixture(
    case: direct.DatasetCase, repo: pathlib.Path, python: pathlib.Path
) -> dict[str, object]:
    omnifuse_rows = [
        _query_row("q1", ["d1", "d2"], ["d1"], 0.25),
        _query_row("q2", ["d3", "d4"], ["d3"], 0.75),
    ]
    synaptic_rows = [
        _query_row("q1", ["d2", "d1"], ["d1"], 0.5),
        _query_row("q2", ["d4", "d3"], ["d3"], 1.5),
    ]
    systems = {
        "omnifuse": _system_result_fixture(omnifuse_rows, ingest_seconds=0.1),
        "synaptic_memory": _system_result_fixture(synaptic_rows, ingest_seconds=0.2),
    }
    if case.hotpot_supporting:
        for system_result in systems.values():
            rows = system_result["queries"]
            hits = sum(
                len(set(row["retrieved_top_10"]) & set(row["relevant"])) for row in rows
            )
            total = sum(len(row["relevant"]) for row in rows)
            system_result["supporting_facts"] = {
                "hits_at_10": hits,
                "total": total,
                "micro_recall_at_10": hits / total if total else 0.0,
            }
    winners = {}
    for metric in direct.METRIC_NAMES:
        omni_value = systems["omnifuse"]["metrics"][metric]
        synaptic_value = systems["synaptic_memory"]["metrics"][metric]
        winners[metric] = (
            "omnifuse"
            if omni_value > synaptic_value
            else "synaptic_memory"
            if synaptic_value > omni_value
            else "tie"
        )
    return {
        "case": {"id": case.id, "name": case.name, "filename": case.filename},
        "selection": {
            "seed": direct.SAMPLE_SEED,
            "original_corpus_count": 4,
            "selected_corpus_count": 4,
            "original_query_count": 2,
            "eligible_query_count_before_max_queries": 2,
            "max_queries": case.max_queries,
            "klue_corpus_sample": case.klue_corpus_sample,
            "selected_corpus_ids_ordered_sha256": provenance.canonical_json_sha256(
                ["d1", "d2", "d3", "d4"]
            ),
            "eligible_query_ids_ordered_sha256": provenance.canonical_json_sha256(
                ["q1", "q2"]
            ),
            "scored_query_count": 2,
            "scored_query_ids_ordered_sha256": provenance.canonical_json_sha256(
                ["q1", "q2"]
            ),
        },
        "document_preprocessing": {
            "text_character_limit": direct.TEXT_LIMIT,
            "input_documents": 4,
            "indexed_documents": 4,
            "skipped_empty_text_documents": 0,
            "truncated_documents": 0,
            "title_fallback_documents": 0,
            "original_text_characters": 40,
            "indexed_text_characters": 40,
            "indexed_document_ids_ordered_sha256": provenance.canonical_json_sha256(
                ["d1", "d2", "d3", "d4"]
            ),
        },
        "systems": systems,
        "winners": winners,
        "runtime": {
            "python_executable": str(python.resolve()),
            "synaptic_package": str(
                (repo / direct.UPSTREAM_PACKAGE_RELATIVE).resolve()
            ),
            "synaptic_version": direct.EXPECTED_TAG.removeprefix("v"),
            "upstream_driver": str((repo / direct.UPSTREAM_DRIVER_RELATIVE).resolve()),
            "upstream_scorer": str((repo / direct.UPSTREAM_SCORER_RELATIVE).resolve()),
            "omnifuse_package": str(
                (direct.SOURCE_ROOT / "omnifuse" / "__init__.py").resolve()
            ),
            "omnifuse_version": getattr(direct.omnifuse_package, "__version__", None),
            "omnifuse_builder_source": str(
                (direct.SOURCE_ROOT / "omnifuse" / "facade.py").resolve()
            ),
        },
    }


def _worker_validation_fixture(
    repo: pathlib.Path, python: pathlib.Path
) -> tuple[dict[str, object], dict[str, object]]:
    case = direct.CASES[0]
    before = _process_record(repo, python, phase="startup")
    after = _process_record(repo, python, phase="runtime")
    lock = {
        "validation": {"status": "ok"},
        "uv_sync_check": {
            "virtual_environment": str(pathlib.Path(before["python_prefix"]).resolve())
        },
    }
    identity = {"tag": direct.EXPECTED_TAG}
    sources = {"harness": {"sha256": "a" * 64}}
    relative_input = (
        pathlib.Path("tests") / "benchmark" / "data" / case.filename
    ).as_posix()
    dataset_path = repo / relative_input
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(
            {
                "corpus": {
                    document_id: {"title": "Title", "text": "x" * 10}
                    for document_id in ("d1", "d2", "d3", "d4")
                },
                "queries": {"q1": "first", "q2": "second"},
                "qrels": {"q1": {"d1": 1}, "q2": {"d3": 1}},
            }
        ),
        encoding="utf-8",
    )
    input_record = {
        **provenance.file_fingerprint(dataset_path, display_path=relative_input),
        "case_id": case.id,
        "case_name": case.name,
        "git_tracked": True,
    }
    tokenizer = {"status": "ok", "kiwi_available": True}
    state = {
        "identity": identity,
        "sources": sources,
        "inputs": {case.id: input_record},
        "worker_environment": {
            "process": before,
            "environment_lock": lock,
            "tokenizer_runtime": tokenizer,
        },
    }
    evidence = {
        "official_tag_identity": identity,
        "sources": sources,
        "input": input_record,
        "environment_lock": lock,
    }
    payload = {
        "schema": direct.WORKER_SCHEMA,
        "schema_version": direct.WORKER_SCHEMA_VERSION,
        "provenance_level": direct.PROVENANCE_LEVEL,
        "generated_at": "2026-07-22T00:00:00+00:00",
        "status": "ok",
        "worker_identity": _worker_identity(),
        "environment": {
            "before": before,
            "after": after,
            "tokenizer_runtime": tokenizer,
        },
        "contract": dict(direct.WORKER_CONTRACT),
        "evidence": {"before": evidence, "after": evidence},
        "result": _direct_result_fixture(case, repo, python),
    }
    return payload, state


def test_scope_matches_all_fourteen_official_external_cases() -> None:
    assert len(direct.CASES) == 14
    assert [case.id for case in direct.CASES] == [
        "ko_strategyqa",
        "autorag_retrieval",
        "klue_mrc",
        "allganize_rag_eval",
        "allganize_rag_ko",
        "hotpotqa_24",
        "hotpotqa_200",
        "publichealthqa_ko",
        "nfcorpus",
        "scifact",
        "fiqa",
        "miracl_retrieval_ko",
        "multilongdoc_ko",
        "xpqa_ko",
    ]
    assert direct.CANDIDATE_LIMIT == direct.K * 2 == 20
    assert direct.TEXT_LIMIT == 2000
    assert direct.SAMPLE_SEED == 42
    assert {case.id for case in direct.CASES if case.hotpot_supporting} == {
        "hotpotqa_24",
        "hotpotqa_200",
    }


def test_klue_preparation_reproduces_seeded_corpus_and_query_sampling() -> None:
    corpus = {
        f"klue_doc_{index}": {"title": "", "text": str(index)} for index in range(600)
    }
    data = {
        "corpus": corpus,
        "queries": {f"klue_{index}": f"q{index}" for index in range(600)},
        "qrels": {f"klue_{index}": {f"klue_doc_{index}": 1} for index in range(600)},
    }
    driver = types.SimpleNamespace(_load_dataset=lambda _filename: data)
    case = direct.CASE_BY_ID["klue_mrc"]

    prepared, selection = direct._prepare_case_data(driver, case)

    random.seed(42)
    expected_ids = {key for key, _value in random.sample(list(corpus.items()), 500)}
    assert set(prepared["corpus"]) == expected_ids
    assert set(prepared["queries"]) == {
        key.replace("klue_doc_", "klue_") for key in expected_ids
    }
    assert set(prepared["qrels"]) == set(prepared["queries"])
    assert selection["original_corpus_count"] == 600
    assert selection["selected_corpus_count"] == 500
    assert selection["eligible_query_count_before_max_queries"] == 500
    assert selection["max_queries"] == 100


def test_document_preprocessing_matches_official_empty_truncate_and_title_rules() -> (
    None
):
    long_text = "x" * 2001
    documents, stats = direct._omnifuse_documents(
        {
            "empty": {"title": "ignored", "text": ""},
            "fallback": {"title": "", "text": long_text},
            "titled": {"title": "Title", "text": "body"},
        }
    )

    assert documents == [
        {"id": "fallback", "title": "x" * 80, "text": "x" * 2000},
        {"id": "titled", "title": "Title", "text": "body"},
    ]
    assert stats["skipped_empty_text_documents"] == 1
    assert stats["truncated_documents"] == 1
    assert stats["title_fallback_documents"] == 1
    assert stats["indexed_text_characters"] == 2004


def test_metric_payload_keeps_upstream_mrr20_separate_from_mrr10() -> None:
    benchmark = types.SimpleNamespace(
        queries=[
            {
                "query_id": "q1",
                "retrieved_top_k": [f"d{index}" for index in range(10)],
                "relevant": ["relevant"],
                "mrr": 1.0 / 11,
                "search_time_ms": 2.0,
            }
        ],
        summary=lambda: _summary(mrr=1.0 / 11),
    )

    result = direct._system_result(
        benchmark,
        ingest_seconds=0.1,
        retrieved_top_20=[[*[f"d{index}" for index in range(10)], "relevant"]],
        hotpot_supporting=True,
    )

    assert result["metrics"]["mrr_at_20"] == pytest.approx(1.0 / 11)
    assert result["metrics"]["mrr_at_10"] == 0.0
    assert result["queries"][0]["retrieved_top_20"][-1] == "relevant"
    assert result["queries"][0]["reciprocal_rank_at_20"] == pytest.approx(1.0 / 11)
    assert result["queries"][0]["reciprocal_rank_at_10"] == 0.0
    assert result["supporting_facts"] == {
        "hits_at_10": 0,
        "total": 1,
        "micro_recall_at_10": 0.0,
    }


def test_hotpot_supporting_recall_is_micro_averaged() -> None:
    benchmark = types.SimpleNamespace(
        queries=[
            {
                "query_id": "q1",
                "retrieved_top_k": ["a"],
                "relevant": ["a", "b"],
                "mrr": 1.0,
                "search_time_ms": 1.0,
            },
            {
                "query_id": "q2",
                "retrieved_top_k": ["x"],
                "relevant": ["c"],
                "mrr": 0.0,
                "search_time_ms": 1.0,
            },
        ],
        summary=lambda: _summary(total=2, mrr=0.5),
    )

    result = direct._system_result(
        benchmark,
        ingest_seconds=0.1,
        retrieved_top_20=[["a"], ["x"]],
        hotpot_supporting=True,
    )

    assert result["supporting_facts"]["hits_at_10"] == 1
    assert result["supporting_facts"]["total"] == 3
    assert result["supporting_facts"]["micro_recall_at_10"] == pytest.approx(1 / 3)


def test_run_case_calls_official_fts_path_and_reuses_its_query_selection(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class Backend:
        async def close(self) -> None:
            calls["closed"] = True

    class Graph:
        backend = Backend()

        async def search(self, query, *, limit):
            calls["synaptic_search"] = (query, limit)
            return types.SimpleNamespace(
                nodes=[types.SimpleNamespace(node=types.SimpleNamespace(id="node-1"))]
            )

    class Driver:
        BenchmarkResult = _Benchmark

        @staticmethod
        def _load_dataset(_filename):
            return {
                "corpus": {"d1": {"title": "", "text": "body"}},
                "queries": {"q1": "body"},
                "qrels": {"q1": {"d1": 1}},
            }

        @staticmethod
        async def _build_graph(corpus, *, no_embedding):
            calls["corpus"] = corpus
            calls["no_embedding"] = no_embedding
            return Graph(), {"d1": "node-1"}

        @staticmethod
        async def _run_benchmark(name, graph, id_map, queries, qrels, *, max_queries):
            calls["upstream"] = (name, graph, id_map, queries, qrels, max_queries)
            search_result = await graph.search("body", limit=20)
            benchmark = _Benchmark()
            benchmark.add(
                "q1",
                "body",
                [item.node.id for item in search_result.nodes],
                {"node-1"},
                k=10,
                description=name,
                search_time_ms=0.1,
            )
            return benchmark

    class Omni:
        def retrieve(self, query, *, limit):
            calls["omnifuse_search"] = (query, limit)
            return [(types.SimpleNamespace(id="d1"), 1.0)]

    monkeypatch.setattr(
        direct,
        "_load_upstream_driver",
        lambda _repo: (
            Driver,
            types.SimpleNamespace(),
            {
                "python_executable": sys.executable,
                "synaptic_package": "synaptic.py",
                "upstream_driver": "driver.py",
                "upstream_scorer": "metrics.py",
            },
        ),
    )
    monkeypatch.setattr(
        direct,
        "build_inmemory",
        lambda _nodes, _triples, documents, *, vector_k: (
            calls.update(documents=documents, vector_k=vector_k) or Omni()
        ),
    )
    monkeypatch.setattr(
        direct,
        "_require_runtime_path",
        lambda _actual, expected, _label: str(pathlib.Path(expected).resolve()),
    )

    result = asyncio.run(
        direct._run_case(direct.CASE_BY_ID["ko_strategyqa"], pathlib.Path("repo"))
    )

    assert calls["no_embedding"] is True
    assert calls["closed"] is True
    assert calls["vector_k"] == 20
    assert calls["omnifuse_search"] == ("body", 20)
    assert calls["synaptic_search"] == ("body", 20)
    assert calls["upstream"][-1] == 100
    assert result["systems"]["synaptic_memory"]["metrics"]["mrr_at_10"] == 1.0
    assert result["systems"]["omnifuse"]["metrics"]["mrr_at_10"] == 1.0
    assert result["winners"]["mrr_at_10"] == "tie"


def test_tag_validation_requires_exact_origin_sha_tag_and_source_hashes(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "synaptic"
    repo.mkdir()
    expected_hashes = {
        direct.UPSTREAM_DRIVER_RELATIVE.as_posix(): direct.UPSTREAM_DRIVER_SHA256,
        direct.UPSTREAM_SCORER_RELATIVE.as_posix(): direct.UPSTREAM_SCORER_SHA256,
        direct.UPSTREAM_PACKAGE_RELATIVE.as_posix(): direct.UPSTREAM_PACKAGE_SHA256,
        direct.UPSTREAM_SQLITE_RELATIVE.as_posix(): direct.UPSTREAM_SQLITE_SHA256,
        direct.UPSTREAM_LOCK_RELATIVE.as_posix(): direct.UPSTREAM_LOCK_SHA256,
    }
    origin = direct.EXPECTED_ORIGIN
    ignored = ""

    def git(_repo, *args, check=True):
        del check
        if args == ("rev-parse", "--show-toplevel"):
            stdout, returncode = str(repo), 0
        elif args in (
            ("remote", "get-url", "origin"),
            ("remote", "get-url", "--push", "origin"),
        ):
            stdout, returncode = origin, 0
        elif args == ("rev-parse", "HEAD") or args == (
            "rev-parse",
            f"refs/tags/{direct.EXPECTED_TAG}^{{}}",
        ):
            stdout, returncode = direct.EXPECTED_TAG_SHA, 0
        elif args == ("describe", "--tags", "--exact-match", "HEAD"):
            stdout, returncode = direct.EXPECTED_TAG, 0
        elif args == ("symbolic-ref", "-q", "HEAD"):
            stdout, returncode = "", 1
        elif args == (
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        ):
            stdout, returncode = ignored, 0
        else:
            stdout, returncode = "", 0
        return subprocess.CompletedProcess(args, returncode, stdout, "")

    def fingerprint(path, *, display_path=None):
        relative = pathlib.Path(display_path).as_posix()
        return {"path": relative, "sha256": expected_hashes[relative], "bytes": 1}

    monkeypatch.setattr(direct, "_git", git)
    monkeypatch.setattr(direct, "file_fingerprint", fingerprint)

    identity = direct._validate_tag_checkout(repo)

    assert identity["head"] == direct.EXPECTED_TAG_SHA
    assert identity["tag"] == direct.EXPECTED_TAG
    assert identity["detached"] is True

    origin = "https://github.com/someone-else/synaptic-memory.git"
    with pytest.raises(provenance.ProvenanceError, match="origin mismatch"):
        direct._validate_tag_checkout(repo)

    origin = direct.EXPECTED_ORIGIN
    ignored = "src/synaptic/ambient.py\0"
    with pytest.raises(provenance.ProvenanceError, match="ignored files outside"):
        direct._validate_tag_checkout(repo)

    ignored = "tests/benchmark/data/generated.json\0"
    identity = direct._validate_tag_checkout(repo)
    assert identity["allowed_ignored_dataset_files"] == [
        "tests/benchmark/data/generated.json"
    ]


def test_official_tag_identity_must_match_shared_repository_fingerprint(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path.resolve())
    identity = {
        "git_root": root,
        "head": direct.EXPECTED_TAG_SHA,
        "origin_fetch": direct.EXPECTED_ORIGIN,
        "tag": direct.EXPECTED_TAG,
    }
    repository = {
        "path": root,
        "git_root": root,
        "sha": direct.EXPECTED_TAG_SHA,
        "origin_fetch_url": direct.EXPECTED_ORIGIN,
        "exact_tags": [direct.EXPECTED_TAG],
    }

    direct._verify_shared_tag_identity(identity, repository)

    repository["exact_tags"] = []
    with pytest.raises(provenance.ProvenanceError, match="shared repository identity"):
        direct._verify_shared_tag_identity(identity, repository)


def test_worker_environment_is_deterministic_and_drops_inherited_pythonpath(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("PYTHONPATH", "poison")

    environment = direct._worker_environment(repo)

    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "poison" not in environment["PYTHONPATH"]
    assert environment["PYTHONPATH"].split(direct.os.pathsep)[0] == str(
        (repo / "src").resolve()
    )
    assert environment["PYTHONPATH"].split(direct.os.pathsep) == (
        direct._worker_pythonpath_entries(repo)
    )


def test_process_environment_requires_no_user_site_and_all_four_variables(
    tmp_path: pathlib.Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    python = pathlib.Path(sys.executable)
    record = _process_record(repo, python, phase="startup")

    direct._validate_process_environment_record(
        record, repo=repo, expected_python=python, phase="startup"
    )

    missing_no_user_site = json.loads(json.dumps(record))
    missing_no_user_site["variables"].pop("PYTHONNOUSERSITE")
    with pytest.raises(provenance.ProvenanceError, match="environment mismatch"):
        direct._validate_process_environment_record(
            missing_no_user_site,
            repo=repo,
            expected_python=python,
            phase="startup",
        )

    enabled_user_site = json.loads(json.dumps(record))
    enabled_user_site["flags"]["no_user_site"] = 0
    with pytest.raises(provenance.ProvenanceError, match="flags violate"):
        direct._validate_process_environment_record(
            enabled_user_site,
            repo=repo,
            expected_python=python,
            phase="startup",
        )


def test_runtime_environment_allows_only_the_official_source_path_transition(
    tmp_path: pathlib.Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    python = pathlib.Path(sys.executable)
    before = _process_record(repo, python, phase="startup")
    after = _process_record(repo, python, phase="runtime")

    direct._validate_runtime_environment_transition(before, after, repo=repo)

    poisoned = json.loads(json.dumps(after))
    poisoned["normalized_sys_path"].append(str(tmp_path / "ambient"))
    with pytest.raises(provenance.ProvenanceError, match="runtime environment"):
        direct._validate_runtime_environment_transition(before, poisoned, repo=repo)


def test_uv_sync_check_requires_exact_selected_extras_and_no_changes(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    uv = tmp_path / "uv.exe"
    uv.write_bytes(b"uv")
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "uv 1.2.3\n", "")
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            f"Would use project environment at: {pathlib.Path(sys.prefix).resolve()}\n"
            "Checked 25 packages in 1ms\nWould make no changes\n",
        )

    monkeypatch.setattr(direct.shutil, "which", lambda _name: str(uv))
    monkeypatch.setattr(direct.subprocess, "run", run)

    record = direct._uv_sync_check(repo)

    assert calls[1] == [str(uv.resolve()), *direct.UV_SYNC_CHECK_ARGUMENTS]
    assert record["selected_extras"] == ["sqlite", "embedding", "korean", "dev"]
    assert record["checked_package_count"] == 25
    assert record["reported_no_changes"] is True
    direct._validate_uv_sync_check_record(record, repo)

    changed = json.loads(json.dumps(record))
    changed["selected_extras"].remove("korean")
    with pytest.raises(provenance.ProvenanceError, match="selected_extras"):
        direct._validate_uv_sync_check_record(changed, repo)


def test_uv_sync_check_rejects_unsynchronized_environment(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    uv = tmp_path / "uv.exe"
    uv.write_bytes(b"uv")

    def run(command, **_kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "uv 1.2.3\n", "")
        return subprocess.CompletedProcess(
            command, 1, "", "kiwipiepy would be installed"
        )

    monkeypatch.setattr(direct.shutil, "which", lambda _name: str(uv))
    monkeypatch.setattr(direct.subprocess, "run", run)

    with pytest.raises(provenance.ProvenanceError, match="not synchronized"):
        direct._uv_sync_check(repo)


def test_distribution_manifest_checks_installed_membership_except_explicit_tools() -> (
    None
):
    lock = {"numpy": ["2.4.3"], "pytest": ["9.0.2"]}
    installed = [
        {"name": "numpy", "version": "2.4.3", "location": "venv"},
        {"name": "pytest", "version": "9.0.2", "location": "venv"},
        {"name": "pip", "version": "99.0", "location": "venv"},
    ]

    validation = direct._validate_installed_distributions(lock, installed)

    assert validation["coverage"] == "installed-distribution-membership"
    assert validation["completeness_enforced_by"] == "uv sync --check"
    assert [item["name"] for item in validation["matched_distributions"]] == [
        "numpy",
        "pytest",
    ]
    assert validation["tool_exceptions"] == [{"name": "pip", "version": "99.0"}]

    wrong_version = [{"name": "numpy", "version": "2.5.1", "location": "venv"}]
    with pytest.raises(provenance.ProvenanceError, match="does not match"):
        direct._validate_installed_distributions(lock, wrong_version)

    unlisted = [{"name": "ambient-plugin", "version": "1", "location": "venv"}]
    with pytest.raises(provenance.ProvenanceError, match="absent from official"):
        direct._validate_installed_distributions(lock, unlisted)


def test_tokenizer_runtime_requires_active_kiwi_and_bound_distribution_modules(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    sqlite_path = repo / direct.UPSTREAM_SQLITE_RELATIVE
    sqlite_path.parent.mkdir(parents=True)
    sqlite_path.write_text("def _normalize_korean(): pass\n", encoding="utf-8")
    prefix = tmp_path / "venv"
    site_packages = prefix / "Lib" / "site-packages"
    kiwi_module = site_packages / "kiwipiepy" / "__init__.py"
    model_module = site_packages / "kiwipiepy_model" / "__init__.py"
    kiwi_module.parent.mkdir(parents=True)
    model_module.parent.mkdir(parents=True)
    kiwi_module.write_text("class Kiwi: pass\n", encoding="utf-8")
    model_module.write_text("MODEL = True\n", encoding="utf-8")
    installed = [
        {
            "name": "kiwipiepy",
            "version": "0.23.1",
            "location": str(site_packages.resolve()),
        },
        {
            "name": "kiwipiepy-model",
            "version": "0.23.0",
            "location": str(site_packages.resolve()),
        },
    ]
    environment_lock = {"installed_distributions": installed}
    sqlite_fingerprint = provenance.file_fingerprint(
        sqlite_path, display_path=direct.UPSTREAM_SQLITE_RELATIVE.as_posix()
    )
    monkeypatch.setattr(direct, "UPSTREAM_SQLITE_SHA256", sqlite_fingerprint["sha256"])
    monkeypatch.setattr(direct, "UPSTREAM_NORMALIZE_KOREAN_SOURCE_SHA256", "a" * 64)
    record = {
        "status": "ok",
        "python_prefix": str(prefix.resolve()),
        "sqlite": {
            **sqlite_fingerprint,
            "runtime_path": str(sqlite_path.resolve()),
            "module": "synaptic.backends.sqlite",
            "normalize_function_source_sha256": "a" * 64,
        },
        "kiwi_available": True,
        "kiwi_instance": {"module": "kiwipiepy", "qualname": "Kiwi"},
        "modules": {
            "kiwipiepy": {
                "module": "kiwipiepy",
                "version": "0.23.1",
                "distribution": installed[0],
                "module_file": provenance.file_fingerprint(kiwi_module),
            },
            "kiwipiepy-model": {
                "module": "kiwipiepy_model",
                "version": "0.23.0",
                "distribution": installed[1],
                "module_file": provenance.file_fingerprint(model_module),
            },
        },
        "functional_probe": {
            "input_sha256": direct.hashlib.sha256(
                direct.TOKENIZER_PROBE_TEXT.encode("utf-8")
            ).hexdigest(),
            "normalized_sha256": "b" * 64,
            "normalized_token_count": 4,
        },
    }
    monkeypatch.setattr(
        direct,
        "_lock_package_versions",
        lambda _path: {"kiwipiepy": ["0.23.1"], "kiwipiepy-model": ["0.23.0"]},
    )

    direct._validate_tokenizer_runtime_record(
        record,
        repo=repo,
        python_prefix=prefix,
        environment_lock=environment_lock,
    )

    fallback = json.loads(json.dumps(record))
    fallback["kiwi_available"] = False
    with pytest.raises(provenance.ProvenanceError, match="active Kiwi"):
        direct._validate_tokenizer_runtime_record(
            fallback,
            repo=repo,
            python_prefix=prefix,
            environment_lock=environment_lock,
        )

    escaped = json.loads(json.dumps(record))
    escaped["modules"]["kiwipiepy"]["module_file"]["path"] = str(
        (tmp_path / "outside.py").resolve()
    )
    with pytest.raises(provenance.ProvenanceError, match="escaped"):
        direct._validate_tokenizer_runtime_record(
            escaped,
            repo=repo,
            python_prefix=prefix,
            environment_lock=environment_lock,
        )


def test_omnifuse_builder_is_bound_to_current_facade_source() -> None:
    record = direct._omnifuse_builder_provenance()

    assert (
        pathlib.Path(record["runtime_path"]).resolve()
        == (direct.SOURCE_ROOT / "omnifuse" / "facade.py").resolve()
    )
    assert record["module"] == "omnifuse.facade"
    assert len(record["function_source_sha256"]) == 64


def test_parent_revalidates_worker_environment_and_suite_evidence(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    python = pathlib.Path(sys.executable)
    payload, state = _worker_validation_fixture(repo, python)
    monkeypatch.setattr(
        direct, "_validate_environment_lock_record", lambda *_args: None
    )
    monkeypatch.setattr(
        direct, "_validate_tokenizer_runtime_record", lambda *_args, **_kwargs: None
    )

    assert (
        direct._validate_worker_payload(
            payload,
            case=direct.CASES[0],
            python=python,
            repo=repo,
            state=state,
            expected_worker_run_id=WORKER_RUN_ID,
        )
        is payload
    )

    wrong_identity = json.loads(json.dumps(payload))
    wrong_identity["worker_identity"]["worker_run_id"] = (
        "00000000000040008000000000000002"
    )
    with pytest.raises(provenance.ProvenanceError, match="does not match"):
        direct._validate_worker_payload(
            wrong_identity,
            case=direct.CASES[0],
            python=python,
            repo=repo,
            state=state,
            expected_worker_run_id=WORKER_RUN_ID,
        )

    missing_tokenizer = json.loads(json.dumps(payload))
    missing_tokenizer["environment"].pop("tokenizer_runtime")
    with pytest.raises(provenance.ProvenanceError, match="strict schema"):
        direct._validate_worker_payload(
            missing_tokenizer,
            case=direct.CASES[0],
            python=python,
            repo=repo,
            state=state,
            expected_worker_run_id=WORKER_RUN_ID,
        )

    changed = json.loads(json.dumps(payload))
    changed["evidence"]["after"]["input"]["sha256"] = "c" * 64
    with pytest.raises(provenance.ProvenanceError, match="source/input evidence"):
        direct._validate_worker_payload(
            changed,
            case=direct.CASES[0],
            python=python,
            repo=repo,
            state=state,
            expected_worker_run_id=WORKER_RUN_ID,
        )


def test_schema_v4_recomputes_direct_result_and_rejects_semantic_tampering(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    python = pathlib.Path(sys.executable)
    payload, state = _worker_validation_fixture(repo, python)
    monkeypatch.setattr(
        direct, "_validate_environment_lock_record", lambda *_args: None
    )
    monkeypatch.setattr(
        direct, "_validate_tokenizer_runtime_record", lambda *_args, **_kwargs: None
    )

    def reject(candidate: dict, pattern: str) -> None:
        with pytest.raises(provenance.ProvenanceError, match=pattern):
            direct._validate_worker_payload(
                candidate,
                case=direct.CASES[0],
                python=python,
                repo=repo,
                state=state,
                expected_worker_run_id=WORKER_RUN_ID,
            )

    changed = json.loads(json.dumps(payload))
    changed["contract"]["candidate_limit"] = direct.CANDIDATE_LIMIT - 1
    reject(changed, "contract")

    changed = json.loads(json.dumps(payload))
    changed["result"]["unexpected"] = True
    reject(changed, "strict schema")

    changed = json.loads(json.dumps(payload))
    changed["result"]["systems"]["omnifuse"]["evaluated_queries"] = 3
    reject(changed, "evaluated query count")

    changed = json.loads(json.dumps(payload))
    changed["result"]["systems"]["omnifuse"]["query_ids_ordered_sha256"] = "0" * 64
    reject(changed, "query order hash")

    changed = json.loads(json.dumps(payload))
    synaptic = changed["result"]["systems"]["synaptic_memory"]
    synaptic["queries"].reverse()
    synaptic["query_ids_ordered_sha256"] = provenance.canonical_json_sha256(
        [row["query_id"] for row in synaptic["queries"]]
    )
    reject(changed, "system query selections")

    changed = json.loads(json.dumps(payload))
    for system_result in changed["result"]["systems"].values():
        system_result["queries"].reverse()
        system_result["query_ids_ordered_sha256"] = provenance.canonical_json_sha256(
            [row["query_id"] for row in system_result["queries"]]
        )
    changed["result"]["selection"]["scored_query_ids_ordered_sha256"] = (
        provenance.canonical_json_sha256(["q2", "q1"])
    )
    reject(changed, "query order or relevant judgments differ from the official input")

    changed = json.loads(json.dumps(payload))
    changed["result"]["systems"]["omnifuse"]["queries"][0]["relevant"] = [
        "d1",
        "d1",
    ]
    reject(changed, "relevant IDs")

    changed = json.loads(json.dumps(payload))
    systems = changed["result"]["systems"]
    for system_result in systems.values():
        original = system_result["queries"][0]
        system_result["queries"][0] = _query_row(
            original["query_id"],
            original["retrieved_top_20"],
            ["d2"],
            original["search_time_ms"],
        )
        system_result["metrics"] = _fixture_metrics(system_result["queries"])
    for metric in direct.METRIC_NAMES:
        omni_value = systems["omnifuse"]["metrics"][metric]
        synaptic_value = systems["synaptic_memory"]["metrics"][metric]
        changed["result"]["winners"][metric] = (
            "omnifuse"
            if omni_value > synaptic_value
            else "synaptic_memory"
            if synaptic_value > omni_value
            else "tie"
        )
    reject(changed, "relevant judgments differ from the official input")

    changed = json.loads(json.dumps(payload))
    changed["result"]["systems"]["omnifuse"]["queries"][0]["relevant"] = []
    reject(changed, "relevant IDs must not be empty")

    changed = json.loads(json.dumps(payload))
    changed["result"]["systems"]["omnifuse"]["queries"][0]["reciprocal_rank_at_10"] = (
        0.5
    )
    reject(changed, "reciprocal rank at 10")

    changed = json.loads(json.dumps(payload))
    changed["result"]["systems"]["omnifuse"]["metrics"]["mrr_at_10"] = 0.5
    reject(changed, "metric mrr_at_10")

    changed = json.loads(json.dumps(payload))
    changed["result"]["winners"]["mrr_at_10"] = "tie"
    reject(changed, "winner for mrr_at_10")

    changed = json.loads(json.dumps(payload))
    changed["result"]["runtime"]["unexpected"] = "value"
    reject(changed, "runtime.*strict schema")


def test_schema_v4_validates_hotpot_supporting_fact_aggregation(
    tmp_path: pathlib.Path,
) -> None:
    case = direct.CASE_BY_ID["hotpotqa_24"]
    result = _direct_result_fixture(
        case, tmp_path / "repo", pathlib.Path(sys.executable)
    )

    direct._validate_worker_result(
        result,
        case=case,
        python=pathlib.Path(sys.executable),
        repo=tmp_path / "repo",
    )

    result["systems"]["omnifuse"]["supporting_facts"]["hits_at_10"] = 0
    with pytest.raises(provenance.ProvenanceError, match="supporting fact counts"):
        direct._validate_worker_result(
            result,
            case=case,
            python=pathlib.Path(sys.executable),
            repo=tmp_path / "repo",
        )


def test_worker_result_is_write_once_before_any_benchmark_work(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    result = tmp_path / "worker.json"
    result.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        direct,
        "_worker_evidence",
        lambda *_args: pytest.fail("worker must not start"),
    )

    with pytest.raises(provenance.OutputExistsError, match="refusing to overwrite"):
        direct._run_worker(
            tmp_path,
            direct.CASES[0],
            result,
            pathlib.Path(sys.executable),
            WORKER_RUN_ID,
        )

    assert result.read_text(encoding="utf-8") == "preserve\n"


def test_default_worker_directory_binds_the_complete_output_path(
    tmp_path: pathlib.Path,
) -> None:
    first_output = tmp_path / "first" / "comparison.json"
    second_output = tmp_path / "second" / "comparison.json"

    first = direct._worker_directory(first_output, None)
    repeated = direct._worker_directory(first_output, None)
    second = direct._worker_directory(second_output, None)

    assert first == repeated
    assert first != second
    assert first.parent == (direct.ROOT / "worklogs").resolve()
    assert first.name.startswith("comparison-")
    assert first.name.endswith("-direct-external-workers")
    identity = first.name.removeprefix("comparison-").removesuffix(
        "-direct-external-workers"
    )
    assert len(identity) == 16
    assert all(character in "0123456789abcdef" for character in identity)
    configured = tmp_path / "explicit-workers"
    assert direct._worker_directory(first_output, configured) == configured.resolve()


def test_preflight_refuses_existing_worker_directory_before_repository_work(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    workers = tmp_path / "workers"
    workers.mkdir()
    monkeypatch.setattr(
        direct,
        "_validate_tag_checkout",
        lambda *_args: pytest.fail("repository work must not start"),
    )

    with pytest.raises(provenance.ProvenanceError, match="refusing to reuse"):
        direct._preflight(
            repo=repo,
            python=pathlib.Path(sys.executable),
            doctor_path=tmp_path / "doctor.json",
            output=tmp_path / "result.json",
            workers_dir=workers,
        )


def test_preflight_keeps_evidence_out_of_the_immutable_tag_checkout(
    tmp_path: pathlib.Path,
) -> None:
    repo = tmp_path / "synaptic"
    repo.mkdir()

    with pytest.raises(provenance.ProvenanceError, match="immutable"):
        direct._preflight(
            repo=repo,
            python=pathlib.Path(sys.executable),
            doctor_path=tmp_path / "doctor.json",
            output=tmp_path / "result.json",
            workers_dir=repo / "workers",
        )


def test_failed_isolated_case_prevents_suite_publication(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    output = tmp_path / "suite.json"
    workers = tmp_path / "workers"
    doctor = {"snapshot": {}}
    monkeypatch.setattr(
        direct,
        "_preflight",
        lambda **_kwargs: {
            "identity": {},
            "repositories": {},
            "sources": {},
            "inputs": {},
            "doctor": doctor,
            "doctor_links": {},
        },
    )
    monkeypatch.setattr(
        direct,
        "run_with_launcher_pid",
        lambda command, **_kwargs: (
            subprocess.CompletedProcess(command, 7, "", "worker exploded"),
            100,
        ),
    )

    with pytest.raises(provenance.ProvenanceError, match="ko_strategyqa failed"):
        direct._run_suite(
            repo=tmp_path,
            python=pathlib.Path(sys.executable),
            doctor_path=tmp_path / "doctor.json",
            output=output,
            workers_dir=workers,
        )

    assert not output.exists()
    assert workers.is_dir()


def test_duplicate_worker_run_id_preserves_partial_raw_and_blocks_report(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    output = tmp_path / "suite.json"
    workers = tmp_path / "workers"
    state = {
        "identity": {},
        "repositories": {},
        "sources": {},
        "inputs": {},
        "doctor": {"snapshot": {}},
        "doctor_links": {},
    }
    monkeypatch.setattr(direct, "_preflight", lambda **_kwargs: state)
    monkeypatch.setattr(direct, "new_worker_run_id", lambda: WORKER_RUN_ID)
    monkeypatch.setattr(
        direct, "_validate_worker_payload", lambda payload, **_kwargs: payload
    )

    def run_worker(command, **_kwargs):
        result_path = pathlib.Path(command[command.index("--worker-result") + 1])
        result_path.write_text(
            json.dumps({"worker_identity": _worker_identity()}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", ""), 100

    monkeypatch.setattr(direct, "run_with_launcher_pid", run_worker)

    with pytest.raises(provenance.ProvenanceError, match="duplicate worker run ID"):
        direct._run_suite(
            repo=tmp_path,
            python=pathlib.Path(sys.executable),
            doctor_path=tmp_path / "doctor.json",
            output=output,
            workers_dir=workers,
        )

    assert not output.exists()
    assert len(list(workers.glob("*.json"))) == 1


def test_successful_suite_publishes_all_write_once_worker_artifacts(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    output = tmp_path / "suite.json"
    workers = tmp_path / "workers"
    state = {
        "identity": {},
        "repositories": {},
        "sources": {},
        "inputs": {},
        "doctor": {"snapshot": {}},
        "doctor_links": {},
    }
    monkeypatch.setattr(direct, "_preflight", lambda **_kwargs: state)
    monkeypatch.setattr(
        direct,
        "_postflight",
        lambda *_args: {"checks": {"postflight_completed_before_publication": True}},
    )
    monkeypatch.setattr(
        direct,
        "_validate_worker_payload",
        lambda payload, **_kwargs: payload,
    )

    metric_names = (
        "mrr_at_20",
        "mrr_at_10",
        "precision_at_10",
        "recall_at_10",
        "f1_at_10",
        "ndcg_at_10",
    )

    def run_worker(command, **_kwargs):
        case_id = command[command.index("--worker-case") + 1]
        result_path = pathlib.Path(command[command.index("--worker-result") + 1])
        worker_run_id = command[command.index("--worker-run-id") + 1]
        metrics = {name: 1.0 for name in metric_names}
        payload = {
            "worker_identity": _worker_identity(worker_run_id, pid=200),
            "result": {
                "case": {"id": case_id},
                "systems": {
                    "omnifuse": {"metrics": metrics},
                    "synaptic_memory": {"metrics": metrics},
                },
                "winners": {name: "tie" for name in metric_names},
            },
        }
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", ""), 100

    monkeypatch.setattr(direct, "run_with_launcher_pid", run_worker)

    report = direct._run_suite(
        repo=tmp_path,
        python=pathlib.Path(sys.executable),
        doctor_path=tmp_path / "doctor.json",
        output=output,
        workers_dir=workers,
    )

    assert report["status"] == "ok"
    assert report["schema_version"] == 5
    assert direct.WORKER_SCHEMA_VERSION == 4
    assert report["summary"]["completed_cases"] == 14
    assert report["summary"]["wins"]["mrr_at_10"]["tie"] == 14
    assert report["worker_process_summary"]["distinct_worker_run_ids"] == 14
    assert report["worker_process_summary"]["launcher_worker_pid_mismatch_count"] == 14
    assert output.is_file()
    assert len(list(workers.glob("*.json"))) == 14
    with pytest.raises(provenance.OutputExistsError, match="refusing to overwrite"):
        provenance.write_json_once(output, report)


def test_suite_rechecks_raw_worker_bytes_immediately_before_publication(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    output = tmp_path / "suite.json"
    workers = tmp_path / "workers"
    state = {
        "identity": {},
        "repositories": {},
        "sources": {},
        "inputs": {},
        "doctor": {"snapshot": {}},
        "doctor_links": {},
    }
    monkeypatch.setattr(direct, "_preflight", lambda **_kwargs: state)
    monkeypatch.setattr(
        direct,
        "_postflight",
        lambda *_args: {"checks": {"postflight_completed_before_publication": True}},
    )
    monkeypatch.setattr(
        direct, "_validate_worker_payload", lambda payload, **_kwargs: payload
    )

    def run_worker(command, **_kwargs):
        case_id = command[command.index("--worker-case") + 1]
        result_path = pathlib.Path(command[command.index("--worker-result") + 1])
        worker_run_id = command[command.index("--worker-run-id") + 1]
        metrics = {name: 1.0 for name in direct.METRIC_NAMES}
        payload = {
            "worker_identity": _worker_identity(worker_run_id, pid=200),
            "result": {
                "case": {"id": case_id},
                "systems": {
                    "omnifuse": {"metrics": metrics},
                    "synaptic_memory": {"metrics": metrics},
                },
                "winners": {name: "tie" for name in direct.METRIC_NAMES},
            },
        }
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", ""), 100

    monkeypatch.setattr(direct, "run_with_launcher_pid", run_worker)
    original_summary = direct._summary

    def tamper_after_report_assembly(worker_records):
        summary = original_summary(worker_records)
        artifact_path = pathlib.Path(worker_records[0]["artifact"]["path"])
        artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")
        return summary

    monkeypatch.setattr(direct, "_summary", tamper_after_report_assembly)

    with pytest.raises(provenance.ProvenanceError, match="worker artifact.*changed"):
        direct._run_suite(
            repo=tmp_path,
            python=pathlib.Path(sys.executable),
            doctor_path=tmp_path / "doctor.json",
            output=output,
            workers_dir=workers,
        )

    assert not output.exists()
    assert len(list(workers.glob("*.json"))) == 14


def test_doctor_binding_covers_every_case_with_hash_and_bytes() -> None:
    inputs = {
        case.id: {
            "path": f"tests/benchmark/data/{case.filename}",
            "sha256": f"{index:064x}",
            "bytes": index,
        }
        for index, case in enumerate(direct.CASES, 1)
    }

    bindings = direct._doctor_inputs(inputs)

    assert len(bindings) == 14
    assert {binding["target_id"] for binding in bindings} == set(direct.CASE_BY_ID)
    assert all(binding["bytes"] > 0 for binding in bindings)
    assert all(len(binding["sha256"]) == 64 for binding in bindings)


def test_per_query_head_to_head_reports_quality_and_latency_losses() -> None:
    def row(query_id: str, retrieved: list[str], search_time_ms: float) -> dict:
        return {
            "query_id": query_id,
            "retrieved_top_10": retrieved,
            "retrieved_top_20": retrieved,
            "relevant": ["gold"],
            "search_time_ms": search_time_ms,
        }

    workers = [
        {
            "case_id": "case-a",
            "payload": {
                "result": {
                    "systems": {
                        "omnifuse": {
                            "queries": [
                                row("win", ["gold", "noise"], 1.0),
                                row("loss", ["noise", "gold"], 3.0),
                            ]
                        },
                        "synaptic_memory": {
                            "queries": [
                                row("win", ["noise", "gold"], 2.0),
                                row("loss", ["gold", "noise"], 2.0),
                            ]
                        },
                    }
                }
            },
        }
    ]

    result = direct._per_query_head_to_head(workers)

    assert result["queries"] == 2
    assert result["questions_with_any_omnifuse_quality_loss"] == 1
    assert result["cases"][0]["quality_loss_query_ids"] == ["loss"]
    assert result["observed_search_ms"]["questions_with_omnifuse_loss"] == 1


def test_suite_cli_requires_doctor_runtime_and_write_once_output(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"preserve": True}), encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        direct.main(
            [
                "--synaptic-repo",
                str(tmp_path),
                "--synaptic-python",
                sys.executable,
                "--doctor-manifest",
                str(tmp_path / "doctor.json"),
                "--out",
                str(output),
                "--workers-dir",
                str(tmp_path / "workers"),
            ]
        )

    assert raised.value.code == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"preserve": True}
