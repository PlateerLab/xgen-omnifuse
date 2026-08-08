from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "eval" / "download_synaptic_datasets.py"
)
SPEC = importlib.util.spec_from_file_location("download_synaptic_datasets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


def test_2wiki_schema_hash_filter_and_streaming(tmp_path: Path) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    examples = [
        {
            "_id": "q1",
            "question": "Who wrote it?",
            "context": {
                "title": ["Alpha", "Empty"],
                "content": [["first", "second"], []],
            },
            "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
        },
        {
            "_id": "q2",
            "question": "Where next?",
            "context": [["Alpha", ["first", "second"]], ["Beta", ["third"]]],
            "supporting_facts": [["Beta", 0]],
        },
        {
            "_id": "drop",
            "question": "No matching support",
            "context": [["Gamma", ["body"]]],
            "supporting_facts": [["Missing", 0]],
        },
    ]

    def fake_load_dataset(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return iter(examples)

    output = tmp_path / "2wiki_dev.json"
    metadata = downloader.build_2wiki(output, load_dataset_fn=fake_load_dataset)
    result = json.loads(output.read_text(encoding="utf-8"))

    alpha_id = hashlib.sha1("Alpha\0first second".encode()).hexdigest()[:16]
    beta_id = hashlib.sha1("Beta\0third".encode()).hexdigest()[:16]
    assert result["corpus"][alpha_id] == {"title": "Alpha", "text": "first second"}
    assert result["qrels"] == {"q1": {alpha_id: 1}, "q2": {beta_id: 1}}
    assert result["queries"] == {"q1": "Who wrote it?", "q2": "Where next?"}
    assert metadata["corpus_size"] == result["corpus_size"] == 3
    assert result["query_size"] == result["qrels_size"] == 2
    assert result["qrels_rows"] == 2
    assert result["source"] == "huggingface: voidful/2WikiMultihopQA/validation"
    assert result["provenance"]["streaming"] is True
    assert calls == [
        (
            ("voidful/2WikiMultihopQA",),
            {"split": "validation", "streaming": True},
        )
    ]
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_trec_covid_uses_all_beir_streams_and_filters_qrels(tmp_path: Path) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dataset(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        if args == ("BeIR/trec-covid", "corpus"):
            return iter(
                [
                    {"_id": "d1", "title": "One", "text": "body one"},
                    {"_id": "d2", "title": None, "text": "body two"},
                    {"_id": "d1", "title": "One", "text": "body one"},
                ]
            )
        if args == ("BeIR/trec-covid", "queries"):
            return iter(
                [
                    {"_id": "q1", "text": "  useful query  "},
                    {"_id": "q2", "text": "  "},
                    {"_id": "q3", "text": "no relevance"},
                ]
            )
        if args == ("BeIR/trec-covid-qrels",):
            return iter(
                [
                    {"query-id": "q1", "corpus-id": "d1", "score": 2},
                    {"query-id": "q1", "corpus-id": "d2", "score": 0},
                    {"query-id": "q2", "corpus-id": "d2", "score": 1},
                    {"query-id": "q3", "corpus-id": "d2", "score": -1},
                ]
            )
        raise AssertionError(args)

    output = tmp_path / "trec_covid.json"
    downloader.build_trec_covid(output, load_dataset_fn=fake_load_dataset)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["corpus"] == {
        "d1": {"title": "One", "text": "body one"},
        "d2": {"title": "", "text": "body two"},
    }
    assert result["queries"] == {"q1": "useful query"}
    assert result["qrels"] == {"q1": {"d1": 2}}
    assert result["corpus_size"] == 2
    assert result["query_size"] == result["qrels_size"] == result["qrels_rows"] == 1
    assert result["source"] == "huggingface: BeIR/trec-covid"
    assert [call[0] for call in calls] == [
        ("BeIR/trec-covid", "corpus"),
        ("BeIR/trec-covid", "queries"),
        ("BeIR/trec-covid-qrels",),
    ]
    assert all(call[1]["streaming"] is True for call in calls)
    assert [call[1]["split"] for call in calls] == ["corpus", "queries", "test"]


def test_atomic_replace_preserves_existing_output_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "2wiki_dev.json"
    output.write_text("sentinel", encoding="utf-8")
    examples = [
        {
            "_id": "q1",
            "question": "question",
            "context": [["Title", ["text"]]],
            "supporting_facts": [["Title", 0]],
        }
    ]

    def reject_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(downloader.os, "replace", reject_replace)
    with pytest.raises(OSError, match="cannot replace"):
        downloader.build_2wiki(
            output, load_dataset_fn=lambda *args, **kwargs: iter(examples)
        )

    assert output.read_text(encoding="utf-8") == "sentinel"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_main_only_writes_the_selected_dataset(tmp_path: Path) -> None:
    examples = [
        {
            "_id": "q1",
            "question": "question",
            "context": [["Title", ["text"]]],
            "supporting_facts": [["Title", 0]],
        }
    ]
    exit_code = downloader.main(
        ["--out-dir", str(tmp_path), "--only", "2wiki"],
        load_dataset_fn=lambda *args, **kwargs: iter(examples),
    )

    assert exit_code == 0
    assert (tmp_path / "2wiki_dev.json").is_file()
    assert not (tmp_path / "trec_covid.json").exists()


def test_optional_dependency_error_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(downloader.importlib, "import_module", missing_module)
    with pytest.raises(
        downloader.MissingDatasetsDependency, match=r"pip install datasets"
    ):
        downloader._resolve_load_dataset()
