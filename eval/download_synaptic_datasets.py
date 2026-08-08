"""Stream the large public inputs declared by synaptic-memory's benchmark.

The generated files retain the mapping-based public-IR schema consumed by both
benchmark suites, while avoiding materializing either source corpus in memory.
Install the optional downloader dependency with ``pip install datasets``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]

DATASET_FILENAMES = {
    "2wiki": "2wiki_dev.json",
    "trec_covid": "trec_covid.json",
}


class MissingDatasetsDependency(RuntimeError):
    """Raised when a real download is requested without Hugging Face datasets."""


class _CorpusFragmentWriter:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._seen: set[str] = set()
        self.count = 0

    def add(self, doc_id: str, title: str, text: str) -> bool:
        if doc_id in self._seen:
            return False
        self._seen.add(doc_id)
        if self.count:
            self._stream.write(",")
        json.dump(doc_id, self._stream, ensure_ascii=False)
        self._stream.write(":")
        json.dump(
            {"title": title, "text": text},
            self._stream,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.count += 1
        return True


def _resolve_load_dataset() -> DatasetLoader:
    try:
        module = importlib.import_module("datasets")
    except ImportError as exc:
        raise MissingDatasetsDependency(
            "Downloading benchmark data requires the optional Hugging Face "
            "'datasets' package; install it with `pip install datasets`."
        ) from exc
    return module.load_dataset


def _load_dataset_fn(loader: DatasetLoader | None) -> DatasetLoader:
    return loader if loader is not None else _resolve_load_dataset()


def _doc_id_2wiki(title: str, text: str) -> str:
    payload = (title + "\0" + text).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _new_fragment(destination: Path) -> tuple[Path, TextIO]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.corpus.",
        suffix=".tmp",
        dir=destination.parent,
    )
    return Path(temporary_name), os.fdopen(
        descriptor, "w", encoding="utf-8", newline=""
    )


def _write_member(stream: TextIO, key: str, value: object, *, first: bool) -> None:
    if not first:
        stream.write(",")
    json.dump(key, stream, ensure_ascii=False)
    stream.write(":")
    json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))


def _write_dataset_atomic(
    destination: Path,
    *,
    metadata: Mapping[str, object],
    corpus_fragment: Path,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write("{")
            first = True
            for key, value in metadata.items():
                _write_member(stream, key, value, first=first)
                first = False
            stream.write(',"corpus":{')
            with corpus_fragment.open(encoding="utf-8") as fragment:
                shutil.copyfileobj(fragment, stream, length=1024 * 1024)
            stream.write('},"queries":')
            json.dump(queries, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write(',"qrels":')
            json.dump(qrels, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _context_columns(context: object) -> tuple[Iterable[object], Iterable[object]]:
    if isinstance(context, Mapping):
        titles = context.get("title")
        contents = context.get("content")
        if isinstance(titles, Iterable) and not isinstance(titles, (str, bytes)):
            if isinstance(contents, Iterable) and not isinstance(
                contents, (str, bytes)
            ):
                return titles, contents
        return (), ()
    if isinstance(context, Iterable) and not isinstance(context, (str, bytes)):
        pairs = list(context)
        return (pair[0] for pair in pairs), (pair[1] for pair in pairs)
    return (), ()


def _supporting_titles(value: object) -> Iterable[object]:
    if isinstance(value, Mapping):
        titles = value.get("title", ())
        if isinstance(titles, Iterable) and not isinstance(titles, (str, bytes)):
            return titles
        return ()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return (fact[0] for fact in value)
    return ()


def _2wiki_metadata(
    *,
    corpus_size: int,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    return {
        "name": "2WikiMultihopQA dev",
        "source": "huggingface: voidful/2WikiMultihopQA/validation",
        "provenance": {
            "provider": "huggingface",
            "streaming": True,
            "inputs": [
                {
                    "role": "examples",
                    "dataset": "voidful/2WikiMultihopQA",
                    "config": None,
                    "split": "validation",
                }
            ],
            "document_id": "sha1(title + '\\0' + text)[:16]",
            "relevance": "supporting_facts.title within each validation example",
        },
        "corpus_size": corpus_size,
        "query_size": len(queries),
        "qrels_size": len(qrels),
        "qrels_rows": sum(len(relevant) for relevant in qrels.values()),
    }


def build_2wiki(
    destination: Path, *, load_dataset_fn: DatasetLoader | None = None
) -> dict[str, object]:
    loader = _load_dataset_fn(load_dataset_fn)
    examples = loader("voidful/2WikiMultihopQA", split="validation", streaming=True)
    fragment_path, fragment_stream = _new_fragment(destination)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    try:
        writer = _CorpusFragmentWriter(fragment_stream)
        try:
            for example in examples:
                qid = str(example["_id"])
                title_to_doc_id: dict[str, str] = {}
                titles, contents = _context_columns(example.get("context"))
                for raw_title, sentences in zip(titles, contents):
                    title = str(raw_title)
                    if isinstance(sentences, list):
                        text = " ".join(str(sentence) for sentence in sentences).strip()
                    else:
                        text = str(sentences).strip()
                    if not text:
                        continue
                    doc_id = _doc_id_2wiki(title, text)
                    writer.add(doc_id, title, text)
                    title_to_doc_id[title] = doc_id

                relevant: dict[str, int] = {}
                for raw_title in _supporting_titles(
                    example.get("supporting_facts", {})
                ):
                    doc_id = title_to_doc_id.get(str(raw_title))
                    if doc_id is not None:
                        relevant[doc_id] = 1
                if relevant:
                    queries[qid] = str(example["question"])
                    qrels[qid] = relevant
        finally:
            fragment_stream.flush()
            os.fsync(fragment_stream.fileno())
            fragment_stream.close()

        if writer.count == 0 or not queries or not qrels:
            raise ValueError(
                "2Wiki validation produced an empty corpus or relevance set"
            )
        metadata = _2wiki_metadata(
            corpus_size=writer.count, queries=queries, qrels=qrels
        )
        _write_dataset_atomic(
            destination,
            metadata=metadata,
            corpus_fragment=fragment_path,
            queries=queries,
            qrels=qrels,
        )
        return metadata
    finally:
        if not fragment_stream.closed:
            fragment_stream.close()
        fragment_path.unlink(missing_ok=True)


def _trec_metadata(
    *,
    corpus_size: int,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    return {
        "name": "BEIR trec-covid test",
        "source": "huggingface: BeIR/trec-covid",
        "provenance": {
            "provider": "huggingface",
            "streaming": True,
            "inputs": [
                {
                    "role": "corpus",
                    "dataset": "BeIR/trec-covid",
                    "config": "corpus",
                    "split": "corpus",
                },
                {
                    "role": "queries",
                    "dataset": "BeIR/trec-covid",
                    "config": "queries",
                    "split": "queries",
                },
                {
                    "role": "qrels",
                    "dataset": "BeIR/trec-covid-qrels",
                    "config": None,
                    "split": "test",
                },
            ],
            "relevance": "positive qrels scores only",
        },
        "corpus_size": corpus_size,
        "query_size": len(queries),
        "qrels_size": len(qrels),
        "qrels_rows": sum(len(relevant) for relevant in qrels.values()),
    }


def build_trec_covid(
    destination: Path, *, load_dataset_fn: DatasetLoader | None = None
) -> dict[str, object]:
    loader = _load_dataset_fn(load_dataset_fn)
    corpus_rows = loader("BeIR/trec-covid", "corpus", split="corpus", streaming=True)
    query_rows = loader("BeIR/trec-covid", "queries", split="queries", streaming=True)
    qrel_rows = loader("BeIR/trec-covid-qrels", split="test", streaming=True)

    qrels: dict[str, dict[str, int]] = {}
    for row in qrel_rows:
        score = int(row.get("score") or 0)
        if score <= 0:
            continue
        qid = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        qrels.setdefault(qid, {})[doc_id] = score

    queries: dict[str, str] = {}
    for row in query_rows:
        qid = str(row["_id"])
        text = str(row.get("text") or "").strip()
        if text and qid in qrels:
            queries[qid] = text
    qrels = {qid: relevant for qid, relevant in qrels.items() if qid in queries}
    required_doc_ids = {doc_id for relevant in qrels.values() for doc_id in relevant}

    fragment_path, fragment_stream = _new_fragment(destination)
    try:
        writer = _CorpusFragmentWriter(fragment_stream)
        try:
            for row in corpus_rows:
                doc_id = str(row["_id"])
                if writer.add(
                    doc_id,
                    str(row.get("title") or ""),
                    str(row.get("text") or ""),
                ):
                    required_doc_ids.discard(doc_id)
        finally:
            fragment_stream.flush()
            os.fsync(fragment_stream.fileno())
            fragment_stream.close()

        if writer.count == 0 or not queries or not qrels:
            raise ValueError("TREC-COVID produced an empty corpus or relevance set")
        if required_doc_ids:
            preview = ", ".join(sorted(required_doc_ids)[:5])
            raise ValueError(
                f"TREC-COVID qrels reference missing corpus ids: {preview}"
            )
        metadata = _trec_metadata(
            corpus_size=writer.count, queries=queries, qrels=qrels
        )
        _write_dataset_atomic(
            destination,
            metadata=metadata,
            corpus_fragment=fragment_path,
            queries=queries,
            qrels=qrels,
        )
        return metadata
    finally:
        if not fragment_stream.closed:
            fragment_stream.close()
        fragment_path.unlink(missing_ok=True)


def _selected_datasets(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DATASET_FILENAMES)
    selected: list[str] = []
    for value in values:
        for name in (part.strip() for part in value.split(",")):
            if not name:
                continue
            if name not in DATASET_FILENAMES:
                available = ", ".join(DATASET_FILENAMES)
                raise ValueError(f"unknown dataset {name!r}; available: {available}")
            if name not in selected:
                selected.append(name)
    if not selected:
        raise ValueError("--only did not select a dataset")
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="directory that will receive 2wiki_dev.json and/or trec_covid.json",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="2wiki or trec_covid; repeat the option or use a comma-separated list",
    )
    return parser


def main(
    argv: Sequence[str] | None = None, *, load_dataset_fn: DatasetLoader | None = None
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        selected = _selected_datasets(args.only)
    except ValueError as exc:
        parser.error(str(exc))

    builders = {"2wiki": build_2wiki, "trec_covid": build_trec_covid}
    for name in selected:
        destination = args.out_dir / DATASET_FILENAMES[name]
        metadata = builders[name](destination, load_dataset_fn=load_dataset_fn)
        print(
            f"{name}: {destination} "
            f"({metadata['corpus_size']} docs, {metadata['query_size']} queries)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
