"""Shared finreg loading, reference-edge derivation, and scoring.

Corpus-agnostic where it matters: the REFERENCES edges are derived from the
``article_no`` values actually present in the corpus (no hand-written finreg
regex), mirroring how a structured-document corpus links to itself.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "finreg" / "raw.jsonl"
QUERIES = HERE / "data" / "queries"
K = 10


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def load_corpus() -> list[dict]:
    with open(RAW, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_queries(name: str) -> list[dict]:
    with open(QUERIES / name, encoding="utf-8") as f:
        return [q for q in json.load(f)["queries"] if q.get("relevant_docs")]


def title_line(a: dict) -> str:
    """The article's canonical heading — a short high-signal field."""
    return nfc(f"{a['law']} {a['article_no']}({a['title']})")


_CITE = re.compile(r"(?:「(?P<scope>[^」]+)」\s*)?(?P<key>제\d+조(?:의\d+)?)")


def build_reference_triples(docs: list[dict]) -> list[tuple[str, str, str]]:
    """(src_doc, 'references', tgt_doc) for every article citation resolved to a
    corpus article — intra-law by default, cross-law when a 「statute」 prefix
    names another law present in the corpus."""
    by_key = {(nfc(a["law"]), nfc(a["article_no"])): a["doc_id"] for a in docs}
    laws = {nfc(a["law"]) for a in docs}
    triples: list[tuple[str, str, str]] = []
    for a in docs:
        src, law, text = a["doc_id"], nfc(a["law"]), nfc(a["text"])
        seen: set[str] = set()
        for m in _CITE.finditer(text):
            scope, key = m.group("scope"), m.group("key")
            tgt_law = scope if (scope and scope in laws) else law
            tgt = by_key.get((tgt_law, key))
            if tgt and tgt != src and tgt not in seen:
                seen.add(tgt)
                triples.append((src, "references", tgt))
    return triples


def to_chunks_nodes(docs: list[dict]):
    """OmniFuse inputs: each article is a node AND a titled chunk (id=doc_id)."""
    nodes = [(a["doc_id"], title_line(a)) for a in docs]
    chunks = [{"id": a["doc_id"], "title": title_line(a), "text": nfc(a["text"])} for a in docs]
    return nodes, chunks


# --- scoring (uses eval/metrics.py, identical to synaptic-memory's) ---

def score_mrr(runner, queries, k: int = K) -> dict:
    from metrics import BenchmarkResult

    bench = BenchmarkResult()
    for q in queries:
        ranked, seen = [], set()
        for d in runner(q["query"]):
            if d and d not in seen:
                seen.add(d)
                ranked.append(d)
        bench.add(query_id=q.get("qid", ""), query=q["query"],
                  retrieved=ranked[:k], relevant=set(q["relevant_docs"]), k=k)
    s = bench.summary()
    s["hits"] = sum(1 for x in bench.queries if x["mrr"] > 0)
    s["n"] = len(bench.queries)
    return s


def score_strict(runner, queries, k: int = K) -> dict:
    """Multi-hop: a query is 'solved' only if ALL its relevant docs are in top-k."""
    from metrics import BenchmarkResult

    bench = BenchmarkResult()
    solved = 0
    for q in queries:
        ranked, seen = [], set()
        for d in runner(q["query"]):
            if d and d not in seen:
                seen.add(d)
                ranked.append(d)
        rel = set(q["relevant_docs"])
        bench.add(query_id=q.get("qid", ""), query=q["query"],
                  retrieved=ranked[:k], relevant=rel, k=k)
        if rel.issubset(set(ranked[:k])):
            solved += 1
    s = bench.summary()
    s["strict"] = solved
    s["n"] = len(bench.queries)
    return s
