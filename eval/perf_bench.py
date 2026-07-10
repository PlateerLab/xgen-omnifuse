"""Efficiency head-to-head, scored by synaptic's own metric.

`metrics.BenchmarkResult` already carries an efficiency number — `mean_search_time_ms`,
fed by `add(..., search_time_ms=…)`. We use that, for both systems, so the efficiency
comparison is no more hand-rolled than the accuracy one. Index build / ingest wall time is
measured around each system's own construction path.

    python eval/perf_bench.py --data-dir <synaptic tests/benchmark/data> \
        --dataset nfcorpus.json --synaptic-repo <path>

Reported per system:
  ingest_s          wall time to go from raw corpus to a queryable index
  mean_search_ms    synaptic's own per-query metric (metrics.BenchmarkResult)
  mrr               so a speed claim can never be read apart from what it retrieves
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import BenchmarkResult  # noqa: E402 — synaptic's own scorer
from omnifuse import build_inmemory  # noqa: E402

K = 10


def _parse(data):
    raw = data.get("corpus", data.get("documents", []))
    corpus = []
    if isinstance(raw, dict):
        for did, doc in raw.items():
            corpus.append((str(did), str(doc.get("title", "")), str(doc.get("text", ""))))
    else:
        for doc in raw:
            did = str(doc.get("doc_id", doc.get("_id", doc.get("id", ""))))
            corpus.append((did, str(doc.get("title", "")), str(doc.get("text", doc.get("content", "")))))
    qrels = data.get("relevant_docs", data.get("qrels", {}))
    queries = data.get("queries", {})
    ql = []
    for qid, text in (queries.items() if isinstance(queries, dict) else []):
        rel = qrels.get(qid, [])
        rel = set(map(str, rel.keys() if isinstance(rel, dict) else rel))
        if rel and text:
            ql.append((str(qid), str(text), rel))
    return corpus, sorted(ql, key=lambda x: x[0])


def run_omnifuse(corpus, ql):
    t0 = time.perf_counter()
    of = build_inmemory([], [], [{"id": d, "title": t, "text": x} for d, t, x in corpus])
    ingest = time.perf_counter() - t0

    bench = BenchmarkResult()
    for qid, text, rel in ql:
        t0 = time.perf_counter()
        hits = of.retrieve(text, limit=K)
        ms = (time.perf_counter() - t0) * 1000.0
        seen = []
        for c, _ in hits:
            if c.id not in seen:
                seen.append(c.id)
        bench.add(query_id=qid, query=text, retrieved=seen[:K], relevant=rel, k=K, search_time_ms=ms)
    s = bench.summary()
    return ingest, s["mean_search_time_ms"], s["mrr"]


async def run_synaptic(repo, corpus, ql):
    sys.path.insert(0, str(repo))
    import tempfile

    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    tmp = tempfile.NamedTemporaryFile(prefix="perf_", suffix=".db", delete=False)
    tmp.close()
    t0 = time.perf_counter()
    backend = SqliteGraphBackend(tmp.name)
    await backend.connect()
    graph = SynapticGraph(backend, embedder=None, reranker=None)
    for doc_id, title, text in corpus:
        if text or title:
            await graph.add(title=title or doc_id, content=text, properties={"doc_id": doc_id})
    ingest = time.perf_counter() - t0

    bench = BenchmarkResult()
    for qid, text, rel in ql:
        t0 = time.perf_counter()
        res = await graph.search(text, limit=K * 2)
        ms = (time.perf_counter() - t0) * 1000.0
        seen = []
        for h in res.nodes:
            d = (h.node.properties or {}).get("doc_id", "")
            if d and d not in seen:
                seen.append(d)
        bench.add(query_id=qid, query=text, retrieved=seen[:K], relevant=rel, k=K, search_time_ms=ms)
    s = bench.summary()
    return ingest, s["mean_search_time_ms"], s["mrr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", default="nfcorpus.json")
    ap.add_argument("--synaptic-repo", type=Path, default=None)
    a = ap.parse_args()

    corpus, ql = _parse(json.load(open(Path(a.data_dir) / a.dataset, encoding="utf-8")))
    print(f"{a.dataset}: corpus={len(corpus)} queries={len(ql)}  (scored by synaptic's metrics.py)")
    print(f"{'system':10}{'ingest_s':>11}{'mean_search_ms':>17}{'mrr':>9}")

    i, ms, mrr = run_omnifuse(corpus, ql)
    print(f"{'OmniFuse':10}{i:>11.2f}{ms:>17.2f}{mrr:>9.4f}")

    if a.synaptic_repo:
        i, ms, mrr = asyncio.run(run_synaptic(a.synaptic_repo, corpus, ql))
        print(f"{'synaptic':10}{i:>11.2f}{ms:>17.2f}{mrr:>9.4f}")


if __name__ == "__main__":
    main()
