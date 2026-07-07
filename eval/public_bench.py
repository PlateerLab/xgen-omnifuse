"""OmniFuse vs synaptic-memory on synaptic's committed PUBLIC IR datasets.

These datasets (HotPotQA, Allganize, KLUE-MRC, PublicHealthQA, AutoRAG,
Ko-StrategyQA) ship inside the synaptic-memory repo under
``tests/benchmark/data/*.json`` (BEIR-style corpus/queries/qrels). Point at a
synaptic-memory checkout with --synaptic-repo.

OmniFuse runs self-contained (zero deps, field-weighted BM25). The synaptic
column is optional and uses synaptic's OWN eval.run_all.run_public_dataset
(FTS-only: embedder=None, reranker=None) so the number is exactly what synaptic
reports on itself. Identical corpus, queries, qrels, metric (eval/metrics.py), k=10.

    python eval/public_bench.py --synaptic-repo /path/to/synaptic-memory
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import BenchmarkResult  # noqa: E402

from omnifuse import build_inmemory  # noqa: E402

DATASETS = [
    ("HotPotQA-24", "hotpotqa_24.json"), ("HotPotQA-200", "hotpotqa.json"),
    ("Allganize RAG-ko", "allganize_rag_ko.json"), ("Allganize RAG-Eval", "allganize_rag_eval.json"),
    ("PublicHealthQA", "publichealthqa_ko.json"), ("AutoRAG", "autorag_retrieval.json"),
    ("KLUE-MRC", "klue_mrc.json"), ("Ko-StrategyQA", "ko_strategyqa.json"),
]
K = 10


def parse_public(data):
    raw = data.get("corpus", data.get("documents", []))
    corpus = []
    if isinstance(raw, dict):
        for did, doc in raw.items():
            if isinstance(doc, dict):
                corpus.append((str(did), str(doc.get("title", "")), str(doc.get("text", doc.get("content", "")))))
            elif isinstance(doc, str):
                corpus.append((str(did), "", doc))
    else:
        for doc in raw:
            if isinstance(doc, dict):
                did = str(doc.get("doc_id", doc.get("_id", doc.get("id", ""))))
                corpus.append((did, str(doc.get("title", "")), str(doc.get("text", doc.get("content", "")))))
    queries = data.get("queries", [])
    qrels = data.get("relevant_docs", data.get("qrels", {}))
    ql = []
    if isinstance(queries, dict):
        for qid, text in queries.items():
            rel = qrels.get(qid, {})
            relevant = set(map(str, rel)) if isinstance(rel, (dict, list)) else set()
            if relevant and text:
                ql.append((str(qid), str(text), relevant))
    else:
        for q in queries:
            qid = str(q.get("qid", q.get("query_id", q.get("_id", ""))))
            text = str(q.get("query", q.get("question", "")))
            rr = q.get("relevant_docs", q.get("answer_ids", q.get("positive_doc_ids", [])))
            relevant = set(map(str, rr)) if isinstance(rr, (dict, list)) else set()
            if relevant and text:
                ql.append((qid, text, relevant))
    return corpus, ql


def omni_mrr(path):
    corpus, ql = parse_public(json.load(open(path, encoding="utf-8")))
    chunks = [{"id": d, "title": t, "text": x} for d, t, x in corpus]
    of = build_inmemory([], [], chunks)
    bench = BenchmarkResult()
    for qid, text, relevant in ql:
        ranked, seen = [], set()
        for c, _ in of.retrieve(text, limit=K * 2):
            if c.id not in seen:
                seen.add(c.id)
                ranked.append(c.id)
        bench.add(query_id=qid, query=text, retrieved=ranked[:K], relevant=relevant, k=K)
    s = bench.summary()
    return s["mrr"], len(corpus)


async def synaptic_mrr(repo, path, name):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tests" / "benchmark"))
    from eval.run_all import DatasetConfig, run_public_dataset

    r = await run_public_dataset(DatasetConfig(name=name, path=path, quick=True), embedder=None, reranker=None)
    return r.mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synaptic-repo", type=Path, required=True)
    args = ap.parse_args()
    bd = args.synaptic_repo / "tests" / "benchmark" / "data"

    print(f"{'dataset':22}{'synaptic':>12}{'OmniFuse':>12}  winner")
    print("-" * 60)
    wins = 0
    for name, fn in DATASETS:
        path = bd / fn
        if not path.exists():
            print(f"{name:22}{'(missing)':>12}")
            continue
        om, _ = omni_mrr(path)
        syn = asyncio.run(synaptic_mrr(args.synaptic_repo, path, name))
        w = "OmniFuse" if om > syn else ("synaptic" if syn > om else "tie")
        wins += om > syn
        print(f"{name:22}{syn:>12.4f}{om:>12.4f}  {w}")
    print("-" * 60)
    print(f"OmniFuse wins {wins}/{len(DATASETS)} public datasets (MRR@10)")


if __name__ == "__main__":
    main()
