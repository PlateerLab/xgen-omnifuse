"""Adaptive-retrieval benchmark — does a system's *memory* improve retrieval?

synaptic-memory learns by Hebbian reinforcement of graph nodes/edges; OmniFuse remembers
the queries a document was confirmed to answer and indexes them as a `memory` field. Both
its own memory eval and ours were, until now, contract smoke-gates: nobody had measured
whether the dynamics improve retrieval quality.

Protocol (identical for both systems, no labels leak into evaluation):

    queries are split deterministically into FEEDBACK (F) and HELD-OUT EVAL (E)
      1. MRR@10 on E                                            (cold)
      2. replay F: search, then feed back which retrieved docs were relevant
      3. MRR@10 on E again — E was never searched for feedback   (warm)

Report ``warm - cold``. A memory that works is positive; one that injects
query-independent noise is not.

    python eval/adaptive_bench.py --dataset nfcorpus.json --data-dir PATH [--split 0]
    python eval/adaptive_bench.py ... --synaptic-repo PATH   # also run synaptic's Hebbian
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Feedback, build_inmemory  # noqa: E402

K = 10


def _rr(retrieved, relevant):
    for i, d in enumerate(retrieved):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


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


def run_omnifuse(corpus, F, E):
    chunks = [{"id": d, "title": t, "text": x} for d, t, x in corpus]

    def ids(of, q):
        return [c.id for c, _ in of.retrieve(q, limit=K)]

    def mrr(of, qs):
        return sum(_rr(ids(of, t), r) for _, t, r in qs) / len(qs)

    cold_store = build_inmemory([], [], chunks)
    cold = mrr(cold_store, E)

    fb = Feedback()
    for _, text, rel in F:
        fb.observe_ranked(ids(cold_store, text), relevant=rel, query=text)

    warm = mrr(build_inmemory([], [], chunks, feedback=fb), E)
    return cold, warm, len(fb)


async def run_synaptic(repo, corpus, F, E):
    sys.path.insert(0, str(repo))
    import tempfile

    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    tmp = tempfile.NamedTemporaryFile(prefix="adaptive_", suffix=".db", delete=False)
    tmp.close()
    backend = SqliteGraphBackend(tmp.name)
    await backend.connect()
    graph = SynapticGraph(backend, embedder=None, reranker=None)
    for doc_id, title, text in corpus:
        if text or title:
            await graph.add(title=title or doc_id, content=text, properties={"doc_id": doc_id})

    async def search(text):
        res = await graph.search(text, limit=K * 2)
        out = []
        for h in res.nodes:
            d = (h.node.properties or {}).get("doc_id", "")
            if d and all(d != x[1] for x in out):
                out.append((h.node.id, d))
        return out[:K]

    async def mrr(qs):
        tot = 0.0
        for _, t, r in qs:
            tot += _rr([d for _, d in await search(t)], r)
        return tot / len(qs)

    cold = await mrr(E)
    for _, text, rel in F:
        hits = await search(text)
        good = [nid for nid, d in hits if d in rel]
        bad = [nid for nid, d in hits if d not in rel]
        if good:
            await graph.reinforce(good, success=True)
        if bad:
            await graph.reinforce(bad, success=False)
    warm = await mrr(E)
    return cold, warm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus.json")
    ap.add_argument("--data-dir", required=True, help="synaptic tests/benchmark/data")
    ap.add_argument("--split", type=int, default=0, choices=(0, 1))
    ap.add_argument("--synaptic-repo", type=Path, default=None)
    a = ap.parse_args()

    corpus, ql = _parse(json.load(open(Path(a.data_dir) / a.dataset, encoding="utf-8")))
    F = [q for i, q in enumerate(ql) if i % 2 == a.split]
    E = [q for i, q in enumerate(ql) if i % 2 != a.split]
    print(f"{a.dataset}  corpus={len(corpus)}  feedback={len(F)}  held-out={len(E)}")

    cold, warm, remembered = run_omnifuse(corpus, F, E)
    print(f"OmniFuse memory : cold={cold:.4f}  warm={warm:.4f}  delta={warm-cold:+.4f}  "
          f"(docs remembered={remembered})")

    if a.synaptic_repo:
        import asyncio

        c, w = asyncio.run(run_synaptic(a.synaptic_repo, corpus, F, E))
        print(f"synaptic Hebbian: cold={c:.4f}  warm={w:.4f}  delta={w-c:+.4f}")


if __name__ == "__main__":
    main()
