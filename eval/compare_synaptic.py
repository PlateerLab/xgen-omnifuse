"""Head-to-head: OmniFuse vs synaptic-memory on finreg — identical corpus,
queries, and metric (eval/metrics.py, k=10). Both single-shot, no LLM.

OmniFuse runs self-contained. The synaptic column is OPTIONAL and requires:
  * pip install "synaptic-memory[sqlite,korean]"
  * a built graph at eval/finreg_graph.sqlite, e.g. via synaptic-memory's
    eval/datasets/ingest_finreg.py against this same eval/data/finreg/raw.jsonl
Point to it with --synaptic-graph PATH. Without it, only OmniFuse is reported.

    python eval/compare_synaptic.py --synaptic-graph /path/to/finreg_graph.sqlite
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    K, build_reference_triples, load_corpus, load_queries, score_mrr, score_strict, to_chunks_nodes,
)

from omnifuse import build_inmemory  # noqa: E402


def run_omnifuse(sh, mh):
    docs = load_corpus()
    nodes, chunks = to_chunks_nodes(docs)
    triples = build_reference_triples(docs)
    of = build_inmemory(nodes, triples, chunks)  # graph-fusion ON

    def run(q):
        return [c.id for c, _ in of.retrieve(q, limit=20)]

    return score_mrr(run, sh), score_strict(run, mh)


async def run_synaptic(graph_path, sh, mh):
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.evidence_search import EvidenceSearch

    backend = SqliteGraphBackend(str(graph_path))
    await backend.connect()
    searcher = EvidenceSearch(backend=backend, embedder=None, reranker=None)
    cache = {}
    for q in sh + mh:
        res = await searcher.search(q["query"], k=K * 2, fts_seed_limit=30)
        got, seen = [], set()
        for ev in res.evidence:
            did = ev.document_id or (ev.node.properties or {}).get("doc_id", "")
            if did and did not in seen:
                seen.add(did)
                got.append(did)
        cache[q["qid"]] = got
    await backend.close()
    byq = {q["query"]: q["qid"] for q in sh + mh}
    run = lambda qt: cache[byq[qt]]
    return score_mrr(run, sh), score_strict(run, mh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synaptic-graph", type=Path, default=None)
    args = ap.parse_args()

    sh = load_queries("finreg.json")
    mh = load_queries("finreg_multihop.json")

    t0 = time.time()
    o_sh, o_mh = run_omnifuse(sh, mh)
    print(f"OmniFuse done [{time.time() - t0:.1f}s]")

    syn = None
    if args.synaptic_graph and args.synaptic_graph.exists():
        t0 = time.time()
        syn = asyncio.run(run_synaptic(args.synaptic_graph, sh, mh))
        print(f"synaptic done [{time.time() - t0:.1f}s]")
    else:
        print("(synaptic column skipped — pass --synaptic-graph PATH to include it)")

    print("\n" + "=" * 74)
    print(f"{'FINREG single-shot retrieval, no LLM, metric=eval/metrics.py, k=10':^74}")
    print("=" * 74)
    col = f"{'metric':26}{'synaptic-memory':>22}{'OmniFuse':>22}"
    print(col)
    print("-" * 74)

    def row(name, syn_v, omni_v, pct=False):
        sv = (f"{syn_v:.4f}" if syn is not None and isinstance(syn_v, float)
              else (str(syn_v) if syn is not None else "—"))
        ov = f"{omni_v:.4f}" if isinstance(omni_v, float) else str(omni_v)
        print(f"{name:26}{sv:>22}{ov:>22}")

    row("single-hop MRR@10", syn[0]["mrr"] if syn else None, o_sh["mrr"])
    row("single-hop nDCG@10", syn[0]["mean_ndcg@k"] if syn else None, o_sh["mean_ndcg@k"])
    row("single-hop hit@10", f"{syn[0]['hits']}/{syn[0]['n']}" if syn else None, f"{o_sh['hits']}/{o_sh['n']}")
    row("multi-hop strict-solved", f"{syn[1]['strict']}/{syn[1]['n']}" if syn else None, f"{o_mh['strict']}/{o_mh['n']}")
    row("multi-hop R@10", syn[1]["mean_recall@k"] if syn else None, o_mh["mean_recall@k"])
    print("=" * 74)


if __name__ == "__main__":
    main()
