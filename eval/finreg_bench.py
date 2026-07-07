"""OmniFuse retrieval benchmark on finreg (Korean financial statutes).

Self-contained: needs only ``omnifuse`` (this repo). Loads 4,417 statute
articles, builds an OmniFuse over the zero-infra in-memory backend through the
public ``build_inmemory`` API, and scores single-shot retrieval — no LLM, no
embedder — with the same IR metric (``eval/metrics.py``) used by synaptic-memory.

    python eval/finreg_bench.py            # graph-fusion ON (default)
    python eval/finreg_bench.py --no-graph # ablation: lexical BM25F only

Two query sets:
  * finreg.json           120 single-hop  (1 relevant article)  -> MRR@10
  * finreg_multihop.json  120 multi-hop   (article + cited article) -> strict-solved
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    K, build_reference_triples, load_corpus, load_queries, score_mrr, score_strict, to_chunks_nodes,
)

from omnifuse import build_inmemory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true", help="disable graph-companion fusion (BM25F only)")
    args = ap.parse_args()
    graph_fusion = not args.no_graph

    docs = load_corpus()
    nodes, chunks = to_chunks_nodes(docs)
    triples = build_reference_triples(docs)
    sh = load_queries("finreg.json")
    mh = load_queries("finreg_multihop.json")

    print(f"corpus: {len(docs)} articles, {len(triples)} REFERENCES edges")
    print(f"queries: {len(sh)} single-hop, {len(mh)} multi-hop   (k={K}, graph_fusion={graph_fusion})\n")

    t0 = time.time()
    of = build_inmemory(nodes, triples, chunks, graph_fusion=graph_fusion)

    def run(q: str):
        return [c.id for c, _ in of.retrieve(q, limit=20)]

    s_sh = score_mrr(run, sh)
    s_mh = score_strict(run, mh)

    print(f"single-hop  MRR@10={s_sh['mrr']:.4f}  nDCG@10={s_sh['mean_ndcg@k']:.4f}  "
          f"hit@10={s_sh['hits']}/{s_sh['n']}")
    print(f"multi-hop   strict-solved={s_mh['strict']}/{s_mh['n']}  R@10={s_mh['mean_recall@k']:.4f}")
    print(f"\n[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
