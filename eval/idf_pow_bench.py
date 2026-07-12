"""Is `idf_pow=1.5` worth its cost? — with synaptic re-run in the same pass.

An ablation that compares OmniFuse's two arms against *recorded* synaptic numbers is not a
head-to-head, it is a memory of one. This bench runs, per dataset, in one process:

    synaptic   via its OWN eval.run_all.run_public_dataset (embedder=None, reranker=None)
    OmniFuse   at idf_pow=1.0 and idf_pow=1.5 (the shipped default)

Identical corpus, queries, qrels, and metric (eval/metrics.py, MRR@10) for all three.

    python eval/idf_pow_bench.py --synaptic-repo /path/to/synaptic-memory
    python eval/idf_pow_bench.py --synaptic-repo PATH --out eval/results/idf_pow_ablation.json

What the two arms decide: `idf_pow` raises IDF to a power, so a rare discriminative term
outweighs several common ones ("entity-burial"). It is the only tuned-looking constant in
the retriever, so it is the one that most deserves an adversarial ablation.
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
from public_bench import parse_public, synaptic_mrr  # noqa: E402 — the same drivers

from omnifuse import build_inmemory  # noqa: E402

# the 8 synaptic-shipped public sets + the 4 extended BEIR/MTEB sets that live in the same dir
DATASETS = [
    ("HotPotQA-24", "hotpotqa_24.json"), ("HotPotQA-200", "hotpotqa.json"),
    ("Allganize RAG-ko", "allganize_rag_ko.json"), ("Allganize RAG-Eval", "allganize_rag_eval.json"),
    ("PublicHealthQA", "publichealthqa_ko.json"), ("AutoRAG", "autorag_retrieval.json"),
    ("KLUE-MRC", "klue_mrc.json"), ("Ko-StrategyQA", "ko_strategyqa.json"),
    ("SciFact", "scifact.json"), ("XPQA-ko", "xpqa_ko.json"),
    ("NFCorpus", "nfcorpus.json"), ("MIRACL-ko", "miracl_retrieval_ko.json"),
    ("FiQA", "fiqa.json"), ("MultiLongDoc-ko", "multilongdoc_ko.json"),
]
K = 10
ARMS = (1.0, 1.2, 1.5)  # band edges + the shipped default


def omni_mrr(path: Path, idf_pow: float) -> float:
    corpus, ql = parse_public(json.load(open(path, encoding="utf-8")))
    chunks = [{"id": d, "title": t, "text": x} for d, t, x in corpus]
    of = build_inmemory([], [], chunks, vector_kwargs={"idf_pow": idf_pow})
    # the exponent must actually reach the index: it is a keyword-only default bound at def
    # time, and a sweep that patched the module constant instead once ran the same value 7x
    assert of.vector._bm25.idf, "empty index"
    bench = BenchmarkResult()
    for qid, text, relevant in ql:
        ranked, seen = [], set()
        for c, _ in of.retrieve(text, limit=K * 2):
            if c.id not in seen:
                seen.add(c.id)
                ranked.append(c.id)
        bench.add(query_id=qid, query=text, retrieved=ranked[:K], relevant=relevant, k=K)
    return bench.summary()["mrr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synaptic-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--datasets", help="comma-separated dataset names to run (default: all)")
    a = ap.parse_args()
    only = {x.strip() for x in a.datasets.split(",")} if a.datasets else None
    bd = a.synaptic_repo / "tests" / "benchmark" / "data"

    print(f"scored by eval/metrics.py (synaptic's own), MRR@10, k={K}. synaptic re-run per dataset.\n")
    print(f"{'dataset':20}{'synaptic':>10}{'omni p=1.0':>12}{'omni p=1.5':>12}{'delta':>9}  who wins")
    print("-" * 76)
    rows, t0 = {}, time.time()
    for name, fn in DATASETS:
        if only and name not in only:
            continue
        path = bd / fn
        if not path.exists():
            print(f"{name:20}{'(missing)':>10}")
            continue
        arms = {p: omni_mrr(path, p) for p in ARMS}
        syn = asyncio.run(synaptic_mrr(a.synaptic_repo, path, name))
        w = [f"p={p} {'wins' if arms[p] > syn else 'LOSES'}" for p in ARMS]
        rows[name] = {"synaptic": round(syn, 4),
                      **{f"omnifuse_idf_pow_{p}": round(arms[p], 4) for p in ARMS},
                      "delta_1.5_minus_1.0": round(arms[1.5] - arms[1.0], 4),
                      "shipped_1.2_beats_synaptic": arms[1.2] > syn,
                      "p1.0_beats_synaptic": arms[1.0] > syn, "p1.5_beats_synaptic": arms[1.5] > syn}
        print(f"{name:20}{syn:>10.4f}{arms[1.0]:>12.4f}{arms[1.5]:>12.4f}"
              f"{arms[1.5]-arms[1.0]:>+9.4f}  {', '.join(w)}")

    print("-" * 76)
    net = sum(r["delta_1.5_minus_1.0"] for r in rows.values())
    w10 = sum(r["p1.0_beats_synaptic"] for r in rows.values())
    w15 = sum(r["p1.5_beats_synaptic"] for r in rows.values())
    lose10 = [k for k, r in rows.items() if not r["p1.0_beats_synaptic"]]
    print(f"net(p=1.5 - p=1.0) over {len(rows)} datasets: {net:+.4f}  (mean {net/len(rows):+.5f})")
    print(f"beats synaptic:  p=1.0 -> {w10}/{len(rows)}   p=1.5 -> {w15}/{len(rows)}")
    if lose10:
        print(f"p=1.0 loses only: {', '.join(f'{k} by {rows[k]['synaptic']-rows[k]['omnifuse_idf_pow_1.0']:.4f}' for k in lose10)}")
    print(f"[{time.time()-t0:.0f}s]")

    if a.out:
        a.out.write_text(json.dumps(
            {"benchmark": "idf_pow ablation with synaptic re-run in the same pass",
             "scorer": "eval/metrics.py (synaptic's own), MRR@10",
             "synaptic_driver": "synaptic's own eval.run_all.run_public_dataset, embedder=None, reranker=None",
             "datasets": rows,
             "net_1.5_minus_1.0": round(net, 4),
             "beats_synaptic": {"idf_pow=1.0": f"{w10}/{len(rows)}", "idf_pow=1.5": f"{w15}/{len(rows)}"}},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
