"""Does a retrieval system's *memory* improve retrieval? — with the controls that make
the question answerable.

This is the axis that defines synaptic-**memory**: it is stateful and learns (Hebbian
reinforcement of graph nodes/edges, feeding resonance-ranked search), while OmniFuse is a
stateless one-shot retriever. Neither project had measured whether that learning improves
retrieval quality — synaptic's own memory eval is a contract smoke-gate, not a benchmark.

Protocol: queries split 50/50 into FEEDBACK (F) and HELD-OUT EVAL (E). Replay F with
relevance feedback, then re-measure MRR@10 on E, which was never searched during feedback.

**The controls are not optional.** A naive run of this benchmark will tell you that a
memory works when it does not. Two placebos are therefore always reported:

  shuffled : the same documents remember the same *volume* of query text, but the
             (query <-> document) pairing is permuted. A query-conditional memory must die.
  random-q : each confirmed document remembers a random *other* feedback query.

and the held-out set is split into

  covered   : queries whose relevant documents were remembered
  uncovered : queries whose relevant documents were NOT remembered. A query-conditional
              memory cannot move these. If they move, the mechanism is a corpus-wide
              artifact (e.g. injecting query text inflates document frequency and deflates
              the IDF of query vocabulary), not memory.

We ran exactly this and it killed our own design. See eval/results/adaptive_memory.json.

    python eval/adaptive_bench.py --data-dir <synaptic tests/benchmark/data> \
        --dataset nfcorpus.json --split 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnifuse import build_inmemory  # noqa: E402

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


def _ids(of, q):
    return [c.id for c, _ in of.retrieve(q, limit=K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus.json")
    ap.add_argument("--data-dir", required=True, help="synaptic tests/benchmark/data")
    ap.add_argument("--split", type=int, default=0, choices=(0, 1))
    ap.add_argument("--seed", type=int, default=20260710)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    corpus, ql = _parse(json.load(open(Path(a.data_dir) / a.dataset, encoding="utf-8")))
    F = [q for i, q in enumerate(ql) if i % 2 == a.split]
    E = [q for i, q in enumerate(ql) if i % 2 != a.split]
    chunks = [{"id": d, "title": t, "text": x} for d, t, x in corpus]

    cold_store = build_inmemory([], [], chunks)
    base = [_rr(_ids(cold_store, t), r) for _, t, r in E]
    cold = sum(base) / len(base)

    pairs = [(t, d) for _, t, rel in F for d in _ids(cold_store, t) if d in rel]
    docs = [d for _, d in pairs]
    remembered = set(docs)
    cov = [i for i, (_, _, r) in enumerate(E) if r & remembered]
    unc = [i for i in range(len(E)) if i not in set(cov)]

    print(f"{a.dataset} split-{a.split}: corpus={len(corpus)} F={len(F)} E={len(E)} "
          f"cold={cold:.4f} remembered={len(remembered)} covered={len(cov)}/{len(E)}")

    shuffled = [q for q, _ in pairs]
    rng.shuffle(shuffled)
    ftexts = [t for _, t, _ in F]
    memories = {
        "real": pairs,
        "shuffled (placebo)": list(zip(shuffled, docs)),
        "random-q (placebo)": [(rng.choice(ftexts), d) for d in docs],
    }

    print(f"{'memory':22}{'warm':>9}{'delta':>10}{'Δcovered':>11}{'Δuncovered':>13}")
    for label, prs in memories.items():
        mem: dict[str, list[str]] = {}
        for q, d in prs:
            mem.setdefault(d, []).append(q)
        aug = [{**c, "text": c["text"] + " " + " ".join(mem.get(c["id"], ()))} for c in chunks]
        of = build_inmemory([], [], aug)
        w = [_rr(_ids(of, t), r) for _, t, r in E]
        warm = sum(w) / len(w)
        dc = (sum(w[i] - base[i] for i in cov) / len(cov)) if cov else 0.0
        du = (sum(w[i] - base[i] for i in unc) / len(unc)) if unc else 0.0
        print(f"{label:22}{warm:>9.4f}{warm-cold:>+10.4f}{dc:>+11.4f}{du:>+13.4f}")

    print("\nRead it this way: if the placebos match `real`, there is no query-conditional\n"
          "memory. If Δuncovered is non-zero, the effect is a corpus-wide scoring artifact.")


if __name__ == "__main__":
    main()
