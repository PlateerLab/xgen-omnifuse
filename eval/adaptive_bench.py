"""Does a retrieval system's *memory* improve retrieval? — with the controls that make the
question answerable.

This is the axis that defines synaptic-**memory**: it is stateful and learns (Hebbian
reinforcement of graph nodes/edges), while OmniFuse's retrieval is otherwise stateless.
Neither project had measured whether the learning improves retrieval quality — synaptic's
own memory eval is a contract smoke-gate.

Two axes, because they answer different questions:

  --split N     DISJOINT queries. Feedback on one half, evaluate on the other. Tests whether
                memory generalizes across *different* questions. It should not, and does not.
  --paraphrase  RE-QUERY. Feedback on the original questions, evaluate on paraphrases of
                them. This is what memory is for: the same need, different words.

**The controls are not optional.** A naive run reports a memory that works when it does
not — it did for us, and we shipped and then reverted the result. Always printed:

  shuffled : the same chunks remember the same volume of query text, but the
             (query <-> chunk) pairing is permuted. Query-conditional memory must die here.
  random-q : each confirmed chunk remembers a random *other* feedback query.

  covered   : eval queries whose relevant chunk remembers something
  uncovered : eval queries whose relevant chunk remembers nothing. Memory cannot move
              these. If it does, the effect is a corpus-wide scoring artifact — e.g.
              injecting query text into the body inflates document frequency and deflates
              the IDF of query vocabulary. `Feedback` indexes memory as an *evidence field*
              precisely so this is structurally impossible.

    python eval/adaptive_bench.py --data-dir <synaptic tests/benchmark/data> --dataset nfcorpus.json
    python eval/adaptive_bench.py --golden golden.json --paraphrase paraphrase.json --corpus corpus.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import reciprocal_rank  # noqa: E402  — synaptic's own scorer, byte-identical
from omnifuse import Feedback, build_inmemory  # noqa: E402

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


def _ids(of, q):
    return [c.id for c, _ in of.retrieve(q, limit=K)]


def _report(chunks, eval_qs, pairs, rng, cold):
    """eval_qs: [(text, relevant_set)]; pairs: [(remembered_query, chunk_id)]"""
    base = [reciprocal_rank(_ids(cold, t), r) for t, r in eval_qs]
    remembered = {d for _, d in pairs}
    cov = [i for i, (_, r) in enumerate(eval_qs) if r & remembered]
    unc = [i for i in range(len(eval_qs)) if i not in set(cov)]
    m = lambda xs, idx: (sum(xs[i] for i in idx) / len(idx)) if idx else 0.0  # noqa: E731

    docs = [d for _, d in pairs]
    shuffled = [q for q, _ in pairs]
    rng.shuffle(shuffled)
    pool = [q for q, _ in pairs] or [""]
    memories = {
        "real": pairs,
        "shuffled (placebo)": list(zip(shuffled, docs)),
        "random-q (placebo)": [(rng.choice(pool), d) for d in docs],
    }

    print(f"cold: all={sum(base)/len(base):.4f}  covered={m(base, cov):.4f} ({len(cov)})  "
          f"uncovered={m(base, unc):.4f} ({len(unc)})  chunks remembering={len(remembered)}")
    print(f"{'memory':22}{'Δall':>9}{'Δcovered':>11}{'Δuncovered':>13}")
    for label, prs in memories.items():
        fb = Feedback()
        for q, d in prs:
            fb.remember(q, [d])
        of = build_inmemory([], [], chunks, feedback=fb)
        w = [reciprocal_rank(_ids(of, t), r) for t, r in eval_qs]
        print(f"{label:22}{sum(w)/len(w) - sum(base)/len(base):>+9.4f}"
              f"{m(w, cov) - m(base, cov):>+11.4f}{m(w, unc) - m(base, unc):>+13.4f}")

    print("\nIf the placebos match `real`, there is no query-conditional memory.\n"
          "If Δuncovered is materially non-zero, the effect is a corpus-wide scoring artifact.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="synaptic tests/benchmark/data (disjoint-query axis)")
    ap.add_argument("--dataset", default="nfcorpus.json")
    ap.add_argument("--split", type=int, default=0, choices=(0, 1))
    ap.add_argument("--corpus", help="chunk list json (re-query axis)")
    ap.add_argument("--golden", help="{queries, relevant_docs} json")
    ap.add_argument("--paraphrase", help="{qid: paraphrased query} json")
    ap.add_argument("--seed", type=int, default=20260710)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    if a.paraphrase:
        corpus = json.load(open(a.corpus, encoding="utf-8"))
        g = json.load(open(a.golden, encoding="utf-8"))
        para = json.load(open(a.paraphrase, encoding="utf-8"))
        chunks = [{"id": c["id"], "title": c["title"], "text": c["text"]} for c in corpus]
        qids = [q for q in g["queries"] if q in para]
        gold = {q: g["relevant_docs"][q][0] for q in qids}
        mem_q = [q for i, q in enumerate(qids) if i % 2 == 1]   # even-indexed stay uncovered
        eval_qs = [(para[q], {gold[q]}) for q in qids]
        pairs = [(g["queries"][q], gold[q]) for q in mem_q]
        print(f"re-query axis: corpus={len(chunks)} eval(paraphrases)={len(eval_qs)}")
    else:
        corpus, ql = _parse(json.load(open(Path(a.data_dir) / a.dataset, encoding="utf-8")))
        F = [q for i, q in enumerate(ql) if i % 2 == a.split]
        E = [q for i, q in enumerate(ql) if i % 2 != a.split]
        chunks = [{"id": d, "title": t, "text": x} for d, t, x in corpus]
        cold0 = build_inmemory([], [], chunks)
        pairs = [(t, d) for _, t, rel in F for d in _ids(cold0, t) if d in rel]
        eval_qs = [(t, r) for _, t, r in E]
        print(f"disjoint-query axis: {a.dataset} split-{a.split} corpus={len(corpus)} "
              f"F={len(F)} E={len(E)}")

    _report(chunks, eval_qs, pairs, rng, build_inmemory([], [], chunks))


if __name__ == "__main__":
    main()
