"""Incremental memory: is `remember()` exact, and what does it cost?

Two questions, and the first one gates the second — a fast update that quietly scores
differently from a rebuilt index is not an optimization, it is a bug with a stopwatch.

  1. EXACTNESS. After k calls to `remember()`, is the index bit-identical to the one a
     full rebuild with the same feedback would have produced? Weights are floats; the bar
     is `==`, not `isclose`.
  2. COST. Does an update track the size of the MEMORY or the size of the CORPUS? Evidence
     is excluded from document frequency, so N and every content term's IDF are fixed. The
     only coupling is that a term seen *only* in evidence takes its IDF from the evidence
     df — and all of that term's postings are evidence-derived. So the answer should be
     "the memory", and this bench prints the number of such terms to show how small the
     coupled set really is.

    python eval/incremental_bench.py --data-dir <synaptic tests/benchmark/data>
    python eval/incremental_bench.py --corpus corpus.json --golden golden.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omnifuse import Feedback, build_inmemory  # noqa: E402
from perf_bench import _parse  # noqa: E402 — one parser for the whole suite


def _identical(a, b) -> str | None:
    x, y = a.vector._bm25, b.vector._bm25
    if set(x._pd) != set(y._pd):
        return f"posting keys differ ({len(set(x._pd) ^ set(y._pd))})"
    if x.idf != y.idf:
        return "idf differs"
    for t in y._pd:
        if list(x._pd[t]) != list(y._pd[t]):
            return f"doc list differs at {t!r}"
        if list(x._pw[t]) != list(y._pw[t]):
            return f"weights differ at {t!r}"
    return None


def run(name: str, chunks: list[dict], memories: list[tuple[str, list[str]]], probes: list[str]) -> dict:
    t0 = time.perf_counter()
    inc = build_inmemory([], [], chunks, feedback=Feedback())
    rebuild_s = time.perf_counter() - t0

    per = []
    for q, ids in memories:
        t0 = time.perf_counter()
        inc.remember(q, ids)
        per.append(time.perf_counter() - t0)

    fb = Feedback()
    for q, ids in memories:
        fb.remember(q, ids)
    full = build_inmemory([], [], chunks, feedback=fb)

    drift = _identical(inc, full)
    same_ranks = ([[c.id for c, _ in inc.retrieve(p, limit=10)] for p in probes]
                  == [[c.id for c, _ in full.retrieve(p, limit=10)] for p in probes])

    # forget: withdraw the odd-indexed half in place, compare against a rebuild with the
    # even half. The inverse operation gets the same bar as the forward one.
    per_f = []
    for q, ids in memories[1::2]:
        t0 = time.perf_counter()
        inc.forget(q, ids)
        per_f.append(time.perf_counter() - t0)
    fb2 = Feedback()
    for q, ids in memories[0::2]:
        fb2.remember(q, ids)
    half = build_inmemory([], [], chunks, feedback=fb2)
    drift_f = _identical(inc, half)
    ms_f = sum(per_f) / len(per_f) * 1000 if per_f else 0.0

    ms = sum(per) / len(per) * 1000
    bm = full.vector._bm25
    out = {
        "corpus_docs": len(chunks), "memories": len(memories),
        "rebuild_s": round(rebuild_s, 3), "ms_per_update": round(ms, 3),
        "speedup_per_memory": round(rebuild_s * 1000 / ms, 1),
        "bit_identical_to_rebuild": drift is None, "drift": drift,
        "identical_top10": same_ranks,
        "ms_per_forget": round(ms_f, 3),
        "forget_bit_identical_to_rebuild": drift_f is None, "forget_drift": drift_f,
        "vocab": len(bm.idf), "evidence_only_terms": len(bm._dfe),
        # does cost drift as memory accumulates? (a postings insert is a memmove)
        "ms_per_update_by_decile": [round(sum(per[i:i + max(1, len(per) // 10)])
                                          / len(per[i:i + max(1, len(per) // 10)]) * 1000, 2)
                                    for i in range(0, len(per), max(1, len(per) // 10))],
    }
    print(f"[{name}] docs={out['corpus_docs']} memories={out['memories']}")
    print(f"  rebuild {out['rebuild_s']}s   remember() {out['ms_per_update']} ms "
          f"-> {out['speedup_per_memory']:,.0f}x per memory")
    print(f"  bit-identical to full rebuild: {'YES' if drift is None else 'NO -> ' + drift}"
          f"   identical top-10: {same_ranks}")
    print(f"  forget() {out['ms_per_forget']} ms — bit-identical to a rebuild without the "
          f"forgotten half: {'YES' if drift_f is None else 'NO -> ' + drift_f}")
    print(f"  evidence-only terms (the only IDF that can move): {out['evidence_only_terms']:,}"
          f" / {out['vocab']:,} vocab")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="synaptic tests/benchmark/data")
    ap.add_argument("--dataset", default="nfcorpus.json")
    ap.add_argument("--corpus", help="private corpus json (id/title/text)")
    ap.add_argument("--golden", help="private golden json (queries/relevant_docs)")
    ap.add_argument("--memories", type=int, default=100)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    results = {}
    if a.data_dir:
        corpus, ql = _parse(json.load(open(Path(a.data_dir) / a.dataset, encoding="utf-8")))
        chunks = [{"id": d, "title": t, "text": x} for d, t, x in corpus]
        mem = [(text, sorted(rel)[:3]) for _, text, rel in ql[:a.memories]]
        results[a.dataset] = run(a.dataset, chunks, mem, [t for _, t, _ in ql[:60]])

        # cost must not follow the corpus: same memories, a tenth of the documents
        keep = {d for _, _, rel in ql[:a.memories] for d in rel}
        sub = [c for c in chunks if c["id"] in keep][:len(chunks) // 10 or 1]
        sub += [c for c in chunks if c["id"] not in keep][:max(0, len(chunks) // 10 - len(sub))]
        ids = {c["id"] for c in sub}
        m2 = [(q, [d for d in ds if d in ids]) for q, ds in mem]
        m2 = [p for p in m2 if p[1]]
        if m2:
            results[f"{a.dataset}:corpus/10"] = run(f"{a.dataset} corpus/10", sub, m2,
                                                    [t for _, t, _ in ql[:60]])

    if a.corpus and a.golden:
        c = json.load(open(a.corpus, encoding="utf-8"))
        g = json.load(open(a.golden, encoding="utf-8"))
        chunks = [{"id": x["id"], "title": x["title"], "text": x["text"]} for x in c]
        qids = sorted(g["queries"])[:a.memories]
        mem = [(g["queries"][q], g["relevant_docs"][q]) for q in qids]
        results["private"] = run("private", chunks, mem, [g["queries"][q] for q in qids[:60]])

    if not results:
        ap.error("pass --data-dir or --corpus/--golden")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
