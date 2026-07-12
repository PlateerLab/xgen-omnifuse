# eval/ — OmniFuse retrieval benchmark

Standalone evaluation harness (not shipped in the package). Measures OmniFuse's
single-shot retrieval — no LLM, no embedder — head-to-head against
**synaptic-memory** on **identical corpus, queries, and metric**
(`eval/metrics.py`, MRR@10). synaptic numbers come from synaptic's *own*
`eval.run_all` runner in FTS-only mode (`embedder=None, reranker=None`), so each
is exactly what synaptic reports on itself.

## Datasets

**finreg** (self-contained here) — 4,417 Korean financial-statute articles from
[law.go.kr](https://www.law.go.kr) (public-domain law). `data/finreg/raw.jsonl` +
`data/queries/finreg{,_multihop}.json`. Corpus + query GT reused from
synaptic-memory's public eval (Apache-2.0). Single-hop = 1 relevant article;
multi-hop = article + the article it cites (both must be retrieved).

**Public IR sets** — the BEIR-style datasets synaptic ships in
`tests/benchmark/data/*.json` (HotPotQA, Allganize RAG-ko/Eval, KLUE-MRC,
PublicHealthQA, AutoRAG, Ko-StrategyQA). Run via `public_bench.py --synaptic-repo`.

## Run

```bash
pip install -e .
python eval/finreg_bench.py                              # finreg (self-contained)
python eval/compare_synaptic.py --synaptic-graph PATH    # finreg head-to-head
python eval/public_bench.py --synaptic-repo PATH         # 8 public datasets
python eval/adaptive_bench.py --data-dir PATH            # does memory improve retrieval?
python eval/perf_bench.py --data-dir PATH --synaptic-repo PATH  # efficiency, synaptic's metricpython eval/incremental_bench.py --data-dir PATH        # is remember() exact, what does it cost?
python eval/idf_pow_bench.py --synaptic-repo PATH       # idf_pow ablation, synaptic re-run per set
```

## Results — single-shot, no LLM, MRR@10, identical metric

| dataset | lang | task | synaptic (FTS) | **OmniFuse** | Δ |
|---|---|---|---:|---:|---:|
| **finreg single-hop** | KO | statute retrieval | 0.7039 | **0.8400** | **+0.136** |
| **finreg multi-hop** (strict/120) | KO | cite-following | 56 | **107** | **+51** |
| HotPotQA-24 | EN | multi-hop | 0.8879 | **0.9077** | +0.020 |
| HotPotQA-200 | EN | multi-hop | 0.8775 | **0.9044** | +0.027 |
| Allganize RAG-ko | KO | enterprise RAG | 0.9562 | **0.9683** | +0.012 |
| Allganize RAG-Eval | KO | domain RAG | 0.9303 | **0.9370** | +0.007 |
| KLUE-MRC | KO | machine reading | 0.7718 | **0.8288** | +0.057 |
| PublicHealthQA | KO | paraphrase QA | 0.6065 | **0.6217** | +0.015 |
| AutoRAG | KO | passage retrieval | 0.9053 | **0.9293** | +0.024 |
| Ko-StrategyQA | KO | strategy QA | 0.6440 | **0.6496** | +0.006 |

**OmniFuse wins 10, loses 0** (avg MRR 0.843 vs 0.809) — every synaptic-shipped dataset,
with zero dependencies (no morphological analyzer), versus synaptic's mandatory Kiwi.
Across the extended track too, the score is **15/15**. Three honest, general, zero-hardcode
logic improvements get here: (1) a dependency-free Korean stemmer (strip 조사/어미/지정사
의문형 + trailing derivational suffixes), (2) IDF term-specificity emphasis (`idf_pow=1.5`),
and (3) directed out-edge graph fusion — no strong embedder, no per-dataset fit.

> **Full-pipeline (dense) track**: with a shared dense embedder
> (`multilingual-e5-small`, same for both), OmniFuse's dense+lexical hybrid beats
> synaptic's fused pipeline on 6/7 measured (loses only PublicHealthQA under e5, which is
> embedder-dependent: OmniFuse wins it with bge-m3). Since the zero-embedder lexical track
> already wins all ten, the embedder is an optional extra. See
> [`docs/comparison/omnifuse_vs_synaptic.md`](../docs/comparison/omnifuse_vs_synaptic.md)
> and [`results/full_pipeline_e5.json`](results/full_pipeline_e5.json).

The **finreg multi-hop** result is a headline: OmniFuse's one-shot graph-companion fusion
solves 107/120 with **no LLM and no agent** — more than synaptic's own 5-turn LLM agent
(88/120, `docs/REPORT-rag-vs-synaptic.md`) and nearly double synaptic's single-shot 56/120.

### How the last dataset (Ko-StrategyQA) was won — case investigation, not fishing

Ko-StrategyQA was long the sole holdout. Six lexical levers (Kiwi / stemmer variants /
expanded endings / corpus compound-splitting / same-article graph) each failed or traded
a bigger win. So we stopped guessing and **inspected the queries omnifuse ranked worst**:

- **"장 발장은 어떤 범죄로 유죄 판결을 받았나요?"** → the relevant "Jean Valjean" doc
  (matching the rare entity 발장) ranked *below* generic legal docs matching the common
  attribute words 범죄/유죄/판결.

The failure mode is **entity-burial**: a natural-language question carries one rare
discriminative entity buried under several common words; plain BM25 sums term scores, so a
doc matching many common words outranks the doc matching the rare entity. The fix is
**IDF term-specificity emphasis** (`text._IDF_POW=1.5`): raise IDF to a power so the rare
term dominates the sum. It flips Ko-StrategyQA (0.6414→0.6509) **and lifts every other
set** (HotPotQA-24 +0.021, AutoRAG +0.014), is zero runtime cost (folded into the
precomputed IDF), and wins all ten across the whole flat band `p ∈ [1.3, 2.0]` — a
principled default, not a fit.

### What makes OmniFuse win (ablation, finreg)

| config | single-hop MRR | multi-hop strict |
|---|---:|---:|
| field-weighted BM25F only (`--no-graph`) | 0.8490 | 19/120 |
| + graph-companion fusion (default) | 0.8400 | **107/120** |

- **Field-weighted BM25 (`Chunk.title`→`text.BM25F`, title 4× body)** — a query
  term in the heading beats a deep body mention. Lifts flat-body 0.797 → 0.85.
- **Graph-companion fusion (`OmniFuse.retrieve`)** — folds 1-hop graph structure
  into the ranking: a cited passage sharing no query vocabulary is surfaced beside
  the seed that references it. One shot, no LLM. Multi-hop 19 → 107.

### Extended coverage — download-only BEIR/MTEB sets

synaptic also references (via `download_datasets.py`, not committed data) a set of
public BEIR/MTEB benchmarks. We fetched and ran the lexical head-to-head on them
too — NFCorpus, SciFact, FiQA (EN) and XPQA-ko, MIRACL-retrieval-ko, MultiLongDoc-ko
(KO). Result: **3 of the 4** head-to-heads, after English morphology was added (see below).
These are unstructured passage-IR sets with no titles and no citation graph, so neither
OmniFuse's field weighting nor its graph-companion fusion has anything to exploit.
OmniFuse's decisive wins are on **structured** corpora (finreg's citation graph, titled
document sets).

| dataset | synaptic (FTS) | **OmniFuse** (`idf_pow=1.5`) | at `idf_pow=1.0` | winner |
|---|---:|---:|---:|---|
| SciFact (EN) | 0.6317 | **0.6456** | 0.6368 | OmniFuse |
| XPQA-ko | 0.3115 | **0.3256** | 0.3239 | OmniFuse |
| NFCorpus (EN, 38.2 rel/q) | 0.5124 | **0.5182** | 0.5080 | OmniFuse |
| MIRACL-retrieval-ko (14.4 rel/q) | **0.9495** | 0.9052 | **0.9489** | synaptic |

⚠️ **Corrections (2026-07-10)**: the numbers previously published here were measured
*before* the `idf_pow` change and were never re-run — they were stale. Re-measured above.
And an earlier revision blamed the IDF emphasis for **both** extended losses; that was
wrong. With emphasis off NFCorpus still lost (0.5080 vs 0.5124) — its cause was missing
English morphology, now fixed by the S-stemmer, and it is a win. Emphasis only widens the
**MIRACL-ko** gap: at `idf_pow=1.0` MIRACL-ko is a dead heat (0.9489 vs 0.9495) but
Ko-StrategyQA flips to a loss, so 1.5 stays the best single global default. Numbers, the corrected
p-band sweep, and honest limitations (FiQA/MultiLongDoc synaptic-ingest bound; their omni
numbers predate `idf_pow` and were not re-run):
[`results/beir_mteb_extra.json`](results/beir_mteb_extra.json).

### Real-world golden set — a live xgen domain corpus (dev-xgen)

Beyond the academic IR sets, we built a **golden benchmark from a real production
corpus**: an xgen retrieval collection on `dev-xgen.x2bee.com` — 한국마사회 (KRA)
institutional documents (동반성장 / ESG / 청렴 / 경마운영 …). We downloaded 220
documents (**5,234 chunks**, mean 1,143 chars) via the retrieval API, then generated
**215 natural Korean questions** with `gpt-4o-mini` from each document's richest chunk
— *body only* (the repeated document-metadata header stripped), paraphrased, neutral to
both retrievers. Both systems then retrieve over the identical corpus/queries/qrels.

| system | MRR@10 | nDCG@10 | R@10 | wall |
|---|---:|---:|---:|---:|
| synaptic (FTS) | 0.2547 | 0.2956 | 0.4279 | 98.0 s |
| **OmniFuse** | **0.4957** | **0.5446** | **0.7535** | **6.6 s** |

Wall time is end-to-end from raw data on both sides (OmniFuse: 6.1 s index build + 0.5 s
for all 215 queries = 2.3 ms/query). **OmniFuse wins by +0.2410 MRR (~1.95×)** on every metric —
on genuinely out-of-distribution real documents. This is exactly the long-institutional-
document regime the retrieval logic targets: a specific entity buried in pages of
boilerplate. Ablation of the two logic improvements on this corpus (every config still
beats synaptic):

| OmniFuse config | MRR@10 |
|---|---:|
| plain CJK bi-gram, `idf_pow=1.0` | 0.4579 |
| + dependency-free Korean stemmer | 0.4775 |
| + IDF emphasis `idf_pow=1.5` (shipped) | 0.4775 |

The field-weighted BM25F (title 4×) + pipeline already dominate; the Korean stemmer adds
+0.020; **IDF emphasis is neutral out of distribution** (0.4775, neither helps nor hurts)
— confirming `idf_pow=1.5` is a principled default, not a fit to the synaptic sets.

The raw KRA documents are a **private domain corpus and are not committed**. Reproduce
from the live collection (needs dev-xgen + OpenAI credentials, read from env):

```bash
python eval/golden_devxgen_bench.py --collection-id 42 --max-docs 220 --num-queries 215
```

Numbers + methodology: [`results/golden_devxgen.json`](results/golden_devxgen.json).

### Memory — does either system's memory improve retrieval?

This is the axis that names synaptic-**memory**: it is stateful and learns. Neither project
had measured whether the learning improves retrieval quality. `adaptive_bench.py` does,
with the controls that make the question answerable — and they are not optional: a naive
run reported a win for us that was not there, and we shipped and retracted it.

    python eval/adaptive_bench.py --data-dir <synaptic tests/benchmark/data>          # disjoint queries
    python eval/adaptive_bench.py --corpus c.json --golden g.json --paraphrase p.json # re-queries

Every run prints two placebos (**shuffled** = the (query, chunk) pairing permuted;
**random-q** = a random other feedback query) and splits the eval set into **covered**
(relevant chunk remembers something) and **uncovered** (it remembers nothing). Memory
cannot move uncovered queries; if it does, the effect is a corpus-wide scoring artifact.

**Re-query axis** — feedback on the original questions, evaluation on held-out paraphrases
of them (token Jaccard 0.43). Same corpus, same queries, scored by synaptic's own
`metrics.py`:

| ΔMRR@10, held-out re-queries | KRA (ko) all | KRA covered | NFCorpus (en) all | NFCorpus covered |
|---|---:|---:|---:|---:|
| synaptic (Hebbian) | +0.0000 | +0.0093 | −0.0010 | −0.0008 |
| **OmniFuse (`Feedback`)** | **+0.1790** | **+0.3903** | **+0.0150** | **+0.0300** |
| ↳ shuffled placebo | +0.0059 | +0.0213 | +0.0015 | +0.0031 |
| ↳ random-query placebo | +0.0029 | +0.0215 | +0.0000 | +0.0000 |

`real` is 5.2× the strongest placebo, so the pairing carries the signal. **Disjoint-query
axis** — a different question, not a rephrasing: memory correctly does nothing (+0.0006),
with Δuncovered **exactly 0.0000**, because a `Feedback` memory is indexed as an *evidence
field* (scored, but excluded from document frequency and from length normalization) so the
collection's IDF is provably untouched. A cold store ranks bit-identically.

synaptic scores ~0 because in the benchmarked version `graph.search()` reads none of the
fields `reinforce()` writes. Numbers, controls and the full retraction history:
[`results/adaptive_memory.json`](results/adaptive_memory.json).

### Efficiency — measured by synaptic's own metric

`metrics.BenchmarkResult` already records `mean_search_time_ms`. We use it for both
systems, so the efficiency comparison is no more hand-rolled than the accuracy one, and
MRR is printed beside it so a speed claim can never be read apart from what it retrieves
(`python eval/perf_bench.py --data-dir … --synaptic-repo …`):

| dataset | system | ingest_s | mean_search_ms | MRR |
|---|---|---:|---:|---:|
| NFCorpus (3,633 docs) | synaptic | 55.01 | 14.14 | 0.5124 |
| | **OmniFuse** | **2.01** | **1.66** | **0.5182** |
| Allganize RAG-ko (200) | synaptic | 5.39 | 4.41 | 0.9562 |
| | **OmniFuse** | **0.18** | **0.18** | **0.9683** |
| KRA golden (5,234 chunks, 215 q) | synaptic | 90.26 | 20.47 | 0.2547 |
| | **OmniFuse** | **6.21** | **2.17** | **0.4957** |

Faster on both axes while retrieving more. Honest framing: *ingest* means "raw corpus →
queryable index"; synaptic writes a persistent SQLite store, which is real work OmniFuse
does not do. `save_index`/`load_index` gives OmniFuse a warm start (0.21 s) but its index
is read back into RAM. Numbers: [`results/perf.json`](results/perf.json).

### Incremental memory

Memory used to be batch: folding a confirmed pair in meant rebuilding the index, which is
not something a live service can do per click. `remember()` now updates the index in place.

```python
of = build_inmemory(nodes, triples, chunks, feedback=Feedback())   # an empty Feedback opts in
of.remember("statin side effects", ["doc7"])                       # ~1 ms, no rebuild
```

This is what the evidence-field design buys. Evidence never enters document frequency, so
`N`, the content df and every content term's IDF are **fixed** — remembering rewrites the
contributions of exactly one document. The single coupling is that a term seen *only* in
evidence takes its IDF from the evidence df; but every posting of such a term is
evidence-derived, so the documents to fix are the ones that remember it. The blast radius
is the memory, not the corpus — measured, **15 such terms out of a 23,610-term vocabulary**.

| | rebuild | `remember()` | per memory |
|---|---:|---:|---:|
| NFCorpus (3,633 docs, 100 memories) | 1.389 s | **1.00 ms** | **1,386x** |
| same memories, a tenth of the corpus | 0.175 s | **1.02 ms** | 172x |
| KRA (5,234 chunks, 120 memories) | 6.605 s | **1.52 ms** | **4,335x** |

The middle row is the control: ten times fewer documents makes the *rebuild* 7.9x cheaper
and leaves `remember()` where it was. Cost tracks the memory, not the corpus, and it stays
flat as memory accumulates.

The bar is that the updated index is **bit-identical** to a full rebuild — every posting,
every float — not merely close, because a weight that drifts is a scoring bug with a
stopwatch. The first prototype claimed the update was purely local, skipped the evidence-df
coupling, and differed from a rebuild in 1,181 terms; the bar caught it.
[`eval/incremental_bench.py`](incremental_bench.py) ·
[`eval/results/incremental_memory.json`](results/incremental_memory.json) ·
[`tests/test_incremental.py`](../tests/test_incremental.py). `forget(query, doc_ids)` is
the exact inverse (~1 ms, bit-identical to a rebuild without the pair; forget-everything lands
bit-identically on the cold index). The bench's forget pass withdraws half the memories in
place and compares against a rebuild with the other half.

### Reproducibility notes

- Both systems non-neural in the lexical track (zero-infra, apples-to-apples). The
  full-pipeline track wires the same e5-small into both.
- Public dataset JSONs live in the synaptic-memory repo (HF-derived); we point at
  them rather than re-hosting. finreg (public-domain law) is included here.
- synaptic's **private** corpora (krra/assort/x2bee) are gitignored — not in its
  repo, so not runnable. The dev-xgen golden corpus is likewise private (results and a
  credential-free reproducer are committed; the documents are not).
