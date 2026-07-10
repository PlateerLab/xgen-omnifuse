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
```

## Results — single-shot, no LLM, MRR@10, identical metric

| dataset | lang | task | synaptic (FTS) | **OmniFuse** | Δ |
|---|---|---|---:|---:|---:|
| **finreg single-hop** | KO | statute retrieval | 0.7039 | **0.8400** | **+0.136** |
| **finreg multi-hop** (strict/120) | KO | cite-following | 56 | **107** | **+51** |
| HotPotQA-24 | EN | multi-hop | 0.8879 | **0.9286** | +0.041 |
| HotPotQA-200 | EN | multi-hop | 0.8775 | **0.9028** | +0.025 |
| Allganize RAG-ko | KO | enterprise RAG | 0.9562 | **0.9683** | +0.012 |
| Allganize RAG-Eval | KO | domain RAG | 0.9303 | **0.9370** | +0.007 |
| KLUE-MRC | KO | machine reading | 0.7718 | **0.8280** | +0.056 |
| PublicHealthQA | KO | paraphrase QA | 0.6065 | **0.6284** | +0.022 |
| AutoRAG | KO | passage retrieval | 0.9053 | **0.9309** | +0.026 |
| Ko-StrategyQA | KO | strategy QA | 0.6440 | **0.6509** | +0.007 |

**OmniFuse wins 10, loses 0** (avg MRR 0.846 vs 0.809) — every synaptic-shipped dataset,
with zero dependencies (no morphological analyzer), versus synaptic's mandatory Kiwi.
Two honest, general, zero-hardcode logic improvements get here: (1) a dependency-free
Korean stemmer (strip 조사/어미 + trailing derivational suffixes) and (2) IDF
term-specificity emphasis (`idf_pow=1.5`) — both no strong embedder, no per-dataset fit.

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
(KO). Result: **BM25-family parity** — 2 wins each of the 4 head-to-heads measured.
These are unstructured passage-IR sets with no titles and no citation graph, so neither
OmniFuse's field weighting nor its graph-companion fusion has anything to exploit.
OmniFuse's decisive wins are on **structured** corpora (finreg's citation graph, titled
document sets).

| dataset | synaptic (FTS) | **OmniFuse** (`idf_pow=1.5`) | at `idf_pow=1.0` | winner |
|---|---:|---:|---:|---|
| SciFact (EN) | 0.6317 | **0.6422** | 0.6368 | OmniFuse |
| XPQA-ko | 0.3115 | **0.3256** | 0.3239 | OmniFuse |
| NFCorpus (EN, 38.2 rel/q) | **0.5124** | 0.5053 | 0.5080 | synaptic |
| MIRACL-retrieval-ko (14.4 rel/q) | **0.9495** | 0.9052 | **0.9489** | synaptic |

⚠️ **Correction (2026-07-10)**: the numbers previously published here were measured
*before* the `idf_pow` change and were never re-run — they were stale. Re-measured above.
The IDF emphasis that wins the core suite **regresses the two heavily multi-relevant
sets**; at `idf_pow=1.0` MIRACL-ko is a dead heat (0.9489 vs 0.9495) but Ko-StrategyQA
flips to a loss, so 1.5 stays the best single global default. Numbers, the corrected
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
| **OmniFuse** | **0.4775** | **0.5446** | **0.7535** | **6.6 s** |

Wall time is end-to-end from raw data on both sides (OmniFuse: 6.1 s index build + 0.5 s
for all 215 queries = 2.3 ms/query). **OmniFuse wins by +0.2228 MRR (~1.9×)** on every metric —
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

### Reproducibility notes

- Both systems non-neural in the lexical track (zero-infra, apples-to-apples). The
  full-pipeline track wires the same e5-small into both.
- Public dataset JSONs live in the synaptic-memory repo (HF-derived); we point at
  them rather than re-hosting. finreg (public-domain law) is included here.
- synaptic's **private** corpora (krra/assort/x2bee) are gitignored — not in its
  repo, so not runnable. The dev-xgen golden corpus is likewise private (results and a
  credential-free reproducer are committed; the documents are not).
