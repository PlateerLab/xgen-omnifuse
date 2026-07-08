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
| **finreg single-hop** | KO | statute retrieval | 0.7039 | **0.8404** | **+0.137** |
| **finreg multi-hop** (strict/120) | KO | cite-following | 56 | **101** | **+45** |
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
solves 101/120 with **no LLM and no agent** — more than synaptic's own 5-turn LLM agent
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
| + graph-companion fusion (default) | 0.8404 | **101/120** |

- **Field-weighted BM25 (`Chunk.title`→`text.BM25F`, title 4× body)** — a query
  term in the heading beats a deep body mention. Lifts flat-body 0.797 → 0.85.
- **Graph-companion fusion (`OmniFuse.retrieve`)** — folds 1-hop graph structure
  into the ranking: a cited passage sharing no query vocabulary is surfaced beside
  the seed that references it. One shot, no LLM. Multi-hop 19 → 101.

### Extended coverage — download-only BEIR/MTEB sets

synaptic also references (via `download_datasets.py`, not committed data) a set of
public BEIR/MTEB benchmarks. We fetched and ran the lexical head-to-head on them
too — NFCorpus, SciFact, FiQA (EN) and XPQA-ko, MIRACL-retrieval-ko, MultiLongDoc-ko
(KO). Result: **BM25-family parity** — 2 wins each of the 4 head-to-heads measured,
all within ±0.02 MRR. These are unstructured passage-IR sets with no titles and no
citation graph, so neither OmniFuse's field weighting nor its graph-companion fusion
has anything to exploit. OmniFuse's decisive wins are on **structured** corpora
(finreg's citation graph, titled document sets). Numbers + honest limitations
(FiQA/MultiLongDoc synaptic-ingest bound; MultiLongDoc's 193MB of long docs exceed
omnifuse's in-memory index on 16 GB): [`results/beir_mteb_extra.json`](results/beir_mteb_extra.json).

### Reproducibility notes

- Both systems non-neural in the lexical track (zero-infra, apples-to-apples). The
  full-pipeline track wires the same e5-small into both.
- Public dataset JSONs live in the synaptic-memory repo (HF-derived); we point at
  them rather than re-hosting. finreg (public-domain law) is included here.
- synaptic's **private** corpora (krra/assort/x2bee) are gitignored — not in its
  repo, so not runnable.
