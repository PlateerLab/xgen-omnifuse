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
| **finreg single-hop** | KO | statute retrieval | 0.7039 | **0.8486** | **+0.145** |
| **finreg multi-hop** (strict/120) | KO | cite-following | 56 | **100** | **+44** |
| HotPotQA-24 | EN | multi-hop | 0.8879 | **0.9077** | +0.020 |
| HotPotQA-200 | EN | multi-hop | 0.8775 | **0.8908** | +0.013 |
| Allganize RAG-ko | KO | enterprise RAG | 0.9562 | **0.9704** | +0.014 |
| Allganize RAG-Eval | KO | domain RAG | 0.9303 | **0.9352** | +0.005 |
| KLUE-MRC | KO | machine reading | 0.7718 | **0.8286** | +0.057 |
| PublicHealthQA | KO | paraphrase QA | 0.6065 | **0.6186** | +0.012 |
| AutoRAG | KO | passage retrieval | 0.9053 | **0.9165** | +0.011 |
| Ko-StrategyQA | KO | strategy QA | **0.6440** | 0.6414 | −0.003 |

**OmniFuse wins 9, loses 1** (avg MRR 0.840 vs 0.809) — with zero dependencies (no
morphological analyzer), versus synaptic's mandatory Kiwi. A dependency-free rule-based
Korean stemmer (strip 조사/어미 + trailing derivational suffixes) flips AutoRAG and
PublicHealthQA — both synaptic wins under a plain CJK bi-gram tokenizer — into OmniFuse
wins. The one loss, Ko-StrategyQA, is a statistical tie (−0.0026, ~1.5 of 592 queries).

> **Full-pipeline (dense) track**: with a shared dense embedder
> (`multilingual-e5-small`, same for both), OmniFuse's dense+lexical hybrid flips its
> only lexical loss — **Ko-StrategyQA becomes a win** — and beats synaptic's fused
> pipeline on 6/7 measured (loses only PublicHealthQA, which is embedder-dependent:
> OmniFuse wins it with bge-m3). Across both tracks OmniFuse is the stronger-or-equal
> system on **9–10 of 10**. See
> [`docs/comparison/omnifuse_vs_synaptic.md`](../docs/comparison/omnifuse_vs_synaptic.md)
> and [`results/full_pipeline_e5.json`](results/full_pipeline_e5.json).

The **finreg multi-hop** result is the headline: OmniFuse's one-shot
graph-companion fusion solves 100/120 with **no LLM and no agent** — more than
synaptic's own 5-turn LLM agent (88/120, `docs/REPORT-rag-vs-synaptic.md`) and
nearly double synaptic's single-shot 56/120.

### Ko-StrategyQA — the one tie, and why we don't chase it

Ko-StrategyQA is the sole set to synaptic, at 0.6414 vs 0.6440 — a −0.0026 MRR gap,
~1.5 of 592 queries, i.e. noise. We tried to cross it the honest way (general,
efficient, zero test-label tuning) and every lever fails or trades a bigger win:

- **Kiwi morphology** wins Ko-StrategyQA but *loses* finreg (0.82 vs 0.85) and HotPotQA.
- **Expanded ending set** nudges it to 0.6418 but regresses Allganize-ko (0.9704→0.9629).
- **Corpus-derived compound splitting** (unsupervised max-match — the same edge Kiwi
  has) moves it ≈0.0000: compound splitting does *not* explain the gap.

Pushing the last 0.0026 would mean fitting a config to the Ko-StrategyQA test labels —
overfitting — so we don't. The shipped full-deriv stemmer is the Pareto-optimal single
choice. (synaptic likewise does not win all of its own benchmarks — see its
AutoRAG/MuSiQue/X2BEE notes.)

### What makes OmniFuse win (ablation, finreg)

| config | single-hop MRR | multi-hop strict |
|---|---:|---:|
| field-weighted BM25F only (`--no-graph`) | 0.8655 | 22/120 |
| + graph-companion fusion (default) | 0.8486 | **100/120** |

- **Field-weighted BM25 (`Chunk.title`→`text.BM25F`, title 4× body)** — a query
  term in the heading beats a deep body mention. Lifts flat-body 0.797 → 0.85.
- **Graph-companion fusion (`OmniFuse.retrieve`)** — folds 1-hop graph structure
  into the ranking: a cited passage sharing no query vocabulary is surfaced beside
  the seed that references it. One shot, no LLM. Multi-hop 22 → 100.

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
