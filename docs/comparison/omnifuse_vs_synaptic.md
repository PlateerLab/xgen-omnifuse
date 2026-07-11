# OmniFuse vs synaptic-memory — reproducible benchmark

Every number here is regenerable from source (`eval/`), single-shot, **no LLM and
no embedder on either side**. synaptic numbers come from synaptic's *own*
`eval.run_all` runner in FTS-only mode (`embedder=None, reranker=None`) — exactly
what synaptic reports on itself. Same corpus, same queries, same qrels, same
metric (`eval/metrics.py`, MRR@10, k=10).

Machine-readable dump: [`eval/results/omnifuse_vs_synaptic.json`](../../eval/results/omnifuse_vs_synaptic.json).
Last updated: 2026-07-10 (OmniFuse v0.5.0).

---

## Headline — finreg (Korean financial statutes, 4,417 articles)

The fully-reproducible, structural benchmark. Corpus is public-domain law
(shipped in `eval/data/finreg/`); reproduce with `python eval/finreg_bench.py`.

| | synaptic (FTS + graph expand) | **OmniFuse** |
|---|---:|---:|
| single-hop MRR@10 | 0.7039 | **0.8400** |
| single-hop nDCG@10 | 0.7410 | **0.8663** |
| single-hop hit@10 | 103/120 | **114/120** |
| multi-hop strict-solved (both docs in top-10) | 56/120 | **107/120** |
| multi-hop R@10 | 0.6667 | **0.9250** |

OmniFuse's **107/120 multi-hop is one-shot, no LLM, no agent** — more than
synaptic's own **5-turn LLM agent** (88/120, `synaptic docs/REPORT-rag-vs-synaptic.md`)
and nearly double synaptic's single-shot 56/120. This is OmniFuse's graph-companion
fusion: a cited article sharing no query vocabulary is pulled in beside the seed
that references it.

---

## Full sweep — finreg + every committed public IR dataset

synaptic ships 8 public BEIR-style datasets in `tests/benchmark/data/`. Run all of
them with `python eval/public_bench.py --synaptic-repo /path/to/synaptic-memory`.

| dataset | lang | task | synaptic | **OmniFuse** | winner |
|---|---|---|---:|---:|---|
| finreg single-hop | ko | statute retrieval | 0.7039 | **0.8400** | OmniFuse |
| finreg multi-hop `/120` | ko | cite-following | 56 | **107** | OmniFuse |
| HotPotQA-24 | en | multi-hop | 0.8879 | **0.9077** | OmniFuse |
| HotPotQA-200 | en | multi-hop | 0.8775 | **0.9044** | OmniFuse |
| Allganize RAG-ko | ko | enterprise RAG | 0.9562 | **0.9683** | OmniFuse |
| Allganize RAG-Eval | ko | domain RAG | 0.9303 | **0.9370** | OmniFuse |
| KLUE-MRC | ko | machine reading | 0.7718 | **0.8288** | OmniFuse |
| PublicHealthQA | ko | paraphrase QA | 0.6065 | **0.6217** | OmniFuse |
| AutoRAG | ko | passage retrieval | 0.9053 | **0.9293** | OmniFuse |
| Ko-StrategyQA | ko | strategy QA | 0.6440 | **0.6496** | OmniFuse |

**OmniFuse: 10 wins, 0 losses** (avg MRR 0.843 vs 0.809) — **zero dependencies** (no
morphological analyzer) vs synaptic's *mandatory* Kiwi. Two honest, general, zero-hardcode
logic improvements — no strong embedder, no per-dataset tuning — clear the whole board:
(1) a dependency-free Korean rule-based stemmer (strip 조사/어미 + trailing derivational
suffixes → stem bi-grams) flips AutoRAG and PublicHealthQA, and (2) **IDF term-specificity
emphasis** (`idf_pow=1.5`) flips the last holdout Ko-StrategyQA while lifting every other
set. Earlier revisions of this file show the pre-stemmer 7-1-2 and the interim 9-1 results.

---

## How Ko-StrategyQA was won — case investigation, not parameter fishing

Ko-StrategyQA was the last holdout. Six lexical levers each **failed or traded a bigger
win** — recorded here because the dead ends are what made the real fix findable:

| technique tried (all via the real pipeline) | Ko-StrategyQA MRR | verdict |
|---|---:|---|
| base CJK bi-grams (no stemmer) | ~0.630 | loses worse |
| Kiwi morphology | wins Ko-Strat | **regresses finreg 0.85→0.82 + HotPotQA** — Pareto loss |
| refined stemmer (keep 성/상/하) | 0.6387 | loses |
| full-deriv stemmer (strip 성/상/하, trailing) | 0.6414 | closest, still short |
| expanded ending set (more 어미/파생) | 0.6418 | **regresses Allganize-ko 0.9704→0.9629** — Pareto loss |
| corpus-derived compound splitting (unsupervised max-match) | 0.6424 | Δ≈0.0000 — the "Kiwi splits compounds" hypothesis does **not** explain the gap |
| same-article graph-companion fusion | 0.6376 | **hurts MRR** (wrong-article siblings promoted above the first hit) though it lifts nDCG 0.616→0.644 |

So we stopped guessing at tokenizers and **inspected the queries omnifuse ranked worst**
(`eval/` diagnostics). The pattern was unmistakable:

> **"장 발장은 어떤 범죄로 유죄 판결을 받았나요?"** — the relevant *Jean Valjean* passage
> (matching the rare entity **발장**) ranked *below* generic legal passages that matched
> the common attribute words **범죄 / 유죄 / 판결**.

**The failure mode is entity-burial.** A natural-language question carries one rare,
discriminative entity buried under several common words. Plain BM25 *sums* per-term
scores, so a passage matching many common words outranks the one passage matching the
rare entity — even though the entity is the whole point of the question.

**The fix is IDF term-specificity emphasis** (`text._IDF_POW`, default 1.5): raise each
term's IDF to a power > 1 so the rare (high-IDF) entity dominates the sum. It is:

- **general** — a property of BM25 scoring, not of Korean or of this dataset;
- **honest / not a fit** — the win holds across the *entire flat band* `p ∈ [1.3, 2.0]`
  (every one of the ten sets wins at any p in that range), so 1.5 is a robust default,
  the opposite of a knife-edge fit to test labels;
- **efficient** — zero runtime cost; the power is folded into the IDF once at index build;
- ~~**strictly additive**~~ — **retracted 2026-07-10.** That claim was measured before the
  English S-stemmer and the Korean copula fix landed. Re-ablated with the tokenizer that
  actually ships, `idf_pow=1.5` is **not** additive: it buys AutoRAG (+0.0143), HotPotQA-200
  (+0.0119), Ko-StrategyQA (+0.0062), SciFact, PublicHealthQA and XPQA, and it **costs**
  MIRACL-ko (0.9812→0.9617), finreg single-hop (0.8533→0.8400, and one multi-hop query:
  108→107), NFCorpus (0.5236→0.5182), Allganize-ko (−0.0017), KLUE (−0.0003). Across the 13
  MRR datasets the net is **+0.0065 — a wash.** Every synaptic number here is re-measured in the
same pass, through synaptic's own `run_public_dataset`; the first version of this table quoted
synaptic from older result files, which is not a head-to-head but a memory of one. (All twelve
re-runs reproduced the recorded values exactly — the column was right, it just had not been
earned.) Runner: [`eval/idf_pow_bench.py`](../../eval/idf_pow_bench.py).

And the part that is uncomfortable to write: **at `idf_pow=1.0` OmniFuse still beats synaptic
on 14 of the 15 datasets.** The single loss is Ko-StrategyQA, by **0.0006 MRR** — on a set of
592 queries, where one query moving from rank 2 to rank 1 is worth 0.0008. The margin that
decides "15/15" is smaller than a single query, and it is bought by measurably degrading
finreg and MIRACL.

We keep 1.5 because the win holds across the entire band `p ∈ [1.3, 2.0]` (Ko-StrategyQA wins
from p ≥ 1.1: 0.6452 / 0.6466 / 0.6500 / 0.6496 / 0.6487 / 0.6470 at 1.1 / 1.2 / 1.3 / 1.5 /
1.7 / 2.0), so 1.5 is a mid-band choice rather than a knife-edge fit. But `idf_pow=1.0`
remains the right call on heavily multi-relevant corpora, and a reader deserves the number
rather than a slogan.

**Is entity-burial real, or a story we tell?** It predicts something checkable: on the queries
`idf_pow` rescues, the gold document should match a *rarer* query term, and *fewer* of them,
than the wrong top-1 — and on the queries it breaks, the reverse. Measured on Ko-StrategyQA:

| | queries | gold max-IDF | wrong top-1 max-IDF | gold overlap | wrong overlap |
|---|---:|---:|---:|---:|---:|
| `idf_pow` rescues | 49 | **6.220** | 6.077 | **3.4** | 5.8 |
| `idf_pow` breaks | 33 | 5.373 | **6.156** | 4.4 | 5.8 |

The sign flips. `idf_pow` bets on rarity: it wins when the gold document is the rare-term
holder and loses when it is not. A mechanism, then — and also exactly why the net is a wash.

**And the per-query diff that closes the question.** For MIRACL the diff exposed one junk
document and a missing copula ending; here it exposes the opposite. At `p=1.0` vs synaptic,
Ko-StrategyQA splits **92 losses / 94 wins / 406 ties**, with reciprocal-rank mass 36.49
against 36.14 — a dead heat. The losses are textbook entity-burial (이기 팝 → *Iggy Pop*,
소니 플레이스테이션 → *PlayStation*, LinkedIn), i.e. exactly the failure `idf_pow` exists to fix,
which is why 1.5 wins the set. There is no clustered defect underneath; the 0.0006 is the
residue of an even disagreement, and a change aimed at flipping it would be label-fitting.
This is where the improve-loop honestly terminates for this dataset.

**A fix that did not work** (recorded, not shipped): the rarest query terms on Ko-StrategyQA
are stemmer residues, not entities — `있나` (idf 7.63), `#연속되` (8.73), `벌은` (7.43). And
`_KO_SUFFIX` contains 하 but not 되, the same closed class with one half missing, so 연속되나요
stops at `연속되`. Adding 되 is right, and immaterial: Ko-StrategyQA at p=1.0 moves 0.6434 →
0.6435 (still a loss) and at p=1.5 drops to 0.6490. The other residues are out of reach — 있나요
hits the two-character stem floor, and 사용되었다 has no listed trailing ending at all.

That closes all ten with **no strong embedder, no morphological analyzer, no hardcoding, and no
fitting to test labels** — with the caveat above stated plainly, because a suite that is 15/15
only by 0.0006 on one set should say so. Numbers:
[`eval/results/idf_pow_ablation.json`](../../eval/results/idf_pow_ablation.json).

---

## Full-pipeline track — dense + lexical (same embedder both sides)

The zero-infra table above is FTS/lexical only. With a dense embedder wired in
(`intfloat/multilingual-e5-small`, the SAME model for both systems, `query:`/
`passage:` prefixes applied by role), OmniFuse gains a dense+lexical hybrid
(`InMemoryVector` RRF fusion, v0.5). synaptic runs its own fused pipeline
(`run_public_dataset(embedder=e5, reranker=None)`). No cross-encoder reranker on
either side (CPU-infeasible at ~2 pairs/sec). MRR@10:

| dataset | OmniFuse (best fusion) | synaptic (dense) | winner |
|---|---:|---:|---|
| HotPotQA-24 | **0.9688** | 0.9667 | OmniFuse |
| HotPotQA-200 | **0.9354** | 0.9282 | OmniFuse |
| Allganize RAG-ko | **0.9667** | 0.9642 | OmniFuse |
| Allganize RAG-Eval | **0.9539** | 0.9383 | OmniFuse |
| **AutoRAG** | **0.8890** | 0.8576 | **OmniFuse** ⤴ |
| **Ko-StrategyQA** | **0.7780** | 0.7374 | **OmniFuse** ⤴ |
| PublicHealthQA | 0.6440 | **0.6763** | synaptic |

⤴ = AutoRAG and Ko-StrategyQA (both already **lexical wins** above) widen further with a
shared embedder — semantic/paraphrase sets where dense recovers what bigrams miss and
OmniFuse's cleaner fusion edges out synaptic's. Since the zero-embedder lexical track
already wins all ten, the embedder is an optional extra here, not what carries the result.

### Reading the two tracks honestly

- **Fused vs fused (the apples-to-apples full-pipeline comparison)** — both
  systems dense+lexical: OmniFuse wins **6 of 7** (HotPot ×2, Allganize ×2,
  AutoRAG, Ko-StrategyQA), loses PublicHealthQA. Notably synaptic's *own* fusion
  underperforms its FTS on AutoRAG (0.9053 FTS → 0.8576 fused) — OmniFuse's
  fusion is better-balanced there.
- **Each system's single best mode per corpus** — the one place this flips a
  result is AutoRAG, where synaptic's **FTS-only** (0.9053) beats every fused
  score on either side. So AutoRAG is a synaptic win under "best-single-mode"
  but an OmniFuse win under "full-pipeline vs full-pipeline".
- **PublicHealthQA** is *embedder-dependent*: e5-small → synaptic (0.6763 vs
  0.6440); **bge-m3** → OmniFuse (0.7511 vs 0.7295).

Net: the **zero-embedder lexical track already wins all ten**. Adding a shared embedder
keeps OmniFuse ahead on 6/7 fused-vs-fused; the one embedder-dependent set is
PublicHealthQA (~0.03 MRR swing — synaptic under e5, OmniFuse under bge-m3).

**Honest caveat**: no *single* fusion config wins all 10 (dense-favoring helps
semantic sets, lexical-favoring helps keyword sets — a real Pareto conflict that
holds for synaptic too), so "best" picks the strongest of {dense, RRF,
union-primary} per corpus. Numbers:
[`eval/results/full_pipeline_e5.json`](../../eval/results/full_pipeline_e5.json).

## Real-world golden set — a live xgen domain corpus (dev-xgen)

The academic sets are synaptic's own. To test on *independent, real* data we built a
golden benchmark from a production xgen retrieval collection (`dev-xgen.x2bee.com`):
한국마사회 (KRA) institutional documents — 동반성장 / ESG / 청렴 / 경마운영 etc.
Downloaded **220 documents / 5,234 chunks** (mean 1,143 chars) via the retrieval API,
then generated **215 natural Korean questions** with `gpt-4o-mini` from each document's
richest chunk — body only (the repeated `<Document-Metadata>` header stripped),
paraphrased, neutral to both retrievers. Same corpus/queries/qrels to both; synaptic
runs its own `run_public_dataset(embedder=None)` FTS path (identical `(title, text)`
per chunk).

| system | MRR@10 | nDCG@10 | R@10 | wall |
|---|---:|---:|---:|---:|
| synaptic (FTS) | 0.2547 | 0.2956 | 0.4279 | 98.0 s |
| **OmniFuse** | **0.4957** | **0.5446** | **0.7535** | **6.6 s** |

Wall time is end-to-end from raw data on both sides (OmniFuse: 6.1 s build + 0.5 s for all
215 queries). **OmniFuse wins by +0.2410 MRR (~1.95×)** on every metric,
on out-of-distribution real documents — the long-institutional-text regime where a
specific entity is buried in boilerplate. Ablation (every config still beats synaptic):

| OmniFuse config | MRR@10 |
|---|---:|
| plain CJK bi-gram, `idf_pow=1.0` | 0.4579 |
| + dependency-free Korean stemmer | 0.4775 |
| + IDF emphasis `idf_pow=1.5` (shipped) | 0.4775 |

Field-weighted BM25F (title 4×) + the retrieve pipeline already dominate; the stemmer
adds +0.020; **IDF emphasis is neutral out of distribution** — it neither helps nor
hurts here, which is the cleanest evidence that `idf_pow=1.5` is a principled default
rather than a fit to the synaptic-shipped datasets. The raw KRA corpus is private and
**not committed**; a credential-free reproducer (`eval/golden_devxgen_bench.py`) and the
numbers (`eval/results/golden_devxgen.json`) are.

## The asymmetry we had not noticed: English had no morphology

OmniFuse shipped a Korean stemmer and treated Latin as raw surface tokens, so a query for
`statin` could never match a document saying `statins` — while synaptic's FTS5 stems
English. That, not the IDF emphasis, was the NFCorpus loss:

| | synaptic | before | **after S-stemmer** |
|---|---:|---:|---:|
| NFCorpus (EN) | 0.5124 | 0.5053 (loss) | **0.5182 (win)** |
| SciFact (EN) | 0.6317 | 0.6422 | **0.6456** |
| HotPotQA-200 (EN) | 0.8775 | 0.9028 | **0.9044** |
| HotPotQA-24 (EN) | 0.8879 | 0.9286 | 0.9077 (still a win) |
| every Korean set, finreg, golden | — | — | **bit-identical** |

`text._en_stem` is **Harman's S-stemmer**: singularize, nothing else. No `-ing`/`-ed`, no
Porter cascade — and crucially **no tunable parameter**, so there is nothing to fit.

Two things we got wrong on the way, recorded because they are the useful part:

- **We first "proved" English stemming was harmful** (NFCorpus 0.5053 → 0.4779). That run
  was void. The harness patched `omnifuse.text.tokenize`, but `backends/memory.py` binds
  `tokenize` at import — so *documents* were indexed with the old tokenizer while
  *queries* used the new one. Any tokenizer experiment must patch both sides and assert
  the document side actually changed.
- **We predicted that augmenting (`surface` + `#stem`, mirroring the Korean design) would
  beat replacing**, on the theory that naive plural-stripping collides distinct lemmas
  (`news→new`, `aids→aid`) and dilutes IDF. Measured: augmenting is *worse*
  (NFCorpus 0.5090 vs 0.5182). The collision cost is real but smaller than the recall it buys.

### A principled alternative we measured and did **not** ship

`idf^p` inflates a rare term even in documents that barely touch it. A better-motivated
emphasis raises the precomputed `(term, doc)` **contribution** to a power `q` — the
contribution carries idf *and* tf-saturation, so `sum(c^q)` (a generalized power mean over
term evidence) rewards terms the document actually matched. Zero query-time cost. It halves
the MIRACL-ko damage (0.9052 → **0.9321** at q=1.2, core still 8/8) — but it is worse on
nine other datasets, and average MRR over the 12 measured sets drops 0.7628 → 0.7610.
Trading nine datasets for two is not an improvement, and picking `q` per corpus is exactly
the fitting we refuse to do. Not shipped; recorded in
[`eval/results/beir_mteb_extra.json`](../../eval/results/beir_mteb_extra.json).

### Prior art, and what is actually ours

Indexing confirmed queries as a *field of the document* is not our idea. BM25F was defined
over fields like title, body and **anchor text** (Robertson & Zaragoza), and putting
**query-click logs in as a field** is standard web-search practice — reported as the single
most important field in machine-learned BM25F work (Svore & Burges, *A machine learning
approach for improved BM25 retrieval*). Blending document frequency across fields is what
Elasticsearch's `cross_fields` already does. We borrowed the mechanism.

What we chose, and did not find in the literature, is the opposite of that blend: the
evidence field is **excluded from document frequency entirely** and is **not
length-normalized**. And we did not choose it to be different. We chose it because the
controls forced it — injecting remembered queries into the body raised the df of query
vocabulary, deflated its IDF corpus-wide, and moved *uncovered* held-out questions, which is
the signature of a scoring artifact rather than memory. Excluding evidence from df is what
makes Δuncovered structurally zero.

The order matters and is worth being plain about: we did not design for incremental updates
and get honesty for free. We designed for honesty, and the frozen df handed us the
incremental update. Two web searches turned up no prior art for the df-exclusion; that is
not the same as there being none, and we make no novelty claim.

## What the evidence field was actually worth: memory that learns in 1 ms

synaptic is a *memory* system: it is stateful and it learns. OmniFuse's answer, `Feedback`,
was for a long time batch-only — a confirmed pair meant a rebuild — and we listed that as an
honest limitation. Removing it turned out not to need a new mechanism, only the consequences
of the one already there.

Because evidence is excluded from document frequency and is not length-normalized, `N`, the
content df, every content term's IDF, and the content fields' avglen are all fixed. So
`remember()` rewrites the contributions of exactly one document. The one thing that *is*
global — a term seen only in evidence takes its IDF from the evidence df — turns out to be
cheap for the same reason: every posting of such a term is evidence-derived, so the documents
to fix are exactly the ones that remember it. Measured on NFCorpus after 100 memories, that
coupled set is **15 terms out of a 23,610-term vocabulary**.

| | rebuild | `remember()` | per memory |
|---|---:|---:|---:|
| NFCorpus (3,633 docs, 100 memories) | 1.389 s | **1.00 ms** | **1,386x** |
| same memories, a tenth of the corpus | 0.175 s | **1.02 ms** | 172x |
| KRA (5,234 chunks, 120 memories) | 6.605 s | **1.52 ms** | **4,335x** |

The middle row is the control that makes the claim falsifiable: ten times fewer documents
makes the *rebuild* 7.9x cheaper and leaves `remember()` exactly where it was.

The correctness bar was bit equality with a full rebuild — every posting, every float. Our
first prototype asserted the update was purely local, ignored the evidence-df coupling, and
differed from a rebuild in 1,181 terms. It was fast, it was plausible, and it was wrong; the
bar is the only reason we know. This is the third time in this file that an assertion died to
a control, and the pattern is not an accident: a memory that "obviously" cannot move anything
global is exactly the kind of claim that must be measured rather than reasoned.

Contrast: synaptic's `reinforce()` is genuinely incremental (a SQLite write), but in the
benchmarked version `graph.search()` reads none of the fields it writes, so its measured
retrieval delta is noise around zero. Being incremental is not the same as being wired in.

The consolidation cascade completes the picture, measured on the same axis: `maintain()`
(consolidate + decay + prune) **does run** — reinforcement feeds it, it promotes 54 nodes and
decays vitality on all 5,234 — and retrieval moves by exactly **+0.0000**, on a cold store and
on a warm one. Being maintained is not the same as being wired into retrieval either. (TTL
expiry cannot trigger on a same-session benchmark; what is measured is promotion + decay +
prune.) And the inverse direction now exists on our side: `forget()` withdraws a remembered
pair in ~1 ms, bit-identical to a rebuild that never saw it — remember everything, forget
everything, and you land bit-identically on the cold index.

## The last loss was ours: a missing copula in the Korean stemmer

MIRACL-ko was the only dataset synaptic still won. Rather than keep trying weightings, we
dumped the **per-query MRR diff** (scored by synaptic's own `metrics.reciprocal_rank`).
OmniFuse lost 29 queries and won 14 — and the losses clustered on one junk document:

> `q: 그리스의 수도는 어디인가?`  omni **0.200** / syn **1.000**
> omni's top-3: **"내 친구의 집은 어디인가"**, "…(영화)", "…(텔레비전 프로그램)"

Every *"…어디인가?"* ("where is…?") question retrieved the film article whose **title is the
question word**, in a 4×-weighted title field. The cause was not the weighting: the copula's
interrogative paradigm (`-인가/-인가요/-입니까/-인지`) was missing from `_KO_SUFFIX`, so
`어디인가` stemmed to the *rare* token `어디인` instead of the common word `어디`, and
`idf_pow` amplified exactly that rarity. Kiwi splits the copula into morphemes — which is
why synaptic never saw the failure.

Adding the paradigm — a closed linguistic class, like the 조사/어미 already shipped — takes
**MIRACL-ko 0.9052 → 0.9617** (synaptic 0.9495) and wins **every dataset in the suite**. It
is not a fit: every subset from `{인가}` alone to a nine-ending superset wins 8/8 of the
Korean-bearing sets.

Two textbook fixes were tried on the *symptom* first, and both are Pareto losses:

| attempted fix | MIRACL-ko | what it breaks |
|---|---:|---|
| coordination-level matching (`score *= coverage**λ`) | **0.9536** (a win) | Ko-StrategyQA 0.6135, HotPotQA-200 0.8774, PublicHealthQA 0.5824, NFCorpus 0.5040 |
| coverage over lexical units only | 0.9441 | same sets, and MIRACL is worse |
| `minimum_should_match ≥ 2` | 0.9167 | Ko-StrategyQA 0.5663 — it filters gold docs that legitimately match one query word |
| **copula endings (shipped)** | **0.9617** | nothing: 15/15 |

The lesson is the same one this file keeps recording: the diff told us in one look what a
week of weighting sweeps could not.

## Graph fusion: direction is the whole signal (and PPR-style propagation loses)

synaptic propagates over its graph with **Personalized PageRank** (damped power iteration,
per-edge-type weights) plus Hebbian edge reinforcement. OmniFuse's fusion looked crude next
to that — one hop, no damping, no normalization: `companion = max(lexical, α·seed)`. So we
tried to replace it with the textbook-principled thing. **It lost, badly.**

| variant (finreg, same harness) | single-hop MRR | multi-hop strict |
|---|---:|---:|
| no graph | 0.8490 | 19/120 |
| damped, **degree-normalized**, additive propagation (best of α × hops sweep) | 0.6294 | 81/120 |
| additive, no degree normalization (best) | 0.8156 | 94/120 |
| `max(lexical, α·seed)` — bidirectional (what shipped) | 0.8371 | 98/120 |
| **`max(lexical, α·seed)` — out-edges only** | **0.8400** | **102/120** |

Degree normalization is what kills it: a cited article receives `α·seed/deg`, far too small
to reach the top-10. finreg's multi-hop needs the cited article promoted *adjacent to* its
seed, which is exactly what the "crude" rule does. Propagating 2–3 hops never helped either
— the evidence is one hop away.

The real defect was **direction**. `build_reference_triples` emits `(article -references->
cited)`, but `InMemoryGraph._adj_ids` was built symmetrically, so a seed also promoted every
article that *cites* it — a crowd, not evidence. Expanding in-edges only yields **24/120**
multi-hop while still costing single-hop; expanding out-edges only dominates the symmetric
version at every α. `retrieve()`'s own docstring already said "a passage that a strong seed
*references*" — the code simply disagreed with it.

Fixed: `neighbor_ids(..., direction="out"|"in"|"both")`, and fusion asks for `"out"`. In the
real pipeline this took finreg multi-hop **101 → 107/120** and R@10 0.8958 → **0.9250**,
with single-hop effectively unchanged (MRR 0.8404 → 0.8400, but hit@10 113 → 114 and nDCG
0.8651 → 0.8663). No parameter was refitted — `fusion_alpha` stays 0.9. Pass
`fusion_direction="both"` when a graph's edges are genuinely symmetric.

## What makes OmniFuse win (ablation, finreg)

| config | single-hop MRR | multi-hop strict |
|---|---:|---:|
| field-weighted BM25F only (`finreg_bench.py --no-graph`) | 0.8490 | 19/120 |
| + graph-companion fusion (default) | 0.8400 | **107/120** |

1. **Field-weighted BM25 (`Chunk.title` → `text.BM25F`, title 4× body)** — a query
   term in the article heading outranks a chunk that only mentions it deep in the
   body. Flat-body baseline 0.797 → 0.85.
2. **Graph-companion fusion (`OmniFuse.retrieve`)** — 1-hop graph structure folded
   into the ranking (previously graph relations fed only the LLM prompt). One shot,
   no agent. Multi-hop 19 → 101, for a ~0.009 single-hop cost.

## Reverification (2026-07-10) — both systems re-run from scratch

Earlier revisions re-measured only OmniFuse and *reused* synaptic's column from prior
runs. That is an assumption, not evidence, so both systems were re-run end to end. Every
core number reproduces **to the decimal**, on both sides:

| | synaptic (re-run) | OmniFuse (re-run) |
|---|---:|---:|
| finreg single-hop MRR / nDCG | 0.7039 / 0.7410 | 0.8400 / 0.8663 |
| finreg multi-hop strict | 56/120 | 107/120 |
| HotPotQA-24 / -200 | 0.8879 / 0.8775 | 0.9286 / 0.9028 |
| Allganize ko / Eval | 0.9562 / 0.9303 | 0.9683 / 0.9370 |
| KLUE-MRC | 0.7718 | 0.8280 |
| PublicHealthQA | 0.6065 | 0.6284 |
| AutoRAG | 0.9053 | 0.9309 |
| Ko-StrategyQA | 0.6440 | 0.6509 |

Scoring is symmetric by construction: both systems read the same dataset file (same
corpus, queries, qrels), each returns its own top-10, and **both are scored by synaptic's
own `metrics.py` (`reciprocal_rank`, k=10)** — not by a metric we wrote.

### A methodology bug we found and fixed

An earlier sweep of `idf_pow` was **invalid**. `idf_pow: float = _IDF_POW` is a
keyword-only default, bound at *function-definition* time, so monkeypatching the module
constant after import silently changed nothing — every "swept" point was secretly running
the same value. Patching `BM25/BM25F.__init__.__kwdefaults__` makes the sweep actually
vary. The corrected sweep is below; it confirms the p-band claim on the core suite **and**
exposes a regression the broken sweep had hidden.

## The IDF-emphasis trade — it is not free

`idf_pow` wins the core suite across a wide flat band, but **regresses heavily
multi-relevant passage-IR corpora**: betting the score on one rare term is the wrong
strategy when dozens of documents are relevant.

| track | p=1.0 | p=1.3 | **p=1.5** (shipped) | p=2.0 |
|---|---|---|---|---|
| Core (8 public) | 7/8 | **8/8** | **8/8** | **8/8** |
| Extended (4 BEIR/MTEB) | 2/4 | 2/4 | 2/4 | 1/4 |
| MIRACL-ko (14.4 rel/query) | **0.9489** | 0.9144 | 0.9052 | 0.8800 |
| NFCorpus (38.2 rel/query) | **0.5080** | 0.5063 | 0.5053 | 0.4961 |

At `p=1.0`, MIRACL-ko is **0.9489 vs synaptic's 0.9495** — a dead heat — but Ko-StrategyQA
flips to a loss. So `1.5` stays the best single global default (13/15 datasets), and
`idf_pow` is now a documented knob:

```python
build_inmemory(nodes, triples, chunks, vector_kwargs={"idf_pow": 1.0})  # multi-relevant corpora
```

## Performance — measured, with the asymmetry stated

The lexical hot path now folds each `(term, doc)` contribution — which is entirely
query-independent — into the inverted index at build time, so a search is a plain
accumulation of precomputed floats. Rankings are **bit-identical** to the previous
implementation (verified on finreg + all 8 public sets).

| scenario | synaptic | **OmniFuse** |
|---|---:|---:|
| golden set, both from raw data (5,234 chunks, 215 queries) | 98.0 s | **6.6 s** (build 6.1 + query 0.5) |
| per-query latency, golden set | — | **2.3 ms** |
| finreg — synaptic reuses a *prebuilt* SQLite graph, omni rebuilds | 11.0 s | **7.4 s** |
| finreg — **before** this optimization, same conditions | 10.9 s | **26.6 s ← OmniFuse was 2.4× SLOWER** |
| omni self-A/B, 8 public sets (same scores) | — | 249.4 s → **38.7 s** (6.4×) |
| golden set, warm start (`load_index`) | — | **0.43 s** load + 0.5 s queries |

**This is not a blanket "OmniFuse is faster".** Only the first row is apples-to-apples
(both index from raw data). The finreg rows run under conditions *unfavourable* to
OmniFuse — synaptic starts from a persisted index, OmniFuse rebuilt its own every run —
and until this optimization OmniFuse **lost** them outright (26.6 s vs 10.9 s). An earlier
revision of these docs advertised "~7.5× faster" from the golden set alone and never
mentioned the finreg loss; that was an omission, and it is corrected here. The gap is now
closed from both ends (inverted-index scoring, and `save_index`/`load_index` warm start),
but the result remains **workload-dependent**.

## The deepest difference is statefulness — and this is where it lands

synaptic-**memory** learns: Hebbian reinforcement of graph nodes and edges on
co-activation, feeding resonance-ranked search. OmniFuse is a stateless one-shot
retriever. That, not the ranking, is the real gap between the projects — and neither side
had ever measured whether the learning improves retrieval. synaptic's own memory eval
(`eval/scripts/memory_operating_poc.py`) is a contract smoke-gate.

So we built the benchmark: split queries 50/50, replay relevance feedback on one half,
re-measure MRR@10 on the other half, which is never searched during feedback.

| ΔMRR@10 on held-out queries | NFCorpus s0 | NFCorpus s1 | MIRACL-ko s0 |
|---|---:|---:|---:|
| synaptic (Hebbian reinforcement) | −0.0002 | **−0.0174** | **−0.0165** |

**Hebbian reinforcement is neutral to harmful in every measurement.** The reason is
structural: reinforcing nodes and edges learns a *query-independent* prior — "this document
tends to be relevant". Relevance is a property of a *(query, document)* pair. A prior
learned on statins is noise for a query about vitamin D.

### We then built a memory of our own, believed we had won, and were wrong

Query-independent priors failed first, exactly as the theory predicts: multiplying scores by
a Beta posterior odds `(hits+1)/(misses+1)` cost **−0.0384**; positive-only **−0.0175**;
empirical-Bayes shrinkage to the corpus base rate — the most "correct" of the three —
**−0.0489**.

So we made memory *query-conditional and textual*: a confirmed query became part of what the
document is about, indexed as its own BM25F field. It looked like a decisive win — NFCorpus
+0.0019/+0.0076, MIRACL-ko **+0.0618/+0.0729**. We shipped it.

Then we ran the controls, and they killed it:

| MIRACL-ko, split 0 | Δ overall | Δ covered | **Δ uncovered** |
|---|---:|---:|---:|
| real memory | +0.0618 | +0.1430 | **+0.0441** |
| shuffled (query ↔ doc pairing permuted) | +0.0555 | +0.1079 | +0.0441 |
| **random query attached to each doc** | **+0.0665** | +0.1430 | +0.0498 |

Two facts end the argument. **Breaking the pairing does not break the gain** — a random
feedback query scores *better* than the true one. And **held-out queries whose relevant
documents remembered nothing still gained +0.0441**, which a query-conditional mechanism
cannot cause.

The real mechanism: injecting query text into documents raises the document frequency of
query vocabulary and therefore *deflates its IDF corpus-wide*. MIRACL-ko is exactly the
corpus that prefers weaker emphasis (0.9052 at `idf_pow=1.5` → 0.9489 at 1.0). Our
"memory" was an accidental, uncontrolled `idf_pow` reduction. The control that pins it: a
memory built from tokens that can never match any query moves the score by **+0.0000**
exactly — the effect needs the text to match, so it flows through term statistics, not
through the pairing.

The feature is reverted. The benchmark stays, with placebos and a covered/uncovered split
made mandatory, because its naive form produces a convincing false positive — it produced
one for us. [`eval/adaptive_bench.py`](../../eval/adaptive_bench.py) ·
[`eval/results/adaptive_memory.json`](../../eval/results/adaptive_memory.json)

### What was actually wrong — and what fixed it

Two facts, found by instrumenting instead of guessing.

**synaptic's reinforcement never reaches its retrieval.** `ResonanceScorer` — the only
consumer of the `success_count`/`access_count` that `reinforce()` writes — is used by
`search.py` and `agent_search.py`, not by `graph.search`. Nothing reads `edge.weight`.
Isolating the channels on NFCorpus: no reinforcement Δ = **0.0000** (a determinism check
that passes exactly), negatives only **0.0000**, positives only **+0.0001**, both −0.0045.
Reinforcement touches retrieval only through the edges it creates. (This also corrects us:
we previously called Hebbian *harmful* at −0.0174. Re-running gave −0.0045 — synaptic's
warm pass is not deterministic. It does not hurt; it is simply not wired in.)

**Our memory failed because it became content.** Injecting the remembered query into the
body raised the document frequency of query vocabulary and deflated its IDF corpus-wide.
So memory is now an **evidence field**: scored, but excluded from document frequency and
from length normalization.

```python
fb = Feedback()
fb.remember("statin side effects", ["doc7"])          # a user confirmed doc7 answered it
of = build_inmemory(nodes, triples, chunks, feedback=fb)
```

Each of those three choices was forced by a measurement, not chosen:

| choice | what happens without it |
|---|---|
| evidence excluded from **df** | the retracted false positive: Δuncovered +0.0441, placebos match `real` |
| **no length normalization** on the evidence field | a memory held by 2% of chunks explodes `fnorm`; covered gain collapses +0.4167 → +0.0742 |
| evidence-only terms take IDF from the **evidence df** | the very words memory exists to contribute — those absent from the chunk — are discarded |

### The result, on the axis memory is actually for

Memory pays when the same need returns in different words. So: feedback on the original
questions, evaluation on held-out **paraphrases** (token Jaccard 0.43). Same corpus, same
queries, scored by synaptic's own `metrics.py`.

| ΔMRR@10, held-out re-queries | KRA (ko) all | KRA covered | NFCorpus (en) all | NFCorpus covered |
|---|---:|---:|---:|---:|
| synaptic (Hebbian) | +0.0000 | +0.0093 | −0.0010 | −0.0008 |
| **OmniFuse (`Feedback`)** | **+0.1790** | **+0.3903** | **+0.0150** | **+0.0300** |
| ↳ shuffled placebo | +0.0059 | +0.0213 | +0.0015 | +0.0031 |
| ↳ random-query placebo | +0.0029 | +0.0215 | +0.0000 | +0.0000 |

`real` is 5.2× the strongest placebo: the `(query, chunk)` pairing carries the signal. And
on the *disjoint-query* axis — a different question, not a rephrasing — memory correctly
does **nothing** (+0.0006), with Δuncovered **exactly 0.0000**, because the collection's
IDF is provably untouched. A cold store ranks bit-identically to one built without
feedback. Nothing is tuned.

## Where OmniFuse lags synaptic (honest)

OmniFuse is a focused retrieval library, not a memory system. It leads on retrieval
quality (13/15 datasets), zero dependencies, one-shot fusion (no agent loop) and, now,
lexical speed. It is genuinely behind here:

| capability | synaptic-memory | OmniFuse |
|---|---|---|
| index persistence | SQLite / PostgreSQL store | ✓ `save_index` / `load_index` (stdlib pickle) |
| persistent *queryable* backend | SQLite (FTS5), PostgreSQL | ✗ index is loaded back into RAM |
| vector DB adapter | Qdrant, Kuzu, MinIO | ✗ roadmap |
| reranker | cross-encoder, ColBERT, LLM | ✗ roadmap |
| query rewriting / HyDE / decomposition | yes | ✗ |
| entity extraction / linking / resolution | yes (ko + en) | ✗ label matching only |
| async API | yes | ✗ sync |
| MCP server / agent loop / CLI | yes | ✗ |
| consolidation / snapshot / activity | yes | ✗ (`Vault` has simple salience) |
| **memory that improves retrieval** (ΔMRR@10, held-out re-queries) | Hebbian: **+0.0093** — not wired into `graph.search` | `Feedback`: **+0.4167**, placebos +0.024 / +0.080 |
| scale ceiling | disk-backed | RAM-bound, but build peak cut 209 MB → **46.6 MB** |

The persistence gap — the one that forced OmniFuse to pay index-build cost on every
process — is now closed:

```python
from omnifuse import build_inmemory, save_index, load_index
save_index(build_inmemory(nodes, triples, chunks), "idx.pkl")
of = load_index("idx.pkl")          # warm start; embedder/LLM re-supplied here
```

On the 5,234-chunk golden corpus: build 6.0 s (peak **46.6 MB**) → **load 0.21 s (~29×
faster)**, index **28.7 MB**, rankings identical after the round-trip. Note `pickle` executes arbitrary code on
load, so only load indexes you produced; and the index is still read *into RAM*, so the
scale ceiling is unchanged — a genuinely disk-resident backend remains future work.

## Reproduce

```bash
pip install -e .
python eval/finreg_bench.py                            # finreg, self-contained
python eval/compare_synaptic.py --synaptic-graph PATH  # finreg head-to-head
python eval/public_bench.py --synaptic-repo PATH       # 8 public datasets
```
