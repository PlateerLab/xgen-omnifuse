# Changelog

## Unreleased

- **`remember()` — memory folds into the live index, bit-identically, in ~1 ms.** Memory was
  batch-only: a confirmed `(query -> documents)` pair required an index rebuild, which no live
  service can afford per click. It is now an in-place update.

  This is what the evidence-field design was always worth. Evidence is excluded from document
  frequency and is not length-normalized, so `N`, the content df, every content term's IDF and
  the content fields' avglen are all **fixed** — remembering rewrites the contributions of
  exactly one document. The single coupling is that a term seen *only* in evidence takes its
  IDF from the evidence df, which grows; but all of that term's postings are evidence-derived,
  so the documents to fix are exactly the ones that remember it. Measured, that coupled set is
  **15 terms out of a 23,610-term vocabulary** (NFCorpus, 100 memories).

  | | rebuild | `remember()` | per memory |
  |---|---:|---:|---:|
  | NFCorpus (3,633 docs, 100 memories) | 1.389 s | **1.00 ms** | **1,386x** |
  | same memories, a tenth of the corpus | 0.175 s | **1.02 ms** | 172x |
  | KRA (5,234 chunks, 120 memories) | 6.605 s | **1.52 ms** | **4,335x** |

  The middle row is the control: ten times fewer documents makes the rebuild 7.9x cheaper and
  leaves `remember()` where it was, so the cost tracks the memory rather than the corpus. It is
  also flat as memory accumulates (1.99 -> 1.69 ms per 50 over 215 KRA memories).

  The bar was **bit equality** with a full rebuild — every posting, every float — not `isclose`,
  because a weight that drifts is a scoring bug with a stopwatch. To meet it the index keeps the
  tfw of evidence-only terms beside their weights, so a moved IDF is recomputed rather than
  rescaled. The first prototype asserted the update was purely local, skipped the evidence-df
  coupling entirely, and differed from a rebuild in 1,181 terms; the bar caught it, and the
  wrong claim is recorded in `eval/results/incremental_memory.json`.

  New: `OmniFuse.remember()`, `InMemoryVector.remember()`, `BM25F.update_evidence()`,
  `eval/incremental_bench.py`, `tests/test_incremental.py` (7 tests, incl. bit-identity at every
  prefix and `remember` after `save_index`/`load_index`). Building with an *empty* `Feedback()`
  now opts a store into the memory field — previously an empty `Feedback` was falsy and silently
  fell back to a non-evidence index. A cold store still ranks identically. Static suite
  unchanged; no `forget()` — evidence may only grow.

- **The Korean copula's interrogative paradigm was missing from the ending list — and it
  was the last loss. OmniFuse now wins all 15 datasets.** A per-query diff against synaptic
  (scored by its own `metrics.reciprocal_rank`) showed OmniFuse losing 29 MIRACL-ko queries
  and winning 14, with the losses clustered on a single junk document: every
  *"…어디인가?"* ("where is…?") question retrieved the article titled **"내 친구의 집은
  어디인가"** — a 4×-weighted title match on nothing but the question word.

  Cause: `-인가/-인가요/-입니까/-인지` were absent from `_KO_SUFFIX`, so `어디인가` stemmed to
  the *rare* token `어디인` rather than the common word `어디`, and `idf_pow` amplified that
  rarity. Kiwi splits the copula into morphemes, which is why synaptic never saw it.

  **MIRACL-ko 0.9052 → 0.9617** (synaptic 0.9495) — the extended track goes 4/4 and the
  suite to **15/15**. Also XPQA-ko 0.3256 → 0.3290, KLUE-MRC 0.8280 → 0.8288, the real-world
  golden set 0.4775 → **0.4957**. Small, still-winning costs: PublicHealthQA 0.6284 → 0.6217,
  AutoRAG 0.9309 → 0.9293, Ko-StrategyQA 0.6509 → 0.6496. finreg, HotPotQA, Allganize,
  NFCorpus and SciFact are bit-identical. Not a fit: every suffix subset from `{인가}` alone
  to a nine-ending superset wins 8/8 of the Korean-bearing sets — a flat band, and a closed
  linguistic class like the 조사/어미 already shipped.

  Two standard fixes were tried on the *symptom* first and are recorded as rejected:
  coordination-level matching (`score *= coverage**λ`) wins MIRACL at 0.9536 but breaks
  Ko-StrategyQA 0.6135, HotPotQA-200 0.8774, PublicHealthQA 0.5824 and NFCorpus 0.5040;
  `minimum_should_match ≥ 2` reaches only 0.9167 and collapses Ko-StrategyQA to 0.5663
  because it filters gold documents that legitimately match one query word.

- **Efficiency is now measured with synaptic's own `mean_search_time_ms`**
  (`eval/perf_bench.py`), not a timer we invented. NFCorpus: ingest **2.01 s vs 55.01 s**,
  mean search **1.66 ms vs 14.14 ms**, MRR 0.5182 vs 0.5124. Allganize RAG-ko: **0.18 s /
  0.18 ms / 0.9683** vs 5.39 s / 4.41 ms / 0.9562.

- **`Feedback` — memory that survives its own placebo, and beats synaptic on the axis that
  names it.** A confirmed query becomes *evidence about* a chunk: it is indexed as a BM25F
  **evidence field** whose terms score the chunk but never enter document frequency, and
  which is not length-normalized. Measured with synaptic's own `metrics.py`, on the same
  corpus, queries and scorer — feedback on the original questions, evaluated on **held-out
  paraphrases** of them:

  | ΔMRR@10, held-out re-queries | KRA (ko) all | KRA covered | NFCorpus (en) all | NFCorpus covered |
  |---|---:|---:|---:|---:|
  | synaptic (Hebbian) | +0.0000 | +0.0093 | −0.0010 | −0.0008 |
  | **OmniFuse (`Feedback`)** | **+0.1790** | **+0.3903** | **+0.0150** | **+0.0300** |
  | ↳ shuffled placebo | +0.0059 | +0.0213 | +0.0015 | +0.0031 |
  | ↳ random-query placebo | +0.0029 | +0.0215 | +0.0000 | +0.0000 |

  Replicated on a **second corpus, second language and a different relevance structure**:
  on NFCorpus both placebos go *negative* while `real` stays positive, and Δuncovered is
  exactly **0.0000**. The effect is smaller there (+0.0460 vs +0.4167 covered) because its
  cold score is already 0.556 — with 38 relevant documents per query there is little
  headroom. Memory pays most where relevance is concentrated and the cold ranking is weak.
  `real` is 5.2× the strongest placebo on KRA and is the only positive variant on NFCorpus,
  so the `(query, chunk)` pairing carries the signal.
  On *unrelated* held-out questions memory correctly does nothing (+0.0006), and
  Δuncovered is **exactly 0.0000** there — the collection's IDF is provably untouched. A
  cold store ranks **bit-identically** to one built with no feedback (verified on finreg and
  the whole public suite). Nothing is tuned.

  Three design decisions, each forced by a measurement: excluding evidence from **df**
  (our retracted version injected the query into the body, which deflated the IDF of query
  vocabulary corpus-wide — an accidental `idf_pow` reduction); **no length normalization**
  on the evidence field (a memory held by 2% of chunks otherwise explodes `fnorm` and the
  covered gain collapses +0.4167 → +0.0742); and giving **evidence-only terms** an IDF from
  the evidence df (otherwise the very words memory exists to contribute are discarded).

- **Fairness correction on synaptic.** An earlier revision reported its Hebbian
  reinforcement as *harmful* (−0.0174). Re-running the same configuration gave −0.0045:
  synaptic's warm pass is not deterministic. Channel isolation shows why the effect is
  noise — in this version `SynapticGraph.search()` reads **none** of the fields
  `reinforce()` writes (`ResonanceScorer`, the only consumer of `success_count`, is used by
  `search.py`/`agent_search.py`, not by `graph.search`; nothing reads `edge.weight`).
  Reinforcement reaches retrieval only through the edges it creates on success: with no
  reinforcement Δ = **0.0000** exactly, with negatives only **0.0000**, with positives only
  **+0.0001**. The fair statement is that Hebbian is *not wired into* this version's
  retrieval — not that it hurts.

- **Full suite re-verified after the `BM25F` change.** finreg (0.8400 / 107 / 0.9250), all
  8 core public sets, all 4 extended sets and the real-world golden set are **bit-identical**
  to before. Index build costs +1.8 % time (6.03 s → 6.14 s) and +0.6 MB peak (46.6 → 47.2 MB)
  for the extra evidence-df bookkeeping; persisted index size and `load_index` are unchanged.
- **`BM25F(evidence_fields=…)`** — fields that describe a document rather than being its
  content: scored, but excluded from document frequency and from length normalization.
  With no evidence fields the class behaves exactly as before.

- **Retracted: `Feedback` (query-conditional memory). The claimed win was not real.**
  A previous entry claimed OmniFuse beat synaptic on the axis that defines it — memory —
  by remembering the queries a document was confirmed to answer and indexing them as a
  BM25F field (NFCorpus +0.0019/+0.0076, MIRACL-ko +0.0618/+0.0729 vs synaptic's
  −0.0002/−0.0174/−0.0165). Placebo controls destroy that claim:
  - permuting the (query ↔ document) pairing keeps the gain (+0.0555 on MIRACL-ko);
  - attaching a **random** feedback query to each confirmed document scores *better* than
    the real one (+0.0665 vs +0.0618);
  - held-out queries whose relevant documents remembered **nothing** still gained +0.0441,
    which a query-conditional mechanism cannot do.

  The real mechanism: injecting query text into documents raises the document frequency of
  query vocabulary and deflates its IDF corpus-wide. MIRACL-ko is precisely the corpus that
  prefers weaker emphasis (0.9052 at `idf_pow=1.5` → 0.9489 at 1.0), so "memory" was an
  accidental, uncontrolled `idf_pow` reduction. A control memory made of tokens that can
  never match a query moves nothing (**+0.0000** exactly), which pins the effect on term
  statistics rather than on the pairing.

  The feature is reverted. The benchmark is kept — with placebo and covered/uncovered
  controls now mandatory, because its naive form yields a convincing false positive:
  [`eval/adaptive_bench.py`](eval/adaptive_bench.py) ·
  [`eval/results/adaptive_memory.json`](eval/results/adaptive_memory.json).

  Our own query-independent designs failed first and are recorded too (Beta posterior odds
  −0.0384, positive-only −0.0175, empirical-Bayes shrinkage −0.0489). **Nothing we tried,
  and nothing synaptic ships, improves held-out retrieval in a query-conditional way.**

- **Index build peaks at ~1/4 the memory, at no cost in build time.** Indexing needs
  corpus-wide document frequency before it can compute a contribution, so per-document
  term counts have to survive from pass 1 to pass 2. They used to survive as dicts of
  strings over the whole corpus — that was the peak. They are now interned **term ids in
  `array('i')`**, and each document is released the moment pass 2 consumes it. `BM25`/
  `BM25F` also accept a zero-arg callable yielding documents, so `InMemoryVector` streams
  tokenization instead of materializing the tokenized corpus. On 5,234 chunks: build peak
  **177.2 MB → 46.6 MB**, build time **6.13 s → 6.03 s**, rankings unchanged (finreg + all
  8 public + NFCorpus verified). A first attempt — stream and simply re-tokenize in pass 2
  — also hit 46.7 MB but doubled build time to 13.6 s; it was rejected.
- **MIRACL-ko stays the single loss, and we stopped trying to fix it.** Seven principled
  variants were measured and none wins it or dominates the shipped configuration:
  `idf_pow` band, `contribution^q` power-mean, emphasis restricted to word tokens,
  Hangul stem-only, Hangul bi-gram-only, bi-grams over the raw surface form (`V4`), and
  `V4 + #raw`. `V4` is notable — it lifts Ko-StrategyQA to 0.6599 (best seen) and KLUE to
  0.8314, but costs AutoRAG, PublicHealthQA and Allganize-Eval, so the 7-set average falls
  0.8355 → 0.8311. MIRACL's best possible score is 0.9489 at `idf_pow=1.0` against
  synaptic's 0.9495 — a 0.13-query gap. Its relevance is diffuse (14.4 relevant/query) and
  emphasis is a precision move; that is a structural trade with Ko-StrategyQA, not a bug.

- **English morphological normalization — NFCorpus flips to a win (0.5053 → 0.5182 vs
  synaptic's 0.5124).** OmniFuse normalized Korean morphology but indexed Latin tokens as
  raw surface forms, so `statin` could never match `statins`. `text._en_stem` adds
  **Harman's S-stemmer** (singularize; no `-ing`/`-ed`, no Porter cascade). It has *no
  tunable parameter*, so there is nothing to fit. SciFact 0.6422 → 0.6456,
  HotPotQA-200 0.9028 → 0.9044, HotPotQA-24 0.9286 → 0.9077 (still a win); every Korean
  dataset, finreg, and the golden set are **bit-identical**. Suite: **14 of 15** datasets
  (Core 10/10, Extended 3/4, Real-world 1/1).
- **Correction.** An earlier revision claimed the IDF emphasis *caused* both the NFCorpus
  and MIRACL-ko losses. That was wrong: with emphasis off, NFCorpus still lost (0.5080 vs
  0.5124). Emphasis widens the MIRACL-ko gap; the NFCorpus gap was missing English
  morphology, now fixed. MIRACL-ko remains the single loss.
- **Recorded negative results** (they are why the real fix was findable):
  - *Replacing graph fusion with PPR-style damped, degree-normalized propagation* — much
    worse (finreg single-hop 0.6294, multi-hop 81/120). Degree normalization starves the
    cited article; finreg's evidence is one out-edge away.
  - *`contribution^q` power-mean instead of `idf^p`* — better on the two Extended losses
    (MIRACL-ko 0.9052 → 0.9321) but worse on nine other datasets; average MRR 0.7628 →
    0.7610. Not shipped: it trades nine datasets for two.
  - *Augmenting Latin tokens with `surface + #stem`* (mirroring the Korean design) — worse
    than plain replacement (NFCorpus 0.5090 vs 0.5182).
  - *An English-stemming run that "proved" stemming harmful* was **void**: the harness
    patched `text.tokenize` but `backends/memory.py` binds `tokenize` at import, so
    documents were indexed with the old tokenizer while queries used the new one. The
    corrected run reverses the conclusion.

- **Graph-companion fusion now follows edge direction — finreg multi-hop 101 → 107/120**
  (R@10 0.8958 → 0.9250), single-hop unchanged (hit@10 113 → 114, nDCG 0.8651 → 0.8663).
  `retrieve()` documents that it surfaces "a passage a strong seed *references*", but
  `InMemoryGraph._adj_ids` was symmetric, so a seed also promoted every node that cited
  *it* — a crowd, not evidence. `GraphStore.neighbor_ids` gains
  `direction="out" | "in" | "both"` (default `"both"`, unchanged); fusion asks for `"out"`
  via the new `OmniFuse(fusion_direction=…)`. Set `"both"` for genuinely symmetric graphs.
  No parameter was refitted (`fusion_alpha` stays 0.9). Public datasets carry no graph, so
  their scores are untouched. Recorded negative results: replacing this rule with damped,
  degree-normalized PPR-style propagation is far *worse* (0.6294 / 81), and multi-hop
  propagation never helps — finreg's evidence is exactly one out-edge away.
- **Index is ~30 % smaller, loads 2× faster, and peaks at half the memory while building.**
  `BM25.tf`, `BM25F.doc_tf` and the per-field length norms existed only to *derive* the
  postings, and afterwards were read by nothing but `score()`.
  - They are no longer **retained**: `score()` reads its precomputed contribution straight
    out of the postings (each `_pd[t]` is ascending, so it is a binary search). Persisted
    index **41.2 MB → 28.7 MB**; `load_index` **0.43 s → 0.21 s**.
  - They are no longer **materialized for the whole corpus** either. `BM25F` build peak
    **68.2 MB → 36.0 MB**; full `build_inmemory` peak 209.3 MB → 177.2 MB here, and then
    → **46.6 MB** once the pass-1 intermediate was compacted (see the entry at the top).
  - Rankings unchanged throughout (verified on finreg + all 8 public sets).
- **Index persistence — `save_index` / `load_index`.** A built in-memory index (graph +
  passage store) round-trips to disk with stdlib `pickle`, so a process starts warm
  instead of re-indexing: on a 5,234-chunk corpus, **load 0.43 s vs a 5.98 s rebuild
  (14×)**, rankings identical. The LLM and the embedder callable are deliberately not
  persisted — pass them to `load_index(..., llm=, embedder=)`. `pickle` executes arbitrary
  code on load, so only load indexes you produced. This closes the one gap that forced
  OmniFuse to pay build cost every run; the index is still read into RAM, so a truly
  disk-resident backend remains future work.
- **Lexical search is ~6.4× faster, with bit-identical rankings.** A term's contribution
  to a document (`idf * tfw(k1+1)/(k1+tfw)` for `BM25F`, `idf*(k1+1)*f/(f+norm)` for
  `BM25`) does not depend on the query, so it is now folded into the inverted index at
  build time: a search is a plain accumulation of precomputed floats over the postings,
  instead of a full scan that re-derived document lengths per term. Verified score-for-score
  against the previous implementation on finreg + all 8 public datasets (8 public: 249.4 s
  → 38.7 s; KLUE-MRC 230.5 s → 28.8 s). `InMemoryGraph._by_label` is a dict lookup, not an
  O(N) scan.
- **`idf_pow` is now a documented knob** (`InMemoryVector`, `build_inmemory(vector_kwargs=…)`).
  It was effectively hardcoded before.
- **Honest correction.** The IDF term-specificity emphasis is a Pareto trade, not a free
  win: it takes the core suite 10/10 but *regresses* heavily multi-relevant passage-IR
  corpora (MIRACL-ko 0.949 → 0.905, NFCorpus 0.508 → 0.505). The previously published
  BEIR/MTEB numbers predated `idf_pow` and were never re-run — they are corrected in
  `eval/results/beir_mteb_extra.json`. An earlier parameter sweep was also invalid
  (keyword-only defaults bind at def time, so monkeypatching the module constant changed
  nothing); the corrected sweep confirms the core p∈[1.3,2.0] band and exposes the
  regression the broken sweep hid.

## 0.5.0

Retrieval quality — the ranking now uses field structure and graph structure,
not just flat body BM25. Measured on the finreg corpus (4,417 Korean statute
articles) against synaptic-memory's own eval metric (`metrics.py`, k=10),
single-shot, no LLM:

| | synaptic FTS-only | OmniFuse 0.5 |
|---|---:|---:|
| single-hop MRR@10 | 0.704 | **0.840** |
| single-hop hit@10 | 103/120 | **114/120** |
| multi-hop strict-solved | 56/120 | **107/120** |

On the full synaptic benchmark suite (finreg + 8 public IR sets, zero-embedder lexical
track), OmniFuse 0.5 wins **all ten datasets** (avg MRR 0.846 vs synaptic 0.809) via two
honest, dependency-free, zero-hardcode logic changes:

- **Dependency-free Korean stemming in `text.tokenize`** — Hangul runs now have their
  common particles (조사), verb/adjective endings (어미), and trailing derivational
  suffixes (적/화/성/상/하/들) stripped by a small rule table before bi-gramming, so a
  query and a document align on the stem the way a morphological analyzer (Kiwi) would
  — but pure Python, and it emits *fewer* tokens than raw bi-grams (stem bi-grams + one
  stem unigram) ⇒ more accurate on Korean *and* more memory-efficient. Suffixes are
  stripped only when *trailing*, so 상황/성별 (with the char leading) are untouched, and
  the emitted stem unigram still lets compound forms match. Hanja/Kana and Latin are
  unchanged. This flips AutoRAG and PublicHealthQA to OmniFuse (7→9 wins).
- **IDF term-specificity emphasis in `text.BM25`/`BM25F`** (`_IDF_POW`, default 1.5) —
  each term's IDF is raised to a power so a rare, discriminative term (a named entity)
  dominates the several common words it is buried under in a long natural-language
  question ("장 발장은 어떤 범죄로 유죄 판결을 받았나요?" — the entity 발장 vs common
  범죄/유죄/판결). Plain BM25 sums per-term scores, so many common matches otherwise
  outrank the one rare-entity match; the power fixes this "entity-burial". Found by
  inspecting the failing queries, not fishing. Zero runtime cost (folded into the
  precomputed IDF once at index build). Flips the last holdout **Ko-StrategyQA
  0.6414→0.6509 (9→10 wins)** and lifts every other set (HotPotQA-24 0.908→0.929,
  AutoRAG 0.917→0.931); the win holds across the whole flat band `p ∈ [1.3, 2.0]`, so it
  is a robust default, not a fit to test labels. Tunable via
  `BM25(..., idf_pow=…)` / `BM25F(..., idf_pow=…)`.
- **`Chunk.title`** — an optional short high-signal field. When any chunk carries
  a title, `InMemoryVector` indexes it with **field-weighted BM25** (`text.BM25F`),
  title weighted 4x over body — a query term in the heading outranks a chunk
  that only mentions it deep in a long passage. No title → identical to before.
- **Hybrid dense + lexical retrieval** — when chunks carry embeddings *and* text,
  `InMemoryVector` min-max normalizes dense cosine and lexical BM25(F) per query and
  combines them `dense_weight·dense + lexical_weight·lexical` (dense recovers
  paraphrase, lexical nails exact terms). The default `lexical_weight=0.8` (vs dense
  1.0) is a *flat* optimum — the aggregate MRR barely moves across 0.4–1.0, so it is
  a single principled setting, not a per-corpus fit. This score fusion beat rank
  fusion (RRF) on every dataset measured. Tunable via
  `build_inmemory(..., vector_kwargs={"lexical_weight":…, "dense_weight":…})`.
  In a full-pipeline benchmark (shared e5-small embedder) it flips omnifuse's two
  lexical-only losses (AutoRAG, Ko-StrategyQA) into wins over synaptic's fused pipeline.
- **Graph-companion fusion** (`OmniFuse.retrieve`) — a new public retrieval API
  that fuses 1-hop graph structure *into the ranking*: a passage referenced/linked
  by a strong lexical seed is surfaced beside it (score `fusion_alpha`×seed), so
  multi-hop evidence sharing no query vocabulary lands in one shot — no agent, no
  LLM. `search()` now builds its chunks/evidence on `retrieve()`. Opt out with
  `graph_fusion=False`. Added `GraphStore.neighbor_ids` (InMemory + Fuseki).

## 0.4.0

- Replaced `Memory` (remember/recall) with **`Vault`** — an omnifuse-native memory:
  - `fuse(text=, facts=)` write / `surface(query)` read (on-brand verbs, not generic remember/recall).
  - **fuse-on-write**: facts deduped & entities coreferenced by label (knowledge merges, not piles).
  - **salience**: each fuse/surface bumps node salience; `surface()` re-ranks results by it (no PPR/Hebbian).
  - `save()/load()` JSONL incl. salience; incremental label set (no per-write re-derivation).
- **Breaking**: `Memory`/`remember`/`recall` removed (the lib is pre-1.0; no released users relied on it).

## 0.3.0

- `Memory` — a growing store built on OmniFuse search. `remember()` facts/notes over time,
  `recall()` via the same one-shot graph+vector fusion. Notes auto-link to known entities by
  label; `save()/load()` JSONL persistence. (synaptic-memory–style memory on omnifuse's engine.)

## 0.2.0

- Convenience loaders so you can give loose data and search immediately (synaptic `from_data` style):
  - `from_triples(triples, chunks=...)` — accepts `(s, p, o)` tuples / dicts / `Triple`; **infers nodes**
    (object of an is-a edge → class) when none are given.
  - `from_jsonl(triples=, nodes=, chunks=)` and `from_csv(triples=, chunks=)` — stdlib json/csv, zero deps.
  - `from_fuseki(query_url, graph_uri, user=, password=)` — one call over any SPARQL endpoint.
- `build_inmemory` now coerces loose tuples/dicts too.

## 0.1.0

Initial extraction of the one-shot GraphRAG fusion algorithm as a backend-agnostic library.

- `OmniFuse.search` — one-shot fusion: vector/lexical + graph label-linking + class
  enumeration + HippoRAG 1-hop, fused with MMR diversity and adaptive top-k, single synthesis.
- `GraphStore` / `VectorStore` / `LLM` protocols (structural typing).
- Zero-infra backends: `InMemoryGraph` (BM25 label search with CJK n-grams, class
  enumeration, 1-hop traversal) and `InMemoryVector` (cosine or BM25 lexical).
- `EchoLLM` so the pipeline runs with no API key; `build_inmemory(...)` one-call setup.
- `FusekiGraph` — stdlib-only SPARQL adapter (any SPARQL 1.1 endpoint); same algorithm runs
  on a real Apache Jena Fuseki store. Self-contained (in-memory) and Jena modes both supported.
- Language-neutral default system prompt (overridable via `system_prompt=`).
- `dependencies = []` core; pytest smoke tests; quickstart + fuseki examples.
