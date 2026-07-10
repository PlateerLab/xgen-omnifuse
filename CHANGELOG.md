# Changelog

## Unreleased

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
