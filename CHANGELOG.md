# Changelog

## 0.5.0

Retrieval quality — the ranking now uses field structure and graph structure,
not just flat body BM25. Measured on the finreg corpus (4,417 Korean statute
articles) against synaptic-memory's own eval metric (`metrics.py`, k=10),
single-shot, no LLM:

| | synaptic FTS-only | OmniFuse 0.5 |
|---|---:|---:|
| single-hop MRR@10 | 0.704 | **0.840** |
| single-hop hit@10 | 103/120 | **115/120** |
| multi-hop strict-solved | 56/120 | **103/120** |

- **Dependency-free Korean stemming in `text.tokenize`** — Hangul runs now have their
  common particles (조사) and endings (어미) stripped by a small rule table before
  bi-gramming, so a query and a document align on the stem the way a morphological
  analyzer (Kiwi) would — but pure Python, and it emits *fewer* tokens than raw
  bi-grams (stem bi-grams + one stem unigram) ⇒ more accurate on Korean *and* more
  memory-efficient. Hanja/Kana and Latin are unchanged. On the synaptic benchmark this
  lifts the lexical (zero-embedder) track from 7→**9 wins / 10** — AutoRAG and
  PublicHealthQA flip to OmniFuse — and raises average MRR 0.829→**0.840**.
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
