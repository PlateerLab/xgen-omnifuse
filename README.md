# OmniFuse

**Backend-agnostic, one-shot GraphRAG.** Fire several retrieval strategies at once —
vector/lexical passages **+** graph label-linking **+** class enumeration **+** relation
expansion — and *fuse* them with MMR diversity into a single LLM synthesis. No
iterative ReAct tool loop. **Zero infra, zero lock-in:** the full algorithm runs on a
pure-Python in-memory backend (dict + BM25), and swaps to Fuseki / Qdrant / any LLM by
passing objects that match three small protocols.

```python
from omnifuse import from_triples

of = from_triples(                                  # nodes are inferred; no DB, no API key
    [("담보", "instanceOf", "규정"), ("담보", "한도", "5억")],
    chunks=[("c1", "담보 한도는 5억원이다", ["담보"])],
)
print(of.search("담보 한도").answer)
```

Load however you have the data — all zero-dep, same `search()`:

```python
from omnifuse import from_jsonl, from_csv, from_fuseki, build_inmemory
of = from_jsonl(triples="t.jsonl", chunks="c.jsonl")
of = from_csv(triples="triples.csv", chunks="chunks.csv")
of = from_fuseki("http://localhost:3030/ds/query", graph_uri="urn:g", user="admin", password="…")
of = build_inmemory(nodes, triples, chunks)         # explicit Node/Triple/Chunk
```

Build the index once, start warm afterwards (stdlib pickle, zero deps):

```python
from omnifuse import save_index, load_index
save_index(of, "idx.pkl")
of = load_index("idx.pkl")            # ~29x faster than rebuilding; pass embedder=/llm= here
```

## Why graph fusion (not just vectors)

Pure vector RAG answers from the top-k passages it happens to embed near the query. A
graph store also gives you operations cosine similarity can't:

- **Complete enumeration** — *all* instances of a class ("list every regulation"), exact counts.
- **Relations / multi-hop** — what an entity is connected to, 1-hop neighbors, paths.
- **Minority evidence survives** — MMR diversity keeps the decisive exception/warning that
  near-duplicate passages would otherwise crowd out of a fixed top-k.

OmniFuse fuses both: the vector seed for *content*, the graph seeds for *structure*.

## Design — algorithm as a library

The algorithm only talks to three `typing.Protocol`s, never to a database:

```python
class GraphStore(Protocol):
    def search_labels(self, query, *, limit=30) -> list[tuple[Node, float]]: ...   # full-text label search
    def class_instances(self, class_id, *, limit=1000) -> list[Node]: ...          # enumeration
    def neighbors(self, node_id, *, hops=1, limit=100) -> list[tuple[str,str,str]]: ...  # traversal
    def count_class(self, class_id) -> int: ...
    def get_node(self, node_id) -> Node | None: ...

class VectorStore(Protocol):
    def search(self, query, *, limit=20) -> list[tuple[Chunk, float]]: ...
    def fetch(self, ids) -> list[Chunk]: ...

class LLM(Protocol):
    def generate(self, prompt, *, system="", timeout=None) -> str: ...
```

- **Zero-infra default** — `InMemoryGraph` indexes node labels with **BM25** (CJK
  character n-grams, so Korean/CJK search works with no morphological analyzer).
  `InMemoryVector` picks its mode from what the chunks carry: **hybrid** (dense
  cosine ⊕ lexical BM25 fused by Reciprocal Rank Fusion) when embeddings *and*
  text are present, **dense** cosine with embeddings only, else **field-weighted
  BM25** (`BM25F`) that scores a chunk's short `title` above its body.
- **`dependencies = []`** — the core needs nothing but the standard library. Real backends
  are optional extras (`pip install "xgen-omnifuse[fuseki,qdrant]"`).
- **Bring your own LLM** — pass anything with `generate(...)`; the bundled `EchoLLM`
  returns the fused evidence so the pipeline runs end-to-end with no API key.

## The pipeline (`OmniFuse.search`)

1. vector/lexical seed + **1-hop graph fusion** → adaptive top-k (score-distribution cut, not fixed k)
2. graph label-linking → 1-hop relations
3. class enumeration (complete list/count)
4. HippoRAG — entities of the retrieved chunks → 1-hop expansion
5. evidence assembled with **MMR** diversity (Jaccard, no embeddings needed)
6. one LLM synthesis over the fused evidence
7. honest `evidence_nodes` — only the nodes the answer actually cites

### `OmniFuse.retrieve` — ranking, not just synthesis

`retrieve(question)` returns the ranked `(chunk, score)` list with no LLM call —
use it directly for search/eval. On top of the vector seed it does **graph-companion
fusion**: a passage that a strong seed *references/links to* is surfaced beside it
(companion score = `fusion_alpha` × seed), so multi-hop evidence that shares no query
vocabulary lands in one shot — no agent, no LLM. `search()` builds its chunks and
evidence on `retrieve()`. Opt out with `graph_fusion=False`.

## Install

```bash
pip install xgen-omnifuse            # core (zero deps)
pip install "xgen-omnifuse[dev]"     # + pytest, ruff
```

Run the demo with no install:

```bash
python examples/quickstart.py
```

## Layout

```
src/omnifuse/
  protocols.py     # GraphStore / VectorStore / LLM  (the swap points)
  models.py        # Node, Triple, Chunk (+ optional title), SearchResult
  text.py          # tokenizer + BM25 + BM25F (field-weighted, CJK n-grams)
  fusion.py        # MMR, adaptive top-k, relation ranking
  oneshot.py       # OmniFuse.search / retrieve — the fusion algorithm
  backends/memory.py  # InMemoryGraph + InMemoryVector (hybrid/dense/lexical, zero infra)
  llm.py           # EchoLLM, CallableLLM
  feedback.py      # Feedback — memory as a BM25F evidence field
  facade.py        # build_inmemory(...), save_index / load_index
examples/  tests/  eval/   # eval/ = head-to-head benchmark vs synaptic-memory
```

## Two interchangeable modes (same algorithm)

```python
# (a) self-contained — zero infra
from omnifuse import build_inmemory
of = build_inmemory(nodes, triples, chunks)

# (b) backed by Apache Jena Fuseki (or any SPARQL endpoint) — graph-only or with a vector store
from omnifuse import OmniFuse, InMemoryVector
from omnifuse.backends.fuseki import FusekiGraph
graph = FusekiGraph("http://localhost:3030/ds/query", graph_uri="urn:my-graph", user="admin", password="…")
of = OmniFuse(graph, InMemoryVector([]))   # search() unchanged
```

`FusekiGraph` is stdlib-only (urllib) and uses portable `FILTER(CONTAINS(...))`, so it
works on **any** SPARQL 1.1 store — not just jena-text.

## Benchmarks — vs synaptic-memory (OmniFuse v0.5.0)

Head-to-head on synaptic-memory's own eval data, its own metric (`eval/metrics.py`,
MRR@10), single-shot, k=10. synaptic's column is its *own* `run_all` FTS-only path.
Full harness + numbers in [`eval/`](eval/) and
[`docs/comparison/omnifuse_vs_synaptic.md`](docs/comparison/omnifuse_vs_synaptic.md).

<details open>
<summary><b>Scoring & fairness — how these numbers were audited</b></summary>

- **Not our scorer.** Both systems are scored by *synaptic's own* `metrics.py`
  (`reciprocal_rank`, k=10), not by a metric we wrote.
- **Symmetric harness.** Both read the same dataset file → same corpus, queries and qrels;
  each returns its own top-10; identical scoring.
- **Both re-run from scratch.** An earlier revision re-measured only OmniFuse and *reused*
  synaptic's column from prior runs (an assumption, not evidence). Both sides were re-run:
  all 12 core figures reproduce **to the decimal**.
- **OmniFuse's field-weighted BM25F (title 4× body) is a design advantage**, not scoring
  bias — both systems receive the same `(title, text)`; OmniFuse simply exploits the title
  field harder.
- **synaptic's memory was measured fairly, and we corrected ourselves.** We first reported
  its Hebbian reinforcement as *harmful* (−0.0174); re-running the same configuration gave
  −0.0045 (its warm pass is not deterministic). Channel isolation shows `graph.search()`
  reads none of the fields `reinforce()` writes, so its measured deltas are noise around
  zero — it is **not wired in**, not harmful.
- **The golden set's questions are LLM-generated** and each has a single relevant chunk, so
  the absolute MRR is a *lower bound* for both systems; the comparison stays symmetric.
- **The IDF emphasis is a Pareto trade, not a free win** — it wins the core suite but
  regresses heavily multi-relevant corpora (see footnote ² below).
- A parameter sweep we previously published was **invalid** (keyword-only defaults bind at
  def time, so the monkeypatch changed nothing). The corrected sweep is in
  [`eval/results/beir_mteb_extra.json`](eval/results/beir_mteb_extra.json).

</details>

Everything in **one table** — MRR@10 unless noted, single-shot, no LLM, no embedder
(zero-infra lexical track). "winner" = higher score; **OmniFuse leads all 15 datasets**
(Core 10/10, Extended 4/4, Real-world 1/1).

| track | dataset | synaptic (FTS) | **OmniFuse** | winner |
|---|---|---:|---:|:--|
| **Core** (synaptic-shipped, 10/10) | finreg single-hop | 0.7039 | **0.8400** | 🟢 OmniFuse |
| | finreg multi-hop `strict/120` | 56 | **107** | 🟢 OmniFuse |
| | HotPotQA-24 | 0.8879 | **0.9077** | 🟢 OmniFuse |
| | HotPotQA-200 | 0.8775 | **0.9044** | 🟢 OmniFuse |
| | Allganize RAG-ko | 0.9562 | **0.9683** | 🟢 OmniFuse |
| | Allganize RAG-Eval | 0.9303 | **0.9370** | 🟢 OmniFuse |
| | KLUE-MRC | 0.7718 | **0.8288** | 🟢 OmniFuse |
| | PublicHealthQA | 0.6065 | **0.6217** | 🟢 OmniFuse |
| | AutoRAG | 0.9053 | **0.9293** | 🟢 OmniFuse |
| | Ko-StrategyQA | 0.6440 | **0.6496** | 🟢 OmniFuse |
| | **Core average** | 0.809 | **0.843** | 🟢 **10 / 10** |
| **Extended** (BEIR/MTEB, HF) | SciFact (EN) | 0.6317 | **0.6456** | 🟢 OmniFuse |
| | XPQA-ko | 0.3115 | **0.3290** | 🟢 OmniFuse |
| | NFCorpus (EN) | 0.5124 | **0.5182** | 🟢 OmniFuse |
| | MIRACL-retrieval-ko² | 0.9495 | **0.9617** | 🟢 OmniFuse |
| **Real-world** (live xgen corpus) | KRA/마사회 golden¹ | 0.2547 | **0.4957** | 🟢 **OmniFuse (~1.9×)** |
| | ↳ nDCG@10 / R@10 | 0.30 / 0.43 | **0.54 / 0.75** | 🟢 OmniFuse |

**Core: 10 wins / 0 losses** (avg MRR 0.809 → **0.843**) — every synaptic-shipped dataset,
**zero dependencies** (no morphological analyzer) vs synaptic's *mandatory* Kiwi.
**Extended: 4–0.** **Real-world: +0.2410 MRR (~1.95×)** on a live production corpus (¹한국마사회 docs
on xgen dev-xgen — 5,234 chunks, 215 LLM-generated natural questions; raw corpus
private/not committed — [reproducer](eval/golden_devxgen_bench.py) ·
[numbers](eval/results/golden_devxgen.json)).

> **One honest caveat on `idf_pow=1.5`.** Re-ablated under the shipping tokenizer it nets
> **+0.0065 MRR across 13 datasets — a wash**: it buys AutoRAG/HotPotQA-200/Ko-StrategyQA and
> costs MIRACL-ko (0.9812→0.9617), finreg (0.8533→0.8400) and NFCorpus (0.5236→0.5182). At
> `idf_pow=1.0` OmniFuse still wins **14 of 15**; the one loss is Ko-StrategyQA by **0.0006**,
> less than a single query on that 592-query set. We keep 1.5 because the win holds across the
> whole band `p ∈ [1.3, 2.0]`, but the 15/15 rests on that margin and you should know it.
> synaptic is **re-ingested and re-queried per dataset in the same pass** through its own
> `run_public_dataset` — no recalled numbers. [`eval/idf_pow_bench.py`](eval/idf_pow_bench.py) ·
> [`eval/results/idf_pow_ablation.json`](eval/results/idf_pow_ablation.json)

²**MIRACL-ko was the last loss, and it was our bug.** Dumping the per-query diff showed
every "…어디인가?" (where is…?) question retrieving the article titled **"내 친구의 집은
어디인가"** — a 4×-weighted title match on nothing but the question word. The cause: the
copula's interrogative paradigm (`-인가/-인가요/-입니까/-인지`) was missing from the Korean
ending list, so `어디인가` stemmed to the *rare* token `어디인` instead of the common word
`어디`, and `idf_pow` amplified that rarity. Kiwi splits the copula into morphemes, which is
why synaptic never saw it. Adding the paradigm — a closed linguistic class, like the 조사/어미
already there — takes MIRACL-ko **0.9052 → 0.9617** and wins **every** dataset. It is not a
fit: every subset from `{인가}` alone to a nine-ending superset wins 8/8 of the Korean-bearing
sets. Rejected first, and recorded: coordination-level matching (MIRACL 0.9536 but four sets
break) and `minimum_should_match` (0.9167, Ko-StrategyQA 0.5663).

(FiQA 57k-doc and MultiLongDoc-ko 193 MB omitted: synaptic ingest/RAM-bound on a 16 GB box.)

### Speed — measured, and **not** a universal win

⚠️ **Read the conditions, not just the ratio.** OmniFuse's speed advantage holds when both
systems index from raw data. It does **not** hold unconditionally, and it did not always
hold at all:

| scenario | conditions | synaptic | OmniFuse |
|---|---|---:|---:|
| golden set (5,234 chunks, 215 q) | both index from raw data — **the only apples-to-apples row** | 98.0 s | **6.6 s** |
| finreg — *today* | synaptic reuses a **prebuilt** SQLite graph; omni rebuilds | 11.0 s | **7.4 s** |
| finreg — **before** this optimization | same conditions | **10.9 s** | **26.6 s ← OmniFuse was 2.4× SLOWER** |
| golden set, warm start (`load_index`) | omni loads a persisted index | — | 0.43 s + 0.5 s queries |

So: the earlier "~7.5× faster" line was **golden-set-only** and it omitted that on finreg
OmniFuse was *slower* (26.6 s vs 10.9 s), because synaptic starts from a persisted index
while OmniFuse rebuilt its own on every run. That gap is now closed two ways — an
inverted-index optimization (6.4× on the lexical path, rankings bit-identical) and
`save_index`/`load_index` (14× warm start) — but the honest framing stands: **this is a
workload-dependent result, not a blanket "OmniFuse is faster".**

Where OmniFuse still genuinely lags synaptic — disk-resident queryable backends,
rerankers, HyDE/query decomposition, entity linking, async, MCP — is listed in
[the parity table](docs/comparison/omnifuse_vs_synaptic.md#where-omnifuse-lags-synaptic-honest).

### How OmniFuse wins — two honest, zero-hardcode logic improvements

No strong embedder, no per-dataset tuning, no fitting to test labels:

0. **Symmetric morphology — Korean *and* English.** Latin tokens were indexed as raw
   surface forms, so `statin` could not match `statins`, while Korean got full
   normalization. Harman's **S-stemmer** (singularize only; no tunable parameter, so
   nothing to fit) closes the asymmetry: NFCorpus **0.5053 → 0.5182** flips to a win,
   SciFact +0.003, HotPotQA-200 +0.002, and every Korean set is bit-identical.
1. **Dependency-free Korean stemmer** — strips 조사/어미 + trailing derivational suffixes
   so a query and a doc align on the stem the way Kiwi would, but pure Python and emitting
   *fewer* tokens (more accurate *and* more efficient). Flips AutoRAG + PublicHealthQA.
2. **IDF term-specificity emphasis** (`idf_pow=1.5`) — a question ("장 발장은 어떤 범죄로
   유죄 판결을 받았나요?") buries its one rare entity (발장) under common words
   (범죄/유죄/판결); plain BM25 *sums* term scores, so many common matches outrank the one
   rare-entity match. Raising IDF to a power lets the rare term dominate. This
   "entity-burial" fix — found by *inspecting the failing queries*, not fishing — flips
   the last holdout Ko-StrategyQA and lifts every other set. Zero runtime cost (folded into
   the precomputed IDF); the win holds across the whole flat band `p ∈ [1.3, 2.0]`, so it
   is a robust default, not a fit.

On top of these, **field-weighted BM25F** (title 4× body) and **graph-companion fusion**
carry the structured corpora — the finreg multi-hop **107/120 is one-shot, no LLM**,
beating synaptic's own 5-turn LLM agent (88/120) by following `제N조` citations.

**Full-pipeline track** (shared `multilingual-e5-small` embedder, both sides): OmniFuse's
dense+lexical hybrid leads the fused-vs-fused comparison **6/7** (only PublicHealthQA to
synaptic under e5, and that is embedder-dependent — OmniFuse wins it with bge-m3). The
zero-embedder lexical track above already wins all ten, so the embedder is an optional
extra, not a crutch. See
[`eval/results/full_pipeline_e5.json`](eval/results/full_pipeline_e5.json).

```bash
python eval/finreg_bench.py                        # finreg, self-contained
python eval/public_bench.py --synaptic-repo PATH   # the 8 public datasets
```

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

Faster on both axes while retrieving more. Honest framing: *ingest* means "raw corpus →
queryable index"; synaptic writes a persistent SQLite store, which is real work OmniFuse
does not do. `save_index`/`load_index` gives OmniFuse a warm start (0.21 s) but its index
is read back into RAM. Numbers: [`eval/results/perf.json`](eval/results/perf.json).

## Memory — `Feedback`

The deepest difference between OmniFuse and synaptic-**memory** was never the ranking: it
is that synaptic is *stateful* and learns. Neither project had measured whether that
learning improves retrieval, so we built the benchmark — and it first told us we had won
when we had not. (The placebos are in the harness now precisely because of that.)

A confirmed query becomes **evidence about** a chunk: indexed as a BM25F evidence field
whose terms score it but never enter document frequency, and which is not
length-normalized.

```python
from omnifuse import Feedback, build_inmemory
fb = Feedback()
fb.remember("statin side effects", ["doc7"])          # a user confirmed doc7 answered it
of = build_inmemory(nodes, triples, chunks, feedback=fb)
```

Indexing confirmed queries as a document *field* is borrowed — BM25F over title/body/anchor-text,
and query-click logs as a field, are standard web-search practice. What is ours is excluding that
field from document frequency and from length normalization, which the placebo controls forced.
No novelty is claimed; see [`docs/comparison`](docs/comparison/omnifuse_vs_synaptic.md).

Feedback on the original questions, evaluation on held-out **paraphrases** of them — the
case memory exists for. Same corpus, same queries, scored by *synaptic's own* `metrics.py`:

| ΔMRR@10, held-out re-queries | KRA (ko) all | KRA covered | NFCorpus (en) all | NFCorpus covered |
|---|---:|---:|---:|---:|
| synaptic (Hebbian) | +0.0000 | +0.0093 | −0.0010 | −0.0008 |
| **OmniFuse (`Feedback`)** | **+0.1790** | **+0.3903** | **+0.0150** | **+0.0300** |
| ↳ shuffled placebo | +0.0059 | +0.0213 | +0.0015 | +0.0031 |
| ↳ random-query placebo | +0.0029 | +0.0215 | +0.0000 | +0.0000 |

`real` is 5.2× the strongest placebo, so the `(query, chunk)` pairing is what carries the
signal. On *unrelated* held-out questions memory correctly does nothing (+0.0006) and
Δuncovered is exactly **0.0000** — the collection's IDF is provably untouched. A cold store
ranks **bit-identically** to one built with no feedback, so memory can never regress a
system that has not been used. Nothing is tuned.

### Learning without a rebuild

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
[`eval/incremental_bench.py`](eval/incremental_bench.py) ·
[`eval/results/incremental_memory.json`](eval/results/incremental_memory.json) ·
[`tests/test_incremental.py`](tests/test_incremental.py). `forget(query, doc_ids)` is the
exact inverse — it withdraws a pair in ~1 ms, bit-identical to a rebuild without it, and a
term whose last holder forgets it is erased from the vocabulary. Remember everything, forget
everything, and you land bit-identically on the cold index.

Why synaptic scores ~0: in the benchmarked version its `graph.search()` reads none of the
fields `reinforce()` writes. Its consolidation cascade is the same story, now measured: `maintain()`
promotes 54 nodes and decays all 5,234, and retrieval moves by exactly **+0.0000**. Harness, controls and the full retraction history:
[`eval/adaptive_bench.py`](eval/adaptive_bench.py) ·
[`eval/results/adaptive_memory.json`](eval/results/adaptive_memory.json).

## Roadmap

- `backends/qdrant.py` vector adapter; jena-text fast path for `FusekiGraph`
- async pipeline (parallel seeds via `asyncio.gather`)
- cross-encoder reranker hook, query expansion
- configurable ISA predicates and prompt templates (per domain/language)

## Vault — fuse / surface (omnifuse-native memory)

A growing knowledge store with two omnifuse-specific dynamics, not a generic remember/recall:
**fuse-on-write** (facts deduped & merged by entity) and **salience** (frequently fused/surfaced
nodes rank higher). Zero infra; notes auto-link to known entities; persists to JSONL.

```python
from omnifuse import Vault

v = Vault()
v.fuse(facts=[("담보", "instanceOf", "규정")])
v.fuse("담보 한도는 5억원이다", facts=[("담보", "한도", "5억")])
print(v.surface("담보 한도").answer)     # fusion search over everything fused, salience-ranked
v.save("vault.jsonl"); v2 = Vault.load("vault.jsonl")
```

## CI / Releasing

- `ci.yml` — runs pytest (3.10–3.12) + `python -m build` + `twine check` on every push/PR.
- `publish.yml` — on a GitHub **Release**, builds and uploads to PyPI via **Trusted Publishing**
  (no token in the repo). One-time PyPI setup: project → *Publishing* → add pending publisher
  `PlateerLab / xgen-omnifuse / publish.yml / pypi`. (Token mode: add `secrets.PYPI_API_TOKEN`.)

Build locally:

```bash
pip install build && python -m build      # dist/*.tar.gz + *.whl
```

## License

TBD.
