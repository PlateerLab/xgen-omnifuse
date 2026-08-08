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

For a static text-only corpus, `build_inmemory(..., auto_link_titles=True)` derives
directed `references` edges when a passage names another passage's unambiguous title:

```python
of = build_inmemory([], [], chunks, auto_link_titles=True)
```

The linker uses a token trie, ignores ambiguous and single-token aliases, and recognizes
conservative name forms such as `Philip V` for `Philip V of Spain`. It is opt-in because
an incrementally mutable corpus needs an equally mutable graph; combining
`auto_link_titles=True` with `mutable=True` is rejected instead of leaving stale edges.

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
  linking.py       # deterministic title-mention graph edges for static text corpora
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

## Benchmarks

### Official HotPotQA retrieval and local-LLM E2E cohort (2026-08-08, v22)

The v22 comparison reproduces Synaptic's tagged
[`test_e2e_qa.py`](https://github.com/PlateerLab/synaptic-memory/blob/v0.27.0/tests/benchmark/test_e2e_qa.py)
on all 24 seed-42 HotPotQA questions and the same 226-document corpus. Retrieval uses the
upstream limits of 10 search results, 8 evidence steps and 2,048 approximate context tokens.
Synaptic runs its native classifier, relation detector, five-phrase extractor and
`build_evidence`; OmniFuse runs native retrieval with the new deterministic title-reference
graph and product MMR. Four isolated workers are counterbalanced AB/BA/AB/BA, with no warm-up.

| retrieval/evidence metric (worker median) | **OmniFuse** | synaptic-memory |
|---|---:|---:|
| Recall | **0.9792** | 0.7292 |
| Hit rate | **1.0000** | 0.9583 |
| all-gold-document rate | **0.9583** | 0.5000 |
| MRR | **0.9077** | 0.8451 |
| nDCG | **0.9006** | 0.6908 |
| answer exact presence / token recall | **0.9583 / 0.8750** | 0.5833 / 0.5250 |
| build / mean retrieval / p95 retrieval (ms) | **14.49 / 2.39 / 2.54** | 58.27 / 65.73 / 82.71 |
| RSS delta (MB) | **1.573** | 1.696 |

OmniFuse wins all **11/11** common retrieval, evidence and efficiency metrics. The derived
graph has 226 document nodes and 94 unambiguous title-reference edges; no query, answer,
qrel or document id is used to create an edge.

The answer pass uses the upstream prompt, `/api/chat` payload (`stream:false`,
`think:false`) and simple correctness function unchanged. Both systems call the same local
Ollama 0.32.6 `qwen3.5:4b` model digest, alternating order for every question. The 120-second
timeout is transport allowance only. All 48 responses are immutable external checkpoints.

| answer metric | **OmniFuse** | synaptic-memory |
|---|---:|---:|
| cohort mean correctness, including zeroes | **0.7542** | 0.5040 |
| upstream nonzero-only mean correctness | **0.9051** | 0.8641 |
| accuracy at 0.5 / exact-score rate / positive-score rate | **0.7500 / 0.7083 / 0.8333** | 0.5000 / 0.5000 / 0.5833 |
| mean / p95 generation (ms) | **37,573 / 55,404** | 42,237 / 65,621 |
| total generation (ms) | **901,760** | 1,013,679 |
| mean retrieval (ms) | **2.308** | 71.914 |
| mean prompt tokens | **1,134.96** | 1,243.54 |

OmniFuse wins all **10/10** common answer-quality and cost metrics. Because local LLM output
is stochastic, this is one complete controlled run, not a cross-model constant. The upstream
mean excludes zero-score answers, so the zero-inclusive cohort mean is reported first.

Evidence:
[`e2e_qa_retrieval_synaptic_tag_v0.27.0_836d536_20260808_v22.json`](eval/results/e2e_qa_retrieval_synaptic_tag_v0.27.0_836d536_20260808_v22.json)
(SHA-256 `860ea0c2e9b4caa1d7598fcda269cb017c24d095fd39038965883a86399e4ded`)
and
[`e2e_qa_answer_synaptic_tag_v0.27.0_836d536_qwen3.5_4b_20260808_v22.json`](eval/results/e2e_qa_answer_synaptic_tag_v0.27.0_836d536_qwen3.5_4b_20260808_v22.json)
(SHA-256 `0c96d722f07700603ad0a44d4ca825954da964323b870f28eca112c905677961`).
Both artifacts record the official tag/SHA, data and source hashes, complete per-question
rows, process identity, model digest and successful postflight checks.

### Official-test-path LongMemEval-S retrieval cohort (2026-08-03, v20)

Synaptic's tagged
[`test_longmemeval.py`](https://github.com/PlateerLab/synaptic-memory/blob/v0.27.0/tests/benchmark/test_longmemeval.py)
mixes memory retrieval with an external answer LLM. The v20 cohort isolates the part both
memory systems can own: the upstream test's plain `graph.search(question, limit=20)` and
mean gold-session recall. It reproduces the upstream seed-42 balanced sampling, exact
turn-pair construction, 2,000-character truncation, session/date/turn metadata, fresh index
per question and deduplicated session ranking. The requested 50-question default produces
48 questions: eight from each of six types, covering 2,296 sessions, 23,668 turns and
11,935 indexed turn pairs.

Two fresh workers per system run in AB/BA order under Python 3.12.10. The controller writes
the selected 48 questions to one immutable 25,708,257-byte artifact before launching workers;
workers do not parse and retain the 277 MB source dataset, so dataset-loader memory is not
misreported as index RSS. Every source, repository, input, sample, worker and postflight
fingerprint is recorded. Rankings are bit-deterministic across both trials.

| worker-median metric | **OmniFuse** | synaptic-memory | winner |
|---|---:|---:|---|
| upstream mean session recall@20 | **0.9705** | 0.8413 | **OmniFuse** |
| session hit rate@20 | **1.0000** | 0.9167 | **OmniFuse** |
| session MRR@20 | **0.8936** | 0.6990 | **OmniFuse** |
| session nDCG@20 | **0.8937** | 0.6898 | **OmniFuse** |
| total build, 48 fresh indexes (ms) | **20.4873** | 82.1059 | **OmniFuse** |
| mean query latency (ms) | **51.6333** | 235.1101 | **OmniFuse** |
| per-trial query p95 (ms) | **58.4504** | 252.8300 | **OmniFuse** |
| maximum per-question RSS delta (MB) | **1.0056** | 2.1770 | **OmniFuse** |

OmniFuse's build value is native lazy-store construction; complete lexical materialization
is charged to that question's single retrieval, not hidden in a warm-up. Synaptic's native
`graph.add` work is charged to build and its native search work to retrieval.

OmniFuse wins all **8/8 common aggregate metrics**. On the upstream recall metric it has
10 question wins, 38 ties and **0 losses**. MRR and nDCG each have two individual-question
losses, so the table is an aggregate result rather than a claim that every rank is better.
Answer generation and answer correctness are deliberately excluded because they depend on
the external LLM, not only the memory implementation.

Evidence:
[`longmemeval_retrieval_synaptic_tag_v0.27.0_836d536_20260803_v20.json`](eval/results/longmemeval_retrieval_synaptic_tag_v0.27.0_836d536_20260803_v20.json)
(SHA-256 `f5d8c551cf658f8b8d2fdeb64f0b46d04542c14f0f8149eedc9c711ced488536`).
The full source dataset is the official cleaned LongMemEval-S file (500 questions,
277,383,467 bytes, SHA-256
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`).

### Official ablation-stage comparison (2026-07-30, v19)

The unmodified tagged `tests/benchmark/test_ablation.py` was allowed to finish all 13
datasets. It ended with **13 skipped in 59,198.80 seconds (16:26:38)**: each case reaches
the external S8 LLM stage and skips when neither the configured remote endpoint nor local
Ollama is available. That raw run has empty stderr and is preserved as execution evidence,
but a skipped suite is not presented as a passing performance result.

The controlled S0-S7 comparison therefore keeps two explicit lanes: S0/S1/S2/S6 and
OmniFuse cold retrieval use no evaluation labels; S3/S4/S5 and OmniFuse feedback are
qrels-supervised. The values below use the best Synaptic value across the listed comparable
stages, making the comparison conservative rather than selecting one weak stage.

| dataset / lane | metric | **OmniFuse** | best comparable Synaptic stage |
|---|---|---:|---:|
| AutoRAG, unlabeled | MRR / nDCG / Recall@10 | **0.8908 / 0.9187 / 1.0000** | 0.8173 / 0.8591 / 0.9900 |
| AutoRAG, unlabeled | mean query / build ms | **6.086 / 1.300** | 200.120 / 6.980 |
| NFCorpus, unlabeled | MRR / nDCG / Recall@10 | **0.5486 / 0.3334 / 0.1674** | 0.5116 / 0.2959 / 0.1466 |
| NFCorpus, unlabeled | mean query / build ms | **5.274 / 2.430** | 184.993 / 15.234 |
| NFCorpus, qrels-supervised | MRR / nDCG / Recall@10 | **0.6276 / 0.5121 / 0.2989** | 0.2908 / 0.1832 / 0.0983 |
| NFCorpus, qrels-supervised | mean query / build ms | **0.287 / 585.841** | 12,128.912 / 2,226,033.641 |

AutoRAG has no query eligible for the upstream feedback simulator's “at least two relevant
documents” rule, so no supervised AutoRAG claim is made. S7 is also excluded from the fair
table: its Ollama provider catches connection errors and returns `[]`, preventing the test's
documented mock-embedding fallback and adding 921 failed embedding attempts in that run.
S8 remains an external-LLM capability, not a retrieval comparison.

Evidence:
[`ablation_autorag_synaptic_tag_v0.27.0_836d536_20260730_v19.json`](eval/results/ablation_autorag_synaptic_tag_v0.27.0_836d536_20260730_v19.json)
(SHA-256 `98910fff89c28c90b13cadd901a446f213496df8f94eefeaeac9aa89d1a8516d`) and
[`ablation_nfcorpus_synaptic_tag_v0.27.0_836d536_20260730_v19.json`](eval/results/ablation_nfcorpus_synaptic_tag_v0.27.0_836d536_20260730_v19.json)
(SHA-256 `e2bd7173586c397f229b924441e939d88bee460905b12e9d2eed4221ea320ac0`).

### Official synaptic-memory `v0.27.0` direct cohort (2026-07-28, v18 revalidation)

The current canonical accuracy result invokes the data preparation and native no-embedding
`MemoryBackend` lexical path
declared by synaptic-memory's official `v0.27.0` tag at
`836d53640e520c88910dd57e098167a4defe50d2`. Both systems receive the same corpus,
selected queries, qrels, 2,000-character document truncation, seed-42 sampling and
20-candidate retrieval contract. Both are scored by the official driver's
[`tests/benchmark/metrics.py`](https://github.com/PlateerLab/synaptic-memory/blob/v0.27.0/tests/benchmark/metrics.py)
`BenchmarkResult`. The local `eval/metrics.py` is independently verified byte-identical
to that tag file
(SHA-256 `3634fe7d…e4ef978`). The official upstream MRR is
computed over all 20 candidates; the other metrics and the separately recomputed MRR@10
use K=10.

The frozen run completed all **14/14 executable cases** in the tag's
[`test_external_datasets.py`](https://github.com/PlateerLab/synaptic-memory/blob/v0.27.0/tests/benchmark/test_external_datasets.py)
and evaluated **2,269 queries per system**.
OmniFuse wins the unweighted dataset macro on every reported metric:

| metric | synaptic-memory | **OmniFuse** | dataset W/L/T |
|---|---:|---:|---:|
| upstream MRR@20 | 0.6553 | **0.7038** | **14/0/0** |
| MRR@10 | 0.6537 | **0.7022** | **14/0/0** |
| Precision@10 | 0.1505 | **0.1776** | **14/0/0** |
| Recall@10 | 0.6606 | **0.7178** | **13/0/1** |
| F1@10 | 0.1919 | **0.2206** | **14/0/0** |
| nDCG@10 | 0.6100 | **0.6747** | **14/0/0** |

| dataset | queries | synaptic MRR@10 | **OmniFuse MRR@10** |
|---|---:|---:|---:|
| Ko-StrategyQA | 100 | 0.6608 | **0.6655** |
| AutoRAG | 114 | 0.8310 | **0.8955** |
| KLUE-MRC | 100 | 0.8555 | **0.9463** |
| Allganize RAG-Eval | 300 | 0.9109 | **0.9363** |
| Allganize RAG-ko | 200 | 0.9468 | **0.9708** |
| HotPotQA-24 | 24 | 0.8750 | **0.9077** |
| HotPotQA-200 | 200 | 0.8340 | **0.8958** |
| PublicHealthQA | 77 | 0.5378 | **0.6249** |
| NFCorpus | 100 | 0.4771 | **0.5058** |
| SciFact | 100 | 0.5245 | **0.6517** |
| FiQA | 100 | 0.2048 | **0.2476** |
| MIRACL-ko | 100 | 0.9287 | **0.9800** |
| MultiLongDoc-ko | 100 | 0.2616 | **0.2959** |
| XPQA-ko | 654 | 0.3027 | **0.3069** |

The v18 revalidation records no dataset losses on MRR@20, MRR@10, Precision@10, F1@10
or nDCG@10; Recall@10 has one tie. The improvement comes from shared query analysis and
scoring behavior, not shortened result lists: no query id, dataset id, per-document exception
or benchmark-specific cutoff is shipped.

The result artifact is
[`direct_external14_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json`](eval/results/direct_external14_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json)
(SHA-256 `b8b0d333c4fb44adb8b32d3fbea01abea232f42f34a0ae71cd667a302e9e51ef`).
Its 14 immutable worker records include every top-10/top-20 ranking and passed pre/post
checks for the official tag, inputs, source trees, Python environment and doctor manifest.
Against the v15 artifact, every macro metric, dataset verdict and all **2,269 OmniFuse
top-20 rankings** are exactly unchanged. A canonical bundle of
`[case_id, query_id, retrieved_top_20]` has SHA-256
`3b6bd4f9d3213036adb128637bff52adfe08803ec00389d1e8a7ec3c742c25c6`
in both runs. The frozen v15 baseline remains
[`direct_external14_fts_synaptic_tag_v0.27.0_836d536_20260724_v15_static_compact_v1.json`](eval/results/direct_external14_fts_synaptic_tag_v0.27.0_836d536_20260724_v15_static_compact_v1.json).
The direct run's latency fields are observational: the upstream-compatible
Windows `time()` clock is too coarse for sub-millisecond OmniFuse queries, so precise speed
claims are kept out of this accuracy table and must use a separate monotonic,
high-resolution repeated protocol.

### Official QA performance contract (2026-07-28, v18)

The QA performance cohort reproduces synaptic-memory
`tests/qa/conftest.py::combined_graph`: the first 50 Korean Wikipedia documents, 50 GitHub
commits and 50 GitHub issues, plus the exact 16-query sequence and thresholds from
`tests/qa/test_performance.py`. Each system runs in four fresh isolated workers in AB/BA/AB/BA
order. The official first pass includes cold tokenizer and index initialization; steady
repeats are reported separately so test-order warming cannot hide initialization cost.
All timings use `perf_counter_ns`.

| worker median | **OmniFuse** | synaptic-memory |
|---|---:|---:|
| build | **0.2015 ms** | 65.4525 ms |
| first query / official p95 | **49.2506 ms** | 3,079.7073 ms |
| official average over 16 queries | **3.1906 ms** | 205.0199 ms |
| steady query p50 | **0.0252 ms** | 8.2988 ms |
| steady query p95 | **0.0524 ms** | 21.4472 ms |
| steady 16-query round | **0.4643 ms** | 159.6240 ms |
| post-query RSS | **35.43 MB** | 534.99 MB |
| lifetime peak RSS | **45.30 MB** | 598.32 MB |

OmniFuse wins all **10/10 common efficiency metrics**. It satisfies the upstream
`p95 < 100 ms` and `average < 50 ms` contracts in **4/4** cold workers; the tagged
synaptic-memory path satisfies each in **0/4**. Rankings are deterministic in 4/4 workers
for both systems. The structural change generalizes the immutable forward-only fast path to
one- and two-field non-evidence indexes, constructing packed vocabulary once after ingestion.
It has no dataset-size threshold, query exception, cache, score-formula change or
benchmark-specific branch. Relative to the pre-change cohort, OmniFuse's median cold p95
falls from 73.6668 to 49.2506 ms (**-33.1%**) while the exact ranking SHA-256 remains
`029157a2197480fcae7055bf267d0a405ea538db55880193714ebbfdf933dec8`.

The upstream batch test measures `MemoryBackend.save_nodes_batch`, whereas OmniFuse exposes
both raw lazy-store construction and full lexical materialization. Those operations are
recorded but excluded from the 10-metric verdict because they are not capability-equivalent.
Evidence:
[`qa_memory_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json`](eval/results/qa_memory_synaptic_tag_v0.27.0_836d536_20260728_v18_forward_fast_v1.json)
(SHA-256 `bcb3e213f4271a341f198d38db925e58c9cc1ab50fc8c85065ffa02ea6ef62ed`).

### Official enterprise native-capability cohort (2026-07-28, v18)

Five repeated runs reproduce the tagged `tests/benchmark/test_enterprise_benchmark.py`
scenario at K=5. `full_native` preserves each product's documented graph and memory adapter,
so internal memory state is not declared equivalent; retrieval outcomes are compared. Accuracy
is deterministic across the five runs.

| `full_native` median | synaptic-memory | **OmniFuse** |
|---|---:|---:|
| MRR | 0.7467 | **0.7689** |
| nDCG@5 | 0.6649 | **0.7637** |
| Recall@5 | 0.7333 | **0.8167** |
| mean latency | 4.4170 ms | **0.0557 ms** |
| build | **0.7840 ms** | 2.7951 ms |

The tiny 12-document fixture leaves build as a disclosed Synaptic win; OmniFuse wins the
three retrieval metrics and query latency. `docs_only` is deliberately excluded from the
verdict: Synaptic still receives dataset intent annotations that select `search` versus
`agent_search`, while OmniFuse receives only the raw query. In that asymmetric ablation,
Synaptic leads MRR (0.8333 vs 0.7611), while OmniFuse leads nDCG@5 (0.7789 vs 0.7044),
Recall@5 (0.9000 vs 0.7333) and median latency (0.2928 vs 1.0855 ms). No product logic was
tuned to this toy fixture.

Evidence:
[`enterprise_synaptic_tag_v0.27.0_836d536_20260728_v18_v1.json`](eval/results/enterprise_synaptic_tag_v0.27.0_836d536_20260728_v18_v1.json)
(SHA-256 `a87a1ab07fb3b0f25fa8b960efe4af6b9145779824ecc4b5e06d446f4a649d67`).

### Static immutable-index footprint (2026-07-24, v15)

Static `InMemoryGraph` and non-feedback `InMemoryVector` lexical stores now use the same
forward-only `CompactPostingsSnapshot`. They retain canonical term/posting/field-length state
but omit reverse document records. Pickle persistence uses the existing validated packed-forward
schema, and SQLite persistence writes the raw posting stream directly. Feedback-backed stores
remain on mutable BM25F.

A deterministic three-run structural diagnostic used 30,000 fielded documents and 30 queries.
Every top-10 id and `float.hex` score matched the BM25F reference (ranking SHA-256
`7d670dcb1e24581e1c05c4c913a88c37c0f76f76ec29f52ef059946e513532ab`).

| median diagnostic | BM25F reference | v15 compact | change |
|---|---:|---:|---:|
| index RSS delta | 28.47 MB | **7.42 MB** | **-73.9%** |
| persistence state | 4,408,135 B | **1,382,373 B** | **-68.6%** |
| serialize | 72.16 ms | **4.05 ms** | **-94.4%** |
| build | **490.04 ms** | 565.10 ms | +15.3% |
| unseen-term query p50 | **0.020 ms** | 0.134 ms | +0.114 ms |
| repeated-query p50 | **0.019 ms** | 0.020 ms | +0.001 ms |

The build and unseen-term regressions are reported rather than hidden; this exact diagnostic is
not a Synaptic capability comparison. The official direct cohort above and persistence cohort below remain
the cross-system evidence. Artifact:
[`static_index_micro_20260724_v15_static_compact_v1.json`](eval/results/static_index_micro_20260724_v15_static_compact_v1.json)
(SHA-256 `87c4eb4f2215dbe68bbc7147080b346e30611175fcc0a1099a0d885687c8aae8`).
Research on partitioned Variable-Byte and block-max query processing informs the next loop; no
SIMD or WAND claim is made by this pure-Python implementation.

### Official `v0.27.0` MemoryBackend performance (2026-07-23, v8)

The high-resolution performance cohort reuses the official case selection, preprocessing,
native no-embedding `MemoryBackend`, K=10, candidate limit 20 and the same frozen Python
environment as the direct14 accuracy run. Each system runs in a fresh process for two
counterbalanced AB/BA trials, with one warm-up and five measured rounds. Fully materialized
raw top-20 retrieval is timed with monotonic `time.perf_counter_ns`; every warm-up and measured
ranking must remain identical inside the run. The official `uv.lock` and installed-package
manifest are verified before and after every worker. Repositories, inputs, scorer, driver and
doctor binding are verified at controller preflight/postflight before atomic write-once
publication. The published canonical hashes were then independently matched to direct14.

The values below are means across the two independent workers. Query p50/p95 first summarize
all measured query calls inside each worker. End-to-end adds native ingest to that worker's
mean duration for one complete measured query-set round, then averages the two workers. Bold
marks the lower observed cost or higher MRR within that row pair.

| dataset | docs / queries | system | ingest mean s | query p50 mean ms | query p95 mean ms | end-to-end mean s | peak RSS mean MB | MRR@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| AutoRAG | 720 / 114 | **OmniFuse** | **0.0010** | **0.4030** | **0.6016** | **0.0483** | **63.50** | **0.8919** |
| | | synaptic | 0.0074 | 206.8121 | 260.0951 | 23.7816 | 614.38 | 0.8310 |
| Allganize RAG-ko | 200 / 200 | **OmniFuse** | **0.0003** | **0.1047** | **0.1884** | **0.0226** | **43.25** | **0.9683** |
| | | synaptic | 0.0016 | 93.3795 | 105.0590 | 16.7036 | 611.19 | 0.9468 |
| NFCorpus | 3,633 / 100 | **OmniFuse** | **0.0044** | **0.1758** | **1.5593** | **0.0532** | **74.06** | **0.5058** |
| | | synaptic | 0.0371 | 271.9918 | 349.2337 | 23.0643 | 74.07 | 0.4771 |

On this host, the Synaptic/OmniFuse ratios are **7.46× / 5.86× / 8.49×** for ingest,
**513.25× / 892.09× / 1,546.95×** for p50 query latency, and
**492.05× / 739.50× / 433.65×** for ingest-plus-mean-query-set-round. OmniFuse's observed
peak RSS is also lower in all three workers; the NFCorpus difference is only 0.014 MB and
should not be treated as a meaningful cross-machine margin.

The structural change behind the removed ingest deficit is lazy materialization for plain
static `InMemoryVector` stores. Construction snapshots the scalar title/text source, while the
immutable BM25/BM25F index is built once, under a lock, on the first lexical query. Mutable and
feedback-backed stores remain eager so their snapshot and evidence semantics do not change.
The first-query cost is not hidden: it is measured separately by the cold CDC round below.

### Large-corpus TREC-COVID observation (2026-07-24)

The completed v14 AB/BA controller used 171,332 documents, 50 scored queries, one warm-up and
five measured rounds in four fresh workers. A v15 OmniFuse-only two-worker follow-up reused the
byte-identical frozen input; it is not presented as a new two-system counterbalanced cohort.

| median | v15 OmniFuse follow-up | canonical Synaptic v0.27.0 |
|---|---:|---:|
| MRR@10 | **0.9083** | 0.7210 |
| query p50 | **109.49 ms** | 2,937.81 ms |
| query p95 | **158.94 ms** | 3,269.43 ms |
| query mean | **110.52 ms** | 2,932.69 ms |
| current RSS | 786.08 MB | **412.88 MB** |
| lifetime peak RSS | 2,134.72 MB | **2,133.99 MB** |

OmniFuse is 26.83x faster at p50 and gains 0.1873 MRR, while current RSS is still 1.90x
Synaptic in this RAM-index-versus-durable-SQLite capability-qualified protocol. v15 reduces its
own current RSS from 1,078.30 MB to 786.08 MB (-27.1%) without changing MRR. Peak RSS is
parser-dominated for both systems. Ingest is excluded from superiority claims because OmniFuse
materializes its lexical index during the unmeasured warm-up in this protocol.

Evidence:
[`perf_trec_covid_sqlite_native_trials2_20260724_v14_raw_df_v1.json`](eval/results/perf_trec_covid_sqlite_native_trials2_20260724_v14_raw_df_v1.json)
(SHA-256 `495e5b8fdf7f7f40dd30ec991f1d92b565398f1cce54182a05f6ddcac86a04dc`) and
[`perf_trec_covid_v15_omnifuse_followup_20260724_v1.json`](eval/results/perf_trec_covid_v15_omnifuse_followup_20260724_v1.json)
(SHA-256 `713390786f95cfcbf1b1b010f60c214b2be7fc01d77bb65c3b64fee469c3f061`).

### Official native SQLite persistence cohort (2026-07-28, v17)

Both systems start from byte-identical frozen inputs and finish with closed, durable,
disk-queryable SQLite artifacts. OmniFuse uses `build_sqlite_index` / `open_sqlite_index`;
Synaptic uses `SqliteGraphBackend.save_nodes_batch` / `SynapticGraph.search`. K=10,
candidate limit 20 and the byte-identical six-metric scorer are shared. Allganize RAG-ko,
AutoRAG and NFCorpus use four fresh workers per system in AB/BA/AB/BA order with one warm-up
and three measured rounds. The 171,332-document TREC-COVID cohort uses two workers per system
in AB/BA order and two measured rounds. `O / S` means OmniFuse / Synaptic.

The v17 harness releases ingestion-only staging before clean open and records post-create,
clean-open and post-query RSS at identical phases. Every value below is a worker median.

| median efficiency metric | Allganize RAG-ko O / S | AutoRAG O / S | NFCorpus O / S | TREC-COVID O / S |
|---|---:|---:|---:|---:|
| durable create (s) | **0.0897 / 3.6958** | **0.6646 / 13.3143** | **1.0521 / 1.1533** | **38.6159 / 42.8383** |
| clean open (ms) | **1.0438 / 2.5157** | **0.9809 / 2.5457** | **1.0310 / 2.4883** | **1.9352 / 2.5216** |
| first query (ms) | **1.1161 / 143.2841** | **4.8718 / 264.8729** | **0.7301 / 81.7479** | **864.7729 / 1,577.6638** |
| steady p50 (ms) | **0.9335 / 137.2669** | **3.9969 / 262.8316** | **1.5333 / 271.5704** | **895.1377 / 1,466.2267** |
| steady p95 (ms) | **1.5236 / 153.5278** | **6.2968 / 324.3645** | **13.1850 / 315.8520** | **1,252.1812 / 1,727.1442** |
| complete query round (s) | **0.1910 / 24.7631** | **0.4771 / 30.4265** | **0.4028 / 21.2739** | **43.8405 / 74.2288** |
| SQLite artifact (bytes) | **499,712 / 524,288** | **2,686,976 / 4,526,080** | **5,627,904 / 18,132,992** | **183,779,328 / 630,202,368** |
| post-run RSS (MB) | **34.92 / 521.45** | **38.70 / 529.04** | **45.51 / 50.02** | **407.19 / 454.45** |
| post-create RSS (MB) | **34.67 / 537.26** | **38.91 / 543.30** | **45.67 / 48.64** | **375.60 / 453.35** |
| clean-open RSS (MB) | **34.78 / 537.29** | **39.00 / 543.33** | **45.74 / 48.82** | **375.94 / 453.38** |
| post-query RSS (MB) | **35.50 / 521.48** | **39.85 / 533.32** | **46.21 / 55.22** | **407.20 / 454.48** |
| lifetime peak RSS (MB) | **38.65 / 599.63** | **44.85 / 601.37** | **49.88 / 55.58** | **1,813.81 / 1,814.21** |
| workload peak RSS (MB) | **35.58 / 538.74** | **44.57 / 547.60** | **49.84 / 55.58** | **463.53 / 595.19** |

| official accuracy metric | Allganize RAG-ko O / S | AutoRAG O / S | NFCorpus O / S | TREC-COVID O / S |
|---|---:|---:|---:|---:|
| MRR@20 | **0.9733 / 0.9585** | **0.8977 / 0.8911** | **0.5173 / 0.5088** | **0.9029 / 0.7210** |
| MRR@10 | **0.9733 / 0.9585** | **0.8977 / 0.8905** | **0.5147 / 0.5080** | **0.9017 / 0.7210** |
| Precision@10 | **0.1227 / 0.1058** | **0.1000 / 0.0991** | **0.3010 / 0.2844** | **0.6800 / 0.5420** |
| Recall@10 | **1.0000 / 0.9950** | **1.0000 / 0.9912** | **0.1521 / 0.1448** | **0.0173 / 0.0132** |
| F1@10 | **0.2088 / 0.1877** | **0.1818 / 0.1802** | **0.1412 / 0.1329** | **0.0334 / 0.0255** |
| nDCG@10 | **0.9802 / 0.9673** | **0.9237 / 0.9155** | **0.2980 / 0.2807** | **0.7135 / 0.5510** |

The arithmetic verdict is **76 strict wins, 0 ties and 0 losses**: thirteen efficiency and
six official accuracy metrics on each of four datasets. The TREC lifetime-peak margin is only
0.41 MB and is not treated as operationally meaningful because frozen JSON parsing dominates
that metric. The larger workload-peak and phase-RSS margins exclude that parser high-water.

v17 also fixes two measurement defects rather than optimizing the benchmark result: it drops
Synaptic's ingestion-only `Node` staging before clean open, and it routes doctor-registered
non-direct datasets such as TREC through the canonical snapshot preflight while preserving the
upstream selection rules for the 14 direct cases. The raw SQLite format, scoring formula,
rankings and artifact bytes are unchanged from v16; there is no cache, dataset switch or
query-specific branch.

Compact evidence:
[`persistence_memory4_synaptic_tag_v0.27.0_836d536_20260728_v17_summary.json`](eval/results/persistence_memory4_synaptic_tag_v0.27.0_836d536_20260728_v17_summary.json)
(SHA-256 `60411946ca671f8e3f1f402786e4fb7aa703a70671426d4374b3a972ac0ec11b`).
The accepted v16 raw-scorer diagnostic and rejected weighted-forward alternative remain in
[`sqlite_raw_scorer_micro_20260727_v16.json`](eval/results/sqlite_raw_scorer_micro_20260727_v16.json)


### Official NFCorpus CDC result (2026-07-23, v8)

The CDC protocol applies 36 inserts, 36 updates, 36 deletes and 36 no-ops, then checks each
system against its own full rebuild at every checkpoint. It uses the same official scorer and
reports all six accuracy metrics used by synaptic-memory. Two fresh AB/BA workers per system
also measure incremental costs and RSS. Rankings and `float.hex` scores match the corresponding
full rebuild exactly in every worker.

| metric | **OmniFuse** | synaptic |
|---|---:|---:|
| MRR@20 | **0.5080** | 0.4799 |
| MRR@10 | **0.5056** | 0.4771 |
| Precision@10 | **0.2960** | 0.2507 |
| Recall@10 | **0.1514** | 0.1323 |
| F1@10 | **0.1401** | 0.1206 |
| nDCG@10 | **0.2927** | 0.2481 |

| p50 cost | **OmniFuse** | synaptic | Synaptic / OmniFuse |
|---|---:|---:|---:|
| initial ingest | **0.0063 s** | 0.0302 s | 4.82× |
| one mutation group | **0.0004 s** | 0.0029 s | 6.86× |
| cold first query-set round | **1.5952 s** | 23.1189 s | 14.49× |
| steady query-set round | **0.0502 s** | 23.1714 s | 461.72× |
| incremental end-to-end | **1.5957 s** | 23.1218 s | 14.49× |

The compact result is
[`perf_cdc_synaptic_tag_v0.27.0_836d536_20260723_v8_summary.json`](eval/results/perf_cdc_synaptic_tag_v0.27.0_836d536_20260723_v8_summary.json)
(SHA-256 `a4dd6e22df1bac13c5ff5d360dfd86cf2b4ff67ad8bb3ee6f2a3e26be48e6019`).
It binds the three static reports and the CDC report by immutable path, byte size and SHA-256,
plus source, input, scorer, environment, worker and ranking provenance. These are current-host
observations for the official no-embedding in-memory scope, not cross-machine constants or
evidence about Synaptic's persistent backends, embedding pipelines, rerankers or unrelated
agent-memory capabilities. Pre-existing host workloads were left running, so absolute timing
should be reproduced on the target host.

### Historical upstream-main snapshot (2026-07-14)

The completed lexical cohort compares OmniFuse HEAD
`cd355ddb5c59f5f8bb20c694a4fba8419567310a` (with the benchmark worktree recorded by
content fingerprint) against synaptic-memory upstream `main` at
`7470e728a7d728dedea5363aabbb73adf6ac666f`. That checkout reports package version
`0.27.0`, but it is **not** the official `v0.27.0` tag
(`836d53640e520c88910dd57e098167a4defe50d2`). Public IR results are single-shot,
FTS-only, and scored at k=10. The enterprise scenario follows the selected main
fixture at k=5. Full harness, provenance, and numbers are in [`eval/`](eval/).

<details open>
<summary><b>Scoring & fairness — how these numbers were audited</b></summary>

- **Same scorer, verified by content.** `eval/metrics.py` and synaptic's
  `tests/benchmark/metrics.py` have the same SHA-256
  (`3634fe7d…e4ef978`) and are byte-identical.
- **Symmetric harness.** Both read the same dataset file → same corpus, queries and qrels;
  each returns its own top-10; identical scoring.
- **Both re-run from raw inputs.** The 2026-07-14 canonical runs re-ingest each dataset for
  synaptic through its own `run_public_dataset(embedder=None, reranker=None)` driver and
  rebuilds OmniFuse from the same corpus in the same pass.
- **Coverage is explicit.** The doctor fingerprints all 22 harness-declared targets and
  records missing/private inputs rather than silently dropping them. Strict public mode
  requires 19 inputs to validate (17 public IR plus two local finreg inputs); that number
  is an input-validation gate, not a count of benchmark wins. The private KRA corpus
  remains a separately reported target and is not converted into a silent skip.
- **OmniFuse's field-weighted BM25F (title 4× body) is a design advantage**, not scoring
  bias — both systems receive the same `(title, text)`; OmniFuse simply exploits the title
  field harder.
- **synaptic's memory was measured fairly, and we corrected ourselves.** We first reported
  its Hebbian reinforcement as *harmful* (−0.0174); re-running the same configuration gave
  −0.0045 (its warm pass is not deterministic). Channel isolation shows `graph.search()`
  reads none of the fields `reinforce()` writes, so its measured deltas are noise around
  zero — it is **not wired in**, not harmful.
- **KRA is historical in this run.** Its questions are LLM-generated and its raw corpus is
  private; the 2026-07-13 doctor marks it `skipped_private`, so it is not counted in the
  current same-pass wins.
- **The IDF emphasis is a Pareto trade, not a free win** — it wins the core suite but
  regresses heavily multi-relevant corpora (see footnote ² below).
- A parameter sweep we previously published was **invalid** (keyword-only defaults bind at
  definition time, so the monkeypatch changed nothing). The current sweep passes
  `idf_pow` into each constructed index and re-runs synaptic in the same pass.

</details>

#### Public8 and extended9 results

The declared matrix is **22 targets**: two local finreg tasks, eight public JSON files tracked
by synaptic-memory, nine upstream-declared or downloader-generated public inputs, one tracked
enterprise fixture, one official QA combined fixture, and one private KRA target. Missing and private inputs remain in the denominator. The public8 and extended9
cohorts below are two separate, complete invocations against the same upstream-main commit;
they are not one 17-dataset same-pass run and are not official-tag results.

The completed 2026-07-14 tracked-public run uses identical corpus/query/qrels files and the
byte-identical scorer at K=10:

| dataset | synaptic FTS | **OmniFuse** |
|---|---:|---:|
| HotPotQA-24 | 0.8879 | **0.9077** |
| HotPotQA-200 | 0.8775 | **0.8958** |
| Allganize RAG-ko | 0.9595 | **0.9683** |
| Allganize RAG-Eval | 0.9303 | **0.9371** |
| PublicHealthQA | 0.6065 | **0.6133** |
| AutoRAG | 0.8994 | **0.9282** |
| KLUE-MRC | 0.7718 | **0.8293** |
| Ko-StrategyQA | 0.6440 | **0.6466** |
| **Mean MRR@10** | 0.8221 | **0.8408** |

OmniFuse wins **8/8** tracked public datasets on MRR@10. It also wins 8/8 on Recall@10
and nDCG@10. Precision@10 improves in macro average but wins 5/8 datasets, so this is not
an "all metrics on every dataset" result. Machine-readable canonical provenance:
[`public_v027_20260714_canonical.json`](eval/results/public_v027_20260714_canonical.json).

The separately completed extended9 invocation evaluates the shipped global
`idf_pow=1.2` setting (the same value for every dataset):

| dataset | synaptic FTS | **OmniFuse** |
|---|---:|---:|
| 2WikiMultihopQA-dev | 0.8244 | **0.9517** |
| MuSiQue-dev | 0.7316 | **0.7791** |
| TREC-COVID | 0.7437 | **0.9083** |
| SciFact | 0.6317 | **0.6441** |
| XPQA-ko | 0.3115 | **0.3277** |
| NFCorpus | 0.5124 | **0.5175** |
| MIRACL-ko | 0.9495 | **0.9750** |
| FiQA | 0.2902 | **0.2920** |
| MultiLongDoc-ko | 0.6326 | **0.6501** |
| **Mean MRR@10** | 0.6253 | **0.6717** |

OmniFuse wins **9/9** extended datasets on MRR@10. This invocation records MRR only;
it does not support claims about the other three public8 metrics. Combining the two
independent cohorts after the fact gives 17/17 MRR@10 wins and an unweighted dataset macro
of **0.7513 vs 0.7179**. Provenance:
[`extended9_v027_20260714_canonical.json`](eval/results/extended9_v027_20260714_canonical.json).

Finreg was also rebuilt locally at the current default and reproduced MRR@10 **0.8471**,
nDCG@10 **0.8738**, hit@10 **115/120**, and multi-hop strict **108/120**. The current
synaptic graph comparison is kept separate because its runner starts from a prebuilt SQLite
graph while OmniFuse rebuilds from raw JSONL. The `108/120` result and synaptic's historical
five-turn agent `88/120` therefore describe different protocols and are context, not a
controlled head-to-head. Provenance:
[`finreg_omnifuse_v027_20260713.json`](eval/results/finreg_omnifuse_v027_20260713.json).

The historical upstream-main enterprise artifact is preserved for audit, but it is superseded
by the official-tag five-run v18 cohort above. Its old full-scenario aggregate did not
reproduce against the selected fixture, and its `docs_only` track is not an equal-input IR
comparison because Synaptic receives intent routing metadata. Historical evidence:
[`enterprise_v027_20260713.json`](eval/results/enterprise_v027_20260713.json).

The historical private KRA result is preserved in
[`golden_devxgen.json`](eval/results/golden_devxgen.json), but its corpus was unavailable in
this session and is not counted as current evidence. The detailed 2026-07-10 investigation,
including rejected variants, is retained as a
[historical comparison](docs/comparison/omnifuse_vs_synaptic.md).

### Historical speed context (2026-07-10) — **not** a universal win

⚠️ **Read the conditions, not just the ratio.** OmniFuse's speed advantage holds when both
systems index from raw data. It does **not** hold unconditionally, and it did not always
hold at all:

| scenario | conditions | synaptic | OmniFuse |
|---|---|---:|---:|
| golden set (5,234 chunks, 215 q) | both index from raw data — **the only apples-to-apples row** | 98.0 s | **6.6 s** |
| finreg — 2026-07-10 | synaptic reuses a **prebuilt** SQLite graph; omni rebuilds | 11.0 s | **7.4 s** |
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

### How OmniFuse wins — structural, zero-hardcode logic improvements

No strong embedder and no per-dataset runtime switch. The global default was selected from
suite-level failure analysis and is disclosed as such:

0. **Symmetric morphology — Korean *and* English.** Latin tokens were indexed as raw
   surface forms, so `statin` could not match `statins`, while Korean got full
   normalization. Harman's **S-stemmer** (singularize only; no tunable parameter, so
   nothing to fit) closes the asymmetry: NFCorpus **0.5053 → 0.5182** flips to a win,
   SciFact +0.003, HotPotQA-200 +0.002, and every Korean set is bit-identical.
1. **Dependency-free Korean stemmer** — strips 조사/어미 + trailing derivational suffixes
   so a query and a doc align on the stem the way Kiwi would, but pure Python and emitting
   *fewer* tokens (more accurate *and* more efficient). Flips AutoRAG + PublicHealthQA.
2. **IDF term-specificity emphasis** (`idf_pow=1.2`) — a question ("장 발장은 어떤 범죄로
   유죄 판결을 받았나요?") buries its one rare entity (발장) under common words
   (범죄/유죄/판결); plain BM25 *sums* term scores, so many common matches outrank the one
   rare-entity match. Raising IDF to a power lets the rare term dominate. This
   "entity-burial" fix — found by *inspecting the failing queries*, not fishing — flips
   Ko-StrategyQA without crossing FiQA's upper constraint. The exponent is folded into
   precomputed IDF, so it adds no query-time work. The global overlap found by the earlier
   suite sweep was `[1.1, 1.3]`; 1.2 is its midpoint. The completed five-arm extended9
   sweep confirms that the one shipped value wins all nine sets. Its p1.1 arm has a slightly
   higher extended9 macro, so p1.2 is described as the stable shipped midpoint, not as the
   post-hoc macro-optimal arm.

On top of these, **field-weighted BM25F** (title 4× body) and **graph-companion fusion**
carry the structured corpora — the finreg multi-hop **108/120 is one-shot, no LLM**,
beating synaptic's own 5-turn LLM agent (88/120) by following `제N조` citations.

**Exploratory full-pipeline track**: historical dense runs cover seven datasets and their
best observed OmniFuse variant leads six. Because the embedder/configuration is not one
fixed predeclared head-to-head across all seven, this is exploratory context rather than a
canonical 6/7 claim. The zero-embedder lexical track above is the controlled comparison. See
[`eval/results/full_pipeline_e5.json`](eval/results/full_pipeline_e5.json).

```bash
python eval/finreg_bench.py                        # finreg, self-contained
python eval/public_bench.py --synaptic-repo PATH   # the 8 public datasets
```

### Historical upstream-main efficiency — isolated workers and whole-process RSS

Accuracy and performance are deliberately separate. Accuracy uses the byte-identical
synaptic `BenchmarkResult`; latency uses direct `perf_counter` measurements. Each system
runs in a fresh process with the same K and candidate limit, deterministic query order,
one warm-up round, and five measured rounds. The report includes p50/p95/mean latency and
the whole worker's peak RSS, with MRR beside the timing result.

Selected 2026-07-13 upstream-main host measurements (`warmup=1`, `repeats=5`, K=10,
candidates=20):

| dataset | system | ingest s | p50 ms | p95 ms | peak RSS MB | MRR@10 |
|---|---|---:|---:|---:|---:|---:|
| PublicHealthQA | **OmniFuse** | **0.06** | **0.13** | **0.23** | **31.7** | **0.6133** |
| | synaptic | 5.52 | 4.20 | 5.27 | 599.8 | 0.6065 |
| NFCorpus | **OmniFuse** | **1.69** | **0.38** | **5.00** | **62.4** | **0.5175** |
| | synaptic | 50.52 | 16.92 | 31.04 | 97.3 | 0.5124 |
| Allganize RAG-ko | **OmniFuse** | **0.21** | **0.19** | **0.39** | **35.6** | **0.9683** |
| | synaptic | 5.80 | 4.57 | 6.19 | 599.9 | 0.9595 |

The ingest values are not capability-equivalent: OmniFuse builds a RAM retrieval index,
while synaptic builds a durable, disk-queryable SQLite graph. Storage is therefore reported
separately as **native artifact footprint**, with the same caveat; it is not described as a
general memory-usage comparison. On the two explicitly selected footprint fixtures,
OmniFuse writes 5.90 MB vs 20.76 MB on NFCorpus (**3.52× smaller**) and 0.31 MB vs
0.76 MB on Allganize (**2.45× smaller**). These two datasets are regression fixtures, not
a representative sample. Reproduce with
[`eval/perf_bench.py`](eval/perf_bench.py) and
[`eval/footprint_bench.py`](eval/footprint_bench.py). These schema-v1 efficiency artifacts
predate the stricter write-once/source-clean canonical contract and are retained as selected
measurements, not official-tag canonical evidence:
[`perf_publichealth_v027_20260713.json`](eval/results/perf_publichealth_v027_20260713.json),
[`perf_nfcorpus_v027_20260713.json`](eval/results/perf_nfcorpus_v027_20260713.json),
[`perf_allganize_v027_20260713.json`](eval/results/perf_allganize_v027_20260713.json), and
[`footprint_v027_20260713.json`](eval/results/footprint_v027_20260713.json).

## Memory — `Feedback`

The deepest difference between OmniFuse and synaptic-**memory** was never the ranking: it
is that synaptic is *stateful* and learns. Synaptic's own suite already tests reinforcement
and consolidation contracts; our narrower question was whether those updates improve
held-out `graph.search` retrieval. The first version told us we had won when we had not,
which is why the placebo controls are now part of the harness.

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
| synaptic (Hebbian) | +0.0000 | +0.0093 | −0.0010 | −0.0002 |
| **OmniFuse (`Feedback`)** | **+0.1843** | **+0.4016** | **+0.1133** | **+0.1785** |
| ↳ shuffled placebo | −0.0184 | −0.0217 | +0.0028 | +0.0044 |
| ↳ random-query placebo | +0.0263 | +0.0590 | +0.0002 | +0.0006 |

On covered KRA queries, `real` is 6.8× the strongest placebo; on NFCorpus it is about 40×,
so the `(query, chunk)` pairing is what carries the signal. On *unrelated* held-out questions
memory correctly does nothing (+0.0006) and
Δuncovered is exactly **0.0000** — the collection's IDF is provably untouched. A cold store
ranks **bit-identically** to one built with no feedback, so memory can never regress a
system that has not been used. There is no per-dataset runtime switch.

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

| | rebuild | `remember()` | `forget()` | per memory |
|---|---:|---:|---:|---:|
| NFCorpus (3,633 docs, 100 memories) | 1.337 s | **0.96 ms** | **0.90 ms** | **1,387x** |
| same memories, a tenth of the corpus | 0.125 s | **0.92 ms** | 1.04 ms | 136x |
| KRA (5,234 chunks, 120 memories) | 6.261 s | **1.51 ms** | **1.42 ms** | **4,145x** |

Both directions are **bit-identical** to a rebuild on all three corpora — remember with the
pair, forget without it.

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
