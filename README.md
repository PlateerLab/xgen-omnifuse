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

## Repository model

[`jinsoo96/js-omnifuse`](https://github.com/jinsoo96/js-omnifuse) is the personal source of
truth. [`PlateerLab/xgen-omnifuse`](https://github.com/PlateerLab/xgen-omnifuse) is the
organization mirror and keeps the published Python package name `xgen-omnifuse`.

Changes land on `js-omnifuse:main` first. The organization repository runs
[`sync-from-js-omnifuse.yml`](.github/workflows/sync-from-js-omnifuse.yml) every 15 minutes
and on manual dispatch. It accepts only a fast-forward from the personal source; it never
force-pushes or silently overwrites an independent organization commit.

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
  cosine and lexical BM25 min-max normalized per query, then weighted) when embeddings *and*
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
  lexical_rerank.py # bounded phrase/surface reranking + Korean zero-hit fallback
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

## How the current ranking works

The current retrieval path is one general algorithm. It contains no benchmark name, query id,
qrel, expected answer or document-specific exception.

1. **Lossless index, query-only cleanup.** Documents keep every token. At query time,
   Korean request endings and English closed-class grammar words are removed so subject
   terms carry the rank. If cleanup would remove everything, OmniFuse falls back to the
   original query tokens.
2. **BM25F candidate admission.** Title and body fields produce a configured bounded
   frontier (`max(limit, pool)`, default pool 40); title matches remain more informative
   than the same token buried in a long passage.
3. **Complete-word coordination.** When Korean word-boundary evidence exists, candidates
   containing the complete subject word are preferred over substring-only matches. At most
   one excluded candidate is restored, and only when its original BM25F score exceeds the
   weakest retained complete-word hit. The final top-K naturally decides whether it stays.
4. **Korean zero-hit recovery.** If ordinary lexical search returns no candidate, a
   character-evidence BM25 fallback scans that query's corpus once. It shares the configured
   BM25 constants, preserves mutable slot IDs, adds no persistent character index, and never
   runs on a normal hit path.
5. **Phrase evidence.** Ordered query bigrams vote inside the bounded frontier. Only matched
   query pairs are retained for each candidate.
6. **Personal-memory surface fusion.** First-person memory questions also receive an
   independent raw phrase/coverage ranking. Normalized lexical rank and surface rank are
   combined as `0.4 / lexical_rank + 0.6 / surface_rank`, an inverse-rank fusion inspired by
   [Reciprocal Rank Fusion](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/).
   It only reorders the already admitted candidate frontier and never scans the corpus.
7. **Question title anchors.** In graph mode, exact title mentions and conservative one-edit
   multi-token typos become graph seeds. A single-token alias is accepted only when it is
   the unambiguous base of a parenthetically disambiguated title. Earlier mentions receive
   slightly higher priority, then the usual directed companion expansion runs.
8. **Lazy owned memory.** Feedback is copied on construction and BM25F evidence materializes
   on first use. `remember()` and `forget()` remain exact incremental updates, while an
   unused store does not pay eager indexing cost.

This separates complementary signals instead of mixing incomparable raw score scales. The
candidate bound, generic linguistic classes and title-edit rule are product contracts, not
per-dataset switches.

The candidate-local policy lives in `omnifuse.lexical_rerank`, separate from index storage
and persistence. Its thresholds and weights are named once rather than scattered through
backend branches, and `pool` is the only configured candidate floor—there is no hidden
minimum that overrides a caller's smaller pool.

The complete iteration history, rejected variants and immutable artifacts are kept outside
this product overview in [`eval/README.md`](eval/README.md) and [`eval/results`](eval/results).

## Benchmark summary

The current canonical comparison uses the official synaptic-memory `v0.27.0` tag at
`836d53640e520c88910dd57e098167a4defe50d2`. Comparable retrieval tracks use the same
corpus, queries, relevance judgments, K and byte-identical scorer. Each artifact binds the
source, inputs, Python environment, isolated workers and postflight state.

| track | OmniFuse | synaptic-memory | verdict |
|---|---:|---:|---|
| Direct14, 2,269 queries | MRR@10 **0.7049**, nDCG@10 **0.6783** | MRR@10 0.6537, nDCG@10 0.6100 | five metrics **14/0/0**, Recall **13/0/1** by dataset |
| HotPotQA retrieval, 24 questions | Recall **0.9792**, nDCG **0.9483**, mean **3.83 ms** | Recall 0.7292, nDCG 0.6908, mean 60.43 ms | **11/0** aggregate; zero per-question quality losses |
| LongMemEval-S, 48 questions | MRR **0.8392**, nDCG **0.8643**, mean **60.22 ms** | MRR 0.6990, nDCG 0.6898, mean 241.61 ms | **8/0** quality and efficiency aggregate |
| Enterprise full-native | MRR **0.7689**, nDCG **0.7637**, mean **0.791 ms** | MRR 0.7467, nDCG 0.6649, mean 5.799 ms | quality, search and build win |
| local-Qwen answer E2E | correctness **0.7938**, mean generation **6,323 ms** | correctness 0.5450, mean generation 6,752 ms | **9/1** aggregate; stochastic p95 generation loses |

The scope is precise:

- Direct14 has no dataset-level loss, but 301/2,269 queries have at least one lower quality
  metric. Synaptic already reaches a metric ceiling on many queries, so “all datasets win”
  is not rewritten as “every query wins.”
- HotPotQA and LongMemEval have zero individual quality losses. LongMemEval still records
  five local efficiency losses even though all aggregate efficiency metrics win.
- Enterprise `full_native` preserves each product's native graph and memory semantics.
  The asymmetric `docs_only` diagnostic is excluded from the equal-input verdict.
- Answer generation is stochastic and is reported separately from deterministic retrieval.

Full protocols, every historical run, rejected variants and reproduction commands are in
[`eval/README.md`](eval/README.md). Machine-readable, write-once evidence is in
[`eval/results`](eval/results); the current artifacts are:

- [Direct14 v34](eval/results/direct_external14_synaptic_tag_v0.27.0_836d536_20260810_v34.json)
- [HotPotQA retrieval v34](eval/results/e2e_qa_retrieval_synaptic_tag_v0.27.0_836d536_20260810_v34.json)
- [LongMemEval v34](eval/results/longmemeval_retrieval_synaptic_tag_v0.27.0_836d536_20260810_v34.json)
- [Enterprise v34](eval/results/enterprise_synaptic_tag_v0.27.0_836d536_20260810_v34.json)
- [local-Qwen answer v26](eval/results/e2e_qa_answer_synaptic_tag_v0.27.0_836d536_qwen3.5_4b_20260809_v26.json)

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

The evidence field scores only the chunk that owns it. It does not enter document frequency
or length normalization, so a cold feedback-enabled store ranks identically to a store with
no feedback and unrelated chunks keep their content IDF.

### Learning without a rebuild

`remember(query, doc_ids)` and `forget(query, doc_ids)` update only the affected evidence
postings. Both directions are bit-identical to a full rebuild; the update cost follows the
changed memory rather than the corpus size.

```python
of = build_inmemory(nodes, triples, chunks, feedback=Feedback())
of.remember("statin side effects", ["doc7"])
of.forget("statin side effects", ["doc7"])
```

Held-out paraphrase, shuffled/random placebo, consolidation and incremental-cost experiments
are archived with their controls and retraction history in
[`eval/README.md`](eval/README.md), [`adaptive_memory.json`](eval/results/adaptive_memory.json)
and [`incremental_memory.json`](eval/results/incremental_memory.json).

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
