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
  facade.py        # build_inmemory(...)
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

**Zero-dependency / lexical track** — every synaptic-shipped dataset:

| dataset | synaptic (FTS) | **OmniFuse** | winner |
|---|---:|---:|---|
| **finreg** single-hop | 0.7039 | **0.8401** | OmniFuse |
| **finreg** multi-hop (strict/120) | 56 | **103** | OmniFuse |
| HotPotQA-24 | 0.8879 | **0.9077** | OmniFuse |
| HotPotQA-200 | 0.8775 | **0.8908** | OmniFuse |
| Allganize RAG-ko | 0.9562 | **0.9679** | OmniFuse |
| Allganize RAG-Eval | 0.9303 | **0.9319** | OmniFuse |
| KLUE-MRC | 0.7718 | **0.8192** | OmniFuse |
| PublicHealthQA | 0.6065 | 0.6065 | tie |
| AutoRAG | **0.9053** | 0.8924 | synaptic |
| Ko-StrategyQA | **0.6440** | 0.6242 | synaptic |
| **average MRR** | 0.809 | **0.829** | **OmniFuse** |

**7 wins, 1 tie, 2 losses** — zero deps (no morphological analyzer) vs synaptic's Kiwi.
The finreg multi-hop **103/120 is one-shot, no LLM** — beating synaptic's own 5-turn
LLM agent (88/120) via graph-companion fusion following `제N조` citations.

**Extended coverage** — synaptic's download-only BEIR/MTEB sets (fetched from HF), lexical:

| dataset | synaptic (FTS) | **OmniFuse** | winner |
|---|---:|---:|---|
| SciFact (EN) | 0.6317 | **0.6368** | OmniFuse |
| XPQA-ko | 0.3115 | **0.3278** | OmniFuse |
| NFCorpus (EN) | **0.5124** | 0.5075 | synaptic |
| MIRACL-retrieval-ko | **0.9495** | 0.9293 | synaptic |
| FiQA (EN, 57k docs) | n/a¹ | 0.2871 | — |
| MultiLongDoc-ko (193 MB) | n/a¹ | 0.3286² | — |

BM25-family **parity** (2–2) on unstructured passage IR — no titles/citation graph for
field weighting or graph fusion to exploit. OmniFuse's decisive wins are on *structured*
corpora. ¹synaptic ingest time/RAM-bound on a 16 GB box; ²omni in-memory index capped
long-doc text (zero-infra RAM bound). Numbers:
[`eval/results/beir_mteb_extra.json`](eval/results/beir_mteb_extra.json).

**Full-pipeline track** (shared `multilingual-e5-small` embedder, both sides): OmniFuse's
dense+lexical hybrid **flips its two lexical losses (AutoRAG, Ko-StrategyQA) to wins**
and leads the fused-vs-fused comparison **6/7** (only PublicHealthQA to synaptic, and
that is embedder-dependent — OmniFuse wins it with bge-m3). See
[`eval/results/full_pipeline_e5.json`](eval/results/full_pipeline_e5.json).

```bash
python eval/finreg_bench.py                        # finreg, self-contained
python eval/public_bench.py --synaptic-repo PATH   # the 8 public datasets
```

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
