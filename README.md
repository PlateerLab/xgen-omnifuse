# OmniFuse

**Backend-agnostic, one-shot GraphRAG.** Fire several retrieval strategies at once —
vector/lexical passages **+** graph label-linking **+** class enumeration **+** relation
expansion — and *fuse* them with MMR diversity into a single LLM synthesis. No
iterative ReAct tool loop. **Zero infra, zero lock-in:** the full algorithm runs on a
pure-Python in-memory backend (dict + BM25), and swaps to Fuseki / Qdrant / any LLM by
passing objects that match three small protocols.

```python
from omnifuse import build_inmemory, Node, Triple, Chunk

of = build_inmemory(nodes, triples, chunks)        # in-memory, no DB, no API key
print(of.search("불법도박 관련 규정").answer)
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
  character n-grams, so Korean/CJK search works with no morphological analyzer), and
  `InMemoryVector` uses cosine when embeddings are present, else BM25 lexical.
- **`dependencies = []`** — the core needs nothing but the standard library. Real backends
  are optional extras (`pip install "xgen-omnifuse[fuseki,qdrant]"`).
- **Bring your own LLM** — pass anything with `generate(...)`; the bundled `EchoLLM`
  returns the fused evidence so the pipeline runs end-to-end with no API key.

## The pipeline (`OmniFuse.search`)

1. vector/lexical seed → adaptive top-k (score-distribution cut, not fixed k)
2. graph label-linking → 1-hop relations
3. class enumeration (complete list/count)
4. HippoRAG — entities of the retrieved chunks → 1-hop expansion
5. evidence assembled with **MMR** diversity (Jaccard, no embeddings needed)
6. one LLM synthesis over the fused evidence
7. honest `evidence_nodes` — only the nodes the answer actually cites

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
  models.py        # Node, Triple, Chunk, SearchResult
  text.py          # tokenizer + BM25 (CJK n-grams)
  fusion.py        # MMR, adaptive top-k, relation ranking
  oneshot.py       # OmniFuse.search — the fusion algorithm
  backends/memory.py  # InMemoryGraph + InMemoryVector (zero infra)
  llm.py           # EchoLLM, CallableLLM
  facade.py        # build_inmemory(...)
examples/  tests/
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

## Roadmap

- `backends/qdrant.py` vector adapter; jena-text fast path for `FusekiGraph`
- async pipeline (parallel seeds via `asyncio.gather`)
- reranker / cross-encoder hook, query expansion
- configurable ISA predicates and prompt templates (per domain/language)

## License

TBD.
