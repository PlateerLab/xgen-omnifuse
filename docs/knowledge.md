# Independent knowledge retrieval (knowledge/v1)

`xgen-omnifuse` is a retrieval library. It does not require Xgen, the graph builder,
PostgreSQL, Qdrant, an LLM, or an application server. `KnowledgeSearch` runs the
existing `OmniFuse` algorithm over these three independent inputs:

| Input | Interface | Operations |
| --- | --- | --- |
| File/directory hierarchy | `HierarchyStore` | resources, children, ancestors, descendants, literal path resolution, name/path search |
| Embeddings | `EmbeddingStore` | declared model profiles, query-vector search |
| Stored knowledge graph | `KnowledgeGraphStore` | entity search, typed facts, graph neighbors, class membership/counts, entity/chunk links |

`ContentStore` resolves shared source chunks and evidence, and supplies lexical
candidates. Graph entities and chunks have separate IDs. The graph adapter
projects entity relationships onto related passages explicitly, so graph fusion
works even when no entity ID equals a chunk ID.

## Small/offline input

```python
from omnifuse import KnowledgeSearch, MemoryKnowledgeProvider, ReadScope
from omnifuse.knowledge import KnowledgeBundle

bundle = KnowledgeBundle.load("knowledge.json")
provider = MemoryKnowledgeProvider(bundle)
search = KnowledgeSearch(provider, mode="lexical")
result = search.search("reporting obligations", scope=ReadScope(root_ids=("policies",)))
print(result.result.relations)
print(result.facts, result.citations)
```

The JSON file can be produced by **any** implementation of knowledge/v1. The
consumer's wheel includes the records, schema and digest lock; it never downloads
code or imports `xgen_ontology` at runtime. `examples/knowledge_search.py` is runnable.

## All three inputs, including real embeddings

```python
search = KnowledgeSearch(
    provider,
    query_encoder=my_encoder,             # str -> numeric vector
    query_profile=my_encoder_profile,     # EmbeddingProfile, not just a dimension
)
result = search.search("reporting obligations", scope=ReadScope(root_ids=("policies",)))
```

The profile identifies model, model revision, preprocessing, dimension and metric.
It must match the stored profile exactly. `auto` chooses hybrid for text plus
embeddings, dense for embeddings without text, lexical for text, graph when only
graph evidence is available, or hierarchy for directory/name navigation. Request
`lexical` explicitly to ignore available embeddings. Missing encoders and mismatched
profiles raise errors rather than silently switching algorithms.

The reference provider implements cosine, dot and Euclidean search. Returned dense
scores are higher-is-better (Euclidean distance is negated). Hybrid uses the same
min-max normalization and default weights as the existing in-memory OmniFuse;
`result.signals` preserves raw candidate scores and the profile. Fusion, graph
label seeds, class enumeration, HippoRAG expansion, MMR, adaptive cutting and optional
synthesis continue to execute inside OmniFuse.

`result` carries the original `SearchResult`, fused ranked source chunks, typed
facts, source/revision/extraction/locator citations, coverage, snapshot ID and actual
retrieval mode. `synthesize=False` is the default. An injected LLM is optional;
providers are checked before and after external synthesis. Hierarchy search returns
`resource_hits` and does not manufacture document content or an answer.

## Database or service adapters

Implement `KnowledgeProvider.open_view(scope, snapshot_id, context)` as a context
manager. It returns a `KnowledgeView` with the three stores, a content store,
`corpus_id`, `snapshot_id`, `capabilities`, and `check()`.

- Each store may be a different DB implementation. A view binds them to one coherent
  source snapshot. Store methods do not need to load a complete bundle.
- `capabilities` declares `hierarchy`, `content`, `embeddings`, `graph` as available;
  add `lexical` when text search is usable. Absent optional stores are `None`.
- Apply authorization and requested scope **before** top-k retrieval, traversal,
  statistics, and pagination. Filters can narrow authorized scope, never widen it.
- Exact class membership counts are required by the legacy class enumeration path.
  Return `Page(exact=False, ...)` if unsupported; that path fails explicitly rather
  than calling a truncated page the complete class.
- Cursor tokens bind to snapshot, scope and query. The reference pages cap at 1,000;
  use `next_cursor`. A search top-k limit does not mean a total record count.
- Embedding search returns unique `(chunk_id, higher_is_better_score)` pairs. Scores
  must be finite. Every hit/fact evidence must resolve inside the same view.
- `check()` enforces lifetime, cancellation, deadline and the host's current access
  policy. The provider owns authentication and transaction isolation. It is a trusted
  adapter, not a sandbox for arbitrary third-party code.
- Integrations should charge provider calls to the shared `OperationContext` and
  configure their transport timeouts. The reference provider does this itself.

The compositional adapter test in `tests/test_knowledge.py` supplies separate objects
for hierarchy, embeddings, content and graph. No inheritance or private Xgen table
schema is required. The in-memory reference materializes a bounded small corpus;
large repositories should implement these protocols using their own indexed stores.

## Updates, deletion and cancellation

`MemoryKnowledgeProvider.publish(new_bundle, expected_snapshot="old")` validates
all components before an atomic pointer replacement. It rejects stale writers,
snapshot rollback, changed content under an immutable chunk ID, and profile/evidence/
fact IDs reused for different meanings. Exact repeated publication is idempotent.
Source revision digests are checked when supplied; the host remains responsible for
binding a revision to the original immutable bytes.

New readers see the complete new snapshot; already-open readers finish on the old
one. Immediate deletion or permission revocation must invalidate existing views via
`authorize`, which is checked before returning results. The reference provider
retains identity tombstones in memory for its lifetime; durable hosts persist their
own revision/generation ledger and authorization policy.

Scoped graph views remove unsupported facts and redact hidden evidence IDs. Extracted
facts with alternative supporting sources survive if visible support remains.
Inferred facts conservatively require all declared supporting evidence to be visible.
Neither the search engine nor the reference provider decides how to rebuild a graph:
the host publishes a new coherent graph from the builder or another producer.

```python
from omnifuse import OperationContext

context = OperationContext(max_calls=1000)
result = await search.asearch("reporting obligations", context=context)
# From another task/thread: context.cancelled.set()
```

`asearch` moves synchronous providers off the event loop and signals cancellation
if the awaiting task is cancelled. It cannot forcibly terminate a blocking external
SDK; use transport timeouts. Cancellation/failure never returns an ordinary empty
successful result. No write or graph rebuild occurs during retrieval.

## Verification and compatibility

- Existing `OmniFuse`, GraphStore/VectorStore and persistence APIs remain available.
- The new contract deliberately rejects unsupported versions and required meanings.
  Custom optional metadata belongs in `extensions`, which survives JSON round trips.
- The neutral exchange uses checksummed JSON with atomic file replacement, never pickle.
  The checksum detects corruption; authenticity belongs to the transport/host.
- `tools/check_package_pair.py` verifies build and search in **different installed
  environments**, with the counterpart package absent from each. CI also runs this
  against the released producer version pinned in its workflow.
- Local JSON/in-memory adapters are reference implementations; this release does not
  add Xgen filestore endpoints, UI, migration, or the next filestore search node.
