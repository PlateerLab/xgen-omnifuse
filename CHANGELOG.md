# Changelog

## 0.1.0 (unreleased)

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
