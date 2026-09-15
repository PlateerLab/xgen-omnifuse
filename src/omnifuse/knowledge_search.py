"""Run the existing OmniFuse algorithm against a portable knowledge view."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .fusion import minmax
from .knowledge import (
    ContractError,
    EmbeddingProfile,
    GraphFact,
    Resource,
    SourceChunk,
    canonical_json,
    validate_vector,
)
from .knowledge_protocols import (
    KnowledgeProvider,
    OperationContext,
    ReadScope,
    UnsupportedCapability,
)
from .models import Chunk, Node, SearchResult
from .oneshot import OmniFuse


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    resource_id: str
    revision: str
    extraction_id: str
    locator: dict
    evidence_ids: tuple[str, ...] = ()


@dataclass
class KnowledgeSearchResult:
    corpus_id: str
    snapshot_id: str
    result: SearchResult
    ranked_chunks: list[tuple[SourceChunk, float]]
    facts: list[GraphFact]
    citations: list[Citation]
    capabilities: tuple[str, ...]
    retrieval_mode: str
    coverage: dict = field(default_factory=dict)
    resource_hits: list[tuple[Resource, float]] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


class _GraphAdapter:
    """Project typed entities/facts onto the legacy algorithm without ID aliases."""
    def __init__(self, view):
        self.view, self.store = view, view.graph
        self.observed: dict[str, GraphFact] = {}
        self._entities = {}

    def _node(self, entity):
        self._entities[entity.id] = entity
        return Node(entity.id, entity.label, entity.kind)

    def search_labels(self, query, *, limit=30):
        self.view.check()
        if self.store is None:
            return []
        return [(self._node(e), s) for e, s in self.store.search_entities(query, limit=limit)]

    def get_node(self, node_id):
        self.view.check()
        if self.store is None:
            return None
        if node_id not in self._entities:
            for entity in self.store.get_entities([node_id]):
                self._node(entity)
        e = self._entities.get(node_id)
        return self._node(e) if e is not None else None

    def _facts(self, node_id, direction="both", limit=100):
        self.view.check()
        if self.store is None:
            return []
        facts = self.store.neighbors(node_id, direction=direction, limit=limit)
        self.observed.update((f.id, f) for f in facts)
        return facts

    def relation(self, fact):
        subject = self.get_node(fact.subject_id)
        target = self.get_node(fact.object_id) if fact.object_id is not None else None
        if subject is None or (fact.object_id is not None and target is None):
            raise ContractError("graph returned an unresolved endpoint")
        literal = fact.literal if isinstance(fact.literal, str) else canonical_json(fact.literal)
        return subject.label, fact.predicate, target.label if target is not None else literal

    def neighbors(self, node_id, *, hops=1, limit=100):
        seen, found, frontier = set(), {}, [node_id]
        for _ in range(hops):
            next_ids = []
            for nid in frontier:
                if nid in seen:
                    continue
                seen.add(nid)
                for f in self._facts(nid, limit=limit):
                    found[f.id] = f
                    next_ids.extend(x for x in (f.subject_id, f.object_id) if x is not None)
                    if len(found) >= limit:
                        return [self.relation(f) for f in found.values()]
            frontier = next_ids
        return [self.relation(f) for f in found.values()]

    def neighbor_ids(self, node_id, *, limit=100, direction="both"):
        ids = set()
        for f in self._facts(node_id, direction, limit):
            ids.update(x for x in (f.subject_id, f.object_id) if x is not None and x != node_id)
        return sorted(ids)[:limit]

    def seed_chunk_ids(self, entity_id, *, limit=100):
        self.view.check()
        return self.store.chunks_for_entities([entity_id], limit=limit) if self.store else []

    def neighbor_chunk_ids(self, chunk_id, *, limit=100, direction="out"):
        self.view.check()
        if self.store is None:
            return []
        entities = self.store.entities_for_chunks([chunk_id]).get(chunk_id, [])
        targets = set()
        for entity in entities:
            targets.update(self.neighbor_ids(entity, direction=direction, limit=limit))
        return [cid for cid in self.store.chunks_for_entities(sorted(targets), limit=limit + 1)
                if cid != chunk_id][:limit]

    def class_instances(self, class_id, *, limit=1000):
        self.view.check()
        if self.store is None:
            return []
        page = self.store.class_members(class_id, limit=min(limit, 1000))
        return [self._node(e) for e in page.items]

    def count_class(self, class_id):
        self.view.check()
        page = self.store.class_members(class_id, limit=1)
        if not page.exact:
            raise UnsupportedCapability("class enumeration requires an exact scoped count")
        return page.total

    def node_specificity(self, node_ids):
        self.view.check()
        if self.store is None:
            return {}
        counts = self.store.entity_chunk_counts(node_ids)
        return {eid: 1.0 / max(1, count) for eid, count in counts.items()}


class _VectorAdapter:
    def __init__(self, view, graph, *, mode, profile, encoder):
        self.view, self.graph = view, graph
        self.mode, self.profile, self.encoder = mode, profile, encoder
        self.ranked = []
        self.signals = {}

    def fetch(self, ids):
        self.view.check()
        sources = self.view.content.get_chunks(ids)
        linked = self.graph.store.entities_for_chunks([c.id for c in sources]) if self.graph.store else {}
        return [Chunk(c.id, c.text, linked.get(c.id, []), title=c.title,
                      meta={"resource_id": c.resource_id, "revision": c.revision,
                            "extraction_id": c.extraction_id, "locator": c.locator}) for c in sources]

    def search(self, query, *, limit=20):
        self.view.check()
        lexical, dense = [], []
        pool = max(40, limit)
        if self.mode in {"lexical", "hybrid"}:
            lexical = self.view.content.search_text(query, limit=pool)
        if self.mode in {"dense", "hybrid"}:
            values = self.encoder(query)
            self.view.check()
            validate_vector(values, self.profile)
            dense = self.view.embeddings.search_vector(list(values), profile_id=self.profile.id, limit=pool)
        if self.mode == "graph":
            for entity, score in self.graph.search_labels(query, limit=limit):
                dense.extend((cid, score) for cid in self.graph.seed_chunk_ids(entity.id, limit=limit))
        self.signals = {
            "lexical": lexical,
            "dense": dense if self.mode != "graph" else [],
            "graph": dense if self.mode == "graph" else [],
            "profile": asdict(self.profile) if self.profile is not None else None,
        }
        import math
        for hits in (lexical, dense):
            if any(not isinstance(cid, str) or type(score) not in (int, float) or not math.isfinite(score)
                   for cid, score in hits):
                raise ContractError("invalid retrieval score or ID")
            if len({i for i, _ in hits}) != len(hits) and self.mode != "graph":
                raise ContractError("provider returned duplicate ranked IDs")
        if self.mode == "hybrid":
            # Same min-max fusion policy and weights as the default InMemoryVector.
            ln, dn = minmax(lexical), minmax(dense)
            scores = {cid: 0.8 * ln.get(cid, 0.0) + dn.get(cid, 0.0) for cid in set(ln) | set(dn)}
        else:
            scores = {}
            for cid, score in lexical + dense:
                scores[cid] = max(scores.get(cid, float("-inf")), score)
        ids = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
        chunks = {c.id: c for c in self.fetch(ids)}
        if set(ids) != set(chunks):
            raise ContractError("retrieval hit cannot resolve in the same snapshot")
        self.ranked = [(chunks[cid], scores[cid]) for cid in ids]
        return self.ranked


class _GuardedLLM:
    def __init__(self, llm, view):
        self.llm, self.view = llm, view

    def generate(self, prompt, **kwargs):
        self.view.check()
        value = self.llm.generate(prompt, **kwargs)
        self.view.check()
        return value


class KnowledgeSearch:
    """Independent retrieval entry point for all three knowledge components.

    ``auto`` uses hybrid when embeddings exist, lexical otherwise, or graph for
    graph-only data. Embeddings require an explicitly matching encoder profile;
    use mode='lexical' to deliberately bypass them. No silent dense fallback.
    """
    def __init__(self, provider: KnowledgeProvider, *, mode="auto",
                 query_encoder: Callable[[str], list[float]] | None = None,
                 query_profile: EmbeddingProfile | None = None, llm=None, **options):
        if mode not in {"auto", "lexical", "dense", "hybrid", "graph", "hierarchy"}:
            raise ValueError("invalid retrieval mode")
        self.provider, self.mode = provider, mode
        self.query_encoder, self.query_profile = query_encoder, query_profile
        self.llm, self.options = llm, options

    def search(self, question: str, *, scope: ReadScope | None = None,
               snapshot_id: str | None = None, synthesize: bool = False,
               context: OperationContext | None = None) -> KnowledgeSearchResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        context = context or OperationContext()
        with self.provider.open_view(scope=scope or ReadScope(), snapshot_id=snapshot_id, context=context) as view:
            view.check()
            if snapshot_id is not None and view.snapshot_id != snapshot_id:
                raise ContractError("provider returned a different snapshot")
            for component in ("hierarchy", "embeddings", "graph"):
                if (getattr(view, component) is not None) != (component in view.capabilities):
                    raise ContractError(f"provider capability mismatch: {component}")
            mode = self.mode
            if mode == "auto":
                if view.embeddings:
                    mode = "hybrid" if "lexical" in view.capabilities else "dense"
                elif "lexical" in view.capabilities:
                    mode = "lexical"
                else:
                    mode = "graph" if view.graph else "hierarchy"
            if mode == "hierarchy":
                if view.hierarchy is None:
                    raise UnsupportedCapability("hierarchy retrieval requires hierarchy")
                hits = view.hierarchy.search_resources(question, limit=self.options.get("vector_k", 20))
                view.check()
                return KnowledgeSearchResult(view.corpus_id, view.snapshot_id,
                                             SearchResult(answer="", question=question), [], [], [],
                                             tuple(sorted(view.capabilities)), mode, getattr(view, "coverage", {}),
                                             resource_hits=hits)
            if mode in {"dense", "hybrid"}:
                if view.embeddings is None or self.query_encoder is None or self.query_profile is None:
                    raise UnsupportedCapability("dense retrieval requires embeddings and a matching query encoder/profile")
                profiles = {p.id: p for p in view.embeddings.profiles()}
                stored = profiles.get(self.query_profile.id)
                if stored is None or asdict(stored) != asdict(self.query_profile):
                    raise ContractError("query encoder profile does not match stored embedding space")
            if mode in {"lexical", "hybrid"} and "content" not in view.capabilities:
                raise UnsupportedCapability("lexical retrieval requires content")
            if mode == "graph" and view.graph is None:
                raise UnsupportedCapability("graph retrieval requires graph")
            graph = _GraphAdapter(view)
            vector = _VectorAdapter(view, graph, mode=mode, profile=self.query_profile, encoder=self.query_encoder)
            engine = OmniFuse(graph, vector, _GuardedLLM(self.llm, view) if self.llm else None, **self.options)
            # Record the exact fused ranking used by search(), avoiding a second
            # query/encoder call and preserving the same read view for all stages.
            ranked = []
            original_retrieve = engine.retrieve

            def retrieve(*args, **kwargs):
                hits = original_retrieve(*args, **kwargs)
                ranked.extend(hits)
                return hits

            engine.retrieve = retrieve
            result = engine.search(question, synthesize=synthesize)
            used_relations = set(result.relations)
            facts = [f for f in graph.observed.values() if " → ".join(graph.relation(f)) in used_relations]
            evidence_ids = sorted({eid for f in facts for eid in f.evidence_ids})
            evidence = view.content.get_evidence(evidence_ids)
            if {e.id for e in evidence} != set(evidence_ids):
                raise ContractError("fact evidence cannot resolve in the same snapshot")
            ids = list(dict.fromkeys([c.id for c, _ in ranked] + [e.chunk_id for e in evidence]))
            sources = {c.id: c for c in view.content.get_chunks(ids)}
            if set(ids) != sources.keys():
                raise ContractError("source citation cannot resolve in the same snapshot")
            citations = [Citation(c.id, c.resource_id, c.revision, c.extraction_id, c.locator,
                                  tuple(e.id for e in evidence if e.chunk_id == c.id)) for c in sources.values()]
            view.check()
            return KnowledgeSearchResult(view.corpus_id, view.snapshot_id, result,
                                         [(sources[c.id], s) for c, s in ranked], facts, citations,
                                         tuple(sorted(view.capabilities)), mode, getattr(view, "coverage", {}), signals=vector.signals)

    async def asearch(self, question: str, **kwargs) -> KnowledgeSearchResult:
        """Run synchronous providers outside the event loop; propagate cancellation.

        Cancellation signals the worker; a blocking third-party call must honor
        its own timeout. No result is returned after cancellation/revocation.
        """
        context = kwargs.pop("context", None) or OperationContext()
        try:
            return await asyncio.to_thread(self.search, question, context=context, **kwargs)
        except asyncio.CancelledError:
            context.cancelled.set()
            raise
