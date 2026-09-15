"""All three inputs, source evidence, scope, mutation and isolated I/O contracts."""
import asyncio
import pathlib
import sys
from dataclasses import asdict, replace
from threading import Event
from time import monotonic

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import (
    KnowledgeSearch,
    MemoryKnowledgeProvider,
    OperationContext,
    ReadScope,
)
from omnifuse.knowledge import (
    BudgetExceeded,
    ChunkEmbedding,
    ContractError,
    EmbeddingProfile,
    Evidence,
    GraphEntity,
    GraphFact,
    KnowledgeBundle,
    OperationCancelled,
    Resource,
    SnapshotConflict,
    SourceChunk,
)
from omnifuse.knowledge_protocols import UnsupportedCapability


def bundle():
    return KnowledgeBundle(
        "library", "s1",
        resources=(Resource("root", "v1", "root", "directory"),
                   Resource("public", "v1", "public", "directory", "root"),
                   Resource("private", "v1", "private", "directory", "root"),
                   Resource("a", "v1", "policy.md", parent_id="public"),
                   Resource("b", "v1", "policy.md", parent_id="public"),
                   Resource("secret", "v1", "secret.md", parent_id="private")),
        chunks=(SourceChunk("c:a:1", "a", "v1", "p1", "violation reporting duty", title="Reporting"),
                SourceChunk("c:b:1", "b", "v1", "p1", "penalty limit is ten dollars", locator={"kind": "page", "page": 3}),
                SourceChunk("c:secret:1", "secret", "v1", "p1", "violation reporting secret password")),
        profiles=(EmbeddingProfile("dense-v1", "example-encoder", "revision-1", 2),),
        embeddings=(ChunkEmbedding("c:a:1", "dense-v1", (1.0, 0.1)),
                    ChunkEmbedding("c:b:1", "dense-v1", (0.2, 1.0)),
                    ChunkEmbedding("c:secret:1", "dense-v1", (1.0, 0.1))),
        entities=(GraphEntity("entity:a", "Reporting", evidence_ids=("ea",)),
                  GraphEntity("entity:b", "Penalty", evidence_ids=("eb",)),
                  GraphEntity("entity:s", "Secret", evidence_ids=("es",))),
        facts=(GraphFact("f:ab", "entity:a", "references", "entity:b", evidence_ids=("ea",)),
               GraphFact("f:bs", "entity:b", "classified_link", "entity:s", evidence_ids=("es",)),
               GraphFact("f:amount", "entity:b", "amount", literal=10, datatype="xsd:integer", evidence_ids=("eb",))),
        evidence=(Evidence("ea", "c:a:1"), Evidence("eb", "c:b:1"), Evidence("es", "c:secret:1")),
        components=("hierarchy", "content", "embeddings", "graph"),
    )


def test_three_components_run_real_fusion_and_resolve_evidence():
    data = bundle()
    seen = []
    def encode(query):
        seen.append(query)
        return [1.0, 0.1]
    engine = KnowledgeSearch(MemoryKnowledgeProvider(data), query_encoder=encode, query_profile=data.profiles[0])
    result = engine.search("violation reporting", scope=ReadScope(root_ids=("public",)))
    assert result.retrieval_mode == "hybrid" and len(seen) == 1
    assert {c.id for c, _ in result.ranked_chunks} == {"c:a:1", "c:b:1"}
    assert "Reporting → references → Penalty" in result.result.relations
    assert "Penalty → amount → 10" in result.result.relations
    assert all(c.resource_id != "secret" for c in result.citations)
    assert not any("Secret" in r for r in result.result.relations)
    assert next(c for c in result.citations if c.resource_id == "b").locator == {"kind": "page", "page": 3}
    assert result.snapshot_id == "s1" and result.result.answer == ""
    assert result.coverage["embedding_chunks"] == {"dense-v1": 2}


def test_entity_chunk_ids_are_distinct_and_graph_expansion_still_works():
    data = bundle()
    engine = KnowledgeSearch(MemoryKnowledgeProvider(data), mode="lexical")
    on = engine.search("violation reporting", scope=ReadScope(resource_ids=("a", "b")))
    off = KnowledgeSearch(MemoryKnowledgeProvider(data), mode="lexical", graph_fusion=False).search(
        "violation reporting", scope=ReadScope(resource_ids=("a", "b")))
    assert "c:b:1" in {c.id for c, _ in on.ranked_chunks}
    assert "c:b:1" not in {c.id for c, _ in off.ranked_chunks}


def test_query_encoder_is_explicit_and_profile_is_more_than_dimension():
    data = bundle()
    with pytest.raises(UnsupportedCapability):
        KnowledgeSearch(MemoryKnowledgeProvider(data)).search("query")
    with pytest.raises(ContractError, match="profile"):
        KnowledgeSearch(MemoryKnowledgeProvider(data), query_encoder=lambda q: [1, 0],
                        query_profile=replace(data.profiles[0], model_revision="wrong")).search("query")
    for values in ([1], [float("nan"), 0], [0, 0]):
        with pytest.raises(ContractError):
            KnowledgeSearch(MemoryKnowledgeProvider(data), query_encoder=lambda q, values=values: values,
                            query_profile=data.profiles[0]).search("query")


def test_hierarchy_pagination_scope_and_cursor_binding():
    provider = MemoryKnowledgeProvider(bundle())
    with provider.open_view(scope=ReadScope(root_ids=("public",))) as view:
        page = view.hierarchy.children("public", limit=1)
        assert page.total == 2 and page.next_cursor
        second = view.hierarchy.children("public", cursor=page.next_cursor, limit=1)
        assert second.items[0].id != page.items[0].id and second.next_cursor is None
        assert not view.hierarchy.get_resources(["secret"])
        assert [r.id for r in view.hierarchy.ancestors("a")] == ["public"]
        assert len(view.hierarchy.descendants("public").items) == 2
        cursor = page.next_cursor
    with provider.open_view() as view, pytest.raises(ContractError, match="cursor"):
        view.hierarchy.children("public", cursor=cursor, limit=1)
    with pytest.raises(RuntimeError, match="closed"):
        view.content.get_chunks(["c:a:1"])


def test_empty_scope_never_means_all_and_allowlist_cannot_be_widened():
    engine = KnowledgeSearch(MemoryKnowledgeProvider(bundle(), allowed_resource_ids=frozenset({"a"})), mode="lexical")
    assert not engine.search("violation", scope=ReadScope(resource_ids=())).ranked_chunks
    result = engine.search("violation", scope=ReadScope(root_ids=("root",)))
    assert {c.resource_id for c in result.citations} == {"a"}
    assert not any("classified" in r for r in result.result.relations)


def test_independent_lexical_graph_and_hierarchy_inputs():
    data = bundle()
    lexical = replace(data, profiles=(), embeddings=(), entities=(), facts=(), evidence=(), components=("content", "hierarchy"))
    assert KnowledgeSearch(MemoryKnowledgeProvider(lexical)).search("violation").retrieval_mode == "lexical"
    graph = replace(data, profiles=(), embeddings=(), components=("content", "hierarchy", "graph"))
    result = KnowledgeSearch(MemoryKnowledgeProvider(graph), mode="graph").search("Reporting")
    assert result.facts and result.citations
    hierarchy = replace(data, chunks=(), profiles=(), embeddings=(), entities=(), facts=(), evidence=(), components=("hierarchy",))
    with MemoryKnowledgeProvider(hierarchy).open_view() as view:
        assert view.hierarchy.children("root").total == 2
    found = KnowledgeSearch(MemoryKnowledgeProvider(hierarchy)).search("policy")
    assert found.retrieval_mode == "hierarchy"
    assert {r.id for r, _ in found.resource_hits} == {"a", "b"}


def test_snapshot_publish_is_atomic_idempotent_and_readers_are_pinned():
    original = bundle()
    provider = MemoryKnowledgeProvider(original)
    changed = replace(original, snapshot_id="s2", resources=tuple(r for r in original.resources if r.id != "secret"),
                      chunks=tuple(c for c in original.chunks if c.resource_id != "secret"),
                      embeddings=tuple(e for e in original.embeddings if e.chunk_id != "c:secret:1"),
                      entities=tuple(e for e in original.entities if e.id != "entity:s"),
                      facts=tuple(f for f in original.facts if f.id != "f:bs"),
                      evidence=tuple(e for e in original.evidence if e.id != "es"))
    with provider.open_view() as old:
        provider.publish(changed, expected_snapshot="s1")
        assert old.snapshot_id == "s1"
        assert old.content.get_chunks(["c:secret:1"])
        with provider.open_view() as new:
            assert new.snapshot_id == "s2" and not new.content.get_chunks(["c:secret:1"])
    provider.publish(changed, expected_snapshot="s1")  # exact retry
    with pytest.raises(SnapshotConflict):
        provider.publish(replace(changed, snapshot_id="s3"), expected_snapshot="s1")
    with pytest.raises(SnapshotConflict):
        provider.publish(original, expected_snapshot="s2")
    with pytest.raises(SnapshotConflict), provider.open_view(snapshot_id="s1"):
        pass
    assert provider.snapshot_id == "s2"


def test_invalid_batch_and_id_reuse_do_not_partially_publish():
    data = bundle()
    provider = MemoryKnowledgeProvider(data)
    changed = replace(data, snapshot_id="s2", chunks=(replace(data.chunks[0], text="different"),) + data.chunks[1:])
    with pytest.raises(ContractError, match="chunk ID reused"):
        provider.publish(changed, expected_snapshot="s1")
    assert provider.snapshot_id == "s1"
    with pytest.raises(ContractError):
        provider.publish(replace(data, snapshot_id="s2", chunks=()), expected_snapshot="s1")
    assert provider.snapshot_id == "s1"


def test_provider_owns_data_and_returns_copies():
    data = bundle()
    provider = MemoryKnowledgeProvider(data)
    data.chunks[1].locator["page"] = 100
    with provider.open_view() as view:
        chunk = view.content.get_chunks(["c:b:1"])[0]
        assert chunk.locator["page"] == 3
        chunk.locator["page"] = 200
        assert view.content.get_chunks(["c:b:1"])[0].locator["page"] == 3


def test_shared_fact_survives_one_evidence_removal_without_hidden_support_ids():
    data = bundle()
    shared = replace(data.facts[0], evidence_ids=("ea", "es"))
    data = replace(data, facts=(shared,) + data.facts[1:])
    with MemoryKnowledgeProvider(data).open_view(scope=ReadScope(root_ids=("public",))) as view:
        fact = next(f for f in view.graph.neighbors("entity:a") if f.id == "f:ab")
        assert fact.evidence_ids == ("ea",)
        assert not view.content.get_evidence(["es"])
    # Inference depending on two sources requires BOTH within the view.
    data = replace(data, facts=(replace(shared, assertion="inferred"),) + data.facts[1:])
    with MemoryKnowledgeProvider(data).open_view(scope=ReadScope(root_ids=("public",))) as view:
        assert not view.graph.neighbors("entity:a")


def test_cancel_timeout_budget_and_access_revocation_never_return_partial_success():
    provider = MemoryKnowledgeProvider(bundle())
    context = OperationContext()
    context.cancelled.set()
    with pytest.raises(OperationCancelled):
        KnowledgeSearch(provider, mode="lexical").search("query", context=context)
    with pytest.raises(TimeoutError):
        KnowledgeSearch(provider, mode="lexical").search("query", context=OperationContext(deadline=monotonic() - 1))
    with pytest.raises(BudgetExceeded):
        KnowledgeSearch(provider, mode="lexical").search("query", context=OperationContext(max_calls=1))
    granted = [True]
    provider = MemoryKnowledgeProvider(bundle(), authorize=lambda: granted[0])
    def encode(query):
        granted[0] = False
        return [1, 0]
    with pytest.raises(PermissionError):
        KnowledgeSearch(provider, query_encoder=encode, query_profile=bundle().profiles[0]).search("query")


def test_async_entrypoint_and_worker_cancellation():
    async def run():
        result = await KnowledgeSearch(MemoryKnowledgeProvider(bundle()), mode="lexical").asearch("Reporting")
        assert result.citations
        entered, release = Event(), Event()
        context = OperationContext()
        def encode(query):
            entered.set()
            release.wait(timeout=2)
            return [1, 0]
        engine = KnowledgeSearch(MemoryKnowledgeProvider(bundle()), query_encoder=encode, query_profile=bundle().profiles[0])
        task = asyncio.create_task(engine.asearch("query", context=context))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert context.cancelled.is_set()
        finally:
            release.set()
    asyncio.run(run())


@pytest.mark.parametrize("mutate", [
    lambda b: replace(b, resources=b.resources + (b.resources[0],)),
    lambda b: replace(b, resources=(replace(b.resources[0], parent_id="public"),) + b.resources[1:]),
    lambda b: replace(b, chunks=(replace(b.chunks[0], revision="v2"),) + b.chunks[1:]),
    lambda b: replace(b, embeddings=(replace(b.embeddings[0], values=(float("nan"), 1)),)),
    lambda b: replace(b, facts=(replace(b.facts[0], evidence_ids=("missing",)),)),
    lambda b: replace(b, facts=(replace(b.facts[0], literal="invalid"),)),
    lambda b: replace(b, contract_version="999.0"),
    lambda b: replace(b, chunks=(replace(b.chunks[0], locator={"kind": "page", "page": 0}),)),
])
def test_invalid_contract_is_rejected(mutate):
    with pytest.raises(ContractError):
        mutate(bundle()).validate()


def test_safe_roundtrip_and_checksums(tmp_path):
    data = bundle()
    data.extensions["example.extra"] = {"unicode": "한글", "nested": [1, False, None]}
    path = tmp_path / "bundle.json"
    data.dump(path)
    loaded = KnowledgeBundle.load(path)
    assert loaded.to_dict() == data.to_dict()
    assert asdict(loaded.facts[-1])["literal"] == 10
    path.write_text(path.read_text().replace("ten dollars", "one dollar"))
    with pytest.raises(ContractError, match="digest"):
        KnowledgeBundle.load(path)


def test_profile_and_evidence_ids_cannot_be_reused_with_new_meaning():
    data = bundle()
    provider = MemoryKnowledgeProvider(data)
    for candidate in (
        replace(data, snapshot_id="s2", profiles=(replace(data.profiles[0], model_revision="new"),)),
        replace(data, snapshot_id="s2", evidence=(replace(data.evidence[0], chunk_id="c:b:1"),) + data.evidence[1:]),
        replace(data, snapshot_id="s2", facts=(replace(data.facts[0], predicate="different"),) + data.facts[1:]),
    ):
        with pytest.raises(ContractError, match="ID reused"):
            provider.publish(candidate, expected_snapshot="s1")
        assert provider.snapshot_id == "s1"


def test_directory_paths_do_not_guess_or_cross_scope():
    with MemoryKnowledgeProvider(bundle()).open_view() as view:
        assert [r.id for r in view.hierarchy.resolve_path(("root", "public", "policy.md"))] == ["a", "b"]
        assert view.hierarchy.resolve_path(("root", "missing")) == []
    with MemoryKnowledgeProvider(bundle()).open_view(scope=ReadScope(root_ids=("public",))) as view:
        assert view.hierarchy.resolve_path(("root", "private", "secret.md")) == []
    with pytest.raises(ValueError):
        ReadScope(resource_ids="secret")


def test_exact_class_count_is_not_the_first_thousand_rows():
    total = 1205
    data = KnowledgeBundle(
        "large", "s1", resources=(Resource("r", "v1", "data.txt"),),
        chunks=(SourceChunk("c", "r", "v1", "p", "list Catalog"),),
        entities=(GraphEntity("class", "Catalog", "class", ("e",)),) + tuple(
            GraphEntity(f"item:{i}", f"Item {i}", evidence_ids=("e",)) for i in range(total)),
        facts=tuple(GraphFact(f"f:{i}", f"item:{i}", "instanceOf", "class", evidence_ids=("e",)) for i in range(total)),
        evidence=(Evidence("e", "c"),), components=("content", "graph"),
    )
    result = KnowledgeSearch(MemoryKnowledgeProvider(data), mode="graph").search(
        "Catalog", context=OperationContext(max_calls=10000))
    assert f"has {total} instances" in result.result.class_seed
    assert "+1055 more" in result.result.class_seed


def test_dense_only_metric_contract_and_signals():
    data = bundle()
    for metric in ("cosine", "dot", "euclidean"):
        profile = replace(data.profiles[0], metric=metric)
        current = replace(data, profiles=(profile,))
        result = KnowledgeSearch(MemoryKnowledgeProvider(current), mode="dense",
                                 query_encoder=lambda q: [0.2, 1.0], query_profile=profile,
                                 graph_fusion=False).search("semantic query")
        assert result.ranked_chunks[0][0].id == "c:b:1"
        assert result.signals["profile"]["metric"] == metric


def test_external_provider_composes_three_distinct_backends():
    from contextlib import contextmanager
    from types import SimpleNamespace

    reference = MemoryKnowledgeProvider(bundle())
    calls = []
    class ForwardingBackend:
        def __init__(self, inner, name):
            self.inner, self.name = inner, name
        def __getattr__(self, method):
            def call(*args, **kwargs):
                calls.append((self.name, method))
                return getattr(self.inner, method)(*args, **kwargs)
            return call
    class ExternalProvider:
        @contextmanager
        def open_view(self, **kwargs):
            with reference.open_view(**kwargs) as view:
                yield SimpleNamespace(
                    corpus_id=view.corpus_id, snapshot_id=view.snapshot_id, capabilities=view.capabilities,
                    content=ForwardingBackend(view.content, "content"),
                    hierarchy=ForwardingBackend(view.hierarchy, "hierarchy"),
                    embeddings=ForwardingBackend(view.embeddings, "embeddings"),
                    graph=ForwardingBackend(view.graph, "graph"), check=view.check,
                )
    result = KnowledgeSearch(ExternalProvider(), query_encoder=lambda q: [1.0, 0.1],
                             query_profile=bundle().profiles[0]).search("Reporting", scope=ReadScope(root_ids=("public",)))
    assert result.citations and result.facts
    assert {name for name, _ in calls} >= {"content", "embeddings", "graph"}
    # Hierarchy was used by the provider to bind scope before the query, and is
    # independently available for directory navigation.
    paths = KnowledgeSearch(ExternalProvider(), mode="hierarchy").search("policy")
    assert paths.resource_hits and ("hierarchy", "search_resources") in calls


def test_finite_extreme_cosine_vectors_do_not_overflow():
    data = bundle()
    data = replace(data, embeddings=tuple(replace(e, values=(1e308, 1e308)) for e in data.embeddings))
    with MemoryKnowledgeProvider(data).open_view() as view:
        hits = view.embeddings.search_vector([1e-300, 1e-300], profile_id="dense-v1", limit=3)
        assert len(hits) == 3 and all(score == pytest.approx(1.0) for _, score in hits)
    with pytest.raises(ContractError):
        replace(data, embeddings=(replace(data.embeddings[0], values=(10 ** 400, 1)),)).validate()


def test_graph_only_record_sources_need_no_document_text_or_encoder():
    data = bundle()
    data = replace(data, chunks=tuple(replace(c, text="", title="") for c in data.chunks),
                   profiles=(), embeddings=(), components=("content", "hierarchy", "graph"))
    result = KnowledgeSearch(MemoryKnowledgeProvider(data)).search("Reporting")
    assert result.retrieval_mode == "graph" and result.facts and result.citations
