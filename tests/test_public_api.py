"""0.6.0 public extension points: prompt builder, synthesize=False, specificity,
graph_rank helpers and the preloaded vector store."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import (  # noqa: E402
    Chunk,
    InMemoryGraph,
    Node,
    OmniFuse,
    PreloadedVectorStore,
    Triple,
    blend_ppr,
    graph_candidate_chunks,
    minmax,
    ppr_chunk_scores,
    ppr_seeds,
    specificity_rerank,
)
from omnifuse.backends.memory import _minmax  # noqa: E402
from omnifuse.oneshot import PromptBuilder  # noqa: E402

NODES = [
    Node("c_reg", "규정", kind="class"),
    Node("n_audit", "감사규정", kind="instance"),
    Node("n_loan", "여신규정", kind="instance"),
    Node("n_hub", "회사", kind="instance"),
]
TRIPLES = [
    Triple("n_audit", "instanceOf", "c_reg"),
    Triple("n_loan", "instanceOf", "c_reg"),
    Triple("n_audit", "관련규정", "n_loan"),
    Triple("n_hub", "소속", "n_audit"),
    Triple("n_hub", "소속", "n_loan"),
]
CHUNKS = [
    Chunk("k1", "감사규정은 내부감사의 절차를 정한다", entities=["n_audit", "n_hub"]),
    Chunk("k2", "여신규정은 대출 한도를 정한다", entities=["n_loan", "n_hub"]),
    Chunk("k3", "회사 소개 자료", entities=["n_hub"]),
    Chunk("k4", "무관한 문단"),
]
DENSE = {"k1": 0.9, "k2": 0.8, "k3": 0.7, "k4": 0.2}


class RankGraph(InMemoryGraph):
    """In-memory graph plus the optional GraphRankStore methods."""

    _mentions = {"n_audit": ["k1"], "n_loan": ["k2"], "n_hub": ["k1", "k2", "k3"], "c_reg": []}

    def node_specificity(self, ids):
        return {u: 1.0 / len(self._mentions[u]) for u in ids if self._mentions.get(u)}

    def entities_of_chunks(self, chunk_ids, limit=400):
        out = {}
        for u, cks in self._mentions.items():
            for ck in cks:
                if ck in chunk_ids:
                    out.setdefault(ck, []).append(u)
        return out

    def chunks_of_entities(self, ids, limit=1000):
        return [(ck, u) for u in ids for ck in self._mentions.get(u, [])]

    def chunk_coverage(self, ids):
        return {u: len(self._mentions.get(u, [])) / 4.0 for u in ids}

    def all_edges(self, limit=80000):
        return [(t.s, t.o) for t in TRIPLES][:limit]


def _fuse(**kw):
    graph = RankGraph(NODES, TRIPLES)
    store = PreloadedVectorStore(CHUNKS, DENSE)
    return OmniFuse(graph, store, vector_k=4, sem_min=1, graph_fusion=False, **kw)


def test_prompt_builder_replaces_default_prompt():
    seen = {}

    def build(question, evidence, relations, class_seed):
        seen["args"] = (question, list(evidence), list(relations), class_seed)
        return f"Q={question}|E={len(evidence)}"

    res = _fuse(prompt_builder=build).search("감사규정 절차", synthesize=False)
    assert res.prompt.startswith("Q=감사규정 절차|E=")
    assert seen["args"][0] == "감사규정 절차"
    assert seen["args"][1] == res.evidence
    assert res.answer == ""


def test_synthesize_false_skips_llm_and_exposes_prompt_and_evidence():
    calls = []

    class Spy:
        def generate(self, prompt, *, system="", timeout=None):
            calls.append(prompt)
            return "answer"

    fuse = _fuse()
    fuse.llm = Spy()
    res = fuse.search("여신규정 한도", synthesize=False)
    assert calls == []
    assert res.answer == "" and res.evidence_nodes == []
    assert res.prompt == OmniFuse.build_prompt("여신규정 한도", res.evidence, res.relations, res.class_seed)
    assert res.system == fuse.system_prompt
    assert res.evidence and res.evidence[0].startswith("여신규정")
    res2 = fuse.search("여신규정 한도")
    assert calls and res2.answer == "answer"


def test_old_private_names_still_resolve():
    assert OmniFuse._prompt is OmniFuse.build_prompt
    assert OmniFuse._cited_nodes is OmniFuse.cited_nodes
    assert _minmax is minmax
    assert PromptBuilder is not None


def test_specificity_weight_demotes_hub_only_chunks():
    plain = _fuse().retrieve("규정", limit=4)
    weighted = _fuse(specificity_weight=0.5).retrieve("규정", limit=4)
    p = {c.id: s for c, s in plain}
    w = {c.id: s for c, s in weighted}
    # k3 mentions only the hub (specificity 1/3) -> loses more than k1 (has n_audit, 1.0)
    assert w["k3"] / p["k3"] < w["k1"] / p["k1"]
    assert w["k1"] == p["k1"]  # max specificity 1.0 keeps the score
    assert w["k4"] == p["k4"]  # no entities -> untouched
    assert [c.id for c, _ in weighted] == sorted(w, key=lambda k: -w[k])


def test_specificity_rerank_without_method_is_identity():
    hits = [(CHUNKS[0], 0.9), (CHUNKS[1], 0.8)]
    assert specificity_rerank(hits, object(), weight=0.5) is hits


def test_graph_candidate_chunks_prefers_rare_entities_and_excludes_hubs():
    graph = RankGraph(NODES, TRIPLES)
    order, score = graph_candidate_chunks(graph, "감사규정", max_coverage=0.5)
    assert order and order[0] == "k1"
    # the hub (coverage 0.75) is never a seed; its chunk k3 only appears via neighbours, if at all
    assert score.get("k3", 0.0) <= score["k1"]


def test_ppr_scores_reach_neighbour_chunks_and_blend_in_place():
    graph = RankGraph(NODES, TRIPLES)
    seeds = ppr_seeds(graph, "감사규정")
    assert seeds and max(seeds, key=seeds.get) == "n_audit"
    ppr = ppr_chunk_scores(graph, seeds, ["k1", "k2", "k3", "k4"])
    assert ppr["k1"] == 1.0 and 0 < ppr["k2"] <= 1.0 and "k4" not in ppr
    scores = {"k1": 0.5, "k2": 0.5, "k4": 0.5}
    touched = blend_ppr(scores, ppr, weight=0.2)
    assert touched == 2 and scores["k1"] == 0.8 * 0.5 + 0.2 * 1.0 and scores["k4"] == 0.5


def test_preloaded_store_fuses_dense_and_lexical():
    store = PreloadedVectorStore(CHUNKS, DENSE)
    hits = store.search("대출 한도", limit=4)
    ids = [c.id for c, _ in hits]
    assert ids[0] == "k2"  # lexical match lifts the second-best dense chunk to the top
    assert set(ids) == {"k1", "k2", "k3", "k4"}
    assert [c.id for c in store.fetch(["k1", "n_audit", "k4"])] == ["k1", "k4"]
