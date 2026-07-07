"""OmniFuse — one-shot, backend-agnostic GraphRAG.

Fire several retrieval strategies in parallel-spirit and *fuse* them into one
synthesis, instead of an iterative tool-calling (ReAct) loop:

  1. vector / lexical passages          (what it says)
  2. graph label-linking -> 1-hop        (how entities connect)
  3. class enumeration                   (complete 'list/count' sets a vector index can't give)
  4. HippoRAG: seed-chunk entities -> 1-hop expansion
  -> evidence assembled with MMR diversity + adaptive top-k
  -> a single LLM synthesis over the fused evidence.

The class lives only against the GraphStore / VectorStore / LLM protocols, so the
same algorithm runs on the in-memory default (zero infra) or on Fuseki/Qdrant.
"""
from __future__ import annotations

import re
from typing import Optional

from .fusion import dynamic_cut, mmr, rank_relations
from .llm import EchoLLM
from .models import SearchResult
from .protocols import LLM, GraphStore, VectorStore
from .text import tokenize

# Language-neutral default — pass system_prompt=... for any language/domain.
SYSTEM = (
    "Answer the question using the evidence below as an explanation, not a raw list dump. "
    "For a conditional question (X related to Y), select only the truly relevant items with "
    "grounds instead of dumping the whole candidate set. Give a complete list only when the "
    "question explicitly asks to enumerate or count. Do not invent facts absent from the evidence."
)


class OmniFuse:
    """One-shot fusion search. Inject a graph store, a vector store and an LLM."""

    def __init__(
        self,
        graph: GraphStore,
        vector: VectorStore,
        llm: Optional[LLM] = None,
        *,
        vector_k: int = 20,
        sem_ratio: float = 0.55,
        sem_min: int = 4,
        sem_max: int = 40,
        mmr_lambda: float = 0.72,
        evidence_k: int = 16,
        rel_limit: int = 40,
        seed_label_limit: int = 30,
        class_list_cap: int = 150,
        graph_fusion: bool = True,
        fusion_alpha: float = 0.9,
        fusion_expand_top: int = 5,
        fusion_neighbor_limit: int = 20,
        system_prompt: str = SYSTEM,
    ):
        self.graph = graph
        self.vector = vector
        self.llm = llm or EchoLLM()
        self.system_prompt = system_prompt
        self.vector_k = vector_k
        self.sem_ratio = sem_ratio
        self.sem_min = sem_min
        self.sem_max = sem_max
        self.mmr_lambda = mmr_lambda
        self.evidence_k = evidence_k
        self.rel_limit = rel_limit
        self.seed_label_limit = seed_label_limit
        self.class_list_cap = class_list_cap
        self.graph_fusion = graph_fusion
        self.fusion_alpha = fusion_alpha
        self.fusion_expand_top = fusion_expand_top
        self.fusion_neighbor_limit = fusion_neighbor_limit

    def retrieve(self, question: str, *, limit: Optional[int] = None) -> list[tuple]:
        """Ranked (chunk, score) fusing lexical/vector seeds with 1-hop graph
        structure — a passage cited/linked by a strong seed is surfaced beside it
        (companion score = ``fusion_alpha`` x seed), so multi-hop evidence that
        shares no query vocabulary still lands in the ranking. Pure retrieval,
        no LLM. ``search()`` builds on this; call it directly for ranking/eval.
        """
        limit = limit or self.vector_k
        vhits = self.vector.search(question, limit=self.vector_k)
        scores: dict[str, float] = {}
        cmap: dict = {}
        for c, sc in vhits:
            scores[c.id] = sc
            cmap[c.id] = c
        if self.graph_fusion and vhits:
            for c, sc in vhits[: self.fusion_expand_top]:
                for tgt in self.graph.neighbor_ids(c.id, limit=self.fusion_neighbor_limit):
                    cand = self.fusion_alpha * sc
                    if cand <= scores.get(tgt, 0.0):
                        continue
                    if tgt not in cmap:
                        got = self.vector.fetch([tgt])
                        if not got:
                            continue
                        cmap[tgt] = got[0]
                    scores[tgt] = cand
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(cmap[i], s) for i, s in ranked[:limit]]

    def search(self, question: str) -> SearchResult:
        # 1) vector / lexical seed + 1-hop graph fusion (adaptive top-k by score)
        vhits = self.retrieve(question, limit=self.vector_k)
        chunks = dynamic_cut(vhits, ratio=self.sem_ratio, min_k=self.sem_min, max_k=self.sem_max)

        triples: list[tuple[str, str, str]] = []

        # 2) graph label-linking -> 1-hop relations
        seeds = self.graph.search_labels(question, limit=self.seed_label_limit)
        for node, _ in seeds:
            triples.extend(self.graph.neighbors(node.id, hops=1))

        # 3) class enumeration — structural 'complete list/count' a vector index can't give
        class_seed = ""
        class_hits = [n for n, _ in seeds if n.kind == "class"]
        if class_hits:
            cn = class_hits[0]
            insts = self.graph.class_instances(cn.id)
            if len(insts) >= 2:
                shown = " | ".join(i.label for i in insts[: self.class_list_cap])
                more = f" (+{len(insts) - self.class_list_cap} more)" if len(insts) > self.class_list_cap else ""
                class_seed = (
                    f"[CLASS '{cn.label}' has {len(insts)} instances: {shown}{more}. "
                    f"List all only if asked to enumerate/count; otherwise pick the relevant ones.]"
                )

        # 4) HippoRAG — entities of the retrieved chunks -> 1-hop expansion
        for ch in chunks:
            for eid in (ch.entities or []):
                triples.extend(self.graph.neighbors(eid, hops=1))

        rel_strings = [f"{s} → {p} → {o}" for s, p, o in triples]
        relations = rank_relations(rel_strings, question, limit=self.rel_limit)

        # 5) evidence = MMR diversity over chunk text (keep minority/decisive passages)
        ev = mmr([(c.text, sc) for c, sc in vhits], lam=self.mmr_lambda, k=self.evidence_k)

        # 6) single synthesis over fused evidence
        prompt = self._prompt(question, ev, relations, class_seed)
        answer = self.llm.generate(prompt, system=self.system_prompt)

        return SearchResult(
            answer=answer,
            question=question,
            chunks=chunks,
            relations=relations,
            evidence_nodes=self._cited_nodes(relations, class_seed, answer),
            class_seed=class_seed,
        )

    @staticmethod
    def _prompt(question: str, evidence: list[str], relations: list[str], class_seed: str) -> str:
        parts = [f"Question: {question}"]
        if class_seed:
            parts.append(class_seed)
        if relations:
            parts.append("[Graph relations]\n" + "\n".join(relations))
        if evidence:
            parts.append("[Evidence passages]\n" + "\n---\n".join(evidence))
        return "\n\n".join(parts)

    @staticmethod
    def _cited_nodes(relations: list[str], class_seed: str, answer: str) -> list[str]:
        """Nodes the answer actually mentions — honest highlight, not keyword spray."""
        cand: set[str] = set()
        for t in relations:
            parts = [p.strip() for p in t.split("→")]
            if len(parts) >= 3:
                cand.add(parts[0])
                cand.add(parts[-1])
        m = re.search(r"'([^']+)'", class_seed)  # class label
        if m:
            cand.add(m.group(1).strip())
        mi = re.search(r"instances:\s*(.+)", class_seed)  # instance list (cut trailing prose)
        if mi:
            body = re.split(r"[.\]]|\(\+", mi.group(1))[0]
            for name in body.split("|"):
                nm = name.strip()
                if 2 <= len(nm) <= 50:
                    cand.add(nm)
        ans = answer or ""
        return sorted(c for c in cand if len(c) >= 2 and c in ans)
