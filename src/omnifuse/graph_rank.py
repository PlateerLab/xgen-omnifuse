"""Graph-side ranking — entity seeds to candidate chunks, and personalized PageRank.

Two ways the graph shapes retrieval beyond label-linked relations:

  * ``graph_candidate_chunks``: question -> label-linked seed entities (weighted by
    label match x rarity) -> 1-hop neighbours -> chunks mentioning them. Widens the
    candidate pool with passages a vector index missed. Ranking the new chunks is
    the caller's job (question vector, lexical hits, ...).
  * ``ppr_seeds`` + ``ppr_chunk_scores``: spread seed mass over the whole edge set and
    read it back on the candidate chunks. Re-ranks an existing pool, never adds to it.

Pure functions over the optional ``GraphRankStore`` methods; no deps.
"""
from __future__ import annotations

from typing import Optional


def graph_candidate_chunks(
    graph,
    question: str,
    *,
    max_coverage: float = 0.5,
    hop_decay: float = 0.4,
    pool_max: int = 500,
    seed_limit: int = 24,
    expand_seeds: int = 6,
    neighbor_limit: int = 10,
) -> tuple[list[str], dict[str, float]]:
    """Candidate chunk ids (best first) and their graph scores.

    Seeds are label hits whose coverage is at most ``max_coverage`` (a node that
    appears in more of the collection is a hub, not a clue), weighted
    ``match * (1 - coverage)``. The top ``expand_seeds`` seeds pull in 1-hop
    neighbours at ``hop_decay`` of their weight. Each chunk scores the sum of the
    weights of the entities it mentions, again discounted by coverage.
    """
    hits = graph.search_labels(question, limit=seed_limit)
    cand = [n.id for n, _ in hits]
    if not cand:
        return [], {}
    match = {n.id: float(sc) for n, sc in hits}
    cov = graph.chunk_coverage(cand)
    seed = sorted((u for u in cand if cov.get(u, 0.0) <= max_coverage),
                  key=lambda u: -(match.get(u, 0.0) * (1.0 - cov.get(u, 1.0))))
    if not seed:
        return [], {}

    weight: dict[str, float] = {u: match.get(u, 0.0) for u in seed}
    expanded = list(seed)
    for nid in seed[:expand_seeds]:
        w = weight.get(nid, 0.0) * hop_decay
        for nb in graph.neighbor_ids(nid, limit=neighbor_limit):
            expanded.append(nb)
            if w > weight.get(nb, 0.0):
                weight[nb] = w

    pairs = graph.chunks_of_entities(list(dict.fromkeys(expanded)))
    all_cov = graph.chunk_coverage(list({u for _, u in pairs}))
    score: dict[str, float] = {}
    for ck, u in pairs:
        score[ck] = score.get(ck, 0.0) + (weight.get(u, 0.0) * (1.0 - all_cov.get(u, 1.0)))
    order = [c for c, _ in sorted(score.items(), key=lambda kv: -kv[1])][:pool_max]
    return order, score


def ppr_seeds(graph, question: str, *, limit: int = 24) -> dict[str, float]:
    """Seed mass for PageRank: label-match strength x node specificity, positives only."""
    hits = graph.search_labels(question, limit=limit)
    if not hits:
        return {}
    spec = graph.node_specificity([n.id for n, _ in hits]) or {}
    seeds = {n.id: float(sc) * spec.get(n.id, 0.0) for n, sc in hits}
    return {k: v for k, v in seeds.items() if v > 0}


def ppr_chunk_scores(
    graph,
    seeds: dict[str, float],
    chunk_ids: list[str],
    *,
    alpha: float = 0.5,
    iters: int = 12,
    max_edges: int = 80000,
) -> dict[str, float]:
    """Personalized PageRank from ``seeds`` over ``graph.all_edges`` (undirected),
    read back on ``chunk_ids`` via ``chunks_of_entities`` and scaled to [0, 1].
    ``alpha`` is the restart probability. Empty dict when anything is missing."""
    if not seeds or not chunk_ids:
        return {}
    try:
        edges = graph.all_edges(limit=max_edges)
    except Exception:
        return {}
    if not edges:
        return {}

    adj: dict[str, list[str]] = {}
    for s_uri, o_uri in edges:
        if not s_uri or not o_uri:
            continue
        adj.setdefault(s_uri, []).append(o_uri)
        adj.setdefault(o_uri, []).append(s_uri)
    if not adj:
        return {}

    total = sum(seeds.values()) or 1.0
    reset = {k: v / total for k, v in seeds.items() if k in adj}
    if not reset:
        return {}
    rank = dict(reset)
    for _ in range(iters):
        nxt: dict[str, float] = {}
        for node, mass in rank.items():
            nbrs = adj.get(node)
            if not nbrs:
                continue
            share = (1.0 - alpha) * mass / len(nbrs)
            if share <= 1e-9:
                continue
            for nb in nbrs:
                nxt[nb] = nxt.get(nb, 0.0) + share
        for k, v in reset.items():
            nxt[k] = nxt.get(k, 0.0) + alpha * v
        rank = nxt
    if not rank:
        return {}

    try:
        pairs = graph.chunks_of_entities(list(rank.keys()))
    except Exception:
        return {}
    want = set(chunk_ids)
    out: dict[str, float] = {}
    for ck, uri in pairs:
        if ck not in want:
            continue
        m = rank.get(uri, 0.0)
        if m > out.get(ck, 0.0):
            out[ck] = m
    if not out:
        return {}
    top = max(out.values()) or 1.0
    return {k: v / top for k, v in out.items()}


def blend_ppr(scores: dict[str, float], ppr: dict[str, float], *, weight: float) -> int:
    """In place: ``score = (1 - weight) * score + weight * ppr`` for chunks PPR reached.
    Returns how many were touched."""
    hit = 0
    for ck, p in ppr.items():
        if ck in scores:
            scores[ck] = (1.0 - weight) * scores[ck] + weight * p
            hit += 1
    return hit


__all__ = ["graph_candidate_chunks", "ppr_seeds", "ppr_chunk_scores", "blend_ppr"]
