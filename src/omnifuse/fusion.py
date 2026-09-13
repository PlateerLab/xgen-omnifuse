"""Fusion primitives — diversity (MMR), adaptive cut, relation ranking.

All pure functions, no deps. This is the part of OmniFuse that makes the fused
evidence *non-redundant*: top-k by relevance alone lets near-duplicate passages
crowd out the decisive minority (warnings, exceptions, reversals); MMR keeps both.
"""
from __future__ import annotations

from .text import tokenize, tokenize_query


def _toks(s: str) -> set[str]:
    return set(tokenize(s))


def minmax(pairs: list[tuple]) -> dict:
    """Per-query [0,1] normalization so scores on different scales (dense cosine,
    lexical BM25) can be summed. ``pairs`` is [(key, score), ...]."""
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {i: (s - lo) / rng for i, s in pairs}


def specificity_rerank(hits: list[tuple], graph, *, weight: float) -> list[tuple]:
    """Re-weight (chunk, score) pairs by the most specific entity each chunk mentions.

    ``score * ((1 - weight) + weight * max_specificity)`` — a chunk whose entities are
    all hubs (low specificity) drops, a chunk naming a rare entity keeps its score.
    Needs ``graph.node_specificity(ids) -> {id: 0..1}``; without it the list is
    returned unchanged. Chunks with no entities are not re-weighted.
    """
    spec_fn = getattr(graph, "node_specificity", None)
    if not callable(spec_fn) or not hits:
        return hits
    ids = [u for c, _ in hits for u in (getattr(c, "entities", None) or [])]
    try:
        spec = spec_fn(ids) or {}
    except Exception:
        spec = {}
    if not spec:
        return hits
    out = []
    for c, sc in hits:
        ents = getattr(c, "entities", None) or []
        if ents:
            sc = sc * ((1.0 - weight) + weight * max((spec.get(u, 0.0) for u in ents), default=0.0))
        out.append((c, sc))
    out.sort(key=lambda kv: -kv[1])
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dynamic_cut(scored: list[tuple], *, ratio: float = 0.55, min_k: int = 4, max_k: int = 40) -> list:
    """Adaptive top-k: keep items scoring >= ratio * top_score (clamped to [min_k, max_k]).

    Replaces a fixed k — a sharp query keeps few strong hits, a broad one keeps more.
    ``scored`` is [(item, score), ...] sorted desc; returns the items.
    """
    if not scored:
        return []
    top = scored[0][1] or 1e-9
    keep = [it for it, sc in scored if sc >= ratio * top]
    keep = keep[:max_k]
    if len(keep) < min_k:
        keep = [it for it, _ in scored[:min_k]]
    return keep


def natural_cut(scored: list[tuple], *, min_k: int = 0, max_k: int = 0) -> list:
    """Self-sizing top-k: cut where the scores split into a high and a low group.

    Picks the k that maximizes the between-group variance of head ``scored[:k]`` vs tail
    (Jenks natural breaks / Otsu with two classes), so the data decides how many items
    survive; there is no ratio or threshold to tune. ``min_k`` / ``max_k`` only bound
    the answer (0 = unbounded). Flat scores have no break and are kept whole.
    ``scored`` is [(item, score), ...] sorted desc; returns the items.
    """
    if not scored:
        return []
    n = len(scored) if max_k <= 0 else min(len(scored), max_k)
    lo = max(1, min_k)
    if n <= lo:
        return [it for it, _ in scored[:n]]
    vals = [float(sc) for _, sc in scored[:n]]
    prefix = [0.0]
    for v in vals:
        prefix.append(prefix[-1] + v)
    total = prefix[-1]
    best, best_k = 0.0, n
    for k in range(lo, n):
        head, tail = prefix[k] / k, (total - prefix[k]) / (n - k)
        var = k * (n - k) * (head - tail) ** 2
        if var > best:
            best, best_k = var, k
    return [it for it, _ in scored[:best_k]]


def mmr(candidates: list[tuple], *, lam: float = 0.72, k: int = 16) -> list:
    """Maximal Marginal Relevance over (text, relevance) candidates.

    Selects up to k items maximizing ``lam*relevance - (1-lam)*max_sim_to_selected``
    where similarity is Jaccard over tokens (no embeddings needed).
    """
    if not candidates:
        return []
    pool = [(text, rel, _toks(text)) for text, rel in candidates]
    rmax = max((r for _, r, _ in pool), default=1.0) or 1.0
    selected: list[tuple] = []
    chosen: list[set[str]] = []
    while pool and len(selected) < k:
        best_i, best_v = 0, -1e18
        for i, (text, rel, tks) in enumerate(pool):
            div = max((jaccard(tks, c) for c in chosen), default=0.0)
            v = lam * (rel / rmax) - (1 - lam) * div
            if v > best_v:
                best_v, best_i = v, i
        text, rel, tks = pool.pop(best_i)
        selected.append((text, rel))
        chosen.append(tks)
    return [t for t, _ in selected]


def rank_relations(triples: list[str], question: str, *, limit: int = 40) -> list[str]:
    """Dedup relation strings and rank by how many query terms they hit."""
    q = set(tokenize_query(question))
    seen: set[str] = set()
    uniq: list[str] = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    def hits(t: str) -> int:
        tl = set(tokenize(t))
        return len(q & tl)

    uniq.sort(key=lambda t: -hits(t))
    return uniq[:limit]
