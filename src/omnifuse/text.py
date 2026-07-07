"""Tokenization + BM25 — pure Python, zero deps.

This is the in-memory replacement for jena-text's Lucene index: CJK is tokenized
into character bi-grams (like Lucene's CJK analyzer) so Korean/Chinese/Japanese
label & passage search works with no morphological analyzer installed; Latin text
falls back to word tokens. BM25 Okapi gives ranked full-text retrieval over both.
"""
from __future__ import annotations

import math
import re

_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[가-힣぀-ヿ一-鿿]+")


def tokenize(text: str) -> list[str]:
    """Latin word tokens + CJK character bi-grams (unigram if length 1)."""
    text = (text or "").lower()
    toks = _WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(self, docs_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for d in docs_tokens:
            c: dict[str, int] = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            self.tf.append(c)
            for t in c:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, q_tokens: list[str], i: int) -> float:
        tf = self.tf[i]
        dl = self.dl[i] or 1
        s = 0.0
        for t in q_tokens:
            f = tf.get(t)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self.N)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]


class BM25F:
    """Field-weighted BM25 — a short high-signal field (title/heading) counts for
    more than the body, with per-field length normalization (Robertson's BM25F).

    ``docs`` is a list of ``{field: tokens}`` dicts; ``weights`` maps field ->
    boost. IDF is document-level (a term counts once across fields), so a query
    term appearing in the title lifts the doc without double-charging IDF.
    """

    def __init__(self, docs: list[dict[str, list[str]]], weights: dict[str, float],
                 *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.fields = list(weights.keys())
        self.w = weights
        self.N = len(docs)
        self.avglen = {}
        for f in self.fields:
            tot = sum(len(d.get(f, ())) for d in docs)
            self.avglen[f] = (tot / self.N) if self.N else 0.0
        self.doc_tf: list[dict[str, dict[str, int]]] = []
        df: dict[str, int] = {}
        self.postings: dict[str, set[int]] = {}
        for i, d in enumerate(docs):
            per_field: dict[str, dict[str, int]] = {}
            present: set[str] = set()
            for f in self.fields:
                c: dict[str, int] = {}
                for t in d.get(f, ()):
                    c[t] = c.get(t, 0) + 1
                per_field[f] = c
                present.update(c)
            self.doc_tf.append(per_field)
            for t in present:
                df[t] = df.get(t, 0) + 1
                self.postings.setdefault(t, set()).add(i)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, q_tokens: list[str], i: int) -> float:
        per_field = self.doc_tf[i]
        s = 0.0
        for t in set(q_tokens):
            idf = self.idf.get(t)
            if not idf:
                continue
            tfw = 0.0
            for f in self.fields:
                tf = per_field[f].get(t, 0)
                if not tf:
                    continue
                dl = sum(per_field[f].values()) or 1
                norm = 1 - self.b + self.b * dl / (self.avglen[f] or 1)
                tfw += self.w[f] * tf / norm
            if tfw:
                s += idf * tfw * (self.k1 + 1) / (self.k1 + tfw)
        return s

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        q = tokenize(query)
        cand: set[int] = set()
        for t in set(q):
            cand |= self.postings.get(t, set())
        scored = [(i, self.score(q, i)) for i in cand]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]
