"""Tokenization + BM25 — pure Python, zero deps.

The in-memory replacement for jena-text's Lucene index. Latin → word tokens.
Hanja/Kana → character bi-grams (Lucene CJK-analyzer style). **Hangul** gets a
lightweight, dependency-free morphological normalization: a rule-based stripper
removes the common particles (조사) and endings (어미) so a query and a document
align on the stem the way a morphological analyzer (Kiwi) would — but pure Python,
and it emits *fewer* tokens than raw bi-grams (bi-grams of the stem + one stem
unigram), so it is both more accurate on Korean and more memory-efficient. A
query term in "외국환거래법상" / "외국환거래법에" both reduce to "외국환거래법".
"""
from __future__ import annotations

import math
import re

_WORD = re.compile(r"[a-z0-9]+")
_HANGUL = re.compile(r"[가-힣]+")
_CJK_OTHER = re.compile(r"[぀-ヿ一-鿿]+")  # kana + hanja: bi-grams, no morphology

# Term-specificity emphasis. A natural-language question ("장 발장은 어떤 범죄로 유죄
# 판결을 받았나요?") carries one rare discriminative term (the entity 발장) buried
# under several common ones (범죄/유죄/판결); plain BM25 sums term scores, so a doc
# matching many common words outranks the one matching the rare entity. Raising IDF to
# a power > 1 makes the rare term dominate the sum, fixing this "entity-burial". 1.0 is
# plain BM25; the benchmark suite wins all ten datasets across the whole flat band
# p∈[1.3, 2.0], so 1.5 is a robust, non-fitted default (zero runtime cost — the power
# is folded into the precomputed IDF once at index build).
_IDF_POW = 1.5

# Korean particles (조사), verb/adjective endings (어미), and derivational suffixes,
# stripped only when trailing (so 상황/성별 — with the char leading — are untouched;
# and the emitted stem unigram still lets compound forms match). Longest first.
_KO_SUFFIX = sorted(set([
    "으로써", "으로서", "이라고", "라고", "에게서", "으로", "에서", "에게", "께서", "한테",
    "부터", "까지", "보다", "처럼", "만큼", "같이", "마다", "조차", "마저", "라도", "이라도",
    "이나", "이며", "이랑", "든지", "이야", "께", "의", "은", "는", "이", "가", "을", "를",
    "에", "도", "만", "과", "와", "로", "나", "랑", "야", "요",
    "습니다", "합니다", "입니다", "ㅂ니다", "는데", "지만", "거나", "어서", "아서", "도록",
    "으면", "면서", "고서", "다가", "든가",
    "하다", "되다", "이다", "하는", "되는", "하고", "되고", "했다", "된다", "한다", "하며",
    "되며", "하여", "되어", "여", "며", "면", "서", "고", "지", "니", "게", "자", "라",
    "았", "었", "겠", "임", "함", "됨", "기", "음",
    "적으로", "성이", "적인", "화된", "적", "화", "성", "상", "하", "들",
]), key=len, reverse=True)


def _ko_stem(word: str) -> str:
    """Iteratively strip trailing josa/eomi; keep the stem at least 2 chars."""
    changed = True
    while changed and len(word) >= 3:
        changed = False
        for s in _KO_SUFFIX:
            if len(word) - len(s) >= 2 and word.endswith(s):
                word = word[: -len(s)]
                changed = True
                break
    return word


def tokenize(text: str) -> list[str]:
    """Latin words + Hanja/Kana bi-grams + Hangul stem bi-grams (+ stem unigram)."""
    text = (text or "").lower()
    toks = _WORD.findall(text)
    for run in _CJK_OTHER.findall(text):
        toks.append(run) if len(run) == 1 else toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    for run in _HANGUL.findall(text):
        st = _ko_stem(run)
        if len(st) == 1:
            toks.append(st)
        else:
            toks.extend(st[i:i + 2] for i in range(len(st) - 1))
            toks.append("#" + st)
    return toks


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(self, docs_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75,
                 idf_pow: float = _IDF_POW):
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
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** idf_pow for t, n in df.items()}

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
                 *, k1: float = 1.5, b: float = 0.75, idf_pow: float = _IDF_POW):
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
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** idf_pow for t, n in df.items()}

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
