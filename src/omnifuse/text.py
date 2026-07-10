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
from array import array
from bisect import bisect_left

_WORD = re.compile(r"[a-z0-9]+")
_HANGUL = re.compile(r"[가-힣]+")
_CJK_OTHER = re.compile(r"[぀-ヿ一-鿿]+")  # kana + hanja: bi-grams, no morphology

# Term-specificity emphasis. A natural-language question ("장 발장은 어떤 범죄로 유죄
# 판결을 받았나요?") carries one rare discriminative term (the entity 발장) buried
# under several common ones (범죄/유죄/판결); plain BM25 sums term scores, so a doc
# matching many common words outranks the one matching the rare entity. Raising IDF to
# a power > 1 makes the rare term dominate the sum, fixing this "entity-burial". 1.0 is
# plain BM25; the core benchmark suite wins all ten datasets across the whole flat band
# p∈[1.3, 2.0], so 1.5 is a robust, non-fitted default (zero runtime cost — the power
# is folded into the precomputed IDF once at index build).
#
# It is a real trade, not free: on *heavily multi-relevant* passage-IR corpora (BEIR
# NFCorpus ~38 relevant/query, MIRACL-ko ~14) the emphasis hurts — MIRACL-ko drops
# 0.949 (p=1.0) -> 0.905 (p=1.5) — because betting on one rare term is wrong when many
# documents are relevant. Pass ``idf_pow=1.0`` for such corpora, via
# ``build_inmemory(..., vector_kwargs={"idf_pow": 1.0})``. See docs/comparison.
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


def _en_stem(word: str) -> str:
    """Harman's S-stemmer — singularize a Latin word, and nothing else.

    Korean already gets morphological normalization; leaving Latin as raw surface forms
    meant "statin" never matched "statins". This is the deliberately conservative rule
    set (no -ing/-ed, no Porter cascade): it has no tunable parameter, so there is
    nothing to fit, and it does not maul word stems the way aggressive stemmers do.
    """
    if len(word) > 3:
        if word.endswith("ies") and not word.endswith(("eies", "aies")):
            return word[:-3] + "y"
        if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
            return word[:-1]
        if word.endswith("s") and not word.endswith(("us", "ss")):
            return word[:-1]
    return word


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
    """Latin word stems + Hanja/Kana bi-grams + Hangul stem bi-grams (+ stem unigram)."""
    text = (text or "").lower()
    toks = [_en_stem(w) for w in _WORD.findall(text)]
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
    """Okapi BM25 over a fixed corpus of pre-tokenized documents.

    Queries hit an inverted index: only documents sharing at least one query term
    can score above zero, so scoring the postings union — rather than all *N*
    documents — is exactly score-preserving and turns a full scan into work
    proportional to the matched postings.
    """

    def __init__(self, docs_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75,
                 idf_pow: float = _IDF_POW):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        dls = [len(d) for d in docs_tokens]
        self.avgdl = (sum(dls) / self.N) if self.N else 0.0
        # Pass 1 — document frequency only. Holding every document's term-count map just
        # to derive IDF would double peak memory, so the maps are rebuilt in pass 2.
        df: dict[str, int] = {}
        for d in docs_tokens:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** idf_pow for t, n in df.items()}
        # per-doc length normalization, constant across queries
        avg = self.avgdl or 1.0
        norms = [self.k1 * (1 - self.b + self.b * (dl or 1) / avg) for dl in dls]
        # A term's contribution to a document, idf*(k1+1)*f/(f+norm), does not depend on
        # the query — only on (term, doc). Fold it into the index so a search is a plain
        # accumulation of floats: no division, no tf lookup, no length math per query.
        # Pass 2 — emit the postings. Everything a query needs lives in _pd/_pw afterwards,
        # so no term-frequency map is ever retained (they are the bulk of the index).
        k1p1 = self.k1 + 1
        self._pd: dict[str, array] = {}
        self._pw: dict[str, array] = {}
        for i, d in enumerate(docs_tokens):
            norm = norms[i]
            c: dict[str, int] = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            for t, f in c.items():
                w = self.idf[t] * k1p1 * f / (f + norm)
                if t not in self._pd:
                    self._pd[t] = array("i")
                    self._pw[t] = array("d")
                self._pd[t].append(i)
                self._pw[t].append(w)

    def score(self, q_tokens: list[str], i: int) -> float:
        """Score of document ``i`` — reads the precomputed contributions straight out of
        the postings (each ``_pd[t]`` is ascending, so membership is a binary search)."""
        s = 0.0
        for t in q_tokens:
            pd = self._pd.get(t)
            if pd is None:
                continue
            k = bisect_left(pd, i)
            if k < len(pd) and pd[k] == i:
                s += self._pw[t][k]
        return s

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        """Term-at-a-time accumulation over the inverted index — touches only the
        documents that actually contain a query term, adding a precomputed weight."""
        qtf: dict[str, int] = {}
        for t in tokenize(query):
            qtf[t] = qtf.get(t, 0) + 1
        scores: dict[int, float] = {}
        for t, qn in qtf.items():
            pd = self._pd.get(t)
            if pd is None:
                continue
            pw = self._pw[t]
            if qn == 1:
                for i, w in zip(pd, pw):
                    scores[i] = scores.get(i, 0.0) + w
            else:
                for i, w in zip(pd, pw):
                    scores[i] = scores.get(i, 0.0) + qn * w
        scored = [(i, s) for i, s in scores.items() if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))  # ties -> lowest doc id (deterministic)
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
        # Pass 1 — document frequency only. Keeping every document's per-field term-count
        # map just to derive IDF doubles peak memory, so pass 2 rebuilds them one doc at a
        # time. Field *lengths* are cheap (a few ints per doc) and are kept.
        df: dict[str, int] = {}
        for d in docs:
            present: set[str] = set()
            for f in self.fields:
                present.update(d.get(f, ()))
            for t in present:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** idf_pow for t, n in df.items()}
        self._fw = [self.w[f] for f in self.fields]
        # BM25F sums over the *unique* query terms, so a term's whole contribution to a
        # document — idf * tfw(k1+1)/(k1+tfw) over the weighted fields — depends only on
        # (term, doc). Fold it into the index: a search becomes a sum of precomputed floats.
        k1p1 = self.k1 + 1
        self._pd: dict[str, array] = {}
        self._pw: dict[str, array] = {}
        for i, d in enumerate(docs):
            # per-doc, per-field length normalization — constant across queries
            fnorm = [1 - self.b + self.b * (len(d.get(f, ())) or 1) / (self.avglen[f] or 1)
                     for f in self.fields]
            counts: list[dict[str, int]] = []
            present = set()
            for f in self.fields:
                c: dict[str, int] = {}
                for t in d.get(f, ()):
                    c[t] = c.get(t, 0) + 1
                counts.append(c)
                present.update(c)
            for t in present:
                tfw = 0.0
                for fi in range(len(self.fields)):
                    tf = counts[fi].get(t, 0)
                    if tf:
                        tfw += self._fw[fi] * tf / fnorm[fi]
                if not tfw:
                    continue
                w = self.idf[t] * tfw * k1p1 / (self.k1 + tfw)
                if t not in self._pd:
                    self._pd[t] = array("i")
                    self._pw[t] = array("d")
                self._pd[t].append(i)
                self._pw[t].append(w)

    def _score(self, q_terms, i: int) -> float:
        s = 0.0
        for t in q_terms:
            pd = self._pd.get(t)
            if pd is None:
                continue
            k = bisect_left(pd, i)
            if k < len(pd) and pd[k] == i:
                s += self._pw[t][k]
        return s

    def score(self, q_tokens: list[str], i: int) -> float:
        """Score of document ``i`` — the precomputed contributions are read out of the
        postings (each ``_pd[t]`` is ascending, so membership is a binary search)."""
        return self._score(set(q_tokens), i)

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):
            pd = self._pd.get(t)
            if pd is None:
                continue
            for i, c in zip(pd, self._pw[t]):
                scores[i] = scores.get(i, 0.0) + c
        scored = [(i, s) for i, s in scores.items() if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))  # ties -> lowest doc id (deterministic)
        return scored[:limit]
