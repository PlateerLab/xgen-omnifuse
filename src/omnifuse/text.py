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
#
# The copula's interrogative paradigm (-인가/-인가요/-입니까/-인지) is part of this closed
# class and was missing. Without it "어디인가" stems to the *rare* token 어디인 instead of
# the common word 어디, and `idf_pow` then amplifies that rarity: on MIRACL-ko every
# "…어디인가?" question retrieved the article titled "내 친구의 집은 어디인가" — a 4x-weighted
# title match on nothing but the question word. Kiwi splits the copula into morphemes, which
# is why synaptic never saw this.
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
    "인가요", "입니까", "인가", "인지",
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


def _passes(docs):
    """Indexing needs two passes. ``docs`` may be a materialized sequence, or a zero-arg
    callable returning a fresh iterator — which lets the caller *stream* tokenization
    instead of holding every tokenized document in memory at once."""
    return (lambda: iter(docs())) if callable(docs) else (lambda: iter(docs))


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents.

    Queries hit an inverted index: only documents sharing at least one query term
    can score above zero, so scoring the postings union — rather than all *N*
    documents — is exactly score-preserving and turns a full scan into work
    proportional to the matched postings.

    ``docs_tokens`` may be a list, or a zero-arg callable yielding token lists (streamed).
    """

    def __init__(self, docs_tokens, *, k1: float = 1.5, b: float = 0.75,
                 idf_pow: float = _IDF_POW):
        self.k1, self.b = k1, b
        stream = _passes(docs_tokens)
        # Pass 1 — tokenize ONCE. IDF needs corpus-wide document frequency, so the
        # per-document term counts must survive until pass 2; we keep them as interned
        # term ids in `array('i')` rather than dicts of strings, which is where the
        # memory went. Pass 2 then frees each document as it consumes it.
        vocab: dict[str, int] = {}
        terms: list[str] = []
        df: list[int] = []
        dls: list[int] = []
        doc_ids: list[array] = []
        doc_cnt: list[array] = []
        for d in stream():
            dls.append(len(d))
            c: dict[str, int] = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            ids, cnt = array("i"), array("i")
            for t, f in c.items():
                tid = vocab.get(t)
                if tid is None:
                    tid = len(terms)
                    vocab[t] = tid
                    terms.append(t)
                    df.append(0)
                df[tid] += 1
                ids.append(tid)
                cnt.append(f)
            doc_ids.append(ids)
            doc_cnt.append(cnt)
        self.N = len(dls)
        self.avgdl = (sum(dls) / self.N) if self.N else 0.0
        idfs = [math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** idf_pow for n in df]
        self.idf = dict(zip(terms, idfs))
        avg = self.avgdl or 1.0
        # A term's contribution to a document, idf*(k1+1)*f/(f+norm), does not depend on
        # the query — only on (term, doc). Fold it into the index so a search is a plain
        # accumulation of floats: no division, no tf lookup, no length math per query.
        k1p1 = self.k1 + 1
        self._pd: dict[str, array] = {}
        self._pw: dict[str, array] = {}
        for i in range(self.N):
            norm = self.k1 * (1 - self.b + self.b * (dls[i] or 1) / avg)
            for tid, f in zip(doc_ids[i], doc_cnt[i]):
                t = terms[tid]
                w = idfs[tid] * k1p1 * f / (f + norm)
                if t not in self._pd:
                    self._pd[t] = array("i")
                    self._pw[t] = array("d")
                self._pd[t].append(i)
                self._pw[t].append(w)
            doc_ids[i] = doc_cnt[i] = None  # release as consumed

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

    ``docs`` is a list of ``{field: tokens}`` dicts — or a zero-arg callable yielding
    them, which lets the caller stream tokenization instead of materializing the whole
    tokenized corpus. ``weights`` maps field -> boost. IDF is document-level (a term
    counts once across fields), so a query term appearing in the title lifts the doc
    without double-charging IDF.

    ``evidence_fields`` names fields that are *evidence about* a document rather than its
    content — e.g. queries it was confirmed to answer. Their terms score the document but
    are excluded from document frequency, so they cannot deflate the IDF of the collection
    (a term seen only in evidence takes its IDF from the evidence df instead). They are
    also not length-normalized: remembering a second query must not dilute the first.
    With no evidence fields the class behaves exactly as before.
    """

    def __init__(self, docs, weights: dict[str, float],
                 *, k1: float = 1.5, b: float = 0.75, idf_pow: float = _IDF_POW,
                 evidence_fields: "frozenset[str] | set[str] | None" = None):
        self.k1, self.b = k1, b
        self.fields = list(weights.keys())
        self.w = weights
        self.evidence_fields = frozenset(evidence_fields or ())
        self._is_ev = [f in self.evidence_fields for f in self.fields]
        stream = _passes(docs)
        nf = len(self.fields)
        # Pass 1 — tokenize ONCE. IDF needs corpus-wide document frequency, so per-document
        # counts must survive to pass 2; they are kept as interned term ids in `array('i')`
        # rather than dicts of strings. Pass 2 frees each document as it consumes it.
        vocab: dict[str, int] = {}
        terms: list[str] = []
        df: list[int] = []
        dfe: list[int] = []          # document frequency seen ONLY in evidence fields
        totals = [0] * nf
        doc_ids: list[list[array]] = []
        doc_cnt: list[list[array]] = []
        doc_len: list[array] = []
        self.N = 0
        for d in stream():
            self.N += 1
            f_ids, f_cnt = [], []
            flen = array("i")
            present: set[int] = set()
            present_ev: set[int] = set()
            for fi, f in enumerate(self.fields):
                toks = d.get(f, ())
                flen.append(len(toks))
                totals[fi] += len(toks)
                c: dict[str, int] = {}
                for t in toks:
                    c[t] = c.get(t, 0) + 1
                ids, cnt = array("i"), array("i")
                ev = self._is_ev[fi]
                for t, tf in c.items():
                    tid = vocab.get(t)
                    if tid is None:
                        tid = len(terms)
                        vocab[t] = tid
                        terms.append(t)
                        df.append(0)
                        dfe.append(0)
                    ids.append(tid)
                    cnt.append(tf)
                    (present_ev if ev else present).add(tid)
                f_ids.append(ids)
                f_cnt.append(cnt)
            for tid in present:
                df[tid] += 1
            for tid in present_ev:
                dfe[tid] += 1
            doc_ids.append(f_ids)
            doc_cnt.append(f_cnt)
            doc_len.append(flen)
        self.avglen = {f: (totals[fi] / self.N if self.N else 0.0) for fi, f in enumerate(self.fields)}
        # Evidence never deflates a content term's IDF: df counts content only. A term seen
        # solely in evidence has content-df 0, so it takes its IDF from the evidence df.
        idfs = [math.log(1 + (self.N - (n or ne) + 0.5) / ((n or ne) + 0.5)) ** idf_pow
                for n, ne in zip(df, dfe)]
        self.idf = dict(zip(terms, idfs))
        self._fw = [self.w[f] for f in self.fields]
        # BM25F sums over the *unique* query terms, so a term's whole contribution to a
        # document — idf * tfw(k1+1)/(k1+tfw) over the weighted fields — depends only on
        # (term, doc). Fold it into the index: a search becomes a sum of precomputed floats.
        k1p1 = self.k1 + 1
        avgl = [self.avglen[f] or 1 for f in self.fields]
        self._pd: dict[str, array] = {}
        self._pw: dict[str, array] = {}
        for i in range(self.N):
            flen = doc_len[i]
            fnorm = [1.0 if self._is_ev[fi] else 1 - self.b + self.b * (flen[fi] or 1) / avgl[fi]
                     for fi in range(nf)]
            tfws: dict[int, float] = {}
            for fi in range(nf):
                wf = self._fw[fi]
                nrm = fnorm[fi]
                for tid, tf in zip(doc_ids[i][fi], doc_cnt[i][fi]):
                    tfws[tid] = tfws.get(tid, 0.0) + wf * tf / nrm
            for tid, tfw in tfws.items():
                if not tfw:
                    continue
                t = terms[tid]
                w = idfs[tid] * tfw * k1p1 / (self.k1 + tfw)
                if t not in self._pd:
                    self._pd[t] = array("i")
                    self._pw[t] = array("d")
                self._pd[t].append(i)
                self._pw[t].append(w)
            doc_ids[i] = doc_cnt[i] = None  # release as consumed

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
