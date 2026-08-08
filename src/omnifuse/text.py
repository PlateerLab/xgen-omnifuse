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
from collections import Counter
from heapq import heappush, heapreplace
from typing import NamedTuple

from .settings import DEFAULT_IDF_POW

_WORD = re.compile(r"[a-z0-9]+")
_HANGUL = re.compile(r"[가-힣]+")
_CJK_OTHER = re.compile(r"[぀-ヿ一-鿿]+")  # kana + hanja: bi-grams, no morphology
_ASCII_WORD_TRANSLATION = str.maketrans(
    {
        codepoint: ord(" ")
        for codepoint in range(128)
        if not (ord("0") <= codepoint <= ord("9") or ord("a") <= codepoint <= ord("z"))
    }
)

# Term-specificity emphasis. A natural-language question ("장 발장은 어떤 범죄로 유죄
# 판결을 받았나요?") carries one rare discriminative term (the entity 발장) buried
# under several common ones (범죄/유죄/판결); plain BM25 sums term scores, so a doc
# matching many common words outranks the one matching the rare entity. Raising IDF to
# a power > 1 makes the rare term dominate the sum, fixing this "entity-burial". The
# mechanism is measured, not asserted: on the queries it rescues the gold document matches
# a rarer query term and fewer of them than the wrong top-1 (max-IDF 6.22 vs 6.08, overlap
# 3.4 vs 5.8); on the queries it breaks that sign flips (5.37 vs 6.16). 1.0 is plain BM25.
#
# The default is the MIDPOINT OF THE WINNING BAND, re-derived whenever the suite grows.
# With 13 datasets the band was p∈[1.3, 2.0] and the default 1.5. Completing coverage of
# synaptic's shipped data (18 targets) moved the band: FiQA (57,638 docs, 2.6 rel/query)
# binds it from ABOVE — it wins at p≤1.3 and loses from 1.4 — and Ko-StrategyQA binds it
# from BELOW (wins from p≥1.1). Overlap [1.1, 1.3], midpoint 1.2, at which every one of
# the 16 measured accuracy targets beats synaptic (LOSSES: none), and MIRACL-ko/finreg/
# golden are strictly better than at 1.5. The emphasis is still a trade: heavily
# multi-relevant corpora prefer less of it (that is exactly FiQA's constraint), so
# ``idf_pow=1.0`` remains available via
# ``build_inmemory(..., vector_kwargs={"idf_pow": 1.0})``. See eval/results/idf_pow_ablation.json.
_IDF_POW = DEFAULT_IDF_POW

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
_KO_SUFFIX = tuple(
    sorted(
        set(
            [
                "으로써",
                "으로서",
                "이라고",
                "라고",
                "에게서",
                "으로",
                "에서",
                "에게",
                "께서",
                "한테",
                "부터",
                "까지",
                "보다",
                "처럼",
                "만큼",
                "같이",
                "마다",
                "조차",
                "마저",
                "라도",
                "이라도",
                "이나",
                "이며",
                "이랑",
                "든지",
                "이야",
                "께",
                "의",
                "은",
                "는",
                "이",
                "가",
                "을",
                "를",
                "에",
                "도",
                "만",
                "과",
                "와",
                "로",
                "나",
                "랑",
                "야",
                "요",
                "습니다",
                "합니다",
                "입니다",
                "ㅂ니다",
                "는데",
                "지만",
                "거나",
                "어서",
                "아서",
                "도록",
                "으면",
                "면서",
                "고서",
                "다가",
                "든가",
                "하다",
                "되다",
                "이다",
                "하는",
                "되는",
                "하고",
                "되고",
                "했다",
                "된다",
                "한다",
                "하며",
                "되며",
                "하여",
                "되어",
                "여",
                "며",
                "면",
                "서",
                "고",
                "지",
                "니",
                "게",
                "자",
                "라",
                "았",
                "었",
                "겠",
                "임",
                "함",
                "됨",
                "기",
                "음",
                "적으로",
                "성이",
                "적인",
                "화된",
                "적",
                "화",
                "성",
                "상",
                "하",
                "들",
                "인가요",
                "입니까",
                "인가",
                "인지",
            ]
        ),
        key=len,
        reverse=True,
    )
)


def _suffixes_by_last(
    suffixes: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {}
    for suffix in suffixes:
        buckets.setdefault(suffix[-1], []).append(suffix)
    return {last: tuple(bucket) for last, bucket in buckets.items()}


_KO_SUFFIX_BY_LAST = _suffixes_by_last(_KO_SUFFIX)


def _en_stem_s(word: str) -> str:
    tail2 = word[-2:]
    if word[-3:] == "ies" and word[-4:] not in ("eies", "aies"):
        return word[:-3] + "y"
    if tail2 == "es" and word[-3:] not in ("aes", "ees", "oes"):
        return word[:-1]
    if tail2 not in ("us", "ss"):
        return word[:-1]
    return word


def _en_stem(word: str) -> str:
    """Harman's S-stemmer — singularize a Latin word, and nothing else.

    Korean already gets morphological normalization; leaving Latin as raw surface forms
    meant "statin" never matched "statins". This is the deliberately conservative rule
    set (no -ing/-ed, no Porter cascade): it has no tunable parameter, so there is
    nothing to fit, and it does not maul word stems the way aggressive stemmers do.
    """
    if len(word) > 3 and word[-1] == "s":
        return _en_stem_s(word)
    return word


def _latin_words(lowered: str) -> list[str]:
    if lowered.isascii():
        return lowered.translate(_ASCII_WORD_TRANSLATION).split()
    return _WORD.findall(lowered)


def _ko_stem(word: str) -> str:
    """Iteratively strip trailing josa/eomi; keep the stem at least 2 chars."""
    changed = True
    while changed and len(word) >= 3:
        changed = False
        for s in _KO_SUFFIX_BY_LAST.get(word[-1], ()):
            if len(word) - len(s) >= 2 and word.endswith(s):
                word = word[: -len(s)]
                changed = True
                break
    return word


# Closed-class query operators carry intent but no document subject.  Query analysis
# removes them from both scoring and candidate admission while leaving the index
# lossless.  Otherwise a rare surface form such as a question operator can receive a
# larger IDF contribution than the subject it modifies.
_KO_QUERY_OPERATORS = frozenset(
    {
        "무엇",
        "뭐",
        "무슨",
        "어떤",
        "어떻",
        "어떻게",
        "대해",
        "대한",
        "대하",
        "관련",
        "관한",
        "관하",
    }
)
_KO_QUERY_PARTICLES = frozenset({"의", "은", "는", "이", "가", "을", "를"})
_KO_QUERY_REQUEST_ROOTS = ("설명", "알려", "말해", "말씀", "답변")
_KO_QUERY_REQUEST_TAILS = frozenset(
    {
        "해",
        "해주세요",
        "해주세",
        "해줘",
        "해주십시오",
        "하세요",
        "하시오",
        "주세요",
        "주세",
        "줘",
        "주십시오",
    }
)


class _QueryAnalysis(NamedTuple):
    terms: list[str]
    anchors: frozenset[str]
    restricted: bool


def _hangul_tokens(stem: str) -> list[str]:
    if len(stem) == 1:
        return [stem]
    return [*(stem[i : i + 2] for i in range(len(stem) - 1)), "#" + stem]


def _is_ko_query_request(stem: str) -> bool:
    for root in _KO_QUERY_REQUEST_ROOTS:
        if stem.startswith(root) and stem[len(root) :] in _KO_QUERY_REQUEST_TAILS:
            return True
    return False


def _ko_query_stem(word: str) -> str | None:
    """Return a subject-bearing Korean query stem, or ``None`` for grammar only."""
    stem = _ko_stem(word)
    if stem.endswith("이란") and len(stem) > 3:
        stem = stem[:-2]
    elif stem.endswith("화란") and len(stem) > 3:
        stem = stem[:-1]
    if (
        stem in _KO_QUERY_OPERATORS
        or stem in _KO_QUERY_PARTICLES
        or _is_ko_query_request(stem)
    ):
        return None
    return stem


def _analyze_query(text: str) -> _QueryAnalysis:
    original = tokenize(text)
    lowered = (text or "").lower()
    terms: list[str] = []
    anchors: list[str] = []
    restricted = False

    for word in _latin_words(lowered):
        term = _en_stem_s(word) if len(word) > 3 and word[-1] == "s" else word
        terms.append(term)
        anchors.append(term)
    if not lowered.isascii():
        for run in _CJK_OTHER.findall(lowered):
            run_terms = (
                [run]
                if len(run) == 1
                else [run[i : i + 2] for i in range(len(run) - 1)]
            )
            terms.extend(run_terms)
            anchors.extend(run_terms)
        for run in _HANGUL.findall(lowered):
            original_stem = _ko_stem(run)
            subject_stem = _ko_query_stem(run)
            if subject_stem is None:
                restricted = True
                continue
            run_terms = _hangul_tokens(subject_stem)
            terms.extend(run_terms)
            anchors.extend(run_terms)
            restricted |= subject_stem != original_stem

    if not anchors:
        return _QueryAnalysis(original, frozenset(original), False)
    return _QueryAnalysis(terms, frozenset(anchors), restricted)


def tokenize_query(text: str) -> list[str]:
    """Return subject-bearing query tokens for candidate generation.

    Search scorers use the same subject-bearing terms, so Korean question operators
    cannot dominate ranking through a rare surface form.  Definition copulas use their
    normalized subject form.  An operator-only query safely falls back to the lossless
    tokenizer.
    """
    analysis = _analyze_query(text)
    return [term for term in analysis.terms if term in analysis.anchors]


def _coordinate_query_scores(
    scores: dict[int, float],
    candidates: set[int],
    complete_candidates: set[int],
) -> dict[int, float]:
    """Keep subject candidates, preferring complete Korean word evidence when present."""
    if candidates:
        scores = {doc_id: scores[doc_id] for doc_id in candidates}
    if complete_candidates:
        complete = {
            doc_id: scores[doc_id]
            for doc_id in complete_candidates
            if doc_id in scores
        }
        if complete:
            return complete
    return scores


def tokenize(text: str) -> list[str]:
    """Latin word stems + Hanja/Kana bi-grams + Hangul stem bi-grams (+ stem unigram)."""
    text = (text or "").lower()
    words = _latin_words(text)
    toks = [
        _en_stem_s(word) if len(word) > 3 and word[-1] == "s" else word
        for word in words
    ]
    if text.isascii():
        return toks
    for run in _CJK_OTHER.findall(text):
        toks.append(run) if len(run) == 1 else toks.extend(
            run[i : i + 2] for i in range(len(run) - 1)
        )
    for run in _HANGUL.findall(text):
        st = _ko_stem(run)
        if len(st) == 1:
            toks.append(st)
        else:
            toks.extend(st[i : i + 2] for i in range(len(st) - 1))
            toks.append("#" + st)
    return toks


def _passes(docs):
    """Indexing needs two passes. ``docs`` may be a materialized sequence, or a zero-arg
    callable returning a fresh iterator — which lets the caller *stream* tokenization
    instead of holding every tokenized document in memory at once."""
    return (lambda: iter(docs())) if callable(docs) else (lambda: iter(docs))


def _top_k_scores(scores: dict[int, float], limit: int) -> list[tuple[int, float]]:
    """Rank positive scores by score descending, then document id ascending.

    Positive limits retain only the requested frontier while scanning candidates.
    The negative-limit branch preserves the historical ``sorted_scores[:limit]``
    behavior for callers outside the typed API contract.
    """
    if limit < 0:
        positive = ((i, score) for i, score in scores.items() if score > 0)
        return sorted(positive, key=lambda item: (-item[1], item[0]))[:limit]
    if not limit:
        return []

    frontier: list[tuple[float, int, int]] = []
    for doc_id, score in scores.items():
        if not score > 0:
            continue
        candidate = (score, -doc_id, doc_id)
        if len(frontier) < limit:
            heappush(frontier, candidate)
        elif candidate > frontier[0]:
            heapreplace(frontier, candidate)
    frontier.sort(key=lambda item: (-item[0], item[2]))
    return [(doc_id, score) for score, _negated_id, doc_id in frontier]


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents.

    Queries hit an inverted index: only documents sharing at least one query term
    can score above zero, so scoring the postings union — rather than all *N*
    documents — is exactly score-preserving and turns a full scan into work
    proportional to the matched postings.

    ``docs_tokens`` may be a list, or a zero-arg callable yielding token lists (streamed).
    """

    def __init__(
        self,
        docs_tokens,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _IDF_POW,
    ):
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
            c = Counter(d)
            ids: list[int] = []
            cnt: list[int] = []
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
            doc_ids.append(array("i", ids))
            doc_cnt.append(array("i", cnt))
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
        analysis = _analyze_query(query)
        qtf: dict[str, int] = {}
        for t in analysis.terms:
            qtf[t] = qtf.get(t, 0) + 1
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        for t, qn in qtf.items():
            pd = self._pd.get(t)
            if pd is None:
                continue
            if analysis.restricted and t in analysis.anchors:
                candidates.update(pd)
            if t in analysis.anchors and t.startswith("#"):
                complete_candidates.update(pd)
            pw = self._pw[t]
            if qn == 1:
                for i, w in zip(pd, pw):
                    scores[i] = scores.get(i, 0.0) + w
            else:
                for i, w in zip(pd, pw):
                    scores[i] = scores.get(i, 0.0) + qn * w
        scores = _coordinate_query_scores(scores, candidates, complete_candidates)
        return _top_k_scores(scores, limit)


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

    def __init__(
        self,
        docs,
        weights: dict[str, float],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _IDF_POW,
        evidence_fields: "frozenset[str] | set[str] | None" = None,
    ):
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
        dfe: list[int] = []  # document frequency seen ONLY in evidence fields
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
                c = Counter(toks)
                ids: list[int] = []
                cnt: list[int] = []
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
                f_ids.append(array("i", ids))
                f_cnt.append(array("i", cnt))
            for tid in present:
                df[tid] += 1
            for tid in present_ev:
                dfe[tid] += 1
            doc_ids.append(f_ids)
            doc_cnt.append(f_cnt)
            doc_len.append(flen)
        self.avglen = {
            f: (totals[fi] / self.N if self.N else 0.0)
            for fi, f in enumerate(self.fields)
        }
        # Evidence never deflates a content term's IDF: df counts content only. A term seen
        # solely in evidence has content-df 0, so it takes its IDF from the evidence df.
        idfs = [
            math.log(1 + (self.N - (n or ne) + 0.5) / ((n or ne) + 0.5)) ** idf_pow
            for n, ne in zip(df, dfe)
        ]
        self.idf = dict(zip(terms, idfs))
        self._fw = [self.w[f] for f in self.fields]
        # BM25F sums over the *unique* query terms, so a term's whole contribution to a
        # document — idf * tfw(k1+1)/(k1+tfw) over the weighted fields — depends only on
        # (term, doc). Fold it into the index: a search becomes a sum of precomputed floats.
        k1p1 = self.k1 + 1
        avgl = [self.avglen[f] or 1 for f in self.fields]
        self._pd: dict[str, array] = {}
        self._pw: dict[str, array] = {}
        # State for `update_evidence`. A term seen in any content field has a fixed df, so
        # its IDF can never move. A term seen ONLY in evidence takes its IDF from the
        # evidence df, which grows as documents remember more — and every one of ITS
        # postings is evidence-derived. Keeping their tfw lets an update recompute exactly.
        ev_tfw: dict[str, array] = {}
        has_ev = bool(self.evidence_fields)
        for i in range(self.N):
            flen = doc_len[i]
            fnorm = [
                1.0
                if self._is_ev[fi]
                else 1 - self.b + self.b * (flen[fi] or 1) / avgl[fi]
                for fi in range(nf)
            ]
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
                if has_ev and not df[tid]:
                    ev_tfw.setdefault(t, array("d")).append(tfw)
            doc_ids[i] = doc_cnt[i] = None  # release as consumed
        if self.evidence_fields:
            self._idf_pow = idf_pow
            self._totals = totals
            self._dfe = {
                terms[tid]: dfe[tid] for tid in range(len(terms)) if not df[tid]
            }
            self._ev_tfw = ev_tfw

    def _idf_of(self, n: int) -> float:
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** self._idf_pow

    def update_evidence(self, i: int, before: dict, after: dict) -> None:
        """Fold changed evidence for document ``i`` into the live index — grow or shrink.

        ``before``/``after`` are ``{field: tokens}`` for that document; the content fields
        must be the tokens it was built from. The result is the index a full rebuild would
        have produced — see ``tests/test_incremental.py``.

        ``N``, the content df and every content term's IDF are fixed by construction, because
        evidence is excluded from document frequency. The one thing that does move globally is
        the IDF of a term seen *only* in evidence, which comes from the evidence df that this
        update grows or shrinks — but all of that term's postings are evidence-derived, so the
        work is bounded by the memory rather than by the corpus. A shrink that leaves a term
        with no evidence holder erases it from the index entirely, exactly as a rebuild would.
        """
        if not self.evidence_fields:
            raise RuntimeError(
                "BM25F was built without evidence_fields; nothing to update"
            )
        k1, k1p1 = self.k1, self.k1 + 1
        ev = [fi for fi, f in enumerate(self.fields) if self._is_ev[fi]]

        # 1. evidence df: +1 for terms this document newly holds, -1 for terms it dropped
        held_before = {t for fi in ev for t in before.get(self.fields[fi], ())}
        held_after = {t for fi in ev for t in after.get(self.fields[fi], ())}
        dirty = set()
        for t in held_after - held_before:
            if t in self.idf and t not in self._dfe:
                continue  # a content term: its df, and so its IDF, is fixed
            self._dfe[t] = self._dfe.get(t, 0) + 1
            dirty.add(t)
        for t in held_before - held_after:
            if t in self._dfe:
                self._dfe[t] -= 1
                dirty.add(t)

        # 2. re-derive the IDF of evidence-only terms, and their postings from the kept tfw
        #    (a term whose evidence df hit 0 is erased below, once this doc's entry is gone)
        for t in dirty:
            if self._dfe.get(t, 0) <= 0:
                continue
            self.idf[t] = idf = self._idf_of(self._dfe[t])
            pw, tfws = self._pw.get(t), self._ev_tfw.get(t)
            if pw is None:
                continue
            for k, tfw in enumerate(tfws):
                pw[k] = idf * tfw * k1p1 / (k1 + tfw)

        # 3. re-derive this document's own contributions (its evidence tf changed)
        tfw_i: dict[str, float] = {}
        for fi, f in enumerate(self.fields):
            toks = after.get(f, ())
            c: dict[str, int] = {}
            for t in toks:
                c[t] = c.get(t, 0) + 1
            nrm = (
                1.0
                if self._is_ev[fi]
                else (1 - self.b + self.b * (len(toks) or 1) / (self.avglen[f] or 1))
            )
            wf = self._fw[fi]
            for t, tf in c.items():
                tfw_i[t] = tfw_i.get(t, 0.0) + wf * tf / nrm
        for t, tfw in tfw_i.items():
            if not tfw:
                continue
            w = self.idf[t] * tfw * k1p1 / (k1 + tfw)
            pd = self._pd.get(t)
            if pd is None:
                self._pd[t] = array("i", [i])
                self._pw[t] = array("d", [w])
                if t in self._dfe:
                    self._ev_tfw[t] = array("d", [tfw])
                continue
            k = bisect_left(pd, i)
            if k < len(pd) and pd[k] == i:
                self._pw[t][k] = w
                if t in self._dfe:
                    self._ev_tfw[t][k] = tfw
            else:
                pd.insert(k, i)
                self._pw[t].insert(k, w)
                if t in self._dfe:
                    self._ev_tfw[t].insert(k, tfw)
        # terms this document no longer holds in ANY field: delete its posting entry
        for t in held_before - held_after:
            if t in tfw_i:
                continue  # still held via a content field
            pd = self._pd.get(t)
            if pd is None:
                continue
            k = bisect_left(pd, i)
            if k < len(pd) and pd[k] == i:
                pd.pop(k)
                self._pw[t].pop(k)
                if t in self._ev_tfw:
                    self._ev_tfw[t].pop(k)
            if not pd and self._dfe.get(t, 0) <= 0:
                # this document was the term's last holder: erase it, as a rebuild would
                del self._pd[t], self._pw[t], self.idf[t]
                self._ev_tfw.pop(t, None)
                self._dfe.pop(t, None)

        # 4. evidence is not length-normalized, so avglen is unused for it — keep it honest anyway
        for fi in ev:
            f = self.fields[fi]
            self._totals[fi] += len(after.get(f, ())) - len(before.get(f, ()))
            self.avglen[f] = self._totals[fi] / self.N if self.N else 0.0

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
        return self._score(dict.fromkeys(q_tokens), i)

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        analysis = _analyze_query(query)
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        for t in dict.fromkeys(analysis.terms):
            pd = self._pd.get(t)
            if pd is None:
                continue
            if analysis.restricted and t in analysis.anchors:
                candidates.update(pd)
            if t in analysis.anchors and t.startswith("#"):
                complete_candidates.update(pd)
            for i, c in zip(pd, self._pw[t]):
                scores[i] = scores.get(i, 0.0) + c
        scores = _coordinate_query_scores(scores, candidates, complete_candidates)
        return _top_k_scores(scores, limit)


def _mutable_doc_id(doc_id: int) -> int:
    if type(doc_id) is not int or doc_id < 0:
        raise ValueError("doc_id must be a non-negative int")
    return doc_id


class _MutableBM25:
    """Opt-in exact mutable counterpart to :class:`BM25`.

    New document ids must increase monotonically and are never reused. Updates retain an
    existing id. Raw term frequencies and lengths change eagerly; query-term contributions
    are derived lazily, so a corpus mutation does not rescore unrelated vocabulary.
    """

    def __init__(
        self,
        docs,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _IDF_POW,
    ):
        self.k1, self.b, self._idf_pow = k1, b, idf_pow
        self.N = 0
        self._total_len = 0
        self._max_doc_id = -1
        self._docs: dict[int, tuple[int, dict[str, int]]] = {}
        self._terms: dict[str, str] = {}
        self._df: dict[str, int] = {}
        self._postings: dict[str, dict[int, int]] = {}
        self._weight_cache: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}
        self._mutation_version = 0
        self.upsert_many(docs)
        self._weight_cache.clear()
        self._mutation_version = 0

    @property
    def avgdl(self) -> float:
        return self._total_len / self.N if self.N else 0.0

    @property
    def idf(self) -> dict[str, float]:
        return {term: self._idf(term) for term in self._postings}

    def _freeze(
        self, tokens, staged_terms: dict[str, str]
    ) -> tuple[int, dict[str, int]]:
        counts: dict[str, int] = {}
        length = 0
        for term in tokens:
            if type(term) is not str:
                raise TypeError("tokens must contain only str values")
            canonical = self._terms.get(term)
            if canonical is None:
                canonical = staged_terms.setdefault(term, term)
            length += 1
            counts[canonical] = counts.get(canonical, 0) + 1
        return length, counts

    def _idf(self, term: str) -> float:
        n = self._df[term]
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** self._idf_pow

    def _term_weights(self, term: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        canonical = self._terms.get(term)
        if canonical is None:
            return (), ()
        cached = self._weight_cache.get(canonical)
        if cached is not None:
            return cached
        posting = self._postings[canonical]

        avg = self.avgdl or 1.0
        idf = self._idf(canonical)
        k1p1 = self.k1 + 1
        ids: list[int] = []
        weights: list[float] = []
        for doc_id in sorted(posting):
            tf = posting[doc_id]
            doc_len = self._docs[doc_id][0]
            norm = self.k1 * (1 - self.b + self.b * (doc_len or 1) / avg)
            weight = idf * k1p1 * tf / (tf + norm)
            ids.append(doc_id)
            weights.append(weight)
        result = tuple(ids), tuple(weights)
        self._weight_cache[canonical] = result
        return result

    def _remove(self, doc_id: int, frozen: tuple[int, dict[str, int]]) -> None:
        length, counts = frozen
        self._total_len -= length
        for term in counts:
            posting = self._postings[term]
            del posting[doc_id]
            self._df[term] -= 1
            if not posting:
                del self._postings[term]
                del self._df[term]

    def _add(self, doc_id: int, frozen: tuple[int, dict[str, int]]) -> None:
        length, counts = frozen
        self._total_len += length
        for term, tf in counts.items():
            self._postings.setdefault(term, {})[doc_id] = tf
            self._df[term] = self._df.get(term, 0) + 1

    def upsert_many(self, docs) -> int:
        """Atomically insert or replace documents; return the number changed.

        The complete batch is frozen and validated before the first index mutation. New
        ids must be strictly increasing in batch order; active ids may appear as updates.
        """
        prepared = []
        staged_terms: dict[str, str] = {}
        seen: set[int] = set()
        next_max = self._max_doc_id
        for doc_id, tokens in docs:
            doc_id = _mutable_doc_id(doc_id)
            if doc_id in seen:
                raise ValueError(f"duplicate batch doc_id {doc_id}")
            seen.add(doc_id)
            before = self._docs.get(doc_id)
            if before is None:
                if doc_id <= next_max:
                    raise ValueError(
                        "new doc_id must be greater than every previously assigned id"
                    )
                next_max = doc_id
            prepared.append((doc_id, self._freeze(tokens, staged_terms), before))

        changed = 0
        touched_terms: set[str] = set()
        for doc_id, frozen, before in prepared:
            if before == frozen:
                continue
            if before is not None:
                touched_terms.update(before[1])
            touched_terms.update(frozen[1])
            if before is None:
                self.N += 1
                self._max_doc_id = doc_id
            else:
                self._remove(doc_id, before)
            self._docs[doc_id] = frozen
            self._add(doc_id, frozen)
            changed += 1
        if changed:
            self._terms.update(staged_terms)
            for term in touched_terms:
                if term not in self._postings:
                    self._terms.pop(term, None)
            self._weight_cache.clear()
            self._mutation_version += 1
        return changed

    def upsert(self, doc_id: int, tokens) -> bool:
        """Insert or replace one document; return ``False`` for a scoring no-op."""
        return bool(self.upsert_many(((doc_id, tokens),)))

    def delete_many(self, doc_ids) -> int:
        """Atomically delete active ids; return the number deleted."""
        prepared = []
        seen: set[int] = set()
        for doc_id in doc_ids:
            doc_id = _mutable_doc_id(doc_id)
            if doc_id in seen:
                raise ValueError(f"duplicate batch doc_id {doc_id}")
            seen.add(doc_id)
            before = self._docs.get(doc_id)
            if before is not None:
                prepared.append((doc_id, before))

        touched_terms: set[str] = set()
        for doc_id, before in prepared:
            touched_terms.update(before[1])
            del self._docs[doc_id]
            self._remove(doc_id, before)
        changed = len(prepared)
        if changed:
            self.N -= changed
            for term in touched_terms:
                if term not in self._postings:
                    self._terms.pop(term, None)
            self._weight_cache.clear()
            self._mutation_version += 1
        return changed

    def delete(self, doc_id: int) -> bool:
        """Delete an active document; deleted ids remain reserved."""
        return bool(self.delete_many((doc_id,)))

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if "_terms" in state:
            return
        self._terms = {term: term for term in self._postings}
        self._docs = {
            doc_id: (
                length,
                {self._terms[term]: tf for term, tf in counts.items()},
            )
            for doc_id, (length, counts) in self._docs.items()
        }
        self._df = {self._terms[term]: value for term, value in self._df.items()}
        self._postings = {
            self._terms[term]: posting for term, posting in self._postings.items()
        }
        self._weight_cache = {
            self._terms[term]: cached
            for term, cached in self._weight_cache.items()
            if term in self._terms
        }

    def score(self, q_tokens: list[str], doc_id: int) -> float:
        doc_id = _mutable_doc_id(doc_id)
        score = 0.0
        for term in q_tokens:
            ids, weights = self._term_weights(term)
            position = bisect_left(ids, doc_id)
            if position < len(ids) and ids[position] == doc_id:
                score += weights[position]
        return score

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        analysis = _analyze_query(query)
        qtf: dict[str, int] = {}
        for term in analysis.terms:
            qtf[term] = qtf.get(term, 0) + 1
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        for term, query_count in qtf.items():
            ids, weights = self._term_weights(term)
            if analysis.restricted and term in analysis.anchors:
                candidates.update(ids)
            if term in analysis.anchors and term.startswith("#"):
                complete_candidates.update(ids)
            if query_count == 1:
                for doc_id, weight in zip(ids, weights):
                    scores[doc_id] = scores.get(doc_id, 0.0) + weight
            else:
                for doc_id, weight in zip(ids, weights):
                    scores[doc_id] = scores.get(doc_id, 0.0) + query_count * weight
        scores = _coordinate_query_scores(scores, candidates, complete_candidates)
        return _top_k_scores(scores, limit)


class _MutableBM25F:
    """Opt-in exact mutable counterpart to :class:`BM25F`.

    Content and evidence document frequencies are retained independently. Consequently a
    term can move between evidence-only and content-backed IDF semantics without rebuilding
    the corpus. Per-term contributions are materialized only when that term is queried.
    """

    def __init__(
        self,
        docs,
        weights: dict[str, float],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        idf_pow: float = _IDF_POW,
        evidence_fields: "frozenset[str] | set[str] | None" = None,
    ):
        self.k1, self.b, self._idf_pow = k1, b, idf_pow
        self.fields = list(weights)
        self.w = dict(weights)
        self._fw = [self.w[field] for field in self.fields]
        self.evidence_fields = frozenset(evidence_fields or ())
        self._is_ev = [field in self.evidence_fields for field in self.fields]
        self.N = 0
        self._totals = [0] * len(self.fields)
        self._max_doc_id = -1
        self._docs: dict[int, tuple[tuple[int, ...], tuple[dict[str, int], ...]]] = {}
        self._terms: dict[str, str] = {}
        self._df: dict[str, int] = {}
        self._dfe: dict[str, int] = {}
        self._postings: dict[str, dict[int, tuple[int, ...]]] = {}
        self._weight_cache: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}
        self._mutation_version = 0
        self.upsert_many(docs)
        self._weight_cache.clear()
        self._mutation_version = 0

    @property
    def avglen(self) -> dict[str, float]:
        return {
            field: (self._totals[index] / self.N if self.N else 0.0)
            for index, field in enumerate(self.fields)
        }

    @property
    def idf(self) -> dict[str, float]:
        return {term: self._idf(term) for term in self._postings}

    def _freeze(
        self, fields, staged_terms: dict[str, str]
    ) -> tuple[tuple[int, ...], tuple[dict[str, int], ...]]:
        lengths: list[int] = []
        counts_by_field: list[dict[str, int]] = []
        for field in self.fields:
            counts: dict[str, int] = {}
            length = 0
            for term in fields.get(field, ()):
                if type(term) is not str:
                    raise TypeError("tokens must contain only str values")
                canonical = self._terms.get(term)
                if canonical is None:
                    canonical = staged_terms.setdefault(term, term)
                length += 1
                counts[canonical] = counts.get(canonical, 0) + 1
            lengths.append(length)
            counts_by_field.append(counts)
        return tuple(lengths), tuple(counts_by_field)

    def _presence(
        self, frozen: tuple[tuple[int, ...], tuple[dict[str, int], ...]]
    ) -> tuple[set[str], set[str]]:
        _lengths, counts_by_field = frozen
        content: set[str] = set()
        evidence: set[str] = set()
        for index, counts in enumerate(counts_by_field):
            (evidence if self._is_ev[index] else content).update(counts)
        return content, evidence

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0) or self._dfe[term]
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) ** self._idf_pow

    def _term_weights(self, term: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        canonical = self._terms.get(term)
        if canonical is None:
            return (), ()
        cached = self._weight_cache.get(canonical)
        if cached is not None:
            return cached
        posting = self._postings[canonical]

        avglen = [
            self._totals[index] / self.N if self.N else 0.0
            for index in range(len(self.fields))
        ]
        avgl = [length or 1 for length in avglen]
        idf = self._idf(canonical)
        k1p1 = self.k1 + 1
        ids: list[int] = []
        weights: list[float] = []
        for doc_id in sorted(posting):
            lengths, _counts_by_field = self._docs[doc_id]
            term_frequencies = posting[doc_id]
            tfw = 0.0
            for index, tf in enumerate(term_frequencies):
                if not tf:
                    continue
                norm = (
                    1.0
                    if self._is_ev[index]
                    else 1 - self.b + self.b * (lengths[index] or 1) / avgl[index]
                )
                tfw = tfw + self._fw[index] * tf / norm
            if not tfw:
                continue
            weight = idf * tfw * k1p1 / (self.k1 + tfw)
            ids.append(doc_id)
            weights.append(weight)
        result = tuple(ids), tuple(weights)
        self._weight_cache[canonical] = result
        return result

    def _remove(
        self,
        doc_id: int,
        frozen: tuple[tuple[int, ...], tuple[dict[str, int], ...]],
    ) -> None:
        lengths, counts_by_field = frozen
        for index, length in enumerate(lengths):
            self._totals[index] -= length
        content, evidence = self._presence(frozen)
        for term in content:
            self._df[term] -= 1
            if not self._df[term]:
                del self._df[term]
        for term in evidence:
            self._dfe[term] -= 1
            if not self._dfe[term]:
                del self._dfe[term]
        for term in content | evidence:
            posting = self._postings[term]
            del posting[doc_id]
            if not posting:
                del self._postings[term]

    def _add(
        self,
        doc_id: int,
        frozen: tuple[tuple[int, ...], tuple[dict[str, int], ...]],
    ) -> None:
        lengths, counts_by_field = frozen
        for index, length in enumerate(lengths):
            self._totals[index] += length
        content, evidence = self._presence(frozen)
        for term in content:
            self._df[term] = self._df.get(term, 0) + 1
        for term in evidence:
            self._dfe[term] = self._dfe.get(term, 0) + 1

        term_frequencies: dict[str, list[int]] = {}
        for index, counts in enumerate(counts_by_field):
            for term, tf in counts.items():
                term_frequencies.setdefault(term, [0] * len(self.fields))[index] = tf
        for term, frequencies in term_frequencies.items():
            self._postings.setdefault(term, {})[doc_id] = tuple(frequencies)

    def upsert_many(self, docs) -> int:
        """Atomically insert or replace fielded documents; return the number changed.

        The complete batch is frozen and validated before the first index mutation. New
        ids must be strictly increasing in batch order; active ids may appear as updates.
        """
        prepared = []
        staged_terms: dict[str, str] = {}
        seen: set[int] = set()
        next_max = self._max_doc_id
        for doc_id, fields in docs:
            doc_id = _mutable_doc_id(doc_id)
            if doc_id in seen:
                raise ValueError(f"duplicate batch doc_id {doc_id}")
            seen.add(doc_id)
            before = self._docs.get(doc_id)
            if before is None:
                if doc_id <= next_max:
                    raise ValueError(
                        "new doc_id must be greater than every previously assigned id"
                    )
                next_max = doc_id
            prepared.append((doc_id, self._freeze(fields, staged_terms), before))

        changed = 0
        touched_terms: set[str] = set()
        for doc_id, frozen, before in prepared:
            if before == frozen:
                continue
            if before is not None:
                content, evidence = self._presence(before)
                touched_terms.update(content)
                touched_terms.update(evidence)
            content, evidence = self._presence(frozen)
            touched_terms.update(content)
            touched_terms.update(evidence)
            if before is None:
                self.N += 1
                self._max_doc_id = doc_id
            else:
                self._remove(doc_id, before)
            self._docs[doc_id] = frozen
            self._add(doc_id, frozen)
            changed += 1
        if changed:
            self._terms.update(staged_terms)
            for term in touched_terms:
                if term not in self._postings:
                    self._terms.pop(term, None)
            self._weight_cache.clear()
            self._mutation_version += 1
        return changed

    def upsert(self, doc_id: int, fields) -> bool:
        """Insert or replace one fielded document; return ``False`` for a scoring no-op."""
        return bool(self.upsert_many(((doc_id, fields),)))

    def delete_many(self, doc_ids) -> int:
        """Atomically delete active ids; return the number deleted."""
        prepared = []
        seen: set[int] = set()
        for doc_id in doc_ids:
            doc_id = _mutable_doc_id(doc_id)
            if doc_id in seen:
                raise ValueError(f"duplicate batch doc_id {doc_id}")
            seen.add(doc_id)
            before = self._docs.get(doc_id)
            if before is not None:
                prepared.append((doc_id, before))

        touched_terms: set[str] = set()
        for doc_id, before in prepared:
            content, evidence = self._presence(before)
            touched_terms.update(content)
            touched_terms.update(evidence)
            del self._docs[doc_id]
            self._remove(doc_id, before)
        changed = len(prepared)
        if changed:
            self.N -= changed
            for term in touched_terms:
                if term not in self._postings:
                    self._terms.pop(term, None)
            self._weight_cache.clear()
            self._mutation_version += 1
        return changed

    def delete(self, doc_id: int) -> bool:
        """Delete an active document; deleted ids remain reserved."""
        return bool(self.delete_many((doc_id,)))

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if "_terms" in state:
            return
        self._terms = {term: term for term in self._postings}
        self._docs = {
            doc_id: (
                lengths,
                tuple(
                    {self._terms[term]: tf for term, tf in counts.items()}
                    for counts in counts_by_field
                ),
            )
            for doc_id, (lengths, counts_by_field) in self._docs.items()
        }
        self._df = {self._terms[term]: value for term, value in self._df.items()}
        self._dfe = {self._terms[term]: value for term, value in self._dfe.items()}
        self._postings = {
            self._terms[term]: posting for term, posting in self._postings.items()
        }
        self._weight_cache = {
            self._terms[term]: cached
            for term, cached in self._weight_cache.items()
            if term in self._terms
        }

    def score(self, q_tokens: list[str], doc_id: int) -> float:
        doc_id = _mutable_doc_id(doc_id)
        score = 0.0
        for term in dict.fromkeys(q_tokens):
            ids, weights = self._term_weights(term)
            position = bisect_left(ids, doc_id)
            if position < len(ids) and ids[position] == doc_id:
                score += weights[position]
        return score

    def search(self, query: str, *, limit: int = 20) -> list[tuple[int, float]]:
        analysis = _analyze_query(query)
        scores: dict[int, float] = {}
        candidates: set[int] = set()
        complete_candidates: set[int] = set()
        for term in dict.fromkeys(analysis.terms):
            ids, weights = self._term_weights(term)
            if analysis.restricted and term in analysis.anchors:
                candidates.update(ids)
            if term in analysis.anchors and term.startswith("#"):
                complete_candidates.update(ids)
            for doc_id, weight in zip(ids, weights):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight
        scores = _coordinate_query_scores(scores, candidates, complete_candidates)
        return _top_k_scores(scores, limit)
