"""Incremental memory: `remember()` must produce the index a full rebuild would.

This is the whole justification for the evidence-field design. Because evidence is excluded
from document frequency, N and every content term's IDF are fixed, so remembering a query
is a local edit. The one coupling — a term seen only in evidence takes its IDF from the
evidence df — touches only documents that remember that term.

The bar is BIT equality of the postings, not "close enough": a float that drifts is a
scoring difference we would not notice until a benchmark moved.
"""
from omnifuse import Feedback, build_inmemory

CHUNKS = [
    {"id": "a", "title": "말리부 해변", "text": "캘리포니아의 해안 도로를 따라 이어진다."},
    {"id": "b", "title": "그리스", "text": "수도는 아테네이며 지중해에 면한다."},
    {"id": "c", "title": "내 친구의 집은 어디인가", "text": "1996년 개봉한 영화."},
    {"id": "d", "title": "Athens", "text": "The capital city of Greece, on the Attic peninsula."},
    {"id": "e", "title": "", "text": "제목이 없는 문서. 본문만 존재한다."},
]
MEMORIES = [
    ("그리스의 수도는 어디인가?", ["b"]),
    ("아테네는 어느 나라의 수도인가요?", ["b", "d"]),
    ("캘리포니아 해안 도로", ["a"]),
    ("그리스의 수도는 어디인가?", ["b"]),          # the same query twice: tf grows, df must not
    ("제목 없는 문서를 찾아줘", ["e"]),
    ("what is the capital of Greece", ["d"]),
]


def _rebuild(memories):
    fb = Feedback()
    for q, ids in memories:
        fb.remember(q, ids)
    return build_inmemory([], [], CHUNKS, feedback=fb)


def _incremental(memories):
    of = build_inmemory([], [], CHUNKS, feedback=Feedback())
    for q, ids in memories:
        of.remember(q, ids)
    return of


def _assert_same_index(a, b):
    x, y = a.vector._bm25, b.vector._bm25
    assert set(x._pd) == set(y._pd)
    assert x.idf == y.idf
    for t in y._pd:
        assert list(x._pd[t]) == list(y._pd[t]), t
        assert list(x._pw[t]) == list(y._pw[t]), t  # bit equality


def test_incremental_equals_full_rebuild():
    _assert_same_index(_incremental(MEMORIES), _rebuild(MEMORIES))


def test_incremental_equals_rebuild_at_every_prefix():
    """Order must not matter to the endpoint, and no prefix may drift."""
    for k in range(1, len(MEMORIES) + 1):
        _assert_same_index(_incremental(MEMORIES[:k]), _rebuild(MEMORIES[:k]))


def test_empty_feedback_scores_like_no_feedback():
    """Opting into incremental memory must cost nothing while the memory is empty."""
    warm = build_inmemory([], [], CHUNKS, feedback=Feedback())
    cold = build_inmemory([], [], CHUNKS)
    q = "그리스의 수도"
    assert [c.id for c, _ in warm.retrieve(q, limit=5)] == [c.id for c, _ in cold.retrieve(q, limit=5)]


def test_remember_lifts_the_remembering_chunk():
    of = build_inmemory([], [], CHUNKS, feedback=Feedback())
    q = "그리스의 수도는 어디인가?"
    of.remember(q, ["b"])
    assert of.retrieve(q, limit=3)[0][0].id == "b"


def test_remembering_cannot_move_a_chunk_that_remembers_nothing():
    """Evidence is excluded from df, so the collection's IDF is untouched. Any movement in
    an uncovered chunk's absolute score would be a corpus-wide scoring artifact."""
    of = build_inmemory([], [], CHUNKS, feedback=Feedback())
    bm = of.vector._bm25
    q = "캘리포니아 해안 도로"
    before = bm.score(["지중해", "아테네"], 1)
    of.remember("제목 없는 문서를 찾아줘", ["e"])
    assert bm.score(["지중해", "아테네"], 1) == before
    of.remember(q, ["a"])
    assert bm.score(["지중해", "아테네"], 1) == before


def test_remember_without_feedback_is_a_clear_error():
    of = build_inmemory([], [], CHUNKS)
    try:
        of.remember("q", ["a"])
    except RuntimeError as e:
        assert "feedback=Feedback()" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_remember_survives_save_load(tmp_path):
    """A persisted index must stay teachable — otherwise `remember` is useless in a service."""
    from omnifuse import load_index, save_index
    of = build_inmemory([], [], CHUNKS, feedback=Feedback())
    of.remember("그리스의 수도는 어디인가?", ["b"])
    p = tmp_path / "ix.pkl"
    save_index(of, p)
    back = load_index(p)
    back.remember("아테네는 어느 나라의 수도인가요?", ["b", "d"])
    _assert_same_index(back, _rebuild(MEMORIES[:2]))
