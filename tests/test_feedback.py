"""Memory as an *evidence* field — the properties that make the earlier version's
false positive impossible.

Our retracted design injected the remembered query into the document body. That deflated
the IDF of query vocabulary corpus-wide, which looked like memory but moved even queries
whose relevant documents remembered nothing. These tests pin the three properties that
rule that out.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, Feedback, build_inmemory  # noqa: E402
from omnifuse.text import BM25F, tokenize  # noqa: E402

CHUNKS = [
    Chunk("trial", title="임상 보고서", text="코호트에서 ldl 수치가 12개월간 감소했다"),
    Chunk("weather", title="기상 기록", text="3월 강수량이 평년을 웃돌았다"),
]


def _ids(of, q):
    return [c.id for c, _ in of.retrieve(q)]


def test_cold_store_is_bit_identical_to_no_feedback():
    q = "ldl 감소"
    plain = build_inmemory([], [], CHUNKS)
    cold = build_inmemory([], [], CHUNKS, feedback=Feedback())
    a = [(c.id, round(s, 12)) for c, s in plain.retrieve(q)]
    b = [(c.id, round(s, 12)) for c, s in cold.retrieve(q)]
    assert a == b and a


def test_evidence_never_changes_the_idf_of_a_content_term():
    """The bug we shipped: remembered text raised df and deflated IDF corpus-wide."""
    plain = build_inmemory([], [], CHUNKS)
    fb = Feedback()
    fb.remember("강수량 감소 추세", ["trial"])       # '감소' also occurs in trial's body
    warm = build_inmemory([], [], CHUNKS, feedback=fb)
    for term in ("#감소", "ldl"):
        assert plain.vector._bm25.idf.get(term) == warm.vector._bm25.idf.get(term)


def test_a_remembered_query_lifts_the_chunk_that_answered_it():
    # "스타틴" occurs in no chunk body — only in the remembered query.
    fb = Feedback()
    fb.remember("스타틴이 콜레스테롤을 낮추나요", ["trial"])
    warm = build_inmemory([], [], CHUNKS, feedback=fb)
    assert _ids(warm, "스타틴 콜레스테롤 효과")[0] == "trial"

    cold = build_inmemory([], [], CHUNKS)
    assert "trial" not in _ids(cold, "스타틴")       # unreachable by content alone


def test_memory_cannot_move_a_query_unrelated_to_what_was_remembered():
    q = "강수량"
    cold = build_inmemory([], [], CHUNKS)
    fb = Feedback()
    fb.remember("스타틴이 콜레스테롤을 낮추나요", ["trial"])
    warm = build_inmemory([], [], CHUNKS, feedback=fb)
    a = [(c.id, round(s, 12)) for c, s in cold.retrieve(q)]
    b = [(c.id, round(s, 12)) for c, s in warm.retrieve(q)]
    assert a == b


def test_evidence_field_is_not_length_normalized():
    """Remembering a second query must not dilute the first."""
    one, many = Feedback(), Feedback()
    one.remember("스타틴 효과", ["trial"])
    many.remember("스타틴 효과", ["trial"])
    for extra in ("무관한 질문 하나", "또 다른 무관한 질문", "세번째 무관한 질문"):
        many.remember(extra, ["trial"])
    q = "스타틴 효과"
    s1 = dict((c.id, s) for c, s in build_inmemory([], [], CHUNKS, feedback=one).retrieve(q))
    s2 = dict((c.id, s) for c, s in build_inmemory([], [], CHUNKS, feedback=many).retrieve(q))
    assert s1["trial"] == s2["trial"]


def test_bm25f_evidence_fields_keep_df_out():
    docs = [
        {"body": tokenize("alpha beta"), "memory": tokenize("gamma")},
        {"body": tokenize("alpha"), "memory": tokenize("gamma")},
    ]
    plain = BM25F([{"body": d["body"], "memory": []} for d in docs],
                  {"body": 1.0, "memory": 1.0}, evidence_fields={"memory"})
    withmem = BM25F(docs, {"body": 1.0, "memory": 1.0}, evidence_fields={"memory"})
    assert plain.idf["alpha"] == withmem.idf["alpha"]    # content IDF untouched
    assert "gamma" in withmem.idf                        # evidence-only term still usable


def test_observe_ranked_only_remembers_confirmed_chunks():
    fb = Feedback()
    fb.observe_ranked(["trial", "weather"], relevant={"trial"}, query="ldl 임상")
    assert fb.queries("trial") == ["ldl 임상"]
    assert fb.queries("weather") == []
    assert len(fb) == 1


def test_remember_is_idempotent_and_skips_blank():
    fb = Feedback()
    fb.remember("같은 질의", ["d"])
    fb.remember("같은 질의", ["d"])
    fb.remember("   ", ["d"])
    assert fb.queries("d") == ["같은 질의"]


def test_round_trip(tmp_path):
    fb = Feedback()
    fb.remember("한국어 질의", ["d1"])
    fb.remember("second", ["d1", "d2"])
    p = tmp_path / "fb.json"
    fb.save(p)
    back = Feedback.load(p)
    assert back.queries("d1") == ["한국어 질의", "second"]
    assert back.queries("d2") == ["second"]
