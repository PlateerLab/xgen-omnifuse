"""Memory — the queries a chunk was confirmed to answer become part of what it is about.

Two properties the mechanism must have:
  * a cold store (no feedback) ranks **bit-identically** to one built without feedback,
    so memory can never regress a system that has not been used yet;
  * a remembered query lifts the chunk that answered it for a *different but related*
    query, without that chunk having the query's vocabulary in its body.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, Feedback, build_inmemory  # noqa: E402

CHUNKS = [
    Chunk("stat", title="Trial report", text="the cohort showed reduced ldl over twelve months"),
    Chunk("noise", title="Weather log", text="rainfall in march exceeded the seasonal average"),
]


def _ids(of, q):
    return [c.id for c, _ in of.retrieve(q)]


def test_cold_store_is_bit_identical_to_no_feedback():
    q = "ldl cohort"
    plain = build_inmemory([], [], CHUNKS)
    cold = build_inmemory([], [], CHUNKS, feedback=Feedback())
    a = [(c.id, round(s, 12)) for c, s in plain.retrieve(q)]
    b = [(c.id, round(s, 12)) for c, s in cold.retrieve(q)]
    assert a == b and a


def test_remembered_query_lifts_the_chunk_that_answered_it():
    # "statin" appears in neither chunk's body — only in the remembered query.
    q = "do statins reduce cholesterol"
    cold = build_inmemory([], [], CHUNKS)
    assert "stat" not in _ids(cold, "statin")  # unreachable by content alone

    fb = Feedback()
    fb.remember("statins and cholesterol outcomes", ["stat"])
    warm = build_inmemory([], [], CHUNKS, feedback=fb)
    assert _ids(warm, q)[0] == "stat"


def test_observe_ranked_only_remembers_confirmed_documents():
    fb = Feedback()
    fb.observe_ranked(["stat", "noise"], relevant={"stat"}, query="ldl trial")
    assert fb.queries("stat") == ["ldl trial"]
    assert fb.queries("noise") == []
    assert len(fb) == 1


def test_remember_is_idempotent_and_skips_blank_queries():
    fb = Feedback()
    fb.remember("same query", ["d"])
    fb.remember("same query", ["d"])
    fb.remember("   ", ["d"])
    assert fb.queries("d") == ["same query"]


def test_round_trip(tmp_path):
    fb = Feedback()
    fb.remember("한국어 질의", ["d1"])
    fb.remember("second", ["d1", "d2"])
    p = tmp_path / "fb.json"
    fb.save(p)
    back = Feedback.load(p)
    assert back.queries("d1") == ["한국어 질의", "second"]
    assert back.queries("d2") == ["second"]
    assert len(back) == 2


def test_unremembered_chunk_has_empty_memory():
    fb = Feedback()
    fb.remember("q", ["stat"])
    assert fb.text("noise") == ""
    assert not Feedback()
