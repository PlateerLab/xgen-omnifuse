import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Memory  # noqa: E402


def _mem():
    m = Memory()
    m.add_fact("itemA", "instanceOf", "Category")
    m.add_fact("itemB", "instanceOf", "Category")
    m.remember("itemA has a credit limit of 5 billion won", triples=[("itemA", "limit", "5B")])
    return m


def test_remember_recall():
    m = _mem()
    r = m.recall("itemA limit")
    assert any("limit" in x for x in r.relations)
    assert m.stats() == {"facts": 3, "notes": 1}


def test_recall_enumeration():
    m = _mem()
    r = m.recall("list all Category")
    assert "itemA" in r.class_seed and "itemB" in r.class_seed


def test_auto_link_note_to_entity():
    m = Memory()
    m.add_fact("itemA", "instanceOf", "Category")
    cid = m.remember("note about itemA performance")  # auto-links to itemA
    assert m._chunks[-1][2] == ["itemA"]
    assert cid


def test_save_load(tmp_path):
    m = _mem()
    p = tmp_path / "mem.jsonl"
    m.save(str(p))
    m2 = Memory.load(str(p))
    assert m2.stats() == {"facts": 3, "notes": 1}
    assert any("limit" in x for x in m2.recall("itemA limit").relations)
