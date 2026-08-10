import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Vault  # noqa: E402


def _vault():
    v = Vault()
    v.fuse(facts=[("itemA", "instanceOf", "Category"), ("itemB", "instanceOf", "Category")])
    v.fuse("itemA has a credit limit of 5 billion", facts=[("itemA", "limit", "5B")])
    return v


def test_fuse_surface():
    v = _vault()
    r = v.surface("itemA limit")
    assert any("limit" in x for x in r.relations)
    assert v.stats() == {"facts": 3, "notes": 1, "entities": 4}


def test_fuse_on_write_dedup():
    v = Vault()
    v.fuse(facts=[("a", "p", "b")])
    v.fuse(facts=[("a", "p", "b")])      # duplicate -> merged, not piled up
    assert v.stats()["facts"] == 1


def test_salience_ranks_surface():
    v = _vault()
    v.fuse(facts=[("itemA", "note", "x"), ("itemA", "note2", "y")])  # itemA fused more -> higher salience
    assert v.salience("itemA") > v.salience("itemB")
    r = v.surface("list all Category")
    # itemA (more salient) should rank before itemB in evidence_nodes
    ev = r.evidence_nodes
    if "itemA" in ev and "itemB" in ev:
        assert ev.index("itemA") < ev.index("itemB")


def test_auto_link_note():
    v = Vault()
    v.fuse(facts=[("itemA", "instanceOf", "Category")])
    v.fuse("a remark about itemA")
    assert v._notes[-1][2] == ["itemA"]


def test_auto_link_order_is_deterministic():
    v = Vault()
    v.fuse(facts=[("zeta", "relatedTo", "alpha")])
    v.fuse("zeta and alpha")
    assert v._notes[-1][2] == ["alpha", "zeta"]


def test_save_load(tmp_path):
    v = _vault()
    p = tmp_path / "vault.jsonl"
    v.save(str(p))
    v2 = Vault.load(str(p))
    assert v2.stats() == {"facts": 3, "notes": 1, "entities": 4}
    assert v2.salience("itemA") > 0
    assert any("limit" in x for x in v2.surface("itemA limit").relations)
