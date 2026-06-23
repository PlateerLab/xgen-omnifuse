import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import from_csv, from_jsonl, from_triples  # noqa: E402
from omnifuse.loaders import derive_nodes, to_chunk, to_triple  # noqa: E402


def test_derive_nodes_infers_class():
    nodes = {n.id: n.kind for n in derive_nodes([to_triple(("a", "instanceOf", "규정"))])}
    assert nodes["규정"] == "class" and nodes["a"] == "instance"


def test_to_chunk_forms():
    assert to_chunk(("c1", "txt", ["e"])).entities == ["e"]
    assert to_chunk({"id": "c2", "text": "t"}).id == "c2"


def test_from_triples_tuples_autonodes():
    of = from_triples(
        [("담보", "instanceOf", "규정"), ("담보", "한도", "5억")],
        chunks=[("c1", "담보 한도는 5억원이다", ["담보"])],
    )
    r = of.search("담보 한도")
    assert any("한도" in x for x in r.relations)


def test_from_triples_dicts():
    of = from_triples([{"s": "감사규정", "p": "instanceOf", "o": "규정"}])
    r = of.search("규정")
    assert "규정" in r.class_seed or r.relations


def test_from_jsonl(tmp_path):
    tp = tmp_path / "t.jsonl"
    tp.write_text(json.dumps({"s": "감사규정", "p": "instanceOf", "o": "규정"}, ensure_ascii=False), encoding="utf-8")
    cp = tmp_path / "c.jsonl"
    cp.write_text(json.dumps({"id": "x", "text": "감사규정은 절차를 정한다", "entities": ["감사규정"]}, ensure_ascii=False), encoding="utf-8")
    of = from_jsonl(triples=str(tp), chunks=str(cp))
    assert of.search("규정").answer


def test_from_csv(tmp_path):
    tp = tmp_path / "t.csv"
    tp.write_text("s,p,o\n감사규정,instanceOf,규정\n경마시행규정,instanceOf,규정\n", encoding="utf-8")
    of = from_csv(triples=str(tp))
    r = of.search("규정 전부")
    assert "규정" in r.class_seed
