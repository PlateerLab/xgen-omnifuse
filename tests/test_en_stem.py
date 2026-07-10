"""English morphological normalization — the symmetric counterpart of the Korean stemmer.

Latin tokens used to be indexed as raw surface forms, so a query for "statin" could not
match a document saying "statins". Harman's S-stemmer singularizes and does nothing else.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, build_inmemory  # noqa: E402
from omnifuse.text import _en_stem, tokenize  # noqa: E402


def test_s_stemmer_rules():
    assert _en_stem("statins") == "statin"
    assert _en_stem("studies") == "study"          # -ies -> -y
    assert _en_stem("diseases") == "disease"       # -es  -> -e...s dropped
    assert _en_stem("cells") == "cell"


def test_s_stemmer_leaves_non_plurals_alone():
    assert _en_stem("virus") == "virus"            # -us guard
    assert _en_stem("access") == "access"          # -ss guard
    assert _en_stem("gas") == "gas"                # too short (len <= 3)
    assert _en_stem("his") == "his"


def test_es_guards_fall_through_to_the_plain_s_rule():
    # The -aes/-ees/-oes guards only block the "-es" rule; the "-s" rule still applies,
    # which is the correct singular ("bees" -> "bee"), not a no-op.
    assert _en_stem("bees") == "bee"
    assert _en_stem("toes") == "toe"


def test_query_matches_plural_document():
    of = build_inmemory([], [], [
        Chunk("hit", text="recent studies suggest that statins reduce risk"),
        Chunk("miss", text="an unrelated passage about weather and traffic"),
    ])
    ids = [c.id for c, _ in of.retrieve("statin study")]
    assert ids[0] == "hit"


def test_hangul_and_digits_unaffected_shape():
    # Korean path still emits stem bi-grams + '#stem'; latin path emits stems.
    toks = tokenize("담보 한도 statins")
    assert "statin" in toks
    assert "#담보" in toks
