from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from omnifuse import Chunk, build_inmemory  # noqa: E402
from omnifuse.linking import derive_title_links, title_query_match  # noqa: E402


def _keys(chunks: list[Chunk]) -> list[tuple[str, str, str]]:
    return [(edge.s, edge.p, edge.o) for edge in derive_title_links(chunks)]


def test_title_mentions_become_directional_retrieval_links() -> None:
    chunks = [
        Chunk("competition", "The winners included Sonya Yoncheva.", title="Operalia"),
        Chunk(
            "soprano",
            "Sonya Yoncheva is a Bulgarian operatic soprano.",
            title="Sonya Yoncheva",
        ),
        Chunk("noise", "An unrelated document.", title="Other Material"),
    ]

    graph = build_inmemory([], [], chunks, auto_link_titles=True)
    hits = graph.retrieve("Which competition listed its winners?", limit=3)

    assert _keys(chunks) == [("competition", "references", "soprano")]
    assert graph.graph.neighbor_ids("competition", direction="out") == ["soprano"]
    assert [chunk.id for chunk, _score in hits][:2] == ["competition", "soprano"]


def test_qualified_and_parenthetical_titles_have_conservative_aliases() -> None:
    chunks = [
        Chunk("council", "It was abolished by Philip V.", title="Council History"),
        Chunk("king", "Philip V resumed the throne.", title="Philip V of Spain"),
        Chunk("review", "Andrea Carroll performed.", title="Competition Review"),
        Chunk("singer", "An American soprano.", title="Andrea Carroll (soprano)"),
    ]

    assert _keys(chunks) == [
        ("council", "references", "king"),
        ("review", "references", "singer"),
    ]


def test_ambiguous_and_single_token_titles_do_not_create_noisy_links() -> None:
    chunks = [
        Chunk("source", "Mercury and Shared Name are mentioned.", title="Source Doc"),
        Chunk("one-word", "A planet.", title="Mercury"),
        Chunk("duplicate-a", "First meaning.", title="Shared Name"),
        Chunk("duplicate-b", "Second meaning.", title="Shared Name"),
    ]

    assert _keys(chunks) == []


def test_title_linking_preserves_explicit_nodes_and_deduplicates_edges() -> None:
    chunks = [
        Chunk("source", "Target Document is cited.", title="Source Document"),
        Chunk("target", "Evidence.", title="Target Document"),
    ]
    graph = build_inmemory(
        [("source", "Custom Source")],
        [("source", "references", "target")],
        chunks,
        auto_link_titles=True,
    )

    assert graph.graph.get_node("source").label == "Custom Source"
    assert len(graph.graph.triples) == 1


def test_mutable_corpus_rejects_stale_derived_links() -> None:
    with pytest.raises(ValueError, match="immutable corpus"):
        build_inmemory(
            [],
            [],
            [Chunk("a", "Target Document", title="Source Document")],
            mutable=True,
            auto_link_titles=True,
        )


def test_query_title_match_accepts_exact_entities_and_one_typo() -> None:
    exact = title_query_match(
        "Compare Patrick Baudry and another astronaut", "Patrick Baudry"
    )
    typo = title_query_match("Why is Minister Pool important?", "Minster Pool")

    assert exact is not None and exact.affinity == 1.0 and exact.offset == 1
    assert typo is not None and typo.affinity == 0.95
    assert title_query_match("Was Prince inducted?", "Prince (musician)") is not None
    assert title_query_match("A prince visited the pool", "Minster Pool") is None


def test_query_title_anchors_precede_generic_phrase_matches() -> None:
    chunks = [
        Chunk(
            "generic", "The hall of fame inducted many artists.", title="Hall of Fame"
        ),
        Chunk(
            "director", "Patty Jenkins is an American director.", title="Patty Jenkins"
        ),
        Chunk(
            "musician", "Prince was an American musician.", title="Prince (musician)"
        ),
    ]
    graph = build_inmemory([], [], chunks, auto_link_titles=True)

    ranked = [
        chunk.id
        for chunk, _score in graph.retrieve(
            "Were both Prince and Patty Jenkins inducted into the hall of fame?",
            limit=3,
        )
    ]

    assert set(ranked[:2]) == {"director", "musician"}
