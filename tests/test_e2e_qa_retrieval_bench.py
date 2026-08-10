from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "e2e_qa_retrieval_bench.py"
SPEC = importlib.util.spec_from_file_location(
    "omnifuse_e2e_qa_retrieval_bench", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2e)


def test_sample_query_ids_matches_upstream_seeded_sample() -> None:
    data = {"queries": {f"q{index}": "question" for index in range(30)}}
    expected = random.Random(e2e.SAMPLE_SEED).sample(
        list(data["queries"]), e2e.MAX_QUESTIONS
    )

    assert e2e._sample_query_ids(data) == expected


def test_ranking_metrics_deduplicate_and_score_two_gold_documents() -> None:
    result = e2e._ranking_metrics(
        ["noise", "gold-a", "gold-a", "gold-b"],
        ["gold-a", "gold-b"],
        k=8,
    )

    assert result["retrieved_documents"] == 3
    assert result["hits"] == 2
    assert result["precision"] == 0.25
    assert result["recall"] == 1.0
    assert result["hit"] is True
    assert result["all_gold"] is True
    assert result["reciprocal_rank"] == 0.5
    assert 0.0 < result["ndcg"] < 1.0


def test_context_support_normalizes_case_spacing_and_unicode() -> None:
    result = e2e._context_support(
        "The population is 300,000. 도시는 서울입니다.",
        "300,000",
    )

    assert result["answer_exact"] is True
    assert result["answer_token_recall"] == 1.0
    assert result["answer_tokens"] == 2


def test_truncate_context_enforces_the_shared_word_budget() -> None:
    value = " ".join(f"word-{index}" for index in range(10))

    assert len(e2e._truncate_context(value, max_tokens=4).split()) == 4
    assert e2e._truncate_context("short text", max_tokens=4) == "short text"


def test_map_selected_texts_preserves_mmr_order_with_duplicate_text() -> None:
    hits = [
        (SimpleNamespace(id="a", text="same"), 3.0),
        (SimpleNamespace(id="b", text="other"), 2.0),
        (SimpleNamespace(id="c", text="same"), 1.0),
    ]

    assert e2e._map_selected_texts(hits, ["same", "same", "other"]) == [
        "a",
        "c",
        "b",
    ]


def test_head_to_head_uses_higher_quality_and_lower_cost() -> None:
    quality = (
        "recall",
        "hit_rate",
        "all_gold_rate",
        "reciprocal_rank",
        "ndcg",
        "answer_exact_rate",
        "answer_token_recall",
    )
    cost = ("build_ms", "mean_retrieval_ms", "p95_retrieval_ms", "rss_delta_mb")
    aggregates = {
        "omnifuse": {
            "metrics": {
                **{name: {"p50": 0.9} for name in quality},
                **{name: {"p50": 1.0} for name in cost},
            }
        },
        "synaptic": {
            "metrics": {
                **{name: {"p50": 0.8} for name in quality},
                **{name: {"p50": 2.0} for name in cost},
            }
        },
    }

    result = e2e._head_to_head(aggregates)

    assert result["verdict"] == {
        "omnifuse": 11,
        "synaptic": 0,
        "ties": 0,
        "common_metrics": 11,
    }


def test_per_question_head_to_head_records_quality_and_latency_losses() -> None:
    def row(query_id: str, *, value: float, retrieval_ms: float) -> dict:
        return {
            "query_id": query_id,
            "retrieval_ms": retrieval_ms,
            "metrics": {
                "precision": value,
                "recall": value,
                "f1": value,
                "hit": bool(value),
                "all_gold": bool(value),
                "reciprocal_rank": value,
                "ndcg": value,
            },
            "context_support": {
                "answer_exact": bool(value),
                "answer_token_recall": value,
            },
        }

    trials = {
        "omnifuse": [
            {
                "result": {
                    "questions": [
                        row("win", value=1.0, retrieval_ms=1.0),
                        row("loss", value=0.0, retrieval_ms=3.0),
                    ]
                }
            }
        ],
        "synaptic": [
            {
                "result": {
                    "questions": [
                        row("win", value=0.0, retrieval_ms=2.0),
                        row("loss", value=1.0, retrieval_ms=2.0),
                    ]
                }
            }
        ],
    }

    result = e2e._per_question_head_to_head(trials)

    assert result["questions"] == 2
    assert result["quality"]["questions_with_any_omnifuse_loss"] == 1
    assert result["quality"]["losses"][0]["query_id"] == "loss"
    assert result["retrieval_ms"]["questions_with_omnifuse_loss"] == 1
    assert result["retrieval_ms"]["loss_query_ids"] == ["loss"]
