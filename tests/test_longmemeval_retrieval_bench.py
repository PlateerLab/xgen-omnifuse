from __future__ import annotations

import importlib.util
import random
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "eval" / "longmemeval_retrieval_bench.py"
)
SPEC = importlib.util.spec_from_file_location(
    "omnifuse_longmemeval_retrieval_bench", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
longmem = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(longmem)


def _instance(
    *,
    question_id: str = "q1",
    question_type: str = "type-a",
    answer_session_ids: list[str] | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "What happened?",
        "haystack_sessions": [
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "assistant", "content": "orphan answer"},
                {"role": "user", "content": "last question"},
            ]
        ],
        "haystack_session_ids": ["session-1"],
        "haystack_dates": ["2025-01-02"],
        "answer_session_ids": (
            ["session-1"] if answer_session_ids is None else answer_session_ids
        ),
    }


def test_balanced_sample_reproduces_upstream_rng_and_type_order() -> None:
    data = [
        _instance(question_id=f"a-{index}", question_type="type-a")
        for index in range(5)
    ] + [
        _instance(question_id=f"b-{index}", question_type="type-b")
        for index in range(5)
    ]
    expected: list[str] = []
    rng = random.Random(longmem.SAMPLE_SEED)
    for prefix in ("a", "b"):
        ids = [f"{prefix}-{index}" for index in range(5)]
        rng.shuffle(ids)
        expected.extend(ids[:2])

    sampled = longmem._balanced_sample(data, max_questions=4)

    assert [row["question_id"] for row in sampled] == expected
    assert [row["question_type"] for row in sampled] == [
        "type-a",
        "type-a",
        "type-b",
        "type-b",
    ]


def test_turn_pair_records_match_upstream_pair_and_orphan_rules() -> None:
    records = longmem._turn_pair_records(_instance())

    assert records == [
        {
            "id": "session-1:0",
            "session_id": "session-1",
            "date": "2025-01-02",
            "turn": 0,
            "title": "first question",
            "content": "[User] first question\n[Assistant] first answer",
            "source": "longmemeval:session-1:0",
        },
        {
            "id": "session-1:1",
            "session_id": "session-1",
            "date": "2025-01-02",
            "turn": 1,
            "title": "orphan answer",
            "content": "[Assistant] orphan answer",
            "source": "longmemeval:session-1:1",
        },
        {
            "id": "session-1:2",
            "session_id": "session-1",
            "date": "2025-01-02",
            "turn": 2,
            "title": "last question",
            "content": "[User] last question",
            "source": "longmemeval:session-1:2",
        },
    ]


def test_retrieval_metrics_deduplicate_sessions_and_rank_gold() -> None:
    metrics = longmem._retrieval_metrics(
        ["noise", "gold-a", "gold-a", "gold-b"],
        ["gold-a", "gold-b"],
    )

    assert metrics["eligible"] is True
    assert metrics["retrieved_unique_sessions"] == 3
    assert metrics["hits"] == 2
    assert metrics["session_recall"] == 1.0
    assert metrics["session_hit"] is True
    assert metrics["reciprocal_rank"] == 0.5
    assert 0.0 < metrics["ndcg"] < 1.0


def test_retrieval_metrics_exclude_questions_without_gold_sessions() -> None:
    metrics = longmem._retrieval_metrics(["noise"], [])

    assert metrics == {
        "eligible": False,
        "gold_sessions": 0,
        "retrieved_unique_sessions": 1,
        "hits": 0,
        "session_recall": None,
        "session_hit": None,
        "reciprocal_rank": None,
        "ndcg": None,
    }


def test_file_identity_uses_the_provenance_fingerprint_contract() -> None:
    assert longmem._file_identity(
        {"path": "data.json", "sha256": "a" * 64, "bytes": 42, "extra": True}
    ) == {"path": "data.json", "sha256": "a" * 64, "bytes": 42}


def test_head_to_head_uses_higher_quality_and_lower_cost() -> None:
    quality = {
        "mean_session_recall": {"p50": 0.8},
        "session_hit_rate": {"p50": 0.9},
        "mrr": {"p50": 0.7},
        "ndcg": {"p50": 0.75},
    }
    cost = {
        "total_build_ms": {"p50": 10.0},
        "average_retrieval_ms": {"p50": 1.0},
        "p95_retrieval_ms": {"p50": 2.0},
        "maximum_rss_delta_mb": {"p50": 3.0},
    }
    aggregates = {
        "omnifuse": {"metrics": {**quality, **cost}},
        "synaptic": {
            "metrics": {
                **{name: {"p50": row["p50"] - 0.1} for name, row in quality.items()},
                **{name: {"p50": row["p50"] + 1.0} for name, row in cost.items()},
            }
        },
    }

    result = longmem._head_to_head(aggregates)

    assert result["verdict"] == {
        "omnifuse_wins_or_ties_all_quality": True,
        "omnifuse_wins_or_ties_all_efficiency": True,
        "omnifuse_strict_wins": 8,
        "synaptic_strict_wins": 0,
        "ties": 0,
        "comparable_metrics": 8,
    }
