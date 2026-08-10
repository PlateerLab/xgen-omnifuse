from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "eval" / "e2e_qa_answer_bench.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "omnifuse_e2e_qa_answer_bench", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2e)


def test_simple_correctness_matches_upstream_three_stage_contract() -> None:
    assert e2e._simple_correctness("300,000", "300,000") == 1.0
    assert e2e._simple_correctness("The answer is 300,000.", "300,000") == 1.0
    assert e2e._simple_correctness("Kahveci Nihat", "Nihat Kahveci") == 0.9
    assert e2e._simple_correctness("unrelated", "Nihat Kahveci") == 0.0


def test_payload_matches_official_ollama_shape() -> None:
    payload = e2e._payload(model="qwen3.5:4b", question="Q?", context="C")

    assert payload == {
        "model": "qwen3.5:4b",
        "messages": [
            {"role": "system", "content": e2e.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Context:\nC\n\nQuestion: Q?\n\nAnswer:",
            },
        ],
        "stream": False,
        "think": False,
    }


def test_warmup_payload_is_unrelated_and_bounded_to_one_token() -> None:
    assert e2e._warmup_payload("qwen3.5:4b") == {
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "stream": False,
        "think": False,
        "options": {"num_predict": 1},
        "keep_alive": "10m",
    }


def test_require_loopback_rejects_remote_or_path_urls() -> None:
    assert e2e._require_loopback_base_url("http://127.0.0.1:11434/") == (
        "http://127.0.0.1:11434"
    )

    for value in ("https://example.com", "http://127.0.0.1:11434/api"):
        try:
            e2e._require_loopback_base_url(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe base URL: {value}")


def test_aggregate_reports_official_and_honest_cohort_means() -> None:
    rows = [
        {
            "correctness": score,
            "generation_ms": 10.0,
            "retrieval_ms": 2.0,
            "ollama": {"prompt_eval_count": 100, "eval_count": 5},
        }
        for score in (1.0, 0.5, 0.0, 0.0)
    ]

    result = e2e._aggregate(rows)

    assert result["cohort_mean_correctness"] == 0.375
    assert result["official_mean_correctness_nonzero_only"] == 0.75
    assert result["accuracy_at_0_5"] == 0.5
    assert result["exact_score_rate"] == 0.25
    assert result["positive_score_rate"] == 0.5


def test_head_to_head_uses_higher_quality_and_lower_cost() -> None:
    quality = (
        "cohort_mean_correctness",
        "official_mean_correctness_nonzero_only",
        "accuracy_at_0_5",
        "exact_score_rate",
        "positive_score_rate",
    )
    cost = (
        "mean_generation_ms",
        "p95_generation_ms",
        "total_generation_ms",
        "mean_retrieval_ms",
        "mean_prompt_tokens",
    )
    aggregates = {
        "omnifuse": {
            **{name: 0.9 for name in quality},
            **{name: 1.0 for name in cost},
        },
        "synaptic": {
            **{name: 0.8 for name in quality},
            **{name: 2.0 for name in cost},
        },
    }

    result = e2e._head_to_head(aggregates)

    assert result["verdict"] == {
        "omnifuse": 10,
        "synaptic": 0,
        "ties": 0,
        "common_metrics": 10,
    }


def test_per_question_head_to_head_reports_correctness_and_cost_losses() -> None:
    def row(query_id: str, correctness: float, generation_ms: float) -> dict:
        return {
            "query_id": query_id,
            "correctness": correctness,
            "generation_ms": generation_ms,
            "retrieval_ms": generation_ms,
            "ollama": {
                "prompt_eval_count": int(generation_ms),
                "eval_count": int(generation_ms),
            },
        }

    results = {
        "omnifuse": [row("win", 1.0, 1.0), row("loss", 0.0, 3.0)],
        "synaptic": [row("win", 0.0, 2.0), row("loss", 1.0, 2.0)],
    }

    result = e2e._per_question_head_to_head(results)

    assert result["questions"] == 2
    assert result["questions_with_omnifuse_correctness_loss"] == 1
    assert result["loss_query_ids"]["correctness"] == ["loss"]
    assert result["loss_query_ids"]["generation_ms"] == ["loss"]


def test_percentile_uses_nearest_rank() -> None:
    assert e2e._percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
