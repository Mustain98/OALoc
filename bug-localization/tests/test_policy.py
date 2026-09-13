"""The policy is the mechanism that turns the quality signal into compute."""
import pytest

from src.quality.policy import choose_budget, fixed_budget, tier_for_score

CFG = {
    "model": "test-model",
    "budgets": {
        "low":    {"max_candidates": 50, "max_hops": 3, "max_samples": 3},
        "medium": {"max_candidates": 25, "max_hops": 2, "max_samples": 2},
        "high":   {"max_candidates": 10, "max_hops": 1, "max_samples": 1},
    },
    "baseline_budget": {"max_candidates": 15, "max_hops": 2, "max_samples": 1},
}


@pytest.mark.parametrize("score,tier", [(0, "low"), (1, "low"), (2, "medium"),
                                        (3, "high"), (4, "high")])
def test_tier_boundaries(score, tier):
    assert tier_for_score(score) == tier


def test_low_quality_gets_more_effort_than_high():
    low, high = choose_budget(0, CFG), choose_budget(4, CFG)
    assert low.max_candidates > high.max_candidates
    assert low.max_hops > high.max_hops
    assert low.max_samples > high.max_samples


def test_budget_reads_config_not_hardcoded_values():
    cfg = {**CFG, "budgets": {**CFG["budgets"],
                              "low": {"max_candidates": 99, "max_hops": 7,
                                      "max_samples": 5}}}
    b = choose_budget(0, cfg)
    assert (b.max_candidates, b.max_hops, b.max_samples) == (99, 7, 5)


def test_fixed_budget_is_the_same_for_every_score():
    budgets = {(fixed_budget(CFG).max_candidates, fixed_budget(CFG).max_hops,
                fixed_budget(CFG).max_samples)}
    assert budgets == {(15, 2, 1)}


def test_model_comes_from_config():
    assert choose_budget(2, CFG).model == "test-model"
    assert fixed_budget(CFG).model == "test-model"
