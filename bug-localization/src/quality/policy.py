# src/quality/policy.py
#
# ROLE 1. Stage 2 of the method: score -> Budget.
#
# This is the mechanism that converts the quality signal into actual compute, and it
# is the genuinely new part of the project. The mapping lives in config.yaml, not in
# this file, so it can be tuned and its sensitivity reported.
from src.schemas import Budget


def tier_for_score(score: int) -> str:
    """Low quality => more effort; high quality => less."""
    if score <= 1:
        return "low"
    if score == 2:
        return "medium"
    return "high"


def _budget_from(b: dict, cfg: dict) -> Budget:
    return Budget(
        max_candidates=b["max_candidates"],
        max_hops=b["max_hops"],
        max_samples=b["max_samples"],
        model=cfg["model"],
    )


def choose_budget(score: int, cfg) -> Budget:
    """THE TREATMENT. Effort is a function of the measured report quality."""
    return _budget_from(cfg["budgets"][tier_for_score(score)], cfg)


def fixed_budget(cfg) -> Budget:
    """THE CONTROL. Same effort for every report, whatever its quality."""
    return _budget_from(cfg["baseline_budget"], cfg)
