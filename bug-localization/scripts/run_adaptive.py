"""Adaptive arm: effort chosen by the measured report quality (our method).

Identical to run_baseline.py except for the scorer+policy below. Everything else —
graph, tools, model, prompts, dataset — is held constant, so any difference in
accuracy or cost is attributable to the adaptation and nothing else.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.localizer.run import parse_args, run_experiment
from src.quality.policy import choose_budget
from src.quality.scorer import score_quality


def budget_for(inst, cfg):
    """THE TREATMENT: score the report, then let the score pick the budget."""
    score = score_quality(inst.problem_statement)
    return choose_budget(score, cfg), score


if __name__ == "__main__":
    run_experiment(
        budget_for,
        "adaptive_predictions.jsonl",
        args=parse_args(__doc__),
    )
