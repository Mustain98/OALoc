"""Baseline arm: FIXED effort for every report (the control).

This is what LocAgent / BLAgent effectively do. Compare with run_adaptive.py —
the two files differ only in the scorer+policy, and that single difference IS the
experiment (04_METHODOLOGY.md, "The experimental design").
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.localizer.run import parse_args, run_experiment
from src.quality.policy import fixed_budget


def budget_for(inst, cfg):
    """THE CONTROL: every report gets the same budget. No report is scored."""
    return fixed_budget(cfg), -1


if __name__ == "__main__":
    run_experiment(
        budget_for,
        "baseline_predictions.jsonl",
        args=parse_args(__doc__),
    )
