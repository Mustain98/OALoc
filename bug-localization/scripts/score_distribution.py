"""Validate the quality scorer BEFORE spending any compute on the experiment.

04_METHODOLOGY.md: "The scorer is the risk... does the score correlate with
localization difficulty on our data? Validate that BEFORE claiming the adaptive
policy works — it is step one of the results, not an assumption."

This script costs nothing: no model calls, no network, runs in seconds. It answers
the question that decides whether the experiment is worth running at all. If almost
every report lands in one bucket, the adaptive policy picks the same budget nearly
every time and both arms will produce the same numbers — that is a finding you want
on day one, not after a 300-instance run.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.loader import load_dataset
from src.quality.policy import tier_for_score
from src.quality.scorer import FEATURE_NAMES, score_quality, score_quality_features


def main():
    cfg = load_config()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["limit"]
    insts = load_dataset(cfg["dataset"], cfg["split"], limit)
    n = len(insts)
    if not n:
        print("no instances loaded")
        return

    scores = Counter()
    tiers = Counter()
    features = Counter()
    for inst in insts:
        s = score_quality(inst.problem_statement)
        scores[s] += 1
        tiers[tier_for_score(s)] += 1
        for name, present in score_quality_features(inst.problem_statement).items():
            if present:
                features[name] += 1

    print(f"\nQuality scores over {n} instances ({cfg['dataset']}, {cfg['split']})\n")
    print("score  count  share")
    for s in range(5):
        c = scores[s]
        bar = "#" * round(40 * c / n)
        print(f"  {s}    {c:5d}  {c / n:5.1%}  {bar}")

    print("\nbudget tier  count  share   (effort the policy would spend)")
    for tier in ("low", "medium", "high"):
        c = tiers[tier]
        b = cfg["budgets"][tier]
        print(f"  {tier:9s}  {c:5d}  {c / n:5.1%}  "
              f"candidates={b['max_candidates']} hops={b['max_hops']} "
              f"samples={b['max_samples']}")

    print("\nfeature fire rate  (a feature at ~0% or ~100% carries no signal)")
    for name in FEATURE_NAMES:
        c = features[name]
        print(f"  {name:14s} {c:5d}  {c / n:5.1%}")

    top = max(tiers.values()) / n if tiers else 0
    print()
    if top > 0.85:
        print(f"WARNING: {top:.0%} of reports fall in a single tier. The adaptive arm "
              f"will behave almost identically to the baseline — retune the score "
              f"thresholds or the features before running the experiment.")
    else:
        print(f"OK: effort is spread across tiers (largest bucket {top:.0%}).")


if __name__ == "__main__":
    main()
