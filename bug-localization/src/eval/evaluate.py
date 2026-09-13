"""Evaluation metrics and the baseline-versus-adaptive experiment report."""
import argparse
import json
import os
import statistics

from src.schemas import Instance, Prediction


def _acc_at_k(pred_files, gold_files, k) -> int:
    if not gold_files:
        return 0
    return int(all(g in pred_files[:k] for g in gold_files))


def _rr(pred_files, gold_files) -> float:
    for i, f in enumerate(pred_files, 1):
        if f in gold_files:
            return 1.0 / i
    return 0.0


def evaluate(preds: list[Prediction], gold: list[Instance]) -> dict:
    gold_by_id = {g.instance_id: g for g in gold}
    rows = []
    for p in preds:
        g = gold_by_id.get(p.instance_id)
        if not g:
            continue
        rows.append({
            "acc@1": _acc_at_k(p.ranked_files, g.gold_files, 1),
            "acc@3": _acc_at_k(p.ranked_files, g.gold_files, 3),
            "acc@5": _acc_at_k(p.ranked_files, g.gold_files, 5),
            "mrr":   _rr(p.ranked_files, g.gold_files),
            "function_acc@1": _acc_at_k(p.ranked_functions, g.gold_functions, 1),
            "function_acc@3": _acc_at_k(p.ranked_functions, g.gold_functions, 3),
            "function_acc@5": _acc_at_k(p.ranked_functions, g.gold_functions, 5),
            "function_mrr":   _rr(p.ranked_functions, g.gold_functions),
            "usd":   p.usd,
            "tokens": p.tokens,
            "quality_score": p.quality_score,
        })

    def agg(k):
        return round(statistics.mean(r[k] for r in rows), 4) if rows else 0.0

    return {"n": len(rows), "acc@1": agg("acc@1"), "acc@3": agg("acc@3"),
            "acc@5": agg("acc@5"), "mrr": agg("mrr"),
            "function_acc@1": agg("function_acc@1"),
            "function_acc@3": agg("function_acc@3"),
            "function_acc@5": agg("function_acc@5"),
            "function_mrr": agg("function_mrr"),
            "avg_usd": agg("usd"), "avg_tokens": agg("tokens"),
            "_rows": rows}


def load_preds(path: str) -> list[Prediction]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Prediction(**json.loads(line)))
    return out


def compare(baseline_metrics: dict, adaptive_metrics: dict) -> None:
    """Print the headline metrics for the controlled comparison."""
    print(f"{'metric':12} {'baseline':>10} {'adaptive':>10}")
    for key in ("acc@1", "acc@3", "acc@5", "mrr", "avg_usd", "avg_tokens"):
        print(f"{key:12} {baseline_metrics[key]:>10} {adaptive_metrics[key]:>10}")


def breakdown_by_quality(metrics: dict) -> None:
    """Print Acc@5 and average cost for each report-quality bucket."""
    for low, high, label in ((0, 1, "low"), (2, 2, "medium"), (3, 4, "high")):
        rows = [row for row in metrics["_rows"]
                if low <= row["quality_score"] <= high]
        if rows:
            accuracy = round(statistics.mean(row["acc@5"] for row in rows), 3)
            tokens = round(statistics.mean(row["tokens"] for row in rows), 1)
            usd = round(statistics.mean(row["usd"] for row in rows), 4)
            print(f"{label:8} n={len(rows):3}  acc@5={accuracy}  "
                  f"avg_tokens={tokens}  avg_usd={usd}")


def plot_cost_accuracy(baseline: dict, adaptive: dict, out_png: str) -> None:
    """Plot file-level accuracy against provider-neutral token cost."""
    import matplotlib.pyplot as plt

    figure = plt.figure()
    plt.scatter(baseline["avg_tokens"], baseline["acc@5"],
                label="baseline (fixed)", s=90)
    plt.scatter(adaptive["avg_tokens"], adaptive["acc@5"],
                label="adaptive (ours)", s=90, marker="^")
    plt.xlabel("average tokens per instance")
    plt.ylabel("Acc@5 (file level)")
    plt.title("Accuracy vs token cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    figure.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _compare_from_outputs(outputs_dir: str) -> None:
    baseline_path = os.path.join(outputs_dir, "baseline_predictions.jsonl")
    adaptive_path = os.path.join(outputs_dir, "adaptive_predictions.jsonl")
    missing = [path for path in (baseline_path, adaptive_path)
               if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing prediction file(s): " + ", ".join(missing))

    from src.config import load_config
    from src.data.loader import load_dataset

    config = load_config()
    gold = load_dataset(config["dataset"], config["split"], config["limit"])
    baseline = evaluate(load_preds(baseline_path), gold)
    adaptive = evaluate(load_preds(adaptive_path), gold)
    compare(baseline, adaptive)
    print("\nBaseline quality breakdown")
    breakdown_by_quality(baseline)
    print("\nAdaptive quality breakdown")
    breakdown_by_quality(adaptive)
    plot_path = os.path.join(outputs_dir, "cost_accuracy.png")
    plot_cost_accuracy(baseline, adaptive, plot_path)
    print(f"\nWrote {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true",
                        help="compare baseline and adaptive prediction files")
    parser.add_argument("--outputs", default=None,
                        help="directory containing prediction JSONL files")
    args = parser.parse_args()
    if not args.compare:
        parser.error("--compare is required")

    if args.outputs:
        outputs_dir = args.outputs
    else:
        from src.config import load_config
        outputs_dir = load_config()["paths"]["outputs"]
    os.makedirs(outputs_dir, exist_ok=True)
    _compare_from_outputs(outputs_dir)


if __name__ == "__main__":
    main()
