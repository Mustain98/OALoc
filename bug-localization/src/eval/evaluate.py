# STUB — owned by FAHIM (03_FAHIM_data_and_evaluation.md §3-4).
# Replace wholesale when their branch lands. Enough of evaluate() is implemented here
# for Taj to see end-to-end numbers; the comparison table, the quality breakdown
# and the plot are Fahim's.
#
# NOTE FOR FAHIM: we run a local model, so `usd` is genuinely 0.0 and the headline
# accuracy-vs-cost plot must use avg_tokens on the x-axis, not avg_usd.
# Prediction.tokens already carries it — no schema change needed.
import json
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
            "usd":   p.usd,
            "tokens": p.tokens,
            "quality_score": p.quality_score,
        })

    def agg(k):
        return round(statistics.mean(r[k] for r in rows), 4) if rows else 0.0

    return {"n": len(rows), "acc@1": agg("acc@1"), "acc@3": agg("acc@3"),
            "acc@5": agg("acc@5"), "mrr": agg("mrr"),
            "avg_usd": agg("usd"), "avg_tokens": agg("tokens"),
            "_rows": rows}


def load_preds(path: str) -> list[Prediction]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Prediction(**json.loads(line)))
    return out
