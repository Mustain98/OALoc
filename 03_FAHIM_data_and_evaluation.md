# Fahim — Dataset Loader, Evaluation & The Experiment

> **You own the numbers.** You load the benchmark, turn raw rows into `Instance`s, extract
> ground-truth locations from the fixing patch, and compute the metrics that decide whether
> the project worked. Read `00_PROJECT_GUIDE.md` first. You do NOT depend on anyone —
> start immediately with the shared `schemas.py`.

---

## What you build

| File | What it does |
|---|---|
| `src/config.py`        | load `config.yaml` + `.env` |
| `src/data/loader.py`   | SWE-bench Lite → `list[Instance]` (with gold locations) |
| `src/eval/evaluate.py` | Acc@k, MRR, cost; baseline-vs-adaptive comparison + plot |

---

## 1. `src/config.py`

```python
import os, yaml
from dotenv import load_dotenv

def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()                                 # pulls ANTHROPIC_API_KEY into env
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for p in cfg["paths"].values():
        os.makedirs(p, exist_ok=True)
    return cfg
```

---

## 2. `src/data/loader.py` — the dataset

Use **SWE-bench Lite** (300 real GitHub issues with fixes). Each row already has the report
and the fixing patch; you extract the gold file/function locations from the patch.

```python
import re
from datasets import load_dataset as hf_load
from src.schemas import Instance

def _files_from_patch(patch: str) -> list[str]:
    """Files the fix touched = ground-truth buggy files."""
    return sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.M)))

def _functions_from_patch(patch: str) -> list[str]:
    """Best-effort 'path:func' from hunk headers like '@@ ... def get(self):'."""
    out, cur = [], None
    for line in patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", line)
        if m: cur = m.group(1)
        h = re.search(r"@@.*@@.*\bdef\s+(\w+)", line)
        if h and cur: out.append(f"{cur}:{h.group(1)}")
    return sorted(set(out))

def load_dataset(name: str, split: str, limit: int) -> list[Instance]:
    ds = hf_load(name, split=split)
    insts = []
    for row in ds.select(range(min(limit, len(ds)))):
        patch = row["patch"]
        insts.append(Instance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            gold_files=_files_from_patch(patch),
            gold_functions=_functions_from_patch(patch),
        ))
    return insts
```

> Note: the fixing patch is used ONLY to derive gold locations here in the loader. It is
> never placed in `problem_statement` and never reaches the localizer.

---

## 3. `src/eval/evaluate.py` — the metrics

Two core metrics, both computed at file level (primary) and function level (secondary):

- **Acc@k** — fraction of instances where **all** gold files appear in the top-k predicted
  (strict, matches LocAgent). Report k = 1, 3, 5.
- **MRR** — mean reciprocal rank of the first correct file.
- **Cost** — average USD and tokens per instance.

```python
import json, statistics
from src.schemas import Prediction, Instance

def _acc_at_k(pred_files, gold_files, k) -> int:
    if not gold_files: return 0
    return int(all(g in pred_files[:k] for g in gold_files))

def _rr(pred_files, gold_files) -> float:
    for i, f in enumerate(pred_files, 1):
        if f in gold_files: return 1.0 / i
    return 0.0

def evaluate(preds: list[Prediction], gold: list[Instance]) -> dict:
    gold_by_id = {g.instance_id: g for g in gold}
    rows = []
    for p in preds:
        g = gold_by_id.get(p.instance_id)
        if not g: continue
        rows.append({
            "acc@1": _acc_at_k(p.ranked_files, g.gold_files, 1),
            "acc@3": _acc_at_k(p.ranked_files, g.gold_files, 3),
            "acc@5": _acc_at_k(p.ranked_files, g.gold_files, 5),
            "mrr":   _rr(p.ranked_files, g.gold_files),
            "usd":   p.usd, "tokens": p.tokens,
            "quality_score": p.quality_score,
        })
    agg = lambda k: round(statistics.mean(r[k] for r in rows), 4) if rows else 0.0
    return {"n": len(rows), "acc@1": agg("acc@1"), "acc@3": agg("acc@3"),
            "acc@5": agg("acc@5"), "mrr": agg("mrr"),
            "avg_usd": agg("usd"), "avg_tokens": agg("tokens"),
            "_rows": rows}

def load_preds(path: str) -> list[Prediction]:
    out = []
    for line in open(path):
        d = json.loads(line); out.append(Prediction(**d))
    return out
```

---

## 4. The headline experiment (accuracy-vs-cost + quality breakdown)

This is what the project is judged on. Add a `--compare` entry point:

```python
def compare(baseline_metrics, adaptive_metrics):
    print(f"{'metric':10} {'baseline':>10} {'adaptive':>10}")
    for k in ["acc@1","acc@3","acc@5","mrr","avg_usd"]:
        print(f"{k:10} {baseline_metrics[k]:>10} {adaptive_metrics[k]:>10}")

def breakdown_by_quality(metrics):
    """The money plot: accuracy by report-quality bucket. Adaptive should win on low quality."""
    rows = metrics["_rows"]
    for lo, hi, label in [(0,1,"low"),(2,2,"medium"),(3,4,"high")]:
        sub = [r for r in rows if lo <= r["quality_score"] <= hi]
        if sub:
            acc = round(sum(r["acc@5"] for r in sub)/len(sub), 3)
            cost = round(sum(r["usd"] for r in sub)/len(sub), 4)
            print(f"{label:8} n={len(sub):3}  acc@5={acc}  avg_usd={cost}")

def plot_cost_accuracy(baseline, adaptive, out_png):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(baseline["avg_usd"], baseline["acc@5"], label="baseline (fixed)", s=90)
    plt.scatter(adaptive["avg_usd"], adaptive["acc@5"], label="adaptive (ours)", s=90, marker="^")
    plt.xlabel("avg cost per instance (USD)"); plt.ylabel("Acc@5 (file level)")
    plt.title("Accuracy vs cost"); plt.legend(); plt.grid(True, alpha=.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
```

Wire these into a `__main__` so `python -m src.eval.evaluate --compare` reads both
prediction files from `outputs/`, prints the table, prints the quality breakdown, and saves
`outputs/cost_accuracy.png`.

---

## 5. Test yourself (no model calls needed)

```python
# tests/test_eval.py
from src.schemas import Prediction, Instance
from src.eval.evaluate import evaluate

def test_metrics():
    gold = [Instance("i1","r","c","report", gold_files=["a.py"], gold_functions=["a.py:f"])]
    preds = [Prediction("i1", ranked_files=["a.py","b.py"], ranked_functions=["a.py:f"])]
    m = evaluate(preds, gold)
    assert m["acc@1"] == 1.0 and m["mrr"] == 1.0

def test_miss():
    gold = [Instance("i1","r","c","report", gold_files=["a.py"])]
    preds = [Prediction("i1", ranked_files=["z.py","a.py"], ranked_functions=[])]
    m = evaluate(preds, gold)
    assert m["acc@1"] == 0.0 and m["acc@3"] == 1.0 and m["mrr"] == 0.5
```

---

## 6. Your definition of done

- `load_dataset("princeton-nlp/SWE-bench_Lite","test",5)` returns 5 `Instance`s, each with
  non-empty `gold_files` derived from the patch.
- `evaluate` matches the hand-computed numbers in `test_eval.py`.
- `python -m src.eval.evaluate --compare` prints the baseline-vs-adaptive table, the
  by-quality breakdown, and writes `cost_accuracy.png`.
- You can read the final plot out loud in one sentence: *"adaptive reaches the same/higher
  Acc@5 at lower average cost, and the gap is largest on low-quality reports."* That sentence
  is the result the team defends.
