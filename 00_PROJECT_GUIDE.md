# Project Guide — Quality-Adaptive LLM Bug Localization

> **Read this file first. All three teammates share it.** It defines the goal, the
> architecture, the exact interfaces you code against, the repo layout, and the setup.
> The three role files (`01_`, `02_`, `03_`) tell each person what to build.

---

## 1. What we are building (one paragraph)

A bug localizer that takes a **bug report** and finds the buggy code (which files /
functions). It uses a **LocAgent-style code graph** to navigate the repository, but adds
one new idea: it **measures the quality of the bug report first** and **adapts how much
search effort to spend** — more effort on vague/low-quality reports (to recover accuracy),
less on clear ones (to save cost). Baseline = fixed effort. Our system = effort that
scales with report quality. We prove it with an **accuracy-vs-cost** comparison.

**Novelty in one line:** existing efficient LLM localizers (LocAgent, BLAgent) spend the
same effort on every report; we condition effort on a measured report-quality signal.

---

## 2. Architecture (data flow)

```
                 ┌─────────────────────────────────────────────────────────┐
  bug report ──► │ (A) QUALITY SCORER  ──►  score 0..4                      │
                 │ (B) BUDGET POLICY   ──►  budget {candidates, hops, samples}│  ← ROLE 1 (you)
                 └───────────────┬─────────────────────────────────────────┘
                                 │  report + budget
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │ (C) CODE GRAPH  (repo → nodes/edges, built once per repo) │
                 │ (D) TOOLS: search_entity / traverse_graph / retrieve      │  ← ROLE 2
                 │ (E) LOCALIZER AGENT loop  ──► ranked files & functions    │
                 └───────────────┬─────────────────────────────────────────┘
                                 │  prediction {ranked_files, ranked_functions, cost}
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │ (F) DATASET loader (SWE-bench Lite)                       │
                 │ (G) EVALUATOR: Acc@k, MRR, cost  ──► results tables/plots │  ← ROLE 3
                 └─────────────────────────────────────────────────────────┘
```

Build order: Role 2 and Role 3 can start immediately (they don't depend on the novel
part). Role 1 wires everything together. Everyone codes against the **interfaces in §4** so
work happens in parallel and only integrates at the end.

---

## 3. Repository layout (create exactly this)

```
bug-localization/
├── README.md
├── requirements.txt
├── config.yaml                 # runtime settings (model, dataset, paths) — NOT code
├── .env                        # ANTHROPIC_API_KEY=...   (never commit this)
├── data/                       # downloaded datasets + cached repos (git-ignored)
├── outputs/                    # predictions.jsonl, metrics.json, plots (git-ignored)
├── src/
│   ├── __init__.py
│   ├── schemas.py              # the shared dataclasses in §4  — AGREE FIRST, DO NOT EDIT ALONE
│   ├── config.py               # loads config.yaml + .env
│   ├── llm.py                  # thin wrapper around the model API (shared helper)
│   ├── quality/                # ROLE 1
│   │   ├── scorer.py
│   │   └── policy.py
│   ├── graph/                  # ROLE 2
│   │   ├── indexer.py
│   │   └── tools.py
│   ├── localizer/              # ROLE 1 owns the loop; uses ROLE 2 tools
│   │   └── agent.py
│   ├── data/                   # ROLE 3
│   │   └── loader.py
│   └── eval/                   # ROLE 3
│       └── evaluate.py
├── scripts/
│   ├── run_baseline.py         # fixed effort (control)
│   └── run_adaptive.py         # quality-adaptive effort (our method)
└── tests/
    ├── test_schemas.py
    ├── test_scorer.py
    ├── test_graph.py
    └── test_eval.py
```

---

## 4. Shared interfaces — `src/schemas.py` (LOCK THIS FIRST)

**This is the contract. All three people import from here. Do not change a field without
telling the other two.** Put this file in place on day 1.

```python
# src/schemas.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Instance:
    """One input: a bug report against one repo snapshot (from SWE-bench Lite)."""
    instance_id: str          # e.g. "django__django-12345"
    repo: str                 # e.g. "django/django"
    base_commit: str          # commit to check out (repo BEFORE the fix)
    problem_statement: str    # raw bug report text  <-- the thing we score
    gold_files: list[str] = field(default_factory=list)      # eval only, never fed to localizer
    gold_functions: list[str] = field(default_factory=list)  # eval only

@dataclass
class Budget:
    """How much effort the localizer may spend on this instance."""
    max_candidates: int       # how many entities to keep from search
    max_hops: int             # graph traversal depth
    max_samples: int          # how many agent runs to aggregate
    model: str                # which model tier to use

@dataclass
class Prediction:
    """Output of the localizer for one instance."""
    instance_id: str
    ranked_files: list[str]        # most-suspicious first
    ranked_functions: list[str]    # "path/file.py:func" strings, most-suspicious first
    quality_score: int = -1        # 0..4, -1 if not scored (baseline)
    tokens: int = 0
    usd: float = 0.0
```

**The two function signatures everyone builds toward:**

```python
# ROLE 1
def score_quality(problem_statement: str) -> int: ...          # returns 0..4
def choose_budget(score: int, cfg) -> Budget: ...

# ROLE 2  (the localizer calls these; ROLE 1 owns the loop that calls them)
def build_graph(repo_dir: str) -> "CodeGraph": ...
def search_entity(graph, keyword: str, detail: str) -> list[dict]: ...
def traverse_graph(graph, seeds: list[str], hops: int, edge_types: list[str]) -> str: ...
def retrieve_entity(graph, entity_id: str) -> dict: ...

# ROLE 1 (top-level; ties it together)
def localize(inst: Instance, budget: Budget, graph) -> Prediction: ...

# ROLE 3
def load_dataset(name: str, split: str, limit: int) -> list[Instance]: ...
def evaluate(preds: list[Prediction], gold: list[Instance]) -> dict: ...  # Acc@k, MRR, cost
```

---

## 5. `config.yaml` (single source of runtime settings)

```yaml
model: "claude-sonnet-4-6"        # default localizer model
dataset: "princeton-nlp/SWE-bench_Lite"
split: "test"
limit: 30                          # start small; raise to 300 for the full run
budgets:                           # ROLE 1's policy reads these
  low:    { max_candidates: 50, max_hops: 3, max_samples: 3 }   # for low-quality reports
  medium: { max_candidates: 25, max_hops: 2, max_samples: 2 }
  high:   { max_candidates: 10, max_hops: 1, max_samples: 1 }   # for clear reports
baseline_budget: { max_candidates: 15, max_hops: 2, max_samples: 1 }  # fixed control
paths:
  data: "./data"
  outputs: "./outputs"
```

---

## 6. Setup (every teammate runs this once)

```bash
git clone <your-repo-url> bug-localization && cd bug-localization
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your ANTHROPIC_API_KEY into .env
```

`requirements.txt`:
```
anthropic>=0.40
datasets>=2.19          # SWE-bench Lite loader
networkx>=3.2           # code graph
tree-sitter>=0.21       # (optional) robust parsing; ast fallback is fine to start
rank-bm25>=0.2          # keyword search over entities
pyyaml>=6
python-dotenv>=1
matplotlib>=3.8         # accuracy-vs-cost plot
pytest>=8
```

---

## 7. Ground rules (so parallel work doesn't collide)

- **Branch per role:** `role1-quality`, `role2-graph`, `role3-eval`. PR into `main`.
- **`schemas.py` is frozen** after day 1. Any change is a 3-person decision.
- **`gold_files` / `gold_functions` are for evaluation only.** Never pass them into the
  scorer, tools, or localizer — that would be cheating and it invalidates every result.
- **Everything returns a `Prediction`**; the evaluator only ever sees `Prediction`s + gold.
- **Start with `limit: 5`** in config to debug end-to-end on 5 instances before the full run.
- **Log cost every run.** Accuracy-vs-cost is the headline result; if cost isn't measured,
  the project has no result.
- Commit `predictions.jsonl` and `metrics.json` for each run so results are reproducible.

---

## 8. Definition of done (what "it works" means)

1. `python scripts/run_baseline.py` → produces `outputs/baseline_metrics.json` (Acc@k, MRR, avg cost).
2. `python scripts/run_adaptive.py` → produces `outputs/adaptive_metrics.json`.
3. `python -m src.eval.evaluate --compare` → prints a table + saves an accuracy-vs-cost plot.
4. Expected story: **adaptive ≈ or > baseline accuracy at lower or equal average cost**,
   with the gain concentrated on low-quality reports. Even a small, clearly-measured effect
   is a valid result — do not fake numbers.
