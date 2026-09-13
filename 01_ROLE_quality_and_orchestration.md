# Role 1 — Quality Scorer, Budget Policy & Orchestration  (YOU)

> **This is the novel core of the project — the part that makes it publishable.**
> You own the report-quality scorer, the budget policy, the agent loop, and the two run
> scripts that tie everyone's work together. Read `00_PROJECT_GUIDE.md` first.
> You depend on Role 2's tools (`src/graph/tools.py`) and Role 3's loader/evaluator, so
> stub them (see §6) until they land, then swap in the real ones.

---

## What you build

| File | What it does |
|---|---|
| `src/quality/scorer.py` | bug report text → quality score 0..4 |
| `src/quality/policy.py`  | score → `Budget` (how much effort to spend) |
| `src/localizer/agent.py` | the agent loop: run tools within the budget → `Prediction` |
| `src/llm.py`             | shared thin API wrapper (you write it; everyone uses it) |
| `scripts/run_baseline.py`| fixed-effort control run |
| `scripts/run_adaptive.py`| quality-adaptive run (our method) |

---

## 1. `src/llm.py` — shared model wrapper (write this first)

Everyone calls the model through this, so cost is measured in one place.

```python
import os, anthropic
from dotenv import load_dotenv
load_dotenv()

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# rough per-Mtoken USD; update from current pricing before final run
_PRICE = {"claude-sonnet-4-6": (3.0, 15.0)}   # (input, output) per 1M tokens

def call(model: str, system: str, user: str, max_tokens: int = 1024):
    """Returns (text, tokens, usd). Single place where all cost is counted."""
    r = _client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    tin, tout = r.usage.input_tokens, r.usage.output_tokens
    pin, pout = _PRICE.get(model, (3.0, 15.0))
    usd = tin/1e6*pin + tout/1e6*pout
    return text, tin + tout, usd
```

---

## 2. `src/quality/scorer.py` — the quality signal

The score is the whole thesis, so make it **objective and cheap**. Use presence-based
features first (fast, deterministic, defensible); keep an optional LLM scorer as a variant
you can compare against. Features are taken from prior work on report quality (Fang et al.,
Chaparro et al. — steps-to-reproduce, observed/expected behaviour, stack trace, code).

```python
import re

def score_quality(problem_statement: str) -> int:
    """Return 0..4 = how much localization signal the report carries."""
    t = problem_statement or ""
    tl = t.lower()
    score = 0
    # 1. stack trace / traceback present
    if re.search(r"traceback|at [\w.$]+\([\w.]+:\d+\)|file \".+\", line \d+", tl):
        score += 1
    # 2. code snippet / fenced block / call syntax
    if "```" in t or re.search(r"\b\w+\.\w+\(", t):
        score += 1
    # 3. names a class or method/function explicitly
    if re.search(r"\b[A-Z][a-zA-Z0-9]+\b", t) and re.search(r"\bdef \w+|\w+\(\)|method|function|class\b", tl):
        score += 1
    # 4. steps to reproduce / observed-vs-expected structure
    if re.search(r"steps to reproduce|to reproduce|expected|actual|observed", tl):
        score += 1
    return min(score, 4)

# Optional variant for comparison (uses the LLM). Same 0..4 output.
def score_quality_llm(problem_statement: str, model: str) -> int:
    from src.llm import call
    sys = ("Rate how much a bug report helps locate the buggy code. "
           "Reply with ONE integer 0-4. 0=vague/no clues, 4=stack trace + code + clear repro.")
    txt, *_ = call(model, sys, problem_statement, max_tokens=5)
    m = re.search(r"[0-4]", txt)
    return int(m.group()) if m else 2
```

**Report both scorers in the paper.** The rule-based one is your defensible default; the LLM
one is a comparison point. Do not let the scorer see gold locations.

---

## 3. `src/quality/policy.py` — score → budget

```python
from src.schemas import Budget

def choose_budget(score: int, cfg) -> Budget:
    """Low quality => more effort; high quality => less. Reads cfg['budgets']."""
    if score <= 1:   tier = "low"
    elif score == 2: tier = "medium"
    else:            tier = "high"
    b = cfg["budgets"][tier]
    return Budget(max_candidates=b["max_candidates"], max_hops=b["max_hops"],
                  max_samples=b["max_samples"], model=cfg["model"])

def fixed_budget(cfg) -> Budget:
    """Baseline control: same effort for every report."""
    b = cfg["baseline_budget"]
    return Budget(max_candidates=b["max_candidates"], max_hops=b["max_hops"],
                  max_samples=b["max_samples"], model=cfg["model"])
```

---

## 4. `src/localizer/agent.py` — the loop (uses Role 2's tools)

Keep it simple and budget-bounded. One pass = extract keywords → search → traverse →
retrieve → rank. Repeat `max_samples` times and merge by how often each entity appears.

```python
import re, json
from collections import Counter
from src.schemas import Instance, Budget, Prediction
from src.llm import call
from src.graph import tools   # ROLE 2

def _keywords(problem_statement: str, model: str) -> list[str]:
    sys = "Extract up to 6 code-relevant keywords (class/function/file names, error terms). Reply as a comma-separated list."
    txt, *_ = call(model, sys, problem_statement, max_tokens=60)
    return [w.strip() for w in txt.split(",") if w.strip()][:6]

def _rank_once(inst, budget, graph):
    kws = _keywords(inst.problem_statement, budget.model)
    seeds, cost_t, cost_u = [], 0, 0.0
    for kw in kws:
        for e in tools.search_entity(graph, kw, detail="preview")[:budget.max_candidates]:
            seeds.append(e["id"])
    tree = tools.traverse_graph(graph, seeds[:budget.max_candidates],
                                hops=budget.max_hops,
                                edge_types=["contain", "invoke", "import", "inherit"])
    sys = ("You are localizing a bug. Given the report and this repo structure, list the most "
           "likely buggy entities as 'path/file.py:function', most-suspicious first, one per line.")
    user = f"REPORT:\n{inst.problem_statement}\n\nREPO STRUCTURE:\n{tree}"
    txt, t, u = call(budget.model, sys, user, max_tokens=400)
    cost_t += t; cost_u += u
    ents = [ln.strip() for ln in txt.splitlines() if ":" in ln][:budget.max_candidates]
    return ents, cost_t, cost_u

def localize(inst: Instance, budget: Budget, graph) -> Prediction:
    votes, files, T, U = Counter(), Counter(), 0, 0.0
    for _ in range(budget.max_samples):
        ents, t, u = _rank_once(inst, budget, graph); T += t; U += u
        for rank, e in enumerate(ents):
            votes[e] += (len(ents) - rank)                 # reciprocal-rank style
            files[e.split(":")[0]] += (len(ents) - rank)
    ranked_functions = [e for e, _ in votes.most_common()]
    ranked_files = [f for f, _ in files.most_common()]
    return Prediction(instance_id=inst.instance_id, ranked_files=ranked_files,
                      ranked_functions=ranked_functions, tokens=T, usd=U)
```

---

## 5. Run scripts (the two experiments)

```python
# scripts/run_baseline.py   — fixed effort (control)
import json, yaml
from src.config import load_config
from src.data.loader import load_dataset          # ROLE 3
from src.graph.indexer import build_graph, checkout_repo  # ROLE 2
from src.quality.policy import fixed_budget
from src.localizer.agent import localize

cfg = load_config()
insts = load_dataset(cfg["dataset"], cfg["split"], cfg["limit"])
preds = []
for inst in insts:
    repo_dir = checkout_repo(inst.repo, inst.base_commit, cfg["paths"]["data"])
    graph = build_graph(repo_dir)
    preds.append(localize(inst, fixed_budget(cfg), graph))
with open(f"{cfg['paths']['outputs']}/baseline_predictions.jsonl", "w") as f:
    for p in preds: f.write(json.dumps(p.__dict__) + "\n")
```

`scripts/run_adaptive.py` is identical **except** these two lines:

```python
from src.quality.scorer import score_quality
from src.quality.policy import choose_budget
...
    score = score_quality(inst.problem_statement)
    budget = choose_budget(score, cfg)
    p = localize(inst, budget, graph); p.quality_score = score
```

That single difference — fixed budget vs. quality-chosen budget — **is** the experiment.

---

## 6. Stubs so you're not blocked

Until Role 2/3 land, drop these in to run end-to-end today:

```python
# temporary fake tools
def search_entity(graph, keyword, detail): return [{"id": f"foo/bar.py:{keyword}"}]
def traverse_graph(graph, seeds, hops, edge_types): return "\n".join(seeds)
def build_graph(repo_dir): return object()
def checkout_repo(repo, commit, data_dir): return "/tmp/fake"
def load_dataset(name, split, limit):
    from src.schemas import Instance
    return [Instance("x__x-1","x/x","abc","App crashes with KeyError in cache.get()",["src/cache.py"],["src/cache.py:get"])]
```

---

## 7. Your definition of done

- `score_quality` returns 0..4 and is covered by `tests/test_scorer.py` (feed it a rich report → high, a vague one → low).
- `run_baseline.py` and `run_adaptive.py` both produce prediction files on `limit: 5`.
- The two scripts differ **only** in scorer+policy — verify by diff.
- Cost is populated on every `Prediction` (non-zero `tokens`/`usd`).
- You can state, in one sentence, why adaptive should beat fixed on low-quality reports.
