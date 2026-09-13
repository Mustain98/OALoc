# Taj — Quality Scorer, Budget Policy & Orchestration  (YOU)

> **This is the novel core of the project — the part that makes it publishable.**
> You own the report-quality scorer, the budget policy, the agent loop, and the two run
> scripts that tie everyone's work together. Read the root `README.md` first (Architecture
> and Methodology sections).
> You depend on Farhan's tools (`src/graph/tools.py`) and Fahim's loader/evaluator, so
> stub them (see §6) until they land, then swap in the real ones.
>
> **Status: done — and §4's loop below has since been replaced with a LangGraph
> ReAct agent.** `src/llm.py`, `src/quality/scorer.py`, `src/quality/policy.py`, and the
> run scripts match §1/§2/§3/§5 below closely (see the root `README.md`'s "Where the
> implementation deviates..." section for the small, deliberate departures — RRF not
> Borda-count voting, robust output parsing, cost counted on the keyword call, etc.).
> `src/llm.py` also grew an `ollama` backend as the default (free, local — see
> `README.md`'s Methodology/Configuration sections, "the cost axis is tokens, not
> dollars"), alongside the `anthropic` one sketched in §1. §4's agent loop is the one
> part that changed structurally, not just in small ways — see the note after §4.

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

## 4. `src/localizer/agent.py` — the loop (uses Farhan's tools)

Keep it simple and budget-bounded. One pass = extract keywords → search → traverse →
retrieve → rank. Repeat `max_samples` times and merge by how often each entity appears.

```python
import re, json
from collections import Counter
from src.schemas import Instance, Budget, Prediction
from src.llm import call
from src.graph import tools   # FARHAN

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

**As built, this sketch was superseded.** The fixed one-pass pipeline above (keywords →
search → traverse → rank) is replaced in `src/localizer/agent.py` by a **LangGraph
ReAct loop**: the LLM is bound all three tools directly and decides step by step which
to call and when it has enough to answer, up to `budget.max_hops` tool calls, after
which a `force_answer` node cuts it off. This is closer to how LocAgent's own agent
actually behaves (it doesn't run a fixed sequence either), and it's a strict superset —
nothing stops the model from choosing to do keywords→search→traverse→rank if that's the
best strategy for a given report. What's unchanged from this sketch: the three budget
dials mean the same thing, reciprocal-rank voting across `max_samples` samples works the
same way (see `README.md` Methodology, Stage 4), and `localize(inst, budget, graph) ->
Prediction` is still the exact call `run.py` makes (it also now accepts an optional
`progress_cb` for the web UI's live trace — CLI callers ignore it). See
`README.md` "Fidelity to LocAgent" for the two tool-level upgrades
(hierarchical search, direction-aware traversal) this loop gets "for free" by calling
Farhan's tools.

---

## 5. Run scripts (the two experiments)

```python
# scripts/run_baseline.py   — fixed effort (control)
import json, yaml
from src.config import load_config
from src.data.loader import load_dataset          # FAHIM
from src.graph.indexer import build_graph, checkout_repo  # FARHAN
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

**As built**, the two scripts don't duplicate this loop — they each call one shared
`run_experiment(budget_fn, out_name, args, cfg)` in `src/localizer/run.py`, passing only
their own `budget_for(inst, cfg) -> (Budget, score)` closure. That guarantees the "differ
only in scorer+policy" requirement by construction rather than by discipline — see
`README.md`'s deviation log. `run_experiment` also now indexes each
repo through Farhan's `load_or_build_graph` disk cache instead of calling `build_graph`
fresh every time (see `02_FARHAN_graph_and_tools.md`).

---

## 6. Stubs so you're not blocked

**No longer needed** — Farhan's and Fahim's real implementations have long since
landed. Left below for history / for anyone bootstrapping a similar project from
scratch.

Until Farhan/3 land, drop these in to run end-to-end today:

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
