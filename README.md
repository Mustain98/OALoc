# Quality-Adaptive LLM Bug Localization

A bug localizer that measures the **quality of a bug report** first and **adapts how much
search effort to spend** — more effort on vague reports, less on clear ones. Baseline =
fixed effort. Our system = effort that scales with report quality. We prove it with an
**accuracy-vs-cost** comparison.

It uses a **LocAgent-style code graph** to navigate the target repository (the same
backbone as [LocAgent, ACL 2025](https://arxiv.org/abs/2503.09089)) and adds one new
idea on top: existing efficient LLM localizers (LocAgent, BLAgent) spend the same effort
on every report; this project conditions effort on a measured report-quality signal.

All code lives in `bug-localization/`. This file is now the single source of truth for
architecture, setup, running it, and the methodology to defend — `01_TAJ_...md`,
`02_FARHAN_...md`, `03_FAHIM_...md` remain as the original per-role build docs (useful
history of who built what and the sketches they started from), but everything current
and load-bearing is here.

---

## Status

All three roles are implemented and merged, plus work added after all three landed.

| Part | Owner | State |
|---|---|---|
| `src/schemas.py`, `src/config.py`, `config.yaml` | shared | done (schemas frozen) |
| `src/llm.py` | Taj | done |
| `src/quality/scorer.py`, `src/quality/policy.py` | Taj | done |
| `src/localizer/agent.py`, `src/localizer/run.py` | Taj | done (rewritten to a LangGraph agent — see Methodology) |
| `scripts/run_baseline.py`, `scripts/run_adaptive.py` | Taj | done |
| `scripts/score_distribution.py` | Taj | done |
| `src/graph/indexer.py`, `src/graph/tools.py` | Farhan | done (+ hierarchical search, direction-aware traversal — see Fidelity to LocAgent) |
| `src/graph/cache.py` (persistent graph cache) | Farhan | done |
| `src/data/loader.py`, `src/eval/evaluate.py` | Fahim | done |
| `app.py`, `src/ui/*`, `src/service/*` (web UI) | shared | done |

---

## Architecture

```
                 ┌─────────────────────────────────────────────────────────┐
  bug report ──► │ (A) QUALITY SCORER  ──►  score 0..4                      │
                 │ (B) BUDGET POLICY   ──►  budget {candidates, hops, samples}│  ← Taj
                 └───────────────┬─────────────────────────────────────────┘
                                 │  report + budget
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │ (C) CODE GRAPH  (repo → nodes/edges, built once per       │
                 │     commit, cached to disk — src/graph/cache.py)          │
                 │ (D) TOOLS: search_entity (4-tier: exact id/name, then     │  ← Farhan
                 │     BM25 on ids, then BM25 on content) / traverse_graph   │
                 │     (out/in/both) / retrieve_entity                       │
                 │ (E) LOCALIZER AGENT: LangGraph ReAct loop — the LLM       │
                 │     autonomously calls (D)'s tools until it answers or    │
                 │     the budget is spent ──► ranked files & functions      │
                 └───────────────┬─────────────────────────────────────────┘
                                 │  prediction {ranked_files, ranked_functions, cost}
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │ (F) DATASET loader (SWE-bench Lite)                       │
                 │ (G) EVALUATOR: Acc@k, MRR, cost  ──► results tables/plots │  ← Fahim
                 └─────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │ (H) WEB UI (app.py, src/ui/, src/service/) — a Streamlit  │
                 │     front end over the exact same (A)-(G) functions:      │
                 │     paste a repo + bug report and watch it run live, or   │
                 │     drive (F)/(G) from the browser instead of the CLI.    │
                 └─────────────────────────────────────────────────────────┘
```

(H) was added after (A)-(G) landed, on top of the frozen interfaces below — it required
no change to any of them.

### Repository layout

```
bug-localization/
├── requirements.txt
├── config.yaml                 # runtime settings (model, dataset, paths) — NOT code
├── .env.example                # copy to .env if you switch provider to "anthropic"
├── app.py                      # Streamlit web UI entry point — streamlit run app.py
├── data/                       # cloned repos + graph cache (git-ignored)
│   └── .graph_cache/           # one pickled CodeGraph per repo@commit (src/graph/cache.py)
├── outputs/                    # predictions.jsonl, cost_accuracy.png (git-ignored)
├── src/
│   ├── __init__.py
│   ├── schemas.py              # the shared dataclasses below — frozen, do not edit alone
│   ├── config.py               # loads config.yaml + .env
│   ├── llm.py                  # shared cost-tracking wrapper (ollama | fake | anthropic)
│   ├── quality/                # Taj
│   │   ├── scorer.py
│   │   └── policy.py
│   ├── graph/                  # Farhan
│   │   ├── indexer.py          # checkout_repo, build_graph, CodeGraph
│   │   ├── tools.py            # search_entity / traverse_graph / retrieve_entity
│   │   └── cache.py            # build-once-per-commit disk cache over indexer.build_graph
│   ├── localizer/               # Taj owns the loop; uses Farhan's tools
│   │   ├── agent.py            # the LangGraph ReAct agent — see Methodology
│   │   └── run.py              # shared driver for both run_*.py scripts
│   ├── data/                   # Fahim
│   │   └── loader.py
│   ├── eval/                   # Fahim
│   │   └── evaluate.py
│   ├── service/                # shared — background-job glue for the web UI, not the CLI
│   │   ├── jobs.py             # in-memory job registry (thread + poll)
│   │   └── adhoc.py            # "paste a repo + report" flow, reuses (C)-(E) as-is
│   └── ui/                     # shared — Streamlit pages
│       ├── localize_page.py
│       └── evaluate_page.py
├── scripts/
│   ├── run_baseline.py         # fixed effort (control)
│   ├── run_adaptive.py         # quality-adaptive effort (our method)
│   └── score_distribution.py   # validates the scorer before spending any compute
└── tests/
    ├── test_agent.py
    ├── test_graph.py
    ├── test_scorer.py
    ├── test_policy.py
    ├── test_llm.py
    ├── test_eval.py
    └── fixtures/mini_repo/     # tiny offline fixture repo used by every test above
```

---

## Setup

Takes about 15 minutes, most of it waiting on a model download. **No API key and no
payment is needed** — it runs a free local model by default.

```bash
git clone https://github.com/Mustain98/OALoc.git
cd OALoc/bug-localization
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check it worked — this needs no model and no network:

```bash
pytest tests/ -q
```

Expected: `74 passed`.

### Ollama (the free local model)

**macOS**
```bash
brew install ollama
ollama serve &                     # leave running; or: brew services start ollama
ollama pull qwen2.5-coder:7b       # ~4.7 GB, this is the slow part
```

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5-coder:7b
```

**Windows** — download the installer from <https://ollama.com/download>, then:
```powershell
ollama pull qwen2.5-coder:7b
```

Verify the server is up:
```bash
curl http://localhost:11434/api/tags
```

**Disk:** the model needs ~5 GB, and SWE-bench repos are cloned into
`bug-localization/data/` (several GB per repo — django and sympy are the big ones).
`data/.graph_cache/` adds one small pickle per repo@commit indexed (KBs to low MBs
each, negligible next to the clones). Keep ~20 GB free and prune `data/` between runs.

**Low on RAM or disk?** `qwen2.5:3b` (~2 GB) works for smoke-testing the pipeline:
```bash
python scripts/run_adaptive.py --limit 3 --model qwen2.5:3b
```

---

## Configuration

Everything tunable is in `bug-localization/config.yaml` — no code changes needed.

```yaml
provider: "ollama"                # ollama (free, local) | fake (offline tests) | anthropic
model: "qwen2.5-coder:7b"

ollama:
  host: "http://localhost:11434"
  num_ctx: 16384                  # DO NOT LOWER — see warning below
  timeout: 600                    # seconds; a local 7B on CPU/GPU can be slow

dataset: "princeton-nlp/SWE-bench_Lite"
split: "test"
limit: 30                          # start small; raise to 300 for the full run

budgets:                           # the adaptive policy reads these
  low:    { max_candidates: 50, max_hops: 3, max_samples: 3 }   # for low-quality reports
  medium: { max_candidates: 25, max_hops: 2, max_samples: 2 }
  high:   { max_candidates: 10, max_hops: 1, max_samples: 1 }   # for clear reports
baseline_budget: { max_candidates: 15, max_hops: 2, max_samples: 1 }  # fixed control

paths:
  data: "./data"
  outputs: "./outputs"
  graph_cache: "./data/.graph_cache"   # persisted CodeGraph, keyed by repo@commit
```

> **Do not lower `num_ctx`.** Ollama defaults it to ~4096 and truncates the prompt
> *silently*. The repo-structure tree is most of the prompt, so at the default a
> `hops=3` tree gets cut down to the size of a `hops=1` tree — the effort knob stops
> doing anything and the experiment measures nothing.

Start with `limit: 5` and debug end-to-end before any full run.

**The cost axis is tokens, not dollars.** We run a local model, so `usd` is genuinely
`0.0`. The headline accuracy-vs-cost plot uses `avg_tokens` on the x-axis —
provider-neutral and exactly measurable, unlike a flat line at zero.

---

## Run

```bash
cd bug-localization
source .venv/bin/activate

# Validate the quality scorer first — free, no model calls, decides whether the
# experiment has any signal at all before you spend real compute on it.
python scripts/score_distribution.py

# Offline smoke test — no model, no network, no key
python scripts/run_baseline.py --limit 3 --provider fake --fresh
python scripts/run_adaptive.py --limit 3 --provider fake --fresh

# Real local model
python scripts/run_baseline.py --limit 5
python scripts/run_adaptive.py --limit 5

pytest tests/ -q
```

Results land in `bug-localization/outputs/` as `.jsonl`.

| Flag | Meaning |
|---|---|
| `--limit N` | How many instances (default: `limit` in `config.yaml`) |
| `--provider` | `ollama` (default) / `fake` (offline) / `anthropic` (paid) |
| `--model NAME` | Override the model for one run |
| `--out PATH` | Where to write predictions |
| `--fresh` | Start over instead of resuming |

Runs **checkpoint after every instance and resume by default** — if a run dies at
instance 180 of 300, rerun the same command and it picks up where it stopped. A
prediction that cost 0 tokens is reported as a loud warning at the end of a run: it
means no model output was used, and the results are invalid until the cause is fixed.

---

## Web UI

```bash
cd bug-localization
source .venv/bin/activate
streamlit run app.py
```

Two pages, both calling the exact same functions the CLI above uses:

- **Localize a bug** — paste a GitHub repo (URL or `owner/repo`) and a bug report, pick
  Adaptive or Baseline effort, and watch the agent's live tool-call trace, then see the
  ranked files/functions, the quality score and budget tier chosen, and the actual source
  of the top-ranked entities. An "Offline/fake mode" checkbox runs the whole flow against
  the bundled fixture repo with no Ollama/network needed, useful as a quick smoke test.
  The repository graph is indexed once per commit and cached to disk
  (`data/.graph_cache/`) — re-querying the same commit is near-instant; a "force
  re-index" checkbox bypasses the cache when needed.
- **Evaluate on dataset** — run the SWE-bench-Lite baseline/adaptive arms (same as
  `scripts/run_baseline.py` / `run_adaptive.py`) from the browser with a live progress
  view, or click "Load existing results" to render the Acc@k/MRR comparison table,
  quality-bucket breakdown, and accuracy-vs-tokens plot from whatever is already in
  `outputs/` without running anything.

---

## Shared interfaces (`bug-localization/src/schemas.py`)

**The contract.** All parts of the system import from here; do not change a field
without checking every caller.

```python
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

The function signatures everything else is built against:

```python
# quality
def score_quality(problem_statement: str) -> int: ...          # returns 0..4
def choose_budget(score: int, cfg) -> Budget: ...
def fixed_budget(cfg) -> Budget: ...                            # the baseline control

# graph
def checkout_repo(repo: str, base_commit: str, data_dir: str) -> str: ...
def build_graph(repo_dir: str) -> "CodeGraph": ...
def load_or_build_graph(repo, commit, repo_dir, cache_dir, force=False) -> "CodeGraph": ...
def search_entity(graph, keyword: str, detail: str = "preview") -> list[dict]: ...
def traverse_graph(graph, seeds: list[str], hops: int = 2,
                   edge_types=(...), direction: str = "out") -> str: ...
def retrieve_entity(graph, entity_id: str) -> dict: ...
# direction: 'out' (default, unchanged) | 'in' | 'both' — see Fidelity to LocAgent below.
# tools.py itself defaults to 'out' for backward compatibility; the agent's tool
# wrapper (agent.py) exposes 'both' as ITS default so the LLM gets the richer behavior.

# localizer
def localize(inst: Instance, budget: Budget, graph, progress_cb=None) -> Prediction: ...
# progress_cb, if given, is called once per new agent message (tool call / tool result /
# final answer) — used by the web UI for a live trace; the CLI leaves it at the default
# None and behaves exactly as before.

# data + eval
def load_dataset(name: str, split: str, limit: int) -> list[Instance]: ...
def evaluate(preds: list[Prediction], gold: list[Instance]) -> dict: ...  # Acc@k, MRR, cost
```

---

## Methodology

*The method end to end: what it does, why each step exists, and how the experiment
proves the claim. This is the version to defend in the viva.*

### The hypothesis

One sentence underneath everything: **localization effort should scale with how much
signal the bug report carries.** A clear report (stack trace, code, repro steps) needs
little searching; a vague one needs more. Current systems ignore this and spend a fixed
amount on every report. So the method is a controlled way to test:

> *Does making effort a function of report quality buy accuracy on hard reports and/or
> save cost on easy ones?*

Everything in the pipeline exists to make one variable — **effort** — respond to one
signal — **report quality** — while holding everything else constant.

### The pipeline, stage by stage

**Stage 1 — Quality scoring.** The bug report text goes to a scorer that returns an
integer **0–4**. It counts objective, presence-based features: stack trace/traceback
present? code snippet present? names a class or method explicitly? steps-to-reproduce /
expected-vs-actual structure? Each present feature adds a point. It is deliberately
**rule-based and cheap** — no model call, fully deterministic — because the score is the
load-bearing part of the novelty and must be *objective and defensible* (features come
from the report-quality literature: Fang et al., Chaparro et al.). An optional
LLM-based scorer (`score_quality_llm`) is kept as a comparison point, but the rule-based
one is the default. *Why it exists:* it turns "report quality," which sounds
subjective, into a number a committee can't argue with.

**Stage 2 — Budget policy.** The score maps to a **budget** — three dials: how many
candidate entities to keep, how many graph hops to traverse, how many agent samples to
run (see the `budgets` table in Configuration above). Low score → big budget; high score
→ small budget. The mapping is a lookup table in `config.yaml`, not buried in code, so
it can be tuned and its sensitivity reported. *Why it exists:* this is the mechanism
that converts the quality signal into actual compute — the genuinely new part, turning
emergent effort variation (as in AutoFL) into an *explicit, signal-driven* policy.

**Stage 3 — Graph construction.** Independently, the target repository (checked out at
the exact **pre-fix commit**) is parsed into a code graph: nodes are
files/classes/functions; edges are `contain` / `import` / `invoke` / `inherit`. This is
the LocAgent backbone — it lets the localizer navigate a repo far too large for a
context window, and follow *dependency* links rather than just text matches. Built once
per repository **commit**, and cached to disk (`src/graph/cache.py`) so the same commit
is never re-parsed on a later run. *Why it exists:* it is how the system reaches buggy
code the report doesn't literally name — code a few call-graph hops away from the
symptom.

**Stage 4 — Localizer agent loop.** The report plus its budget drive the search. **One
sample** = the LLM is handed all three tools (`search_entity`, `traverse_graph`,
`retrieve_entity`) and runs an autonomous ReAct-style loop: it decides step by step
which tool to call and when it has enough evidence to answer, up to `max_hops` tool
calls, after which it is forced to give its best answer immediately. In practice this
usually still looks like keywords → search → traverse → rank, but the model — not a
fixed script — chooses the order and can skip or repeat a step. This is closer to how
LocAgent's own agent actually behaves than a fixed pipeline would be.

Two of the three tools are themselves closer to LocAgent's design than a first pass
needs to be: `search_entity` tries an exact-ID match, then an exact-name match, then
BM25 over entity IDs, then BM25 over entity content, in that order (the paper's "sparse
hierarchical entity indexing," §3.1) — so a keyword that's a literal identifier resolves
immediately, and one that only appears inside a function body (e.g. a global variable)
is still findable. `traverse_graph` can walk a seed's out-edges (what it uses), in-edges
(what uses/calls/imports it), or both — so the agent can ask "who else touches this" and
not just "what does this touch."

The loop repeats **`max_samples`** times and merges runs by **reciprocal-rank voting**
(`1/(rank+1)`, not Borda count — an entity ranked high across several runs floats to the
top, and a file's score is the best entity per file per sample, not the sum, so a file
isn't pushed up by a long tail of low-ranked functions). Output: ranked files and
functions, plus token/dollar cost.

The three budget dials (candidates, hops, samples) **are** the effort knobs. On a
low-quality report the loop searches wider, walks farther, and votes across more runs;
on a high-quality one it does the minimum. **That difference is the entire method.**
*Why the multi-sample step:* it improves stability and is itself a knob — running a
vague report 3× and aggregating is one concrete way "more effort" cashes out (borrowed
from AutoFL's confidence aggregation, honest to cite). It only works if repeated passes
actually differ — see "Things that will silently ruin the results" below.

### The experimental design (what a committee actually grades)

Run the **same** system twice on the **same** instances, changing exactly one thing:

- **Baseline (control):** every report gets a **fixed** budget (`baseline_budget` in
  `config.yaml`) — what LocAgent / BLAgent effectively do.
- **Adaptive (treatment):** each report's budget is **chosen by its quality score**.

The graph, tools, model, prompts, and dataset are identical between the two runs, so any
difference in accuracy or cost is attributable to the adaptation and nothing else. That
is what makes it a **controlled experiment**, not a demo. As built, the two run scripts
don't duplicate the loop to guarantee this — they both call one shared
`run_experiment(budget_fn, out_name, args, cfg)` in `src/localizer/run.py`, passing only
their own `budget_for(inst, cfg) -> (Budget, score)` closure, so "differ only in
scorer+policy" is true by construction. Verify any time the scripts are touched:

```bash
diff bug-localization/scripts/run_baseline.py bug-localization/scripts/run_adaptive.py
```

### How success is measured

Two standard metrics, at file level (primary) and function level (secondary):

- **Acc@k** — did *all* gold files land in the top-k (strict, matches LocAgent)? Report
  k = 1, 3, 5.
- **MRR** — how high the first correct file ranks.
- **Cost** — average tokens (and USD, genuinely `0.0` on the local provider) per
  instance.

**Ground truth** comes from the fixing patch in SWE-bench Lite: the files/functions the
real fix touched are the gold locations. The patch is used **only** to derive gold
labels in the evaluator — it never touches the localizer.

**Headline result:** a single **accuracy-vs-cost plot** — baseline point vs adaptive
point (`outputs/cost_accuracy.png`, `avg_tokens` on the x-axis). **The money result:**
the **breakdown by quality bucket** — adaptive should show its biggest accuracy gain on
the **low-quality** reports, the failure class current tools ignore. If that bucket
moves, the thesis is demonstrated; if only cost drops, that's the efficiency half.

### Honest methodological points (say these before they're asked)

- **What's held constant vs varied.** One variable (the budget-selection rule);
  everything else fixed. State it explicitly — it is the validity argument.
- **The scorer is the risk.** If the quality score is noisy, reports route to the wrong
  budget and you lose on both axes. So the method's first internal check is: *does the
  score correlate with localization difficulty on our data?* Validate that **before**
  claiming the adaptive policy works (`scripts/score_distribution.py`) — it is step one
  of the results, not an assumption.
- **This is a controlled comparison, not a SOTA chase.** The agent loop is intentionally
  simple. The claim is not "we beat LocAgent's 92.7%"; it is that *within one system*,
  quality-adaptive effort dominates fixed effort on the accuracy–cost trade-off. Scoped
  that way, it is defensible.

### Relation to prior work (one line each)

- **LocAgent (ACL 2025):** supplies the graph + tools backbone; spends fixed effort per
  report — the gap this method fills.
- **AutoFL (FSE 2024):** shows effort *already* varies with difficulty, but only
  emergently; this method makes it explicit and signal-driven.
- **Fang et al. (Soft Computing 2021):** conditions on report quality by *gating* (skip
  bad reports); this method instead *adapts effort to recover* them — opposite
  direction.

---

## Fidelity to LocAgent

This project reimplements the core of [LocAgent](https://arxiv.org/abs/2503.09089): the
heterogeneous code graph, the three navigation tools, and reciprocal-rank confidence
aggregation across multiple agent samples. Two gaps have been closed to match the paper
more closely:

- **`search_entity` is a 4-tier hierarchy** (paper §3.1): exact entity-ID match, then
  exact entity-name match, then BM25 over entity IDs, then BM25 over entity content —
  instead of a single flat BM25 index. Each hit carries a `"match"` field saying which
  tier found it. See `bug-localization/src/graph/tools.py::search_entity`.
- **`traverse_graph` is direction-aware** (paper Table 2 / Figure 7): the agent can walk
  `out` edges (what an entity uses), `in` edges (what uses/calls/imports it, rendered
  with a `-by` suffix like `invoke-by`), or `both`. See
  `bug-localization/src/graph/tools.py::traverse_graph`.

Repository graphs are also now indexed once per commit and cached to disk
(`bug-localization/src/graph/cache.py`, `data/.graph_cache/`) rather than rebuilt on
every run.

What's still simplified relative to the paper: no fine-tuned model (a stock
`qwen2.5-coder:7b`/`32b` via Ollama, not SFT'd on agent trajectories), and evaluation
covers file/function Acc@k + MRR on SWE-bench-Lite only (no NDCG, no module-level
accuracy, no Loc-Bench). What this project adds that LocAgent doesn't have: the
quality-adaptive effort budget (`src/quality/`) — see Methodology above.

---

## Where the implementation deviates from the original role-doc sketches

`01_TAJ_quality_and_orchestration.md`, `02_FARHAN_graph_and_tools.md`, and
`03_FAHIM_data_and_evaluation.md` are the original per-role specs with code sketches.
Each was followed except where doing so would have broken the result, or where the
design changed after they were written. Each change is commented at the site in code;
the summary:

- **The agent loop is no longer the sketched fixed pipeline.** `01_TAJ` §4 sketched
  keywords → search → traverse → rank as one fixed pass. `src/localizer/agent.py` is
  instead a LangGraph agent that autonomously decides which tool to call and when to
  stop (see Methodology, Stage 4) — closer to how LocAgent's own agent actually behaves,
  and a strict superset of the fixed pipeline.
- **`search_entity`/`traverse_graph` grew beyond `02_FARHAN` §3's sketch** into the
  4-tier hierarchy and direction-aware traversal described above (Fidelity to LocAgent).
- **`load_dataset` reads a local parquet file, not the HF `datasets` library.**
  `03_FAHIM` §2 sketched `datasets.load_dataset(...)`; `src/data/loader.py` instead
  loads `swe_bench_lite_test.parquet` (checked into the repo root) via `pandas` — one
  less runtime dependency on the HF Hub being reachable.
- **Keyword-extraction cost is counted.** The `01_TAJ` sketch discarded it
  (`txt, *_ = call(...)`) while cost is the headline metric.
- **Reciprocal-rank voting** (`1/(rank+1)`), as Methodology Stage 4 describes. The
  sketch used `len(ents) - rank` (Borda count) — a different method than the one
  defended here.
- **Robust output parsing.** The sketch kept any line containing `":"`. A local 7B emits
  numbered lists, bullets, fences and a closing pleasantry; unparsed junk can never
  match a gold path, so a run would score zero for reasons unrelated to the method.
- **The scorer's "names a class/method" feature was tightened.**
  `\b[A-Z][a-zA-Z0-9]+\b` matches any capitalised word including "The" and "When", so
  the feature fired on nearly every report and carried no signal. It now requires a
  real identifier (CamelCase with an internal capital, a dotted path, or a
  `def`/`class` declaration).
- **The baseline/adaptive loop lives in `src/localizer/run.py`**, shared by both
  scripts, so "differ only in scorer+policy" is true by construction (see Methodology,
  Experimental design).
- **Temperature is not pinned in the original design** — see the open issue flagged
  below, since the current agent implementation does pin it.
- Repository graphs are now indexed once per commit and cached to disk
  (`src/graph/cache.py`) instead of rebuilt on every run.
- A Streamlit web UI (`app.py`, `src/ui/`, `src/service/`) was added on top of
  everything above, calling the exact same functions as the CLI.

---

## Things that will silently ruin the results

These are not style notes. Each one produces numbers that look fine and mean nothing.

**Never let gold data reach the localizer.** `gold_files` and `gold_functions` are for
evaluation only. The scorer, the tools and the agent must never see them — passing them
in would invalidate every result. Enforced structurally: none of `build_graph`,
`checkout_repo`, `search_entity`, `traverse_graph`, `retrieve_entity` even accept an
`Instance`.

**Don't lower `num_ctx` in `config.yaml`.** See Configuration above — at the Ollama
default (~4096) the repo-structure tree gets truncated *silently* and the `max_hops`
effort knob stops doing anything.

**A prediction that cost 0 tokens is not a result.** It means no model output was used.
The run scripts print a loud warning when this happens; don't ignore it. (This already
happened once: with the model not yet pulled, every call 404'd and the run still
reported "3 ok" and wrote a perfectly well-formed predictions file built from nothing.)

**Never pin `temperature=0`.** `max_samples` is one of the three effort knobs and only
works if repeated passes differ. Pinning temperature makes every sample identical and
turns the knob into a silent no-op.

> ⚠️ **Open issue, not yet fixed:** the current `src/localizer/agent.py` (the
> LangGraph rewrite) constructs `ChatOllama(model=budget.model, temperature=0.0)` in
> both `build_langgraph_app` and `force_answer_node` — i.e. it pins exactly the value
> this rule warns against. This makes `max_samples > 1` close to a no-op (repeated
> passes will mostly agree), which quietly weakens the RRF voting story in Methodology
> Stage 4. Whoever picks this up next should either remove `temperature=0.0` (matching
> the original design) or, if it was pinned deliberately for more reliable tool-call
> formatting on a 7B model, document that trade-off explicitly and re-validate that
> `max_samples` still does something measurable.

**Don't let the two run scripts drift apart.** See "The experimental design" above —
verify with `diff bug-localization/scripts/run_baseline.py
bug-localization/scripts/run_adaptive.py` any time they're touched.

**`data/` and `outputs/` are git-ignored on purpose.** Repo clones and graph-cache
pickles are large; predictions are cheap to regenerate. Reproducibility comes from
re-running the scripts with the same `config.yaml` and `--limit`, not from committing
run artifacts.

---

## Definition of done

1. `python scripts/run_baseline.py` → produces `outputs/baseline_predictions.jsonl`.
2. `python scripts/run_adaptive.py` → produces `outputs/adaptive_predictions.jsonl`.
3. `python -m src.eval.evaluate --compare` → reads both, prints the Acc@k/MRR/cost table
   and the by-quality breakdown, and saves `outputs/cost_accuracy.png`. (Or from the
   browser: `streamlit run app.py` → "Evaluate on dataset" → "Load existing results".)
4. Expected story: **adaptive ≈ or > baseline accuracy at lower or equal average cost**,
   with the gain concentrated on low-quality reports. Even a small, clearly-measured
   effect is a valid result — do not fake numbers.
5. `pytest tests/ -q` passes fully offline (74 tests as of this writing, no model or
   network needed).

---

## Troubleshooting

**`cannot reach Ollama at http://localhost:11434`**
The server isn't running. `ollama serve &`

**`model 'qwen2.5-coder:7b' is not available in Ollama`**
Not pulled yet. `ollama pull qwen2.5-coder:7b`

**`WARNING: n/n predictions cost 0 tokens`**
No model output was used — the results are invalid. Usually the server died mid-run or
the model isn't pulled. Fix the cause and rerun with `--fresh`.

**`ModuleNotFoundError: No module named 'src'`**
Run the scripts from inside `bug-localization/`, not from the repo root.

**Runs are very slow**
Normal for a local model on a long prompt. Lower `limit`, or use `--model qwen2.5:3b`
while developing. First call after `ollama serve` also pays a model-load cost.

---

## Optional: using a paid API instead

Only if the local model turns out too weak. One line in `config.yaml`:

```yaml
provider: "anthropic"
model: "claude-sonnet-5"
```

Then `pip install anthropic` and put `ANTHROPIC_API_KEY=sk-ant-...` in
`bug-localization/.env` (copy `.env.example`). That file is git-ignored — **never
commit a key**.

Rough cost: ~$0.13 per instance, so ~$8 for a 30-instance baseline+adaptive pair and
~$80 for the full 300. A Claude Pro/Max subscription does **not** cover this — the API
is billed separately.
