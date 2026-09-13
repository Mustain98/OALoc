# Read this before you write any code

Taj (quality scorer, budget policy, agent loop, run scripts) is **done and on `main`**.
Roles 2 and 3 are **stubbed** — your job is to replace the stubs.

This file is the short version. `00_PROJECT_GUIDE.md` is the full spec; `04_METHODOLOGY.md`
is the method to defend in the viva.

---

## 1. Two decisions that change what you write

### The cost axis is TOKENS, not dollars

We run a **free local model through Ollama**, not a paid API. So `Prediction.usd` is
genuinely `0.0` on every run.

**Fahim:** the headline accuracy-vs-cost plot must use `avg_tokens` on the x-axis.
`03_FAHIM_data_and_evaluation.md §4` currently plots `avg_usd` — that will be a flat line at
zero. `Prediction.tokens` already carries what you need, so **no schema change is required**;
just plot the other field.

Tokens are the better axis anyway — provider-neutral and exactly measurable.

### `schemas.py` is frozen

`bug-localization/src/schemas.py` is the contract all three of us import. It is **exactly**
what `00_PROJECT_GUIDE.md §4` specifies, and it is frozen. If you think you need a new field,
raise it with all three of us first — do not edit it on your branch.

---

## 2. What is stubbed and waiting for you

Every stub file starts with `# STUB — owned by FARHAN` or `# STUB — owned by FAHIM`. They
implement the real signatures from `00_PROJECT_GUIDE.md §4`, so **replace them wholesale** —
none of Taj's code touches their internals.

| File | Owner | What the stub currently does |
|---|---|---|
| `bug-localization/src/graph/indexer.py` | **Farhan** | Indexes a local fixture dir with `ast`. Does not clone anything. |
| `bug-localization/src/graph/tools.py` | **Farhan** | Substring matching instead of BM25. |
| `bug-localization/src/data/loader.py` | **Fahim** | Returns 3 hand-written instances, not SWE-bench. |
| `bug-localization/src/eval/evaluate.py` | **Fahim** | `evaluate()` works; the comparison table, quality breakdown and plot are missing. |

The four signatures Taj calls, which must keep working:

```python
# Farhan
def checkout_repo(repo: str, base_commit: str, data_dir: str) -> str: ...
def build_graph(repo_dir: str) -> "CodeGraph": ...
def search_entity(graph, keyword: str, detail: str) -> list[dict]: ...   # dicts need an "id" key
def traverse_graph(graph, seeds: list[str], hops: int, edge_types: list[str]) -> str: ...
def retrieve_entity(graph, entity_id: str) -> dict: ...

# Fahim
def load_dataset(name: str, split: str, limit: int) -> list[Instance]: ...
def evaluate(preds: list[Prediction], gold: list[Instance]) -> dict: ...
```

---

## 3. Branches

| Branch | Purpose |
|---|---|
| `main` | Stable. Only merge things that pass `pytest`. |
| `demo` | **Integration branch.** Open your PRs against this one; Mustain merges here. |
| `taj` | Taj's working branch. |

Make your own branch off `demo` — `farhan-graph` or `fahim-eval`, per
`00_PROJECT_GUIDE.md §7` — and PR into `demo`.

```bash
git fetch origin
git checkout -b farhan-graph origin/demo
# ... work ...
git push -u origin farhan-graph
```

---

## 4. Things that will silently ruin the results

These are not style notes. Each one produces numbers that look fine and mean nothing.

**Never let gold data reach the localizer.** `gold_files` and `gold_functions` are for
evaluation only. The scorer, the tools and the agent must never see them —
`00_PROJECT_GUIDE.md §7` is explicit that this invalidates every result.

**Don't lower `num_ctx` in `config.yaml`.** Ollama defaults it to ~4096 and truncates the
prompt *silently*. The repo-structure tree is the bulk of our prompt, so at the default a
`hops=3` tree gets cut to the size of a `hops=1` tree — the effort knob stops doing anything
and the experiment measures nothing.

**A prediction that cost 0 tokens is not a result.** It means no model output was used. The
run scripts print a loud warning when this happens; don't ignore it. (This already bit us
once: with the model not yet pulled, every call 404'd and the run still reported "3 ok" and
wrote a perfectly well-formed predictions file built from nothing.)

**Never pin `temperature=0`.** `max_samples` is one of the three effort knobs and only works
if repeated passes differ. Pinning temperature makes every sample identical and turns the
knob into a silent no-op.

**Don't let the two run scripts drift apart.** The whole experiment rests on baseline and
adaptive differing *only* in scorer+policy. The shared loop lives in
`src/localizer/run.py` so this is true by construction. Verify any time you touch them:

```bash
diff bug-localization/scripts/run_baseline.py bug-localization/scripts/run_adaptive.py
```

---

## 5. Fahim: run this first, before anything else

```bash
python scripts/score_distribution.py
```

It costs nothing — no model calls, runs in seconds — and it decides whether the experiment
has any signal at all. If most reports land in one quality tier, the adaptive arm picks the
same budget nearly every time and both arms produce identical numbers. `04_METHODOLOGY.md`
calls this validation *"step one of the results, not an assumption."*

It needs your real loader to be meaningful, so it is the first thing worth running once
`load_dataset` is real.

---

## 6. Where Taj deviates from `01_TAJ_quality_and_orchestration.md`

The role doc's code sketches were followed except where they would have broken the result.
Each is commented at the site, and the full list is in `bug-localization/README.md`. The
short version:

- The keyword-extraction call's cost is now counted (the sketch discarded it, while cost is
  the headline metric).
- Voting is reciprocal rank `1/(rank+1)`, as `04_METHODOLOGY.md` Stage 4 describes — the
  sketch used Borda count.
- Model output is parsed robustly; a small local model emits numbered lists, bullets and
  fences, and unparsed junk can never match a gold path.
- The scorer's "names a class/method" regex was tightened; the original matched any
  capitalised word including "The", so it fired on nearly every report and carried no signal.
