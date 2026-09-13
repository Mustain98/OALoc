# Quality-Adaptive LLM Bug Localization

A bug localizer that measures the **quality of a bug report** first and **adapts how much
search effort to spend** — more effort on vague reports, less on clear ones. Baseline =
fixed effort. Our system = effort that scales with report quality.

See `../00_PROJECT_GUIDE.md` for the architecture and `../04_METHODOLOGY.md` for the method.

---

## Status

| Part | Owner | State |
|---|---|---|
| `src/schemas.py`, `src/config.py`, `config.yaml` | shared | done (schemas frozen) |
| `src/llm.py` | Taj | done |
| `src/quality/scorer.py`, `src/quality/policy.py` | Taj | done |
| `src/localizer/agent.py`, `src/localizer/run.py` | Taj | done |
| `scripts/run_baseline.py`, `scripts/run_adaptive.py` | Taj | done |
| `scripts/score_distribution.py` | Taj | done |
| `src/graph/indexer.py`, `src/graph/tools.py` | **Farhan** | **STUB** |
| `src/data/loader.py`, `src/eval/evaluate.py` | **Fahim** | **STUB** |

Every stub carries a `# STUB — owned by FARHAN` or `# STUB — owned by FAHIM` header and
implements the exact signatures from `00_PROJECT_GUIDE.md §4`. Replace them wholesale; none
of Taj's code touches their internals.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API key is needed. The default backend is **Ollama, running locally and free**:

```bash
brew install ollama
ollama serve &
ollama pull qwen2.5-coder:7b
```

## Run

```bash
# Offline smoke test — no model, no network, no key
python scripts/run_baseline.py --limit 3 --provider fake --fresh
python scripts/run_adaptive.py --limit 3 --provider fake --fresh

# Real local model
python scripts/run_baseline.py --limit 5
python scripts/run_adaptive.py --limit 5

# Validate the scorer before spending compute (free, no model calls)
python scripts/score_distribution.py

pytest tests/ -q
```

Flags: `--limit N`, `--provider ollama|fake|anthropic`, `--model NAME`, `--out PATH`,
`--fresh`. Runs checkpoint after every instance and **resume by default** — rerun the same
command after a crash and it picks up where it stopped. Pass `--fresh` to start over.

A prediction that cost 0 tokens is reported as a loud warning at the end of a run: it means
no model output was used, and the results are invalid until the cause is fixed.

---

## Two things the team needs to know

**1. The cost axis is tokens, not dollars.** We run a local model, so `usd` is genuinely
`0.0`. The headline accuracy-vs-cost plot must use `avg_tokens` on the x-axis.
`Prediction.tokens` already carries it, so **no schema change is needed** — but
`03_FAHIM_data_and_evaluation.md §4` currently plots `avg_usd`, so Fahim must adjust.
Tokens are the better axis anyway: provider-neutral and exactly measurable.

**2. `num_ctx` must stay set in `config.yaml`.** Ollama defaults it to ~4096 and truncates
the prompt *silently*. The repo-structure tree is the bulk of our prompt, so at the default
a `hops=3` tree gets cut down to the size of a `hops=1` tree — the effort knob stops doing
anything and the experiment measures nothing.

---

## Where this deviates from `01_TAJ_quality_and_orchestration.md`

The role doc's code sketches were followed except where they would have broken the result.
Each change is commented at the site:

- **Keyword-extraction cost is counted.** The sketch discarded it (`txt, *_ = call(...)`)
  while cost is the headline metric.
- **Reciprocal-rank voting** (`1/(rank+1)`), as `04_METHODOLOGY.md` Stage 4 describes. The
  sketch used `len(ents) - rank`, which is Borda count — a different method than the one the
  write-up defends. File score is the best entity per file per sample, not the sum, so a file
  isn't pushed up by a long tail of low-ranked functions.
- **Robust output parsing.** The sketch kept any line containing `":"`. A local 7B emits
  numbered lists, bullets, fences and a closing pleasantry; unparsed junk can never match a
  gold path, so the run would score zero for reasons unrelated to the method.
- **Feature 3 of the scorer tightened.** `\b[A-Z][a-zA-Z0-9]+\b` matches any capitalised
  word including "The" and "When", so the feature fired on nearly every report and carried no
  signal. Now requires a real identifier.
- **The loop lives in `src/localizer/run.py`**, shared by both scripts. Taj's DoD requires
  the two arms differ only in scorer+policy; sharing the driver makes that true by
  construction instead of by discipline. Verify with
  `diff scripts/run_baseline.py scripts/run_adaptive.py`.
- **Temperature is never pinned.** `max_samples` only works as an effort knob if repeated
  passes differ; `temperature=0` would make them identical and silently neuter the knob.
