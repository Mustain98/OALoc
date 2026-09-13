# Setting up the project

Every teammate runs this once. Takes about 15 minutes, most of it waiting on a download.
**No API key and no payment is needed** — we run a free local model.

---

## 1. Clone

```bash
git clone https://github.com/Mustain98/OALoc.git
cd OALoc
```

All code lives in the `bug-localization/` subdirectory; the shared docs sit at the root.

---

## 2. Python environment

Python 3.10 or newer (developed on 3.13).

```bash
cd bug-localization
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check it worked — this needs no model and no network:

```bash
pytest tests/ -q
```

Expected: `36 passed`.

---

## 3. Ollama (the free local model)

### macOS

```bash
brew install ollama
ollama serve &                     # leave running; or: brew services start ollama
ollama pull qwen2.5-coder:7b       # ~4.7 GB, this is the slow part
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5-coder:7b
```

### Windows

Download the installer from <https://ollama.com/download>, then in PowerShell:

```powershell
ollama pull qwen2.5-coder:7b
```

Verify the server is up:

```bash
curl http://localhost:11434/api/tags
```

**Disk:** the model needs ~5 GB, and SWE-bench repos are cloned into `bug-localization/data/`
(several GB per repo — django and sympy are the big ones). Keep ~20 GB free and prune
`data/` between runs.

**Low on RAM or disk?** `qwen2.5:3b` (~2 GB) works for smoke-testing the pipeline:

```bash
python scripts/run_adaptive.py --limit 3 --model qwen2.5:3b
```

---

## 4. Run it

```bash
cd bug-localization
source .venv/bin/activate

# Offline smoke test — no model, no network, no key. Start here.
python scripts/run_baseline.py --limit 3 --provider fake --fresh
python scripts/run_adaptive.py --limit 3 --provider fake --fresh

# Real local model
python scripts/run_baseline.py --limit 5
python scripts/run_adaptive.py --limit 5

# Validate the quality scorer (free, no model calls)
python scripts/score_distribution.py
```

Results land in `bug-localization/outputs/` as `.jsonl`.

### Flags

| Flag | Meaning |
|---|---|
| `--limit N` | How many instances (default: `limit` in `config.yaml`) |
| `--provider` | `ollama` (default) / `fake` (offline) / `anthropic` (paid) |
| `--model NAME` | Override the model for one run |
| `--out PATH` | Where to write predictions |
| `--fresh` | Start over instead of resuming |

Runs **checkpoint after every instance and resume by default** — if a run dies at instance
180 of 300, rerun the same command and it picks up where it stopped.

---

## 5. Configuration

Everything tunable is in `bug-localization/config.yaml` — no code changes needed.

```yaml
provider: "ollama"
model: "qwen2.5-coder:7b"

ollama:
  host: "http://localhost:11434"
  num_ctx: 16384        # DO NOT LOWER — see below
  timeout: 600

dataset: "princeton-nlp/SWE-bench_Lite"
split: "test"
limit: 30               # raise to 300 for the full run

budgets:                # the adaptive policy reads these
  low:    { max_candidates: 50, max_hops: 3, max_samples: 3 }
  medium: { max_candidates: 25, max_hops: 2, max_samples: 2 }
  high:   { max_candidates: 10, max_hops: 1, max_samples: 1 }

baseline_budget: { max_candidates: 15, max_hops: 2, max_samples: 1 }
```

> **Do not lower `num_ctx`.** Ollama defaults it to ~4096 and truncates the prompt
> *silently*. The repo-structure tree is most of our prompt, so at the default a `hops=3`
> tree gets cut down to the size of a `hops=1` tree — the effort knob stops doing anything
> and the experiment measures nothing.

Start with `limit: 5` and debug end-to-end before any full run.

---

## 6. Troubleshooting

**`cannot reach Ollama at http://localhost:11434`**
The server isn't running. `ollama serve &`

**`model 'qwen2.5-coder:7b' is not available in Ollama`**
Not pulled yet. `ollama pull qwen2.5-coder:7b`

**`WARNING: n/n predictions cost 0 tokens`**
No model output was used — the results are invalid. Usually the server died mid-run or the
model isn't pulled. Fix the cause and rerun with `--fresh`.

**`ModuleNotFoundError: No module named 'src'`**
Run the scripts from inside `bug-localization/`, not from the repo root.

**Runs are very slow**
Normal for a local model on a long prompt. Lower `limit`, or use `--model qwen2.5:3b` while
developing. First call after `ollama serve` also pays a model-load cost.

---

## 7. Optional: using a paid API instead

Only if we later decide the local model is too weak. One line in `config.yaml`:

```yaml
provider: "anthropic"
model: "claude-sonnet-5"
```

Then `pip install anthropic` and put `ANTHROPIC_API_KEY=sk-ant-...` in
`bug-localization/.env` (copy `.env.example`). That file is git-ignored — **never commit a
key**.

Rough cost: ~$0.13 per instance, so ~$8 for a 30-instance baseline+adaptive pair and ~$80
for the full 300. Note that a Claude Pro/Max subscription does **not** cover this — the API
is billed separately.
