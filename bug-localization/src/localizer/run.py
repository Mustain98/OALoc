# src/localizer/run.py
#
# TAJ. The shared experiment driver.
#
# Both run scripts call run_experiment() with a different budget function and nothing
# else. Taj's definition of done requires that baseline and adaptive "differ only
# in scorer+policy — verify by diff"; keeping the loop in one place makes that true by
# construction, and removes the risk of the two loops quietly drifting apart, which
# would invalidate the controlled comparison.
import argparse
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

import subprocess

from src import llm
from src.config import load_config
from src.data.loader import load_dataset                    # FAHIM
from src.graph.cache import load_or_build_graph
from src.graph.indexer import checkout_repo                 # FARHAN
from src.localizer.agent import localize
from src.service.errors import JobCancelled


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--limit", type=int, default=None,
                   help="number of instances (default: config.yaml `limit`)")
    p.add_argument("--provider", default=None,
                   help="override the LLM backend: ollama | fake | anthropic")
    p.add_argument("--model", default=None,
                   help="override the model (default: config.yaml `model`)")
    p.add_argument("--repo", default=None,
                   help="filter instances by repository name (e.g., astropy/astropy)")
    p.add_argument("--out", default=None, help="output .jsonl path")
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing output file instead of resuming")
    return p.parse_args()


def _done_ids(path: str) -> set[str]:
    """instance_ids already written, so a killed run can be resumed."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["instance_id"])
            except Exception:
                continue
    return done


def _resolve_commit(repo_dir: str) -> str:
    """The exact commit actually checked out — the CodeGraph cache key.

    SWE-bench instances mostly already pin a full SHA, but resolving it explicitly
    (rather than trusting the string in the dataset) keeps the cache correct even for
    a short SHA or, via the ad-hoc UI (src/service/adhoc.py), a branch name.
    """
    r = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def run_experiment(budget_fn, out_name: str, args=None, cfg=None,
                   progress_cb=None,
                   cancelled_fn: Callable[[], bool] | None = None) -> str:
    """Run one arm of the experiment.

    budget_fn(inst, cfg) -> (Budget, quality_score)
        The ONLY thing that differs between baseline and adaptive.
        quality_score is -1 when the arm does not score reports.

    progress_cb(i, total, instance_id, tokens, ranked_files_top1), if given, is called
    right after each instance finishes — the same information already printed, just
    exposed programmatically so a UI can show live progress without parsing stdout.

    cancelled_fn(), if given, is polled once per agent sample (not just once per
    instance) via an internal callback passed to localize() — this is what lets a
    UI's Cancel button stop the run at the next graph-step boundary inside the
    *current* instance, rather than only between whole instances.
    """
    cfg = cfg or load_config()
    if args is not None:
        if args.provider:
            cfg["provider"] = args.provider
        if args.model:
            cfg["model"] = args.model
        if args.limit is not None:
            cfg["limit"] = args.limit
    llm.configure(cfg)

    out_path = (args.out if args and args.out
                else os.path.join(cfg["paths"]["outputs"], out_name))
    if args and args.fresh and os.path.exists(out_path):
        os.remove(out_path)
    done = _done_ids(out_path)

    repo_filter = getattr(args, "repo", None) if args else None
    if repo_filter:
        # A repo filter narrows a dataset that's sorted arbitrarily, not by repo — the
        # matching rows could be anywhere, so cfg["limit"] can't be applied before the
        # filter without risking zero matches. Load everything, filter, then cap.
        repos = {repo_filter} if isinstance(repo_filter, str) else set(repo_filter)
        insts = load_dataset(cfg["dataset"], cfg["split"], limit=1_000_000)
        insts = [i for i in insts if i.repo in repos][: cfg["limit"]]
    else:
        insts = load_dataset(cfg["dataset"], cfg["split"], cfg["limit"])

    # Group by repo so the expensive clone/index happens once per repository.
    insts.sort(key=lambda i: (i.repo, i.base_commit))

    print(f"[{out_name}] {len(insts)} instances | provider={cfg['provider']} "
          f"model={cfg['model']}")
    if done:
        print(f"[{out_name}] resuming — {len(done)} already done")

    graph_cache: dict = {}
    total_tokens, total_usd, n_done, n_failed, n_empty = 0, 0.0, 0, 0, 0
    tiers: Counter[int] = Counter()
    t0 = time.time()
    run_id = uuid.uuid4().hex[:12]

    def _sample_cb(sample_idx, entry):
        # Internal per-sample check, separate from the external progress_cb above
        # (different signature/purpose): this is what makes cancellation take effect
        # mid-instance instead of only between whole instances.
        if cancelled_fn is not None and cancelled_fn():
            raise JobCancelled("Job was cancelled by the user.")

    with open(out_path, "a") as fout:
        for i, inst in enumerate(insts, 1):
            if inst.instance_id in done:
                continue
            try:
                key = f"{inst.repo}@{inst.base_commit}"
                if key not in graph_cache:
                    graph_cache.clear()      # instances are repo-sorted; keep memory flat
                    repo_dir = checkout_repo(inst.repo, inst.base_commit,
                                             cfg["paths"]["data"])
                    commit = _resolve_commit(repo_dir)
                    # Disk-backed: a second process (e.g. run_adaptive.py right after
                    # run_baseline.py) reuses this instead of re-parsing the repo.
                    graph_cache[key] = load_or_build_graph(
                        inst.repo, commit, repo_dir, cfg["paths"]["graph_cache"])
                graph = graph_cache[key]

                budget, score = budget_fn(inst, cfg)
                pred = localize(inst, budget, graph, progress_cb=_sample_cb)
                pred.quality_score = score
                pred.run_id = run_id
                pred.timestamp = datetime.now(timezone.utc).isoformat()
                pred.provider = cfg["provider"]
                pred.model = cfg["model"]

                fout.write(json.dumps(pred.__dict__) + "\n")
                fout.flush()                 # checkpoint every instance

                total_tokens += pred.tokens
                total_usd += pred.usd
                n_done += 1
                if pred.tokens == 0:
                    n_empty += 1
                tiers[score] += 1
                print(f"  [{i}/{len(insts)}] {inst.instance_id} "
                      f"score={score} budget=({budget.max_candidates},{budget.max_hops},"
                      f"{budget.max_samples}) tokens={pred.tokens} "
                      f"top={pred.ranked_files[:1]}")
                if progress_cb:
                    progress_cb(i, len(insts), inst.instance_id, pred.tokens,
                               pred.ranked_files[:1])
            except JobCancelled:
                print(f"\n[{out_name}] cancelled by user at instance {i}/{len(insts)}")
                raise   # propagate — the `with` block above still closes fout cleanly
            except Exception as e:
                n_failed += 1
                print(f"  [{i}/{len(insts)}] {inst.instance_id} FAILED: {e}")

    dt = time.time() - t0
    print(f"\n[{out_name}] done: {n_done} ok, {n_failed} failed, {dt:.0f}s")
    if n_done:
        print(f"[{out_name}] avg tokens/instance = {total_tokens / n_done:.0f}")
        print(f"[{out_name}] avg usd/instance    = {total_usd / n_done:.6f}")
    if any(s >= 0 for s in tiers):
        print(f"[{out_name}] quality scores: "
              + ", ".join(f"{s}:{c}" for s, c in sorted(tiers.items())))
    print(f"[{out_name}] wrote {out_path}")

    if n_empty:
        # A prediction that cost nothing was produced by nothing. Saying so loudly
        # matters more than it looks: a whole run of these still writes a well-formed
        # jsonl and a plausible metrics file, and would quietly become a fake result.
        print(f"\n[{out_name}] WARNING: {n_empty}/{n_done} predictions cost 0 tokens — "
              f"no model output was used for them. Treat these results as invalid "
              f"until the cause is fixed.")
    return out_path
