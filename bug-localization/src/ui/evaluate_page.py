# src/ui/evaluate_page.py
#
# "Run/inspect the dataset evaluation" page. Reuses run_experiment (src.localizer.run)
# and the exact budget_for closures from scripts/run_baseline.py / run_adaptive.py, and
# the metric computation in src.eval.evaluate — this page is a view onto the same CLI
# pipeline, not a reimplementation of it.
import glob
import json
import os
import time
import uuid
from types import SimpleNamespace

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.eval.evaluate import evaluate, load_preds, plot_cost_accuracy
from src.data.loader import load_dataset
from src.localizer.run import run_experiment
from src.service import jobs
from src.service.errors import JobCancelled
from scripts.run_adaptive import budget_for as adaptive_budget_for
from scripts.run_baseline import budget_for as baseline_budget_for

_JOB_KEY = "evaluate_job_id"
_METRIC_KEYS = ("acc@1", "acc@3", "acc@5", "mrr", "avg_tokens", "avg_usd")


def _list_prediction_files(outputs_dir: str, prefix: str) -> list[str]:
    """Every {prefix}_predictions*.jsonl file, most recently modified first —
    covers both the legacy fixed name and the per-run timestamped/run-id names."""
    pattern = os.path.join(outputs_dir, f"{prefix}_predictions*.jsonl")
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)


def _describe_prediction_file(path: str) -> str:
    """Label a prediction file with its run provenance so two runs are never
    confused with each other in a picker."""
    name = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [l for l in tail.splitlines() if l.strip()]
        rec = json.loads(lines[-1]) if lines else {}
        provider_model = f"{rec.get('provider') or '?'}/{rec.get('model') or '?'}"
        timestamp = rec.get("timestamp") or ""
        return f"{name}  —  {provider_model}  {timestamp}"
    except Exception:
        return name


@st.cache_data
def _all_instances(dataset: str, split: str) -> list:
    """Load all instances to calculate dynamic limits based on repo filters."""
    return load_dataset(dataset, split, limit=1_000_000)


def _run_eval_job(job, cfg: dict, limit: int, repo_filter: list[str],
                  model: str, do_baseline: bool, do_adaptive: bool,
                  accumulate: bool, fresh: bool) -> None:
    """accumulate=False (the default) writes each run to its own run-id-suffixed
    file, so two Evaluate runs never silently mix in the same .jsonl — the root
    cause of "past and current outputs are confusing to compare". accumulate=True
    restores the old shared-file behavior for a user who wants a running total,
    with `fresh` still meaning "clear that shared file first"."""
    run_id = uuid.uuid4().hex[:12]

    def cancelled_fn() -> bool:
        return getattr(job, "cancelled", False)

    def _make_progress_cb(arm: str):
        def progress_cb(i, n, instance_id, tokens, top_files):
            if cancelled_fn():
                raise JobCancelled("Job was cancelled by the user.")
            job.trace.append({"arm": arm, "i": i, "n": n, "instance_id": instance_id,
                              "tokens": tokens, "top": top_files})
        return progress_cb

    suffix = "" if accumulate else f"_{run_id}"
    args = SimpleNamespace(provider=None, model=model, limit=limit,
                           repo=(repo_filter or None), out=None,
                           fresh=(fresh if accumulate else False))

    paths = {}
    if do_baseline:
        job.trace.append({"arm": "baseline", "status": "starting"})
        args.out = os.path.join(cfg["paths"]["outputs"], f"baseline_predictions{suffix}.jsonl")
        paths["baseline"] = run_experiment(
            baseline_budget_for, "baseline_predictions.jsonl",
            args=args, cfg=cfg, progress_cb=_make_progress_cb("baseline"),
            cancelled_fn=cancelled_fn)
    if do_adaptive:
        job.trace.append({"arm": "adaptive", "status": "starting"})
        args.out = os.path.join(cfg["paths"]["outputs"], f"adaptive_predictions{suffix}.jsonl")
        paths["adaptive"] = run_experiment(
            adaptive_budget_for, "adaptive_predictions.jsonl",
            args=args, cfg=cfg, progress_cb=_make_progress_cb("adaptive"),
            cancelled_fn=cancelled_fn)

    job.result = {"paths": paths, "run_id": run_id}


def _metrics_table(baseline: dict | None, adaptive: dict | None):
    rows = {}
    if baseline:
        rows["baseline"] = {k: baseline[k] for k in _METRIC_KEYS}
    if adaptive:
        rows["adaptive"] = {k: adaptive[k] for k in _METRIC_KEYS}
    return rows


def _render_quality_breakdown(label: str, metrics: dict) -> None:
    st.write(f"**{label} — accuracy by report-quality bucket**")
    buckets = []
    for low, high, name in ((0, 1, "low"), (2, 2, "medium"), (3, 4, "high")):
        bucket_rows = [r for r in metrics["_rows"] if low <= r["quality_score"] <= high]
        if bucket_rows:
            n = len(bucket_rows)
            acc5 = round(sum(r["acc@5"] for r in bucket_rows) / n, 3)
            tokens = round(sum(r["tokens"] for r in bucket_rows) / n, 1)
            buckets.append({"bucket": name, "n": n, "acc@5": acc5, "avg_tokens": tokens})
    if buckets:
        st.dataframe(buckets, hide_index=True)
    else:
        st.caption("no quality scores on this arm (baseline doesn't score reports)")


def _render_per_instance_table(label: str, metrics: dict) -> None:
    rows = metrics.get("_rows") or []
    if not rows:
        return
    with st.expander(f"{label} — per-instance predicted vs gold"):
        table = [{
            "instance_id": r["instance_id"],
            "predicted_files": ", ".join(r["ranked_files"][:5]),
            "gold_files": ", ".join(r["gold_files"]),
            "acc@5": r["acc@5"],
        } for r in rows]
        st.dataframe(table, hide_index=True)


def _load_and_render_results(cfg: dict, baseline_path: str | None,
                             adaptive_path: str | None) -> None:
    have_baseline = bool(baseline_path and os.path.exists(baseline_path))
    have_adaptive = bool(adaptive_path and os.path.exists(adaptive_path))
    if not have_baseline and not have_adaptive:
        st.info("No prediction file selected. Run an evaluation first.")
        return

    gold = load_dataset(cfg["dataset"], cfg["split"], cfg["limit"])
    baseline = evaluate(load_preds(baseline_path), gold) if have_baseline else None
    adaptive = evaluate(load_preds(adaptive_path), gold) if have_adaptive else None

    st.subheader("Baseline vs adaptive")
    st.dataframe(_metrics_table(baseline, adaptive))

    if baseline:
        _render_quality_breakdown("Baseline", baseline)
        _render_per_instance_table("Baseline", baseline)
    if adaptive:
        _render_quality_breakdown("Adaptive", adaptive)
        _render_per_instance_table("Adaptive", adaptive)

    if baseline and adaptive:
        plot_path = os.path.join(cfg["paths"]["outputs"], "cost_accuracy.png")
        plot_cost_accuracy(baseline, adaptive, plot_path)
        st.image(plot_path, caption="Accuracy vs token cost")


def render(cfg: dict) -> None:
    st.header("Evaluate on dataset")
    st.caption("Runs the same baseline/adaptive arms as scripts/run_baseline.py and "
              "scripts/run_adaptive.py against SWE-bench-Lite.")

    instances = _all_instances(cfg["dataset"], cfg["split"])
    all_repos = sorted(list(set(i.repo for i in instances)))
    
    repo_filter = st.multiselect(
        "Repos (optional — empty runs across all repos)",
        options=all_repos)
        
    if repo_filter:
        max_limit = sum(1 for i in instances if i.repo in repo_filter)
    else:
        max_limit = len(instances)

    with st.form("evaluate_form"):
        col1, col2 = st.columns(2)
        limit = col1.number_input("Instance limit", min_value=1, max_value=max(1, max_limit),
                                  value=min(int(cfg.get("limit", 30)), max(1, max_limit)))
        model_choice = col2.selectbox("Model", [
            "qwen2.5-coder:7b (Ollama)",
            "openai/gpt-oss-120b (Groq)"
        ])
        
        col5, col6, col7 = st.columns(3)
        do_baseline = col5.checkbox("Run baseline", value=True)
        do_adaptive = col6.checkbox("Run adaptive", value=True)
        accumulate = col7.checkbox(
            "Accumulate into one shared file",
            value=False,
            help="Off (default): each run writes its own timestamped output file, "
                 "so runs with different settings never silently mix. On: append "
                 "to the same baseline/adaptive file every run, like before.")
        fresh = st.checkbox(
            "Discard the existing shared file first (only applies when accumulating)",
            value=False, disabled=not accumulate)
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        if jobs.any_job_running():
            st.warning("Another job is already running — wait for it to finish first.")
        elif not do_baseline and not do_adaptive:
            st.error("Select at least one arm to run.")
        else:
            job = jobs.create_job("evaluate")

            provider = "ollama" if "Ollama" in model_choice else "groq"
            model_name = model_choice.split(" (")[0]
            cfg_override = dict(cfg, provider=provider, model=model_name)

            jobs.start_thread(job, _run_eval_job, cfg_override, int(limit), repo_filter,
                              model_name, do_baseline, do_adaptive, accumulate, fresh)
            st.session_state[_JOB_KEY] = job.id

    job_id = st.session_state.get(_JOB_KEY)
    job = jobs.get_job(job_id) if job_id else None
    if job:
        st.divider()
        st.subheader("Progress")
        for arm in ("baseline", "adaptive"):
            arm_rows = [e for e in job.trace if e.get("arm") == arm and "instance_id" in e]
            if arm_rows:
                st.write(f"**{arm}**")
                last = arm_rows[-1]
                st.progress(min(1.0, last["i"] / max(1, last["n"])),
                           text=f"{last['i']}/{last['n']}")
                st.dataframe(arm_rows[-10:], hide_index=True)
        if job.status == "running":
            if st.button("Stop/Cancel Job", key=f"cancel_{job.id}", type="primary"):
                job.request_cancel()
                st.rerun()
            st_autorefresh(interval=1500, limit=None, key=f"refresh_{job.id}")
            st.info("Running... (Job will auto-refresh every 1.5s until complete)")
        elif job.status == "cancelling":
            st.button("Cancelling…", key=f"cancel_{job.id}", disabled=True)
            st_autorefresh(interval=1000, limit=None, key=f"refresh_{job.id}")
            st.warning("Cancelling — waiting for the current step to finish...")
        elif job.status == "cancelled":
            st.warning("Job was cancelled.")
        elif job.status == "error":
            st.error(job.error)
        elif job.status == "done":
            run_id = (job.result or {}).get("run_id")
            st.success(f"Done. Run `{run_id}`." if run_id else "Done.")

    st.divider()
    st.subheader("Load existing results")
    outputs_dir = cfg["paths"]["outputs"]
    baseline_files = _list_prediction_files(outputs_dir, "baseline")
    adaptive_files = _list_prediction_files(outputs_dir, "adaptive")

    col_b, col_a = st.columns(2)
    sel_baseline = col_b.selectbox(
        "Baseline file", baseline_files, format_func=_describe_prediction_file,
        index=0 if baseline_files else None,
        disabled=not baseline_files) if baseline_files else None
    sel_adaptive = col_a.selectbox(
        "Adaptive file", adaptive_files, format_func=_describe_prediction_file,
        index=0 if adaptive_files else None,
        disabled=not adaptive_files) if adaptive_files else None

    if not baseline_files and not adaptive_files:
        st.caption("No prediction files found yet in outputs/. Run an evaluation first.")
    elif st.button("Load selected results"):
        _load_and_render_results(cfg, sel_baseline, sel_adaptive)
