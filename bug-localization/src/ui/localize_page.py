# src/ui/localize_page.py
#
# "Paste a GitHub link + a bug report" page. Kicks off src.service.adhoc as a
# background job (src.service.jobs) and polls it, showing the agent's live trace.
import time

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.service import jobs
from src.service.adhoc import run_adhoc_localization

_JOB_KEY = "localize_job_id"


@st.cache_data
def get_dataset_instances(cfg: dict, limit: int):
    from src.data.loader import load_dataset
    return load_dataset(cfg["dataset"], cfg["split"], limit)


def _render_trace_entry(entry: dict) -> None:
    kind = entry.get("type")
    if kind == "status":
        st.write(f"🔧 {entry['text']}")
    elif kind == "tool_call":
        for call in entry.get("calls", []):
            st.write(f"**→ {call['tool']}**")
            st.code(str(call.get("args", {})), language="python")
    elif kind == "tool_result":
        with st.expander(f"← {entry.get('tool', '?')} result", expanded=False):
            st.code(entry.get("output", ""))
    elif kind == "final_answer":
        st.write("**✓ final answer**")
        st.code(entry.get("text", ""))


def _render_result(result: dict) -> None:
    pred = result["prediction"]
    st.caption(f"Run `{pred.get('run_id') or '?'}` · {pred.get('timestamp') or '?'} · "
              f"{pred.get('provider') or '?'}/{pred.get('model') or '?'}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Quality score", result["quality_score"])
    col2.metric("Tokens used", pred["tokens"])
    budget = result["budget"]
    col3.metric("Budget (cand/hops/samples)",
               f"{budget['max_candidates']}/{budget['max_hops']}/{budget['max_samples']}")

    snippets = result.get("snippets") or {}

    st.subheader("Ranked files")
    if pred["ranked_files"]:
        for nid in pred["ranked_files"]:
            with st.expander(nid):
                st.code(snippets.get(nid) or "_(no source available)_", language="python")
    else:
        st.write("_(no files ranked)_")

    st.subheader("Ranked functions")
    if pred["ranked_functions"]:
        for nid in pred["ranked_functions"]:
            with st.expander(nid):
                st.code(snippets.get(nid) or "_(no source available)_", language="python")
    else:
        st.write("_(no functions ranked)_")

    gold_files = result.get("gold_files") or []
    gold_functions = result.get("gold_functions") or []
    if gold_files or gold_functions:
        st.subheader("Predicted vs Ground Truth")
        gold_files_set, gold_functions_set = set(gold_files), set(gold_functions)
        if gold_files:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Predicted files**")
                for f in pred["ranked_files"]:
                    st.write(("✅ " if f in gold_files_set else "❌ ") + f)
            with col_b:
                st.write("**Ground-truth files**")
                for f in gold_files:
                    st.write(f)
        if gold_functions:
            col_c, col_d = st.columns(2)
            with col_c:
                st.write("**Predicted functions**")
                for fn in pred["ranked_functions"]:
                    st.write(("✅ " if fn in gold_functions_set else "❌ ") + fn)
            with col_d:
                st.write("**Ground-truth functions**")
                for fn in gold_functions:
                    st.write(fn)

    stats = result.get("graph_stats") or {}
    if stats:
        with st.expander("Graph coverage stats"):
            st.json(stats)


def render(cfg: dict) -> None:
    st.header("Localize a bug")
    st.caption("Paste a GitHub repo and a bug report; the same agent/graph/tools the "
              "evaluation pipeline uses will localize it, with a live trace.")

    st.write("**Option A: Load from dataset**")
    
    # Load all instances for the dropdown
    instances = get_dataset_instances(cfg, 3000)
    repos = sorted(list(set(i.repo for i in instances)))

    col1, col2 = st.columns(2)
    with col1:
        selected_repo = st.selectbox("Select Repo", [""] + repos)
    
    selected_instance = ""
    with col2:
        if selected_repo:
            repo_instances = [i.instance_id for i in instances if i.repo == selected_repo]
            selected_instance = st.selectbox("Select Instance (from dropdown)", [""] + repo_instances)
        else:
            st.selectbox("Select Instance", ["(Select a repo first)"], disabled=True)
            
    manual_instance = st.text_input("Or type Dataset Instance ID manually", placeholder="e.g., astropy__astropy-14365")
    final_instance_id = manual_instance.strip() if manual_instance.strip() else selected_instance

    with st.form("localize_form"):
        st.write("**Option B: Custom bug report**")
        repo_input = st.text_input(
            "GitHub repo", placeholder="https://github.com/owner/repo or owner/repo")
        ref_input = st.text_input(
            "Branch / commit (optional)",
            placeholder="leave blank for the repo's default branch")
        problem_statement = st.text_area("Bug report", height=200,
                                         placeholder="Paste the issue text here...")
        
        st.divider()
        model_choice = st.selectbox("LLM Model", [
            "qwen2.5-coder:7b (Ollama)",
            "openai/gpt-oss-120b (Groq)"
        ])
        mode = st.radio("Effort mode",
                        ["Adaptive (quality-scored)", "Baseline (fixed budget)"])
        force_reindex = st.checkbox("Force re-index (ignore cached graph)")
        submitted = st.form_submit_button("Localize", type="primary")

    if submitted:
        if jobs.any_job_running():
            st.warning("Another job is already running — wait for it to finish first.")
        elif not final_instance_id.strip() and (not repo_input.strip() or not problem_statement.strip()):
            st.error("Select an Instance ID from Option A OR provide both a GitHub repo and bug report in Option B.")
        else:
            job = jobs.create_job("localize")
            mode_key = "adaptive" if mode.startswith("Adaptive") else "baseline"
            
            provider = "ollama" if "Ollama" in model_choice else "groq"
            model_name = model_choice.split(" (")[0]
            cfg_override = dict(cfg, provider=provider, model=model_name)
            
            jobs.start_thread(
                job, run_adhoc_localization,
                final_instance_id, repo_input, ref_input, problem_statement, mode_key, cfg_override,
                force_reindex,
            )
            st.session_state[_JOB_KEY] = job.id

    job_id = st.session_state.get(_JOB_KEY)
    if not job_id:
        return

    job = jobs.get_job(job_id)
    if job is None:
        return

    st.divider()
    st.subheader("Trace")
    trace_box = st.container()
    with trace_box:
        for entry in job.trace:
            _render_trace_entry(entry)

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
    elif job.status == "done" and job.result:
        st.divider()
        st.subheader("Result")
        _render_result(job.result)
