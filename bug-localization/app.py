"""QALoc web UI.

    streamlit run app.py

Two pages: paste a GitHub repo + a bug report and watch the agent localize it live, or
run/inspect the SWE-bench-Lite baseline-vs-adaptive evaluation. Both pages call the
exact same functions the CLI/eval pipeline uses (src.localizer.agent.localize,
src.localizer.run.run_experiment) — this UI is a window onto that system, not a
separate implementation of it.
"""
import streamlit as st

from src.config import load_config
from src.ui import evaluate_page, localize_page

st.set_page_config(page_title="QALoc", page_icon="🔎", layout="wide")

st.sidebar.title("QALoc")
page = st.sidebar.radio("Page", ["Localize a bug", "Evaluate on dataset"],
                        label_visibility="collapsed")

cfg = load_config()

if page == "Localize a bug":
    localize_page.render(cfg)
else:
    evaluate_page.render(cfg)
