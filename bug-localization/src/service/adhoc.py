# src/service/adhoc.py
#
# Ad-hoc, single-repo localization for the "paste a GitHub link + bug report" web UI.
# Reuses the exact same pieces the offline evaluation pipeline uses — checkout_repo,
# load_or_build_graph, the quality scorer/policy, and localize() — so the UI
# demonstrates the same system that gets measured, not a parallel implementation.
import re
import subprocess
import uuid
from datetime import datetime, timezone

from src import llm
from src.graph.cache import load_or_build_graph
from src.graph.indexer import checkout_repo
from src.localizer.agent import localize
from src.quality.policy import choose_budget, fixed_budget
from src.quality.scorer import score_quality
from src.schemas import Instance
from src.service.errors import JobCancelled

_GITHUB_URL = re.compile(
    r"^(?:https?://)?(?:git@)?(?:www\.)?github\.com[:/]"
    r"(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)"
    r"(?:\.git)?"
    r"(?:/tree/(?P<ref>[\w./-]+))?"
    r"/?$"
)
_BARE_SLUG = re.compile(r"^(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?$")


def parse_repo_ref(text: str) -> tuple[str, str | None]:
    """Normalize a pasted GitHub URL or bare 'owner/repo' into ("owner/repo", ref).

    Accepts: https://github.com/owner/repo[.git], https://github.com/owner/repo/tree/
    <branch>, git@github.com:owner/repo.git, and a bare "owner/repo". `ref` is the
    branch parsed out of a /tree/<branch> URL segment, or None if not present.
    """
    text = (text or "").strip()
    m = _GITHUB_URL.match(text)
    if m:
        return f"{m.group('owner')}/{m.group('repo')}", m.group("ref")
    m = _BARE_SLUG.match(text)
    if m:
        return f"{m.group('owner')}/{m.group('repo')}", None
    raise ValueError(
        f"Could not parse a GitHub repo from {text!r}. "
        "Expected a GitHub URL or 'owner/repo'."
    )


def resolve_ref(repo: str, ref: str | None) -> str:
    """The commit-ish to check out. If the user didn't specify one, resolve the
    remote's actual default branch rather than trusting local `HEAD` — after any
    prior checkout of a specific commit, local HEAD no longer means "default tip".
    """
    if ref:
        return ref
    r = subprocess.run(
        ["git", "ls-remote", "--symref", f"https://github.com/{repo}.git", "HEAD"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"could not reach https://github.com/{repo}: {r.stderr.strip()}")
    m = re.search(r"^ref:\s*refs/heads/(\S+)\s+HEAD", r.stdout, re.M)
    if not m:
        raise RuntimeError(f"could not determine the default branch of {repo}")
    return m.group(1)


def _resolve_commit(repo_dir: str) -> str:
    r = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def run_adhoc_localization(job, instance_id_input: str, repo_input: str, ref_input: str, problem_statement: str,
                           mode: str, cfg: dict, force_reindex: bool = False) -> None:
    """Background-thread target for the Localize page. Populates job.trace as it runs
    and job.result when done (job.status/error are set by jobs.start_thread).

    src/llm.py's provider is process-global, so this (re-)configures it from `cfg`.
    Safe only because jobs.any_job_running() keeps the UI to one job at a time.
    """
    def note(text: str) -> None:
        job.trace.append({"type": "status", "text": text})

    llm.configure(cfg)

    gold_files: list[str] = []
    gold_functions: list[str] = []

    instance_id_input = (instance_id_input or "").strip()
    if instance_id_input:
        from src.data.loader import load_dataset
        note(f"loading instance {instance_id_input} from dataset ...")
        # Load all instances to find the specific one requested
        insts = load_dataset(cfg["dataset"], cfg["split"], limit=1_000_000)
        inst_obj = next((i for i in insts if i.instance_id == instance_id_input), None)
        if not inst_obj:
            job.status = "error"
            job.error = f"Instance {instance_id_input} not found in dataset {cfg['dataset']}"
            return
        repo = inst_obj.repo
        ref = inst_obj.base_commit
        problem_statement = inst_obj.problem_statement
        gold_files = inst_obj.gold_files
        gold_functions = inst_obj.gold_functions

        note(f"checking out {repo}@{ref} ...")
        repo_dir = checkout_repo(repo, ref, cfg["paths"]["data"])
        commit = _resolve_commit(repo_dir)
        instance_id_to_use = instance_id_input
    else:
        repo, url_ref = parse_repo_ref(repo_input)
        ref = (ref_input or "").strip() or url_ref
        ref = resolve_ref(repo, ref)

        note(f"checking out {repo}@{ref} ...")
        repo_dir = checkout_repo(repo, ref, cfg["paths"]["data"])
        commit = _resolve_commit(repo_dir)
        instance_id_to_use = f"adhoc-{uuid.uuid4().hex[:8]}"

    note(f"indexing repository at {commit[:12]} "
        f"({'forced re-index' if force_reindex else 'cached if available'}) ...")
    graph = load_or_build_graph(repo, commit, repo_dir, cfg["paths"]["graph_cache"],
                                force=force_reindex)
    note(f"graph ready: {graph.stats.get('files', 0)} files, "
        f"{graph.stats.get('nodes', 0)} nodes, {graph.stats.get('edges', 0)} edges")

    score = score_quality(problem_statement)
    budget = choose_budget(score, cfg) if mode == "adaptive" else fixed_budget(cfg)
    note(f"quality score={score} -> budget "
        f"(candidates={budget.max_candidates}, hops={budget.max_hops}, "
        f"samples={budget.max_samples})")

    inst = Instance(
        instance_id=instance_id_to_use,
        repo=repo,
        base_commit=commit,
        problem_statement=problem_statement,
    )

    def _progress_cb(i, e):
        if getattr(job, "cancelled", False):
            raise JobCancelled("Job was cancelled by the user.")
        job.trace.append({**e, "sample": i})

    pred = localize(inst, budget, graph, progress_cb=_progress_cb)
    pred.run_id = uuid.uuid4().hex[:12]
    pred.timestamp = datetime.now(timezone.utc).isoformat()
    pred.provider = llm.current_provider()
    pred.model = budget.model

    # Snippets for every ranked entity (not just the first few) — code_of() is a
    # cheap string slice, no LLM/network call, so this costs nothing even for a long
    # ranked list, and lets the result view show source for anything the user expands.
    snippets = {}
    for nid in dict.fromkeys(pred.ranked_files + pred.ranked_functions):
        snippets[nid] = graph.code_of(nid)

    job.result = {
        "prediction": pred.__dict__,
        "quality_score": score,
        "budget": budget.__dict__,
        "snippets": snippets,
        "graph_stats": graph.stats,
        "repo": repo,
        "commit": commit,
        "gold_files": gold_files,
        "gold_functions": gold_functions,
    }
