# src/graph/cache.py
#
# Persistent, build-once-per-commit CodeGraph cache.
#
# build_graph() (indexer.py) walks every .py file with the AST module and
# search_entity's first call builds two BM25 corpuses (tools.py) — cheap on the
# fixture repo, not cheap on django/astropy-sized ones. Today that work happens fresh
# every time a repo is queried, even across separate process invocations of
# run_baseline.py / run_adaptive.py against the same commit. Caching the whole
# CodeGraph to disk, keyed by the exact resolved commit SHA, makes "load a project" a
# one-time cost: the same commit is reused instantly, and a new commit (the project
# being updated) naturally busts the cache without any extra bookkeeping.
import os
import pickle

from src.graph.indexer import CodeGraph, build_graph


def cache_path(repo: str, commit: str, cache_dir: str) -> str:
    safe_repo = repo.replace("/", "__")
    return os.path.join(cache_dir, safe_repo, f"{commit}.pkl")


def load_or_build_graph(repo: str, commit: str, repo_dir: str, cache_dir: str,
                        force: bool = False) -> CodeGraph:
    """Return the CodeGraph for `repo`@`commit`, from disk if already indexed.

    force=True bypasses and overwrites the cache — an explicit "re-index" action
    (e.g. the branch was force-pushed and points at new content under the same name),
    never something that happens silently on an ordinary query.
    """
    path = cache_path(repo, commit, cache_dir)
    if not force and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # a corrupt/incompatible cache entry: fall through and rebuild

    cg = build_graph(repo_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(cg, f)
    os.replace(tmp_path, path)  # atomic: a reader never sees a half-written cache file
    return cg
