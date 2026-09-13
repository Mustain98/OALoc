# src/localizer/agent.py
#
# TAJ owns this loop; it calls FARHAN's tools.
#
# One pass = keywords -> search_entity -> traverse_graph -> rank.
# Repeat max_samples times and merge by reciprocal-rank voting.
#
# The three budget dials (candidates, hops, samples) ARE the effort knobs. On a
# low-quality report the loop searches wider, walks farther and votes across more
# runs; on a high-quality one it does the minimum. That difference is the method.
#
# The prompts here are identical in both arms of the experiment — only the Budget
# differs — which is what makes the comparison controlled.
import re
from collections import Counter, defaultdict

from src.graph import tools                      # FARHAN
from src.llm import LLMError, call
from src.schemas import Budget, Instance, Prediction

# How many characters of repo tree we allow per hop. The cap has to scale with
# max_hops: a flat ceiling would truncate the hops=3 tree down to the size of the
# hops=1 tree and erase the independent variable. The per-hop figure is generous
# enough that a hops=1 tree is essentially never cut.
_TREE_CHARS_PER_HOP = 12_000

_KEYWORD_SYS = (
    "Extract up to 6 code-relevant keywords (class/function/file names, error terms). "
    "Reply as a comma-separated list."
)
_RANK_SYS = (
    "You are localizing a bug. Given the report and this repo structure, list the most "
    "likely buggy entities as 'path/file.py:function', most-suspicious first, one per line."
)

# Matches a 'path/to/file.py:name' entity anywhere in a line, tolerating the
# numbering, bullets, backticks and prose a small local model wraps around it.
_RE_ENTITY = re.compile(r"([\w./\-]+\.py)\s*:\s*([A-Za-z_][\w.]*)")


def _parse_entities(text: str, limit: int) -> list[str]:
    """Pull 'path.py:name' entities out of whatever the model actually emitted.

    A 7B model returns numbered lists, bullets, fenced blocks and a closing
    pleasantry. Keeping every line containing ':' (as the role-doc sketch did) lets
    that noise through, and unparsed junk can never match a gold path — the run would
    score zero for reasons that have nothing to do with the method.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith(("```", "~~~")):
            continue
        m = _RE_ENTITY.search(line)
        if not m:
            continue
        path, name = m.group(1), m.group(2)
        for ent in _entity_aliases(path, name):
            if ent not in seen:
                seen.add(ent)
                out.append(ent)
        if len(out) >= limit:
            break
    return out[:limit]


def _entity_aliases(path: str, name: str) -> list[str]:
    """'path:Class.method' also yields 'path:method'.

    Fahim derives gold functions from diff hunk headers, which give a bare function
    name ('path/file.py:get'). Emitting both spellings lets a qualified prediction
    still match.
    """
    ents = [f"{path}:{name}"]
    if "." in name:
        bare = name.rsplit(".", 1)[-1]
        if bare:
            ents.append(f"{path}:{bare}")
    return ents


def _truncate_tree(tree: str, max_hops: int) -> str:
    cap = _TREE_CHARS_PER_HOP * max(1, max_hops)
    if len(tree) <= cap:
        return tree
    kept = tree[:cap].rsplit("\n", 1)[0]
    dropped = tree[len(kept):].count("\n")
    return f"{kept}\n... ({dropped} more lines truncated)"


def _keywords(problem_statement: str, model: str) -> tuple[list[str], int, float]:
    """Returns (keywords, tokens, usd) — the cost of this call counts too.

    The role-doc sketch discarded it (`txt, *_ = call(...)`) while reporting cost as
    the headline metric.
    """
    txt, t, u = call(model, _KEYWORD_SYS, problem_statement or "", max_tokens=60)
    kws = [w.strip().strip("`'\"") for w in (txt or "").split(",")]
    kws = [w for w in kws if w][:6]
    if not kws:
        # Never let a bad keyword reply silently produce an empty search.
        kws = re.findall(r"[A-Za-z_]{4,}", problem_statement or "")[:6]
    return kws, t, u


def _rank_once(inst: Instance, budget: Budget, graph) -> tuple[list[str], int, float]:
    kws, T, U = _keywords(inst.problem_statement, budget.model)

    # Spread the candidate allowance across the keywords rather than taking
    # max_candidates from each and truncating, which would let keyword #1 crowd
    # out every other search term.
    per_kw = max(1, budget.max_candidates // max(1, len(kws)))
    seeds: list[str] = []
    seen: set[str] = set()
    for kw in kws:
        try:
            hits = tools.search_entity(graph, kw, detail="preview")
        except Exception:
            continue
        for e in hits[:per_kw]:
            eid = e.get("id")
            if eid and eid not in seen:
                seen.add(eid)
                seeds.append(eid)
    seeds = seeds[:budget.max_candidates]

    if not seeds:
        return [], T, U

    tree = tools.traverse_graph(
        graph, seeds,
        hops=budget.max_hops,
        edge_types=["contain", "invoke", "import", "inherit"],
    )
    tree = _truncate_tree(tree or "", budget.max_hops)

    user = f"REPORT:\n{inst.problem_statement}\n\nREPO STRUCTURE:\n{tree}"
    txt, t, u = call(budget.model, _RANK_SYS, user, max_tokens=400)
    T += t
    U += u
    return _parse_entities(txt, budget.max_candidates), T, U


def localize(inst: Instance, budget: Budget, graph) -> Prediction:
    """Run the budgeted search and merge samples by reciprocal-rank voting."""
    fn_votes: Counter[str] = Counter()
    file_votes: Counter[str] = Counter()
    T, U = 0, 0.0

    for _ in range(max(1, budget.max_samples)):
        try:
            ents, t, u = _rank_once(inst, budget, graph)
        except LLMError:
            # The backend itself is misconfigured (server down, model not pulled).
            # That is not an instance-level problem and must NOT be swallowed —
            # swallowing it produces a full run of empty predictions with zero cost
            # that still looks like a successful experiment.
            raise
        except Exception as e:
            # One bad sample must not lose the whole instance, let alone the run.
            print(f"    [warn] sample failed for {inst.instance_id}: {e}")
            continue
        T += t
        U += u

        # Reciprocal rank, as described in 04_METHODOLOGY.md Stage 4. (The role-doc
        # sketch used len(ents) - rank, which is Borda count — a different method
        # than the one the write-up defends.)
        best_per_file: dict[str, float] = defaultdict(float)
        for rank, e in enumerate(ents):
            rr = 1.0 / (rank + 1)
            fn_votes[e] += rr
            f = e.split(":")[0]
            # Best entity in this file, for this sample. Summing every entity would
            # let a file with a long tail of low-ranked functions outrank a file whose
            # top entity is the actual bug.
            best_per_file[f] = max(best_per_file[f], rr)
        for f, rr in best_per_file.items():
            file_votes[f] += rr

    return Prediction(
        instance_id=inst.instance_id,
        ranked_files=[f for f, _ in file_votes.most_common()],
        ranked_functions=[e for e, _ in fn_votes.most_common()],
        tokens=T,
        usd=U,
    )
