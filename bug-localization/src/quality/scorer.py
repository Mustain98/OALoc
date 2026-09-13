# src/quality/scorer.py
#
# TAJ. Stage 1 of the method (04_METHODOLOGY.md): bug report text -> 0..4.
#
# Deliberately rule-based, deterministic and free — no model call. The score is the
# load-bearing part of the novelty, so it has to be objective and defensible. The four
# features come straight from the report-quality literature (Fang et al.,
# Chaparro et al.): steps-to-reproduce, observed/expected behaviour, stack trace, code.
#
# The scorer must never see gold locations. It takes a string, nothing else.
import re

# 1. A stack trace / traceback.
_RE_TRACE = re.compile(
    r"traceback"                       # Python
    r"|at [\w.$]+\([\w.]+:\d+\)"       # Java-style frame
    r"|file \".+\", line \d+"          # Python frame
    r"|^\s*at \w",                     # generic "at ..." frame
    re.M,
)

# 2. A code snippet: a fenced block, an indented block, or call syntax.
_RE_CODE_FENCE = re.compile(r"```|~~~")
_RE_CODE_CALL = re.compile(r"\b\w+\.\w+\(|\bdef \w+\(|\bimport \w+|>>> ")

# 3. Names a class, method or function EXPLICITLY.
#
#    Deviation from 01_TAJ_quality_and_orchestration.md §2, which used
#    `\b[A-Z][a-zA-Z0-9]+\b` — that matches any capitalised word, including the first
#    word of a sentence ("The", "When", "This"), so the feature fired on nearly every
#    report and carried no signal. 04_METHODOLOGY.md names the scorer as *the* risk of
#    the method; a feature that is always on is exactly that risk. We require a real
#    identifier instead: internal-capital CamelCase, a dotted path, or a def/class
#    declaration.
_RE_IDENTIFIER = re.compile(
    r"\b[A-Za-z_]\w*[a-z]\w*[A-Z]\w*\b"      # CamelCase with an INTERNAL capital
    r"|\b\w+\.\w+\b"                          # dotted path: module.attr
    r"|\bdef\s+\w+"                           # def foo
    r"|\bclass\s+\w+"                         # class Foo
    r"|\b\w+_\w+\("                           # snake_case call
)
_RE_CODE_WORD = re.compile(r"\bmethod\b|\bfunction\b|\bclass\b|\battribute\b|\w+\(\)")

# 4. Steps to reproduce / observed-vs-expected structure.
_RE_REPRO = re.compile(
    r"steps to reproduce|to reproduce|reproduction|expected|actual|observed|instead of"
)

FEATURE_NAMES = ("stack_trace", "code_snippet", "named_entity", "repro_steps")


def score_quality_features(problem_statement: str) -> dict[str, bool]:
    """The four presence-based features, individually.

    Exposed so the scorer can be validated before the policy is trusted — see
    scripts/score_distribution.py. 04_METHODOLOGY.md calls that validation "step one
    of the results, not an assumption".
    """
    t = problem_statement or ""
    tl = t.lower()
    return {
        "stack_trace":  bool(_RE_TRACE.search(tl)),
        "code_snippet": bool(_RE_CODE_FENCE.search(t) or _RE_CODE_CALL.search(t)),
        "named_entity": bool(_RE_IDENTIFIER.search(t) and _RE_CODE_WORD.search(tl)),
        "repro_steps":  bool(_RE_REPRO.search(tl)),
    }


def score_quality(problem_statement: str) -> int:
    """Return 0..4 = how much localization signal the report carries."""
    return min(sum(score_quality_features(problem_statement).values()), 4)


def score_quality_llm(problem_statement: str, model: str) -> int:
    """Optional variant for comparison. Same 0..4 output.

    The rule-based scorer above is the defensible default; this is the comparison
    point the paper reports alongside it.
    """
    from src.llm import call

    sys = ("Rate how much a bug report helps locate the buggy code. "
           "Reply with ONE integer 0-4. 0=vague/no clues, 4=stack trace + code + clear repro.")
    txt, *_ = call(model, sys, problem_statement or "", max_tokens=8)
    m = re.search(r"[0-4]", txt or "")
    return int(m.group()) if m else 2
