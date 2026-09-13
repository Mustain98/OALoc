# src/schemas.py
#
# THE SHARED CONTRACT. All three roles import from here.
# Frozen as of day 1 (00_PROJECT_GUIDE.md §4 and §7) — do not change a field
# without telling the other two.
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Instance:
    """One input: a bug report against one repo snapshot (from SWE-bench Lite)."""
    instance_id: str          # e.g. "django__django-12345"
    repo: str                 # e.g. "django/django"
    base_commit: str          # commit to check out (repo BEFORE the fix)
    problem_statement: str    # raw bug report text  <-- the thing we score
    gold_files: list[str] = field(default_factory=list)      # eval only, never fed to localizer
    gold_functions: list[str] = field(default_factory=list)  # eval only


@dataclass
class Budget:
    """How much effort the localizer may spend on this instance."""
    max_candidates: int       # how many entities to keep from search
    max_hops: int             # graph traversal depth
    max_samples: int          # how many agent runs to aggregate
    model: str                # which model tier to use


@dataclass
class Prediction:
    """Output of the localizer for one instance."""
    instance_id: str
    ranked_files: list[str]        # most-suspicious first
    ranked_functions: list[str]    # "path/file.py:func" strings, most-suspicious first
    quality_score: int = -1        # 0..4, -1 if not scored (baseline)
    tokens: int = 0
    usd: float = 0.0
    # Provenance — defaulted to "" so old .jsonl rows (written before these fields
    # existed) still deserialize via Prediction(**json.loads(line)).
    run_id: str = ""                # shared by every instance written in one run
    timestamp: str = ""             # ISO 8601 UTC, set when this instance finished
    provider: str = ""              # "ollama" | "groq" | ...
    model: str = ""
