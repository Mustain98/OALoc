import re
import pandas as pd
import os
from src.schemas import Instance

def _files_from_patch(patch: str) -> list[str]:
    """Files the fix touched = ground-truth buggy files."""
    return sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.M)))

def _functions_from_patch(patch: str) -> list[str]:
    """Best-effort 'path:func' from hunk headers like '@@ ... def get(self):'."""
    out, cur = [], None
    for line in patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", line)
        if m: cur = m.group(1)
        h = re.search(r"@@.*@@.*\bdef\s+(\w+)", line)
        if h and cur: out.append(f"{cur}:{h.group(1)}")
    return sorted(set(out))

def load_dataset(name: str, split: str, limit: int) -> list[Instance]:
    parquet_path = os.path.join(os.path.dirname(__file__), "../../../swe_bench_lite_test.parquet")
    df = pd.read_parquet(parquet_path)
    insts = []
    for _, row in df.head(limit).iterrows():
        patch = row["patch"]
        insts.append(Instance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            gold_files=_files_from_patch(patch),
            gold_functions=_functions_from_patch(patch),
        ))
    return insts
