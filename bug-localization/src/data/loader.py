# STUB — owned by FAHIM (03_FAHIM_data_and_evaluation.md §2).
# Replace wholesale when their branch lands; the real version loads SWE-bench Lite
# and derives gold locations from the fixing patch.
#
# These hand-written instances deliberately span the quality range, so all three
# budget tiers get exercised when you run the pipeline offline. They point at the
# bundled fixture repo (tests/fixtures/mini_repo).
from src.schemas import Instance

_STUB_INSTANCES = [
    # Expected score 0 -> "low" tier (widest search).
    Instance(
        instance_id="stub__vague-1",
        repo="stub/mini",
        base_commit="0000000",
        problem_statement=(
            "The caching is broken. Sometimes things go wrong when I look stuff up. "
            "Please fix it, it has been happening since last week."
        ),
        gold_files=["src/cache.py"],
        gold_functions=["src/cache.py:get"],
    ),
    # Expected score ~2 -> "medium" tier.
    Instance(
        instance_id="stub__medium-1",
        repo="stub/mini",
        base_commit="0000000",
        problem_statement=(
            "Calling the lookup function with a missing key does not behave as "
            "expected. I expected None but got an error instead. The Cache class "
            "seems to be the problem."
        ),
        gold_files=["src/store.py"],
        gold_functions=["src/store.py:lookup"],
    ),
    # Expected score 4 -> "high" tier (minimum effort).
    Instance(
        instance_id="stub__rich-1",
        repo="stub/mini",
        base_commit="0000000",
        problem_statement=(
            "Cache.get raises KeyError for absent keys.\n\n"
            "Steps to reproduce:\n"
            "```python\n"
            "from src.cache import Cache\n"
            "Cache().get('missing')\n"
            "```\n\n"
            "Traceback (most recent call last):\n"
            '  File "src/cache.py", line 11, in get\n'
            "    return self._store[k]\n"
            "KeyError: 'missing'\n\n"
            "Expected: the method returns None. Actual: it raises KeyError."
        ),
        gold_files=["src/cache.py"],
        gold_functions=["src/cache.py:get"],
    ),
]


def load_dataset(name: str, split: str, limit: int) -> list[Instance]:
    """STUB: ignores name/split and returns up to `limit` fixture instances."""
    out = []
    while len(out) < limit:
        out.extend(_STUB_INSTANCES)
    return out[:limit]
