"""Role 1 definition of done: a rich report scores high, a vague one scores low."""
from src.quality.scorer import (
    FEATURE_NAMES,
    score_quality,
    score_quality_features,
)

RICH = """Cache.get raises KeyError for absent keys.

Steps to reproduce:
```python
from src.cache import Cache
Cache().get('missing')
```

Traceback (most recent call last):
  File "src/cache.py", line 11, in get
    return self._store[k]
KeyError: 'missing'

Expected: the method returns None. Actual: it raises KeyError.
"""

VAGUE = (
    "The caching is broken. Sometimes things go wrong when I look stuff up. "
    "Please fix it, it has been happening since last week."
)


def test_rich_report_scores_high():
    assert score_quality(RICH) == 4


def test_vague_report_scores_low():
    assert score_quality(VAGUE) <= 1


def test_score_is_always_in_range():
    for text in (RICH, VAGUE, "", None, "x" * 5000):
        assert 0 <= score_quality(text) <= 4


def test_features_sum_to_score():
    feats = score_quality_features(RICH)
    assert set(feats) == set(FEATURE_NAMES)
    assert sum(feats.values()) == score_quality(RICH)


def test_individual_features_fire_correctly():
    assert score_quality_features(RICH)["stack_trace"]
    assert score_quality_features(RICH)["code_snippet"]
    assert score_quality_features(RICH)["repro_steps"]
    assert not score_quality_features(VAGUE)["stack_trace"]
    assert not score_quality_features(VAGUE)["code_snippet"]


def test_named_entity_ignores_sentence_initial_capitals():
    """The regression the tightened feature 3 exists to prevent.

    The role-doc sketch used \\b[A-Z][a-zA-Z0-9]+\\b, which matches "The" and "When",
    so the feature fired on essentially every report and carried no signal.
    """
    prose = "The function stopped working. When I call it, nothing happens."
    assert not score_quality_features(prose)["named_entity"]
    assert score_quality_features("The Cache class is wrong")["named_entity"]


def test_scorer_is_deterministic():
    assert [score_quality(RICH) for _ in range(5)] == [score_quality(RICH)] * 5
