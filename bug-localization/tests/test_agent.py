"""Agent loop, exercised entirely offline via the `fake` LLM backend."""
import pytest

from src import llm
from src.graph.indexer import build_graph, checkout_repo
from src.localizer.agent import _parse_entities, _truncate_tree, localize
from src.schemas import Budget, Instance


@pytest.fixture(autouse=True)
def fake_backend():
    llm.set_provider("fake")
    llm.reset_fake_log()
    yield
    llm.reset_fake_log()


@pytest.fixture
def graph():
    return build_graph(checkout_repo("stub/mini", "0000000", "./data"))


INST = Instance(
    instance_id="t1",
    repo="stub/mini",
    base_commit="0000000",
    problem_statement="Cache.get raises KeyError. File \"src/cache.py\", line 11.",
    gold_files=["src/cache.py"],
)


def _budget(samples=1, candidates=10, hops=2):
    return Budget(max_candidates=candidates, max_hops=hops, max_samples=samples,
                  model="fake-model")


# --- parsing --------------------------------------------------------------- #

def test_parses_messy_model_output():
    """Numbered, bulleted, fenced and prose-wrapped — what a 7B actually emits."""
    text = (
        "Here are the most likely locations:\n"
        "```\n"
        "1. src/cache.py:Cache.get\n"
        "- src/store.py:lookup\n"
        "* `src/cache.py:helper`\n"
        "```\n"
        "Hope this helps!"
    )
    ents = _parse_entities(text, limit=20)
    assert "src/cache.py:Cache.get" in ents
    assert "src/store.py:lookup" in ents
    assert "src/cache.py:helper" in ents
    assert not any("Hope this helps" in e for e in ents)


def test_qualified_names_also_emit_bare_form():
    """Fahim's gold functions come from diff hunk headers, i.e. bare names."""
    ents = _parse_entities("src/cache.py:Cache.get", limit=20)
    assert "src/cache.py:Cache.get" in ents
    assert "src/cache.py:get" in ents


def test_parser_rejects_prose_with_colons():
    ents = _parse_entities("Note: this is broken. See also: the docs.", limit=20)
    assert ents == []


def test_parser_respects_limit():
    text = "\n".join(f"src/m{i}.py:f{i}" for i in range(50))
    assert len(_parse_entities(text, limit=5)) == 5


# --- truncation ------------------------------------------------------------ #

def test_truncation_scales_with_hops():
    """A flat cap would erase the hops=1 / hops=3 difference — the independent variable."""
    tree = "\n".join(f"src/file{i}.py [file]" for i in range(20_000))
    assert len(_truncate_tree(tree, 3)) > len(_truncate_tree(tree, 1))


def test_small_tree_is_untouched():
    tree = "src/cache.py [file]\n  src/cache.py:get [function]"
    assert _truncate_tree(tree, 1) == tree


# --- the loop -------------------------------------------------------------- #

def test_localize_returns_populated_prediction(graph):
    p = localize(INST, _budget(), graph)
    assert p.instance_id == "t1"
    assert p.ranked_files and p.ranked_functions
    assert p.tokens > 0, "cost must be recorded on every prediction"


def test_max_samples_drives_the_number_of_llm_calls(graph):
    localize(INST, _budget(samples=1), graph)
    one = len(llm.fake_call_log())
    llm.reset_fake_log()
    localize(INST, _budget(samples=3), graph)
    three = len(llm.fake_call_log())
    assert three == 3 * one, "max_samples is an effort knob; it must actually bind"


def test_keyword_call_cost_is_counted(graph):
    """The role-doc sketch discarded it while reporting cost as the headline metric."""
    p = localize(INST, _budget(samples=1), graph)
    calls = llm.fake_call_log()
    assert len(calls) == 2, "one keyword call + one ranking call"
    keyword_tokens = llm._call_fake("fake-model", calls[0]["system"],
                                    calls[0]["user"], 60)[1]
    assert p.tokens > keyword_tokens


def test_more_samples_cost_more(graph):
    cheap = localize(INST, _budget(samples=1), graph)
    dear = localize(INST, _budget(samples=3), graph)
    assert dear.tokens > cheap.tokens


def test_reciprocal_rank_puts_repeated_top_hits_first(graph):
    p = localize(INST, _budget(samples=3), graph)
    # The fake backend always ranks src/cache.py:Cache.get first.
    assert p.ranked_functions[0] == "src/cache.py:Cache.get"
    assert p.ranked_files[0] == "src/cache.py"


def test_no_duplicate_entries(graph):
    p = localize(INST, _budget(samples=3), graph)
    assert len(p.ranked_files) == len(set(p.ranked_files))
    assert len(p.ranked_functions) == len(set(p.ranked_functions))


def test_failing_tool_does_not_crash_the_instance(graph, monkeypatch):
    from src.graph import tools

    monkeypatch.setattr(tools, "traverse_graph",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    p = localize(INST, _budget(), graph)
    assert p.instance_id == "t1"
    assert p.ranked_files == []


def test_broken_backend_raises_instead_of_faking_a_result(graph, monkeypatch):
    """A dead backend must not yield a full run of empty, zero-cost 'successes'.

    Swallowing LLMError produced a well-formed predictions file and a plausible
    metrics file built from nothing at all — the worst possible failure mode here.
    """
    import src.localizer.agent as agent

    def dead(*a, **k):
        raise llm.LLMError("model not pulled")

    monkeypatch.setattr(agent, "call", dead)
    with pytest.raises(llm.LLMError):
        localize(INST, _budget(samples=3), graph)
