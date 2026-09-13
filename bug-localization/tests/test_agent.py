"""Agent loop tests.

The loop is a LangGraph ReAct-style graph over ChatOllama (src/localizer/agent.py).
Run entirely offline: the `fake` provider path needs nothing, and the LangGraph path is
exercised by monkeypatching ChatOllama with a small scripted stand-in whose `.invoke()`
returns a queued response — no network, no real Ollama, no model. The stand-in's
`bind_tools()` returns itself and does not restrict which tools LangGraph's ToolNode may
call, so tool calls it emits execute for real against the mini_repo fixture graph.
"""
import os

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage

from src import llm
from src.graph.indexer import build_graph
from src.localizer import agent as agent_mod
from src.localizer.agent import _parse_entities, localize
from src.schemas import Budget, Instance
from src.service.errors import JobCancelled

MINI_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "mini_repo")


@pytest.fixture
def graph():
    return build_graph(MINI_REPO)


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


# --- parsing (unchanged from before the LangGraph rewrite) ----------------- #

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


# --- the fake/offline provider short-circuit -------------------------------- #

def test_fake_provider_returns_a_well_formed_prediction(graph):
    llm.set_provider("fake")
    try:
        p = localize(INST, _budget(), graph)
    finally:
        llm.set_provider("ollama")
    assert p.instance_id == "t1"
    assert p.ranked_files and p.ranked_functions
    assert p.tokens > 0


# --- the LangGraph loop, driven by a scripted fake chat model --------------- #

class _ScriptedChatModel:
    """Stand-in for ChatOllama. Every instance (bound or not) pops from one shared
    class-level queue, so a script can span both the tool-calling loop and a later
    force_answer_node call regardless of which one constructs it."""

    queue: list = []
    instantiations = 0

    def __init__(self, *args, **kwargs):
        _ScriptedChatModel.instantiations += 1

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if _ScriptedChatModel.queue:
            return _ScriptedChatModel.queue.pop(0)
        return AIMessage(content="```\nsrc/cache.py:Cache.get\n```",
                         response_metadata={"prompt_eval_count": 5, "eval_count": 5})


@pytest.fixture(autouse=True)
def _reset_scripted_model():
    _ScriptedChatModel.queue = []
    _ScriptedChatModel.instantiations = 0
    yield
    _ScriptedChatModel.queue = []
    _ScriptedChatModel.instantiations = 0


def _ai(content="", tool_calls=None, tokens=(5, 5)):
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        response_metadata={"prompt_eval_count": tokens[0], "eval_count": tokens[1]},
    )


def test_tool_calls_execute_against_the_real_graph(monkeypatch, graph):
    """A scripted tool_call for search_entity must return real BM25 hits from the
    fixture graph, and the model's final answer (informed by that) must parse."""
    monkeypatch.setattr(agent_mod, "ChatOllama", _ScriptedChatModel)
    _ScriptedChatModel.queue = [
        _ai(tool_calls=[{"name": "search_entity",
                         "args": {"keyword": "Cache.get"}, "id": "1"}]),
        _ai(content="```\nsrc/cache.py:Cache.get\n```"),
    ]

    p = localize(INST, _budget(samples=1, hops=3), graph)

    assert p.ranked_functions and p.ranked_functions[0] == "src/cache.py:Cache.get"
    assert p.ranked_files[0] == "src/cache.py"
    assert p.tokens > 0


def test_progress_cb_reports_tool_call_and_result(monkeypatch, graph):
    monkeypatch.setattr(agent_mod, "ChatOllama", _ScriptedChatModel)
    _ScriptedChatModel.queue = [
        _ai(tool_calls=[{"name": "search_entity",
                         "args": {"keyword": "Cache.get"}, "id": "1"}]),
        _ai(content="```\nsrc/cache.py:Cache.get\n```"),
    ]

    trace = []
    localize(INST, _budget(samples=1, hops=3), graph,
            progress_cb=lambda i, e: trace.append((i, e)))

    kinds = [e["type"] for _, e in trace]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "final_answer" in kinds
    assert all(i == 0 for i, _ in trace), "single sample: every entry is sample 0"


def _rate_limit_error() -> groq.RateLimitError:
    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://x"))
    return groq.RateLimitError("rate limited", response=resp, body=None)


def test_groq_rotation_advances_past_an_exhausted_key():
    """The direct-invoke path used by the LangGraph nodes (agent_node,
    force_answer_node) must rotate keys on a 429 too, not just src.llm's
    standalone _call_groq — this is the actual call path a real Groq run uses."""
    llm.configure({"groq": {"api_keys": ["key-a", "key-b", "key-c"]}})
    try:
        build_calls = []

        class _Model:
            def invoke(self, messages):
                if len(build_calls) < 3:
                    raise _rate_limit_error()
                return "ok"

        def build_model():
            build_calls.append(llm.current_groq_key())
            return _Model()

        result = agent_mod._invoke_with_groq_rotation(build_model, [])
        assert result == "ok"
        assert build_calls == ["key-a", "key-b", "key-c"]
    finally:
        llm.configure({})


def test_groq_rotation_raises_once_every_key_is_exhausted():
    llm.configure({"groq": {"api_keys": ["key-a", "key-b"]}})
    try:
        def build_model():
            return type("M", (), {"invoke": lambda self, m: (_ for _ in ()).throw(_rate_limit_error())})()

        with pytest.raises(groq.RateLimitError):
            agent_mod._invoke_with_groq_rotation(build_model, [])
    finally:
        llm.configure({})


def test_job_cancelled_propagates_instead_of_being_swallowed(monkeypatch, graph):
    """A progress_cb that raises JobCancelled must stop localize() outright — not
    be caught by the per-sample `except Exception` and treated as a skippable
    failure that lets the loop continue to the next sample."""
    monkeypatch.setattr(agent_mod, "ChatOllama", _ScriptedChatModel)
    _ScriptedChatModel.queue = [
        _ai(tool_calls=[{"name": "search_entity",
                         "args": {"keyword": "Cache.get"}, "id": "1"}]),
        _ai(content="```\nsrc/cache.py:Cache.get\n```"),
    ]

    calls = {"n": 0}

    def progress_cb(i, e):
        calls["n"] += 1
        if calls["n"] == 2:
            raise JobCancelled("cancelled by test")

    with pytest.raises(JobCancelled):
        localize(INST, _budget(samples=2, hops=3), graph, progress_cb=progress_cb)


def test_budget_exhaustion_forces_an_answer(monkeypatch, graph):
    """A model that keeps requesting tools past max_hops must still get cut off and
    forced to answer, not run away (should_continue's force_answer branch)."""
    monkeypatch.setattr(agent_mod, "ChatOllama", _ScriptedChatModel)
    keep_calling = {"name": "search_entity", "args": {"keyword": "cache"}, "id": "x"}
    _ScriptedChatModel.queue = [
        _ai(tool_calls=[keep_calling]),   # tool_count 0 -> 1
        _ai(tool_calls=[keep_calling]),   # tool_count 1 -> 2 == max_hops: next check trips
        _ai(tool_calls=[keep_calling]),   # last_msg still wants a tool -> force_answer
        _ai(content="```\nsrc/cache.py:Cache.get\n```"),  # force_answer_node's own call
    ]

    p = localize(INST, _budget(samples=1, hops=2), graph)

    # force_answer_node instantiates a second, separate ChatOllama (unbound, no tools).
    assert _ScriptedChatModel.instantiations == 2
    # _parse_entities also emits the bare form ("...get") alongside the qualified one.
    assert p.ranked_functions[0] == "src/cache.py:Cache.get"


def test_reciprocal_rank_aggregates_across_samples(monkeypatch, graph):
    """Two samples that agree on the top hit but disagree on the second must rank the
    agreed-upon one first (04_METHODOLOGY.md Stage 4: RRF, not Borda count)."""
    monkeypatch.setattr(agent_mod, "ChatOllama", _ScriptedChatModel)
    _ScriptedChatModel.queue = [
        _ai(content="```\nsrc/cache.py:Cache.get\nsrc/store.py:lookup\n```"),
        _ai(content="```\nsrc/cache.py:Cache.get\n```"),
    ]

    p = localize(INST, _budget(samples=2, hops=2), graph)

    assert p.ranked_functions[0] == "src/cache.py:Cache.get"
    assert p.ranked_files[0] == "src/cache.py"
    assert len(p.ranked_files) == len(set(p.ranked_files))
    assert len(p.ranked_functions) == len(set(p.ranked_functions))


def test_failing_tool_does_not_crash_the_instance(monkeypatch, graph):
    monkeypatch.setattr(agent_mod, "ChatOllama", _ScriptedChatModel)
    from src.graph import tools as farhan_tools
    monkeypatch.setattr(farhan_tools, "traverse_graph",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _ScriptedChatModel.queue = [
        _ai(tool_calls=[{"name": "traverse_graph",
                         "args": {"seeds": ["src/cache.py"]}, "id": "1"}]),
        _ai(content="```\nsrc/cache.py:Cache.get\n```"),
    ]

    p = localize(INST, _budget(), graph)
    assert p.instance_id == "t1"
    assert p.ranked_functions[0] == "src/cache.py:Cache.get"
