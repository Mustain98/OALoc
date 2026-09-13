"""FARHAN's definition of done (02_FARHAN_graph_and_tools.md §5).

Runs entirely offline against the bundled fixture repo — no cloning, no network,
no model.
"""
import os
import textwrap

import pytest

from src.graph.indexer import CodeGraph, build_graph
from src.graph.tools import retrieve_entity, search_entity, traverse_graph

MINI_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "mini_repo")


@pytest.fixture(scope="module")
def cg():
    return build_graph(MINI_REPO)


@pytest.fixture
def nested(tmp_path):
    """A second repo with imports, inheritance and cross-file calls."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "base.py").write_text(textwrap.dedent('''
        class Base:
            def run(self):
                return 1
    '''))
    (pkg / "impl.py").write_text(textwrap.dedent('''
        from pkg.base import Base

        class Impl(Base):
            def run(self):
                return helper()

        def helper():
            return 2
    '''))
    return build_graph(str(tmp_path))


# --- node ids -------------------------------------------------------------- #

def test_node_ids_always_use_forward_slashes(cg):
    """Gold locations come out of a git patch, which uses '/' on every platform.

    A node id like 'src\\cache.py' can never match one, so on Windows every
    prediction would miss and the run would score zero for reasons unrelated to
    the method.
    """
    assert all("\\" not in nid for nid in cg.g.nodes)
    assert "src/cache.py" in cg


def test_file_class_and_function_nodes_exist(cg):
    assert cg.node("src/cache.py")["type"] == "file"
    assert cg.node("src/cache.py:Cache")["type"] == "class"
    assert cg.node("src/cache.py:Cache.get")["type"] == "function"
    assert cg.node("src/cache.py:helper")["type"] == "function"


def test_methods_are_qualified_by_their_class(cg):
    """'get' alone is ambiguous across a repo; 'Cache.get' is not."""
    assert "src/cache.py:Cache.get" in cg
    assert "src/cache.py:get" not in cg


def test_directory_nodes_are_created(cg):
    assert cg.node("src/")["type"] == "dir"


def test_line_numbers_are_recorded(cg):
    d = cg.node("src/cache.py:Cache.get")
    assert d["start_line"] < d["end_line"]


def test_code_is_sliced_from_the_file(cg):
    code = cg.code_of("src/cache.py:Cache.get")
    assert "def get" in code and "self._store[k]" in code
    assert "def put" not in code, "an entity's source must stop at its own end_line"


# --- edges ----------------------------------------------------------------- #

def test_contain_edges_link_file_class_method(cg):
    assert ("src/cache.py", "src/cache.py:Cache", "contain") in cg.out_edges("src/cache.py")
    assert ("src/cache.py:Cache", "src/cache.py:Cache.get", "contain") \
        in cg.out_edges("src/cache.py:Cache")


def test_invoke_edge_follows_a_call(cg):
    """helper() calls c.get() — the edge the localizer walks to reach the bug."""
    targets = [b for _, b, t in cg.out_edges("src/cache.py:helper") if t == "invoke"]
    assert "src/cache.py:Cache.get" in targets


def test_import_edge_between_files(cg):
    targets = [b for _, b, t in cg.out_edges("src/store.py") if t == "import"]
    assert "src/cache.py" in targets


def test_inherit_edge(nested):
    targets = [b for _, b, t in nested.out_edges("pkg/impl.py:Impl") if t == "inherit"]
    assert "pkg/base.py:Base" in targets


def test_calls_attach_to_the_innermost_function(nested):
    """Impl.run calls helper(); the edge must leave Impl.run, not the class."""
    assert "pkg/impl.py:helper" in [
        b for _, b, t in nested.out_edges("pkg/impl.py:Impl.run") if t == "invoke"]
    assert "pkg/impl.py:helper" not in [
        b for _, b, t in nested.out_edges("pkg/impl.py:Impl") if t == "invoke"]


def test_stats_report_resolution_coverage(cg):
    assert cg.stats["files"] == 2
    assert cg.stats["invoke_resolved"] >= 1
    assert cg.stats["invoke_attempted"] >= cg.stats["invoke_resolved"]


# --- robustness ------------------------------------------------------------ #

def test_unparsable_file_still_yields_a_file_node(tmp_path):
    (tmp_path / "broken.py").write_text("def oops(:\n")
    g = build_graph(str(tmp_path))
    assert "broken.py" in g
    assert g.stats["unparsed_files"] == 1


def test_skips_vendored_directories(tmp_path):
    (tmp_path / "real.py").write_text("def a(): pass\n")
    vendor = tmp_path / "node_modules"
    vendor.mkdir()
    (vendor / "junk.py").write_text("def b(): pass\n")
    g = build_graph(str(tmp_path))
    assert "real.py" in g
    assert not any("node_modules" in nid for nid in g.g.nodes)


def test_empty_repo_does_not_crash(tmp_path):
    g = build_graph(str(tmp_path))
    assert g.g.number_of_nodes() == 0
    assert search_entity(g, "anything") == []
    assert traverse_graph(g, [], hops=2) == "(no matching entities)"


# --- search_entity --------------------------------------------------------- #

def test_search_finds_the_entity(cg):
    hits = search_entity(cg, "cache get", detail="preview")
    assert any(h["id"] == "src/cache.py:Cache.get" for h in hits)


def test_every_hit_has_an_id(cg):
    """Taj's agent reads e['id'] off each dict — that key is the contract."""
    assert all("id" in h for h in search_entity(cg, "cache"))


def test_search_is_ranked_best_first(cg):
    scores = [h["score"] for h in search_entity(cg, "lookup")]
    assert scores == sorted(scores, reverse=True)


def test_camel_case_is_split_for_matching(cg):
    """A report says "get"; the node is named `Cache.get` inside CamelCase."""
    assert any(h["id"] == "src/cache.py:Cache" for h in search_entity(cg, "Cache"))


def test_detail_levels(cg):
    assert "preview" not in search_entity(cg, "cache", detail="id")[0]
    assert len(search_entity(cg, "cache", detail="preview")[0]["preview"]) <= 200
    assert "def get" in [h for h in search_entity(cg, "cache get", detail="full")
                         if h["id"] == "src/cache.py:Cache.get"][0]["code"]


def test_irrelevant_keyword_returns_nothing(cg):
    assert search_entity(cg, "zzzznonexistent") == []


def test_index_is_cached_not_rebuilt(cg):
    """The agent searches up to 6 keywords x max_samples per instance."""
    search_entity(cg, "cache")
    assert cg._search_index is not None


# --- traverse_graph -------------------------------------------------------- #

def test_traverse_renders_an_indented_tree(cg):
    out = traverse_graph(cg, ["src/cache.py"], hops=2)
    assert "src/cache.py [file]" in out
    assert "  src/cache.py:Cache [class]" in out
    assert "    src/cache.py:Cache.get [function]" in out


def test_more_hops_reach_more_entities(cg):
    """max_hops is one of the three effort knobs — it MUST actually bind.

    If a hops=3 tree is the same size as a hops=1 tree, the experiment's independent
    variable does nothing and both arms measure the same thing.
    """
    one = traverse_graph(cg, ["src/store.py"], hops=1)
    three = traverse_graph(cg, ["src/store.py"], hops=3)
    assert len(three.splitlines()) > len(one.splitlines())


def test_hops_zero_returns_only_the_seeds(cg):
    out = traverse_graph(cg, ["src/cache.py"], hops=0)
    assert out.splitlines() == ["src/cache.py [file]"]


def test_edge_type_filter_is_respected(cg):
    out = traverse_graph(cg, ["src/cache.py:helper"], hops=2, edge_types=["contain"])
    assert "src/cache.py:Cache.get" not in out


def test_traversal_terminates_on_a_cycle(tmp_path):
    (tmp_path / "loop.py").write_text(textwrap.dedent('''
        def ping():
            return pong()
        def pong():
            return ping()
    '''))
    g = build_graph(str(tmp_path))
    out = traverse_graph(g, ["loop.py:ping"], hops=50)
    assert "loop.py:pong" in out


def test_unknown_seed_is_ignored(cg):
    assert traverse_graph(cg, ["nope.py:nothing"], hops=2) == "(no matching entities)"


# --- retrieve_entity ------------------------------------------------------- #

def test_retrieve_returns_code_and_lines(cg):
    d = retrieve_entity(cg, "src/cache.py:Cache")
    assert d["type"] == "class"
    assert d["path"] == "src/cache.py"
    assert "class Cache" in d["code"]
    assert d["start_line"] >= 1 and d["end_line"] >= d["start_line"]


def test_retrieve_missing_entity_reports_an_error(cg):
    assert retrieve_entity(cg, "nope.py:gone")["error"] == "not found"


# --- the gold-leak rule ---------------------------------------------------- #

def test_tools_never_receive_gold_data():
    """00_PROJECT_GUIDE.md §7: gold locations must never reach the graph side.

    Enforced structurally — none of these functions accepts an Instance.
    """
    import inspect

    from src.graph import indexer, tools
    for fn in (indexer.build_graph, indexer.checkout_repo, tools.search_entity,
               tools.traverse_graph, tools.retrieve_entity):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"inst", "instance", "gold_files", "gold_functions"}
