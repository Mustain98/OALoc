# STUB — owned by ROLE 2 (02_ROLE_graph_and_tools.md).
# Replace this file wholesale when their branch lands. Do not build on its internals;
# Role 1 only depends on the signatures in 00_PROJECT_GUIDE.md §4.
#
# This placeholder indexes a local directory with `ast` so the pipeline runs today.
# It does NOT clone anything — checkout_repo returns the bundled fixture repo.
import ast
import os

_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "fixtures", "mini_repo",
)


class CodeGraph:
    """Minimal stand-in for Role 2's networkx-backed graph."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add(self, nid, **attrs):
        self.nodes[nid] = attrs

    def link(self, a, b, etype):
        self.edges.append((a, b, etype))

    def out_edges(self, nid):
        return [(a, b, t) for a, b, t in self.edges if a == nid]


def checkout_repo(repo: str, base_commit: str, data_dir: str) -> str:
    """STUB: ignores repo/commit and returns the bundled fixture repo."""
    return _FIXTURE


def build_graph(repo_dir: str) -> CodeGraph:
    cg = CodeGraph()
    for root, _, files in os.walk(repo_dir):
        if os.sep + ".git" in root:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            abspath = os.path.join(root, fn)
            path = os.path.relpath(abspath, repo_dir)
            with open(abspath, encoding="utf-8", errors="ignore") as f:
                src = f.read()
            cg.add(path, type="file", path=path, name=fn, code=src,
                   start_line=1, end_line=src.count("\n") + 1)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nid = f"{path}:{node.name}"
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    cg.add(nid, type=kind, path=path, name=node.name,
                           code=ast.get_source_segment(src, node) or "",
                           start_line=node.lineno,
                           end_line=getattr(node, "end_lineno", node.lineno))
                    cg.link(path, nid, "contain")
    return cg
