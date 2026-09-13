# src/graph/indexer.py
#
# FARHAN (02_FARHAN_graph_and_tools.md). The LocAgent backbone: turn a repository
# into a navigable code graph.
#
# Two public functions, both frozen by 00_PROJECT_GUIDE.md §4:
#
#     checkout_repo(repo, base_commit, data_dir) -> local path
#     build_graph(repo_dir)                      -> CodeGraph
#
# Nothing here ever sees gold_files / gold_functions — these signatures do not even
# accept an Instance, and that is deliberate (00_PROJECT_GUIDE.md §7).
import ast
import os
import subprocess

import networkx as nx

# Directories that are never part of the project's own source. Walking them wastes
# minutes on a big repo and pollutes search results with vendored code.
_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".tox", ".nox", ".eggs", ".mypy_cache", ".pytest_cache",
    "__pycache__", ".venv", "venv", "env", "node_modules", "build", "dist",
    "site-packages", ".idea", ".vscode",
}

# The two-file repo under tests/fixtures/ — see the stub branch in checkout_repo.
_FIXTURE_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "fixtures", "mini_repo",
)


def checkout_repo(repo: str, base_commit: str, data_dir: str) -> str:
    """Clone `repo` into data_dir and hard-checkout `base_commit`. Returns local path.

    The commit is the state of the repository BEFORE the fix — checking out anything
    else would leak the answer into the graph.
    """
    if repo.startswith("stub/"):
        # Escape hatch for the offline smoke test documented in README.md / SETUP.md
        # (`--provider fake`). FAHIM's stub loader emits repo="stub/mini", which is not
        # a real GitHub repository; without this the smoke test would try to clone it.
        # Delete this branch once the real SWE-bench loader lands.
        return _FIXTURE_REPO

    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, repo.replace("/", "__"))

    if not os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", dest],
            check=True,
        )

    # -f discards anything a previous instance left behind, so consecutive commits of
    # the same repo can reuse one clone.
    r = subprocess.run(["git", "-C", dest, "checkout", "-f", base_commit],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # Shallow or stale clone: the commit may not be present locally yet.
        subprocess.run(["git", "-C", dest, "fetch", "--all", "--tags"], check=False)
        subprocess.run(["git", "-C", dest, "checkout", "-f", base_commit], check=True)

    return dest


class CodeGraph:
    """A repository as a directed multigraph of code entities.

    Node ids are fully qualified and ALWAYS use forward slashes:

        src/                        dir
        src/cache.py                file
        src/cache.py:Cache          class
        src/cache.py:Cache.get      function (method, qualified by its class)

    Node attributes: type, path, name, start_line, end_line (+ `code` on file nodes).
    Entity source is sliced from its file on demand rather than stored per node —
    storing it three times over (file, class, method) triples memory on a repo the
    size of django for no gain.

    Edge types: contain | import | invoke | inherit.
    """

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.repo_dir: str = ""
        # Populated by build_graph; read by tools.py. Reported by score/eval as
        # graph coverage (02_FARHAN_graph_and_tools.md §1: "log how many you resolve").
        self.stats: dict[str, int] = {}
        # Lazily-built BM25 index, cached here so search_entity does not rebuild it on
        # every keyword. The agent searches up to 6 keywords x max_samples per
        # instance; rebuilding each time costs minutes per instance on a big repo.
        self._search_index = None

    # -- construction ------------------------------------------------------- #
    def add(self, nid, **attrs):
        self.g.add_node(nid, **attrs)

    def link(self, a, b, etype):
        self.g.add_edge(a, b, type=etype)

    # -- access ------------------------------------------------------------- #
    def __contains__(self, nid) -> bool:
        return nid in self.g

    def node(self, nid) -> dict:
        return self.g.nodes[nid]

    def out_edges(self, nid):
        """(src, dst, edge_type) triples leaving `nid`."""
        return [(a, b, d.get("type")) for a, b, d in self.g.out_edges(nid, data=True)]

    def code_of(self, nid) -> str:
        """Source text of one entity, sliced out of its file."""
        d = self.g.nodes.get(nid)
        if not d:
            return ""
        if d.get("type") == "file":
            return d.get("code", "") or ""
        src = (self.g.nodes.get(d.get("path"), {}) or {}).get("code", "") or ""
        if not src:
            return ""
        lines = src.splitlines()
        start = max(1, int(d.get("start_line") or 1))
        end = min(len(lines), int(d.get("end_line") or start))
        return "\n".join(lines[start - 1:end])


# --------------------------------------------------------------------------- #
# AST walking
# --------------------------------------------------------------------------- #
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _direct_calls(node) -> list[str]:
    """Names called directly in `node`'s body, not in a nested def.

    A call inside a nested function belongs to that function, not to this one —
    attributing it here would draw invoke edges from the wrong node.
    """
    names: list[str] = []

    def visit(n, top=False):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, _DEF_TYPES) and not top:
                continue
            if isinstance(child, _DEF_TYPES):
                continue
            if isinstance(child, ast.Call):
                f = child.func
                if isinstance(f, ast.Name):
                    names.append(f.id)          # foo()
                elif isinstance(f, ast.Attribute):
                    names.append(f.attr)        # obj.foo()
            visit(child)

    visit(node, top=True)
    return names


def _imported_modules(tree) -> list[str]:
    """Dotted module names imported by a file, plus relative-import markers.

    A relative import is returned with its leading dots intact ('.mod', '..pkg.mod')
    so it can be resolved against the importing file's own package.
    """
    mods: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.extend(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            base = "." * (n.level or 0) + (n.module or "")
            if base:
                mods.append(base)
    return mods


def _add_dir_chain(cg: CodeGraph, path: str) -> str | None:
    """Create dir nodes for every parent of `path` and the contain edges between them.

    Returns the immediate parent dir id, or None for a file at the repo root.
    """
    parts = path.split("/")[:-1]
    parent = None
    for i in range(len(parts)):
        did = "/".join(parts[:i + 1]) + "/"
        if did not in cg.g:
            cg.add(did, type="dir", path=did, name=parts[i],
                   start_line=1, end_line=1)
        if parent:
            cg.link(parent, did, "contain")
        parent = did
    return parent


def _index_file(cg: CodeGraph, path: str, src: str, pending: dict) -> None:
    """Add the file node and every class/function inside it."""
    parent_dir = _add_dir_chain(cg, path)
    cg.add(path, type="file", path=path, name=path.rsplit("/", 1)[-1], code=src,
           start_line=1, end_line=src.count("\n") + 1)
    if parent_dir:
        cg.link(parent_dir, path, "contain")

    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        # Python 2 files and generated fixtures appear in real repos. Skipping the
        # body still leaves a searchable file node.
        pending["unparsed"] += 1
        return

    pending["imports"].append((path, _imported_modules(tree)))

    def walk(node, scope: list[str], owner: str):
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, _DEF_TYPES):
                continue
            qual = ".".join(scope + [child.name])
            nid = f"{path}:{qual}"
            is_class = isinstance(child, ast.ClassDef)
            cg.add(nid, type="class" if is_class else "function", path=path,
                   name=child.name, qualname=qual, start_line=child.lineno,
                   end_line=getattr(child, "end_lineno", child.lineno))
            cg.link(owner, nid, "contain")

            if is_class:
                for base in child.bases:
                    bname = (base.id if isinstance(base, ast.Name)
                             else base.attr if isinstance(base, ast.Attribute)
                             else None)
                    if bname:
                        pending["inherit"].append((nid, path, bname))
            else:
                for called in _direct_calls(child):
                    pending["invoke"].append((nid, path, called))

            walk(child, scope + [child.name], nid)

    walk(tree, [], path)

    # Module-level calls belong to the file itself.
    for called in _direct_calls(tree):
        pending["invoke"].append((path, path, called))


# --------------------------------------------------------------------------- #
# Best-effort name resolution (invoke / inherit / import)
# --------------------------------------------------------------------------- #
def _resolve_by_name(name: str, from_path: str, by_file, by_name) -> str | None:
    """Prefer a definition in the same file; fall back to a globally unique one.

    Python name resolution needs full type inference to do properly. Same-file-first
    plus unique-global is the honest approximation — ambiguous names are dropped
    rather than guessed, and the drop is counted so coverage can be reported.
    """
    local = by_file.get((from_path, name))
    if local:
        return local
    candidates = by_name.get(name) or []
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_module(mod: str, from_path: str, files: set[str]) -> str | None:
    """Dotted (or relative) module name -> the file node that defines it."""
    if mod.startswith("."):
        depth = len(mod) - len(mod.lstrip("."))
        rest = mod.lstrip(".")
        pkg = from_path.split("/")[:-1]
        pkg = pkg[:len(pkg) - (depth - 1)] if depth > 1 else pkg
        parts = pkg + (rest.split(".") if rest else [])
    else:
        parts = mod.split(".")
    if not parts:
        return None

    stem = "/".join(parts)
    for cand in (f"{stem}.py", f"{stem}/__init__.py"):
        if cand in files:
            return cand

    # Repos commonly nest the package one level down (src/pkg/..., lib/pkg/...), so
    # the dotted path is a suffix of the real path rather than the whole of it.
    for suffix in (f"/{stem}.py", f"/{stem}/__init__.py"):
        matches = [f for f in files if f.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
    return None


def build_graph(repo_dir: str) -> CodeGraph:
    """Parse every Python file under `repo_dir` into a CodeGraph."""
    cg = CodeGraph()
    cg.repo_dir = repo_dir
    pending = {"invoke": [], "inherit": [], "imports": [], "unparsed": 0}

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            abspath = os.path.join(root, fn)
            # Forward slashes, always. Gold locations come out of a git patch, which
            # uses '/' on every platform; a node id like 'src\cache.py' can never
            # match one, so on Windows every prediction would miss and the run would
            # score zero for reasons that have nothing to do with the method.
            path = os.path.relpath(abspath, repo_dir).replace(os.sep, "/")
            try:
                with open(abspath, encoding="utf-8", errors="ignore") as f:
                    src = f.read()
            except OSError:
                continue
            _index_file(cg, path, src, pending)

    # --- second pass: resolve names now that every definition is known -------
    file_paths = {n for n, d in cg.g.nodes(data=True) if d.get("type") == "file"}
    by_file: dict[tuple[str, str], str] = {}
    by_name: dict[str, list[str]] = {}
    for nid, d in cg.g.nodes(data=True):
        if d.get("type") not in ("class", "function"):
            continue
        key = (d["path"], d["name"])
        by_file.setdefault(key, nid)
        by_name.setdefault(d["name"], []).append(nid)

    resolved = {"invoke": 0, "inherit": 0, "import": 0}
    attempted = {"invoke": len(pending["invoke"]),
                 "inherit": len(pending["inherit"]),
                 "import": 0}

    for src_nid, from_path, name in pending["invoke"]:
        tgt = _resolve_by_name(name, from_path, by_file, by_name)
        if tgt and tgt != src_nid:
            cg.link(src_nid, tgt, "invoke")
            resolved["invoke"] += 1

    for src_nid, from_path, name in pending["inherit"]:
        tgt = _resolve_by_name(name, from_path, by_file, by_name)
        if tgt and tgt != src_nid:
            cg.link(src_nid, tgt, "inherit")
            resolved["inherit"] += 1

    for path, mods in pending["imports"]:
        for mod in mods:
            attempted["import"] += 1
            tgt = _resolve_module(mod, path, file_paths)
            if tgt and tgt != path:
                cg.link(path, tgt, "import")
                resolved["import"] += 1

    cg.stats = {
        "files": len(file_paths),
        "unparsed_files": pending["unparsed"],
        "nodes": cg.g.number_of_nodes(),
        "edges": cg.g.number_of_edges(),
        "invoke_resolved": resolved["invoke"],
        "invoke_attempted": attempted["invoke"],
        "inherit_resolved": resolved["inherit"],
        "inherit_attempted": attempted["inherit"],
        "import_resolved": resolved["import"],
        "import_attempted": attempted["import"],
    }
    return cg
