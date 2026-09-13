# Role 2 — Code Graph Indexer & Navigation Tools

> **You build the LocAgent backbone: turn a repository into a searchable graph, and expose
> the three tools the localizer uses to explore it.** Read `00_PROJECT_GUIDE.md` first.
> You do NOT depend on anyone — start immediately. Role 1's agent imports your functions.

---

## What you build

| File | What it does |
|---|---|
| `src/graph/indexer.py` | check out a repo at a commit; parse it into a `CodeGraph` |
| `src/graph/tools.py`   | `search_entity`, `traverse_graph`, `retrieve_entity` |

Your output is consumed only through the four function signatures in
`00_PROJECT_GUIDE.md` §4. As long as those behave, your internals are yours.

---

## 1. The graph model

Parse each Python repo (via the standard-library `ast` module — no external parser needed to
start) into a directed graph. Use `networkx.MultiDiGraph`.

**Nodes** — one per code entity, with a fully-qualified id:
- directory → `src/`
- file → `src/cache.py`
- class → `src/cache.py:Cache`
- function/method → `src/cache.py:Cache.get`

**Node attributes:** `type` (dir/file/class/function), `path`, `start_line`, `end_line`,
`code` (source text of that entity), `name`.

**Edges** (typed):
- `contain` — dir→file, file→class, class→method, file→function
- `import`  — file→file it imports
- `invoke`  — function→function it calls (and function→class it instantiates)
- `inherit` — class→base class

`invoke` and `import` are resolved best-effort by name; it's fine to miss some (log how
many you resolve — Role 3 may report graph coverage).

---

## 2. `src/graph/indexer.py`

```python
import os, ast, subprocess, networkx as nx

def checkout_repo(repo: str, base_commit: str, data_dir: str) -> str:
    """Clone <repo> into data_dir and hard-checkout base_commit. Returns local path."""
    dest = os.path.join(data_dir, repo.replace("/", "__"))
    if not os.path.isdir(dest):
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git", dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "-f", base_commit], check=True)
    return dest

class CodeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
    def add(self, nid, **attrs): self.g.add_node(nid, **attrs)
    def link(self, a, b, etype): self.g.add_edge(a, b, type=etype)

def build_graph(repo_dir: str) -> CodeGraph:
    cg = CodeGraph()
    for root, _, files in os.walk(repo_dir):
        if "/.git" in root: continue
        for fn in files:
            if not fn.endswith(".py"): continue
            path = os.path.relpath(os.path.join(root, fn), repo_dir)
            src = open(os.path.join(root, fn), encoding="utf-8", errors="ignore").read()
            cg.add(path, type="file", path=path, name=fn, code=src,
                   start_line=1, end_line=src.count("\n") + 1)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nid = f"{path}:{node.name}"
                    seg = ast.get_source_segment(src, node) or ""
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    cg.add(nid, type=kind, path=path, name=node.name, code=seg,
                           start_line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno))
                    cg.link(path, nid, "contain")
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                cg.link(nid, f"{path}:{base.id}", "inherit")
                    for sub in ast.walk(node):     # invoke edges (best effort)
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                            cg.link(nid, sub.func.id, "invoke")   # target resolved by name in tools
    return cg
```

Keep the whole graph in memory — SWE-bench repos are large but a single-repo graph fits fine.
Build the graph **once per instance** (Role 1 calls `build_graph` per instance; you may add a
cache keyed by `repo@commit` if it's slow).

---

## 3. `src/graph/tools.py` — the three tools

These are exactly what the agent calls. Return plain dicts/strings (JSON-serializable).

```python
from rank_bm25 import BM25Okapi

def _corpus(cg):
    ids = [n for n, d in cg.g.nodes(data=True) if d.get("type") in ("file","class","function")]
    docs = [(cg.g.nodes[i].get("name","") + " " + i).lower().split() for i in ids]
    return ids, BM25Okapi(docs)

def search_entity(graph, keyword: str, detail: str = "preview") -> list[dict]:
    """keyword -> ranked entities. detail: 'id' | 'preview' | 'full'."""
    ids, bm25 = _corpus(graph)
    scores = bm25.get_scores(keyword.lower().split())
    ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
    out = []
    for nid, sc in ranked[:20]:
        if sc <= 0: break
        d = graph.g.nodes[nid]
        item = {"id": nid, "type": d.get("type"), "path": d.get("path")}
        if detail == "preview": item["preview"] = (d.get("code","")[:200])
        if detail == "full":    item["code"] = d.get("code","")
        out.append(item)
    return out

def traverse_graph(graph, seeds: list[str], hops: int = 2,
                   edge_types=("contain","invoke","import","inherit")) -> str:
    """BFS from seeds up to `hops`, rendered as an indented tree (LLM-friendly)."""
    seen, lines = set(), []
    def walk(nid, depth):
        if nid in seen or depth > hops or nid not in graph.g: return
        seen.add(nid)
        d = graph.g.nodes[nid]
        lines.append("  " * depth + f"{nid} [{d.get('type','?')}]")
        for _, nbr, data in graph.g.out_edges(nid, data=True):
            if data.get("type") in edge_types:
                walk(nbr, depth + 1)
    for s in seeds:
        walk(s, 0)
    return "\n".join(lines) if lines else "(no matching entities)"

def retrieve_entity(graph, entity_id: str) -> dict:
    """Full attributes (code + line numbers) for one entity."""
    if entity_id not in graph.g: return {"id": entity_id, "error": "not found"}
    d = graph.g.nodes[entity_id]
    return {"id": entity_id, "path": d.get("path"), "type": d.get("type"),
            "start_line": d.get("start_line"), "end_line": d.get("end_line"),
            "code": d.get("code","")}
```

---

## 4. Test yourself with a tiny fixture (don't wait for SWE-bench)

```python
# tests/test_graph.py
import os, tempfile, textwrap
from src.graph.indexer import build_graph
from src.graph.tools import search_entity, traverse_graph, retrieve_entity

def _mini_repo():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "cache.py"), "w").write(textwrap.dedent('''
        class Cache:
            def get(self, k):
                return self._store.get(k)
        def helper():
            c = Cache()
            return c.get("x")
    '''))
    return d

def test_graph_finds_entity():
    cg = build_graph(_mini_repo())
    hits = search_entity(cg, "cache get", detail="preview")
    assert any("Cache.get" in h["id"] or "cache.py:get" in h["id"] for h in hits)
    assert "cache.py" in traverse_graph(cg, ["cache.py"], hops=2)
    assert retrieve_entity(cg, "cache.py:Cache")["type"] == "class"
```

`pytest tests/test_graph.py` must pass before you hand off.

---

## 5. Your definition of done

- `checkout_repo` clones + checks out a real SWE-bench instance (test with one id, e.g.
  `astropy/astropy` at its `base_commit`).
- `build_graph` returns a `CodeGraph` with file/class/function nodes and `contain` edges at
  minimum (`invoke`/`import`/`inherit` best-effort).
- The three tools match the §4 signatures in the guide and pass `test_graph.py`.
- `search_entity` returns something sensible for a keyword that appears in the repo.
- You never read `gold_files` — your code doesn't even receive them.
