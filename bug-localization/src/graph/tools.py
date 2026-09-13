# src/graph/tools.py
#
# FARHAN (02_FARHAN_graph_and_tools.md §3). The three tools the localizer calls.
#
#     search_entity(graph, keyword, detail) -> list[dict]   (each dict has an "id")
#     traverse_graph(graph, seeds, hops, edge_types) -> str
#     retrieve_entity(graph, entity_id) -> dict
#
# Everything returned is JSON-serializable: the agent puts traverse_graph's string
# straight into a prompt, so it has to be readable by a model, not by a debugger.
import re

from rank_bm25 import BM25Okapi

# How many entities search_entity returns per keyword before the agent's own
# max_candidates budget trims further.
_SEARCH_LIMIT = 20

# Ceiling on traverse_graph, per hop. On a real repo, three hops of `invoke` edges
# from 50 seeds reaches most of the codebase; without a cap the tool would spend
# minutes producing a tree far past anything the model can read.
#
# The cap SCALES WITH HOPS on purpose. max_hops is one of the three effort knobs
# (04_METHODOLOGY.md Stage 4) — a flat ceiling would make the hops=3 tree the same
# size as the hops=1 tree and quietly erase the experiment's independent variable.
_MAX_NODES_PER_HOP = 400

_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(text: str) -> list[str]:
    """'src/cache.py:Cache.get' -> ['src', 'cache', 'py', 'cache', 'get'].

    Identifiers are split on separators AND on camelCase boundaries, so a report
    saying "get cache" matches a node named `Cache.get` and a keyword `KeyError`
    matches `key` / `error` too.
    """
    out: list[str] = []
    for part in _SPLIT.split(text or ""):
        if not part:
            continue
        for piece in _CAMEL.split(part):
            if piece:
                out.append(piece.lower())
    return out


def _index(graph):
    """Build (and cache on the graph) the BM25 index over searchable entities.

    Cached because the agent calls search_entity once per keyword, up to 6 keywords
    per sample and max_samples samples per instance. Rebuilding the index each time
    would dominate the runtime on any repo bigger than the test fixture.
    """
    cached = getattr(graph, "_search_index", None)
    if cached is not None:
        return cached

    ids, docs = [], []
    for nid, d in graph.g.nodes(data=True):
        if d.get("type") not in ("file", "class", "function"):
            continue
        ids.append(nid)
        # The id carries the directory and file name; qualname carries the class it
        # belongs to. Both are signal a bug report can match against.
        docs.append(_tokens(f"{nid} {d.get('qualname', '')} {d.get('name', '')}"))

    index = (ids, BM25Okapi(docs) if docs else None)
    try:
        graph._search_index = index
    except AttributeError:
        pass          # a graph object that doesn't allow attributes: just don't cache
    return index


def search_entity(graph, keyword: str, detail: str = "preview") -> list[dict]:
    """Rank entities by BM25 relevance to `keyword`.

    detail: 'id' -> id/type/path only | 'preview' -> + first 200 chars of source
            | 'full' -> + the entity's complete source.
    """
    ids, bm25 = _index(graph)
    query = _tokens(keyword)
    if not ids or bm25 is None or not query:
        return []

    scores = bm25.get_scores(query)
    ranked = sorted(zip(ids, scores), key=lambda x: -x[1])[:_SEARCH_LIMIT]

    out: list[dict] = []
    for nid, sc in ranked:
        if sc <= 0:
            break          # scores are sorted; nothing below this point matches
        d = graph.g.nodes[nid]
        item = {"id": nid, "type": d.get("type"), "path": d.get("path"),
                "score": round(float(sc), 3)}
        if detail == "preview":
            item["preview"] = graph.code_of(nid)[:200]
        elif detail == "full":
            item["code"] = graph.code_of(nid)
        out.append(item)
    return out


def traverse_graph(graph, seeds: list[str], hops: int = 2,
                   edge_types=("contain", "invoke", "import", "inherit")) -> str:
    """Walk out from `seeds` up to `hops` edges, rendered as an indented tree.

    The output goes straight into the model's prompt, so it is text, not a data
    structure: one line per entity, indented by distance from its seed, annotated
    with its type and the edge that led to it.
    """
    hops = max(0, int(hops))
    wanted = set(edge_types or ())
    budget = _MAX_NODES_PER_HOP * max(1, hops)

    seen: set[str] = set()
    lines: list[str] = []
    truncated = False

    def walk(nid: str, depth: int, via: str | None):
        nonlocal truncated
        if nid in seen or depth > hops or nid not in graph.g:
            return
        if len(lines) >= budget:
            truncated = True
            return
        seen.add(nid)
        d = graph.g.nodes[nid]
        tag = f" <-{via}" if via else ""
        lines.append("  " * depth + f"{nid} [{d.get('type', '?')}]{tag}")
        if depth == hops:
            return
        # Deterministic order, and `contain` first so a file's own members are listed
        # before the graph wanders off into call targets in other files.
        order = {"contain": 0, "inherit": 1, "invoke": 2, "import": 3}
        edges = [(b, t) for _, b, t in graph.out_edges(nid) if t in wanted]
        edges.sort(key=lambda e: (order.get(e[1], 9), e[0]))
        for nbr, etype in edges:
            walk(nbr, depth + 1, etype)

    for s in seeds or ():
        walk(s, 0, None)

    if not lines:
        return "(no matching entities)"
    if truncated:
        lines.append(f"... (traversal truncated at {budget} entities)")
    return "\n".join(lines)


def retrieve_entity(graph, entity_id: str) -> dict:
    """Full attributes — including source and line numbers — for one entity."""
    if entity_id not in graph.g:
        return {"id": entity_id, "error": "not found"}
    d = graph.g.nodes[entity_id]
    return {
        "id": entity_id,
        "path": d.get("path"),
        "type": d.get("type"),
        "name": d.get("name"),
        "start_line": d.get("start_line"),
        "end_line": d.get("end_line"),
        "code": graph.code_of(entity_id),
    }
