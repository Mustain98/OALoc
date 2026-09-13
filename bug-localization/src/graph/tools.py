# STUB — owned by ROLE 2 (02_ROLE_graph_and_tools.md §3).
# Replace wholesale when their branch lands. Role 1 depends only on these three
# signatures from 00_PROJECT_GUIDE.md §4.
#
# Substring matching stands in for Role 2's BM25 ranking — enough to exercise the
# agent loop end-to-end, not enough to draw conclusions from.


def search_entity(graph, keyword: str, detail: str = "preview") -> list[dict]:
    kw = (keyword or "").lower()
    hits = []
    for nid, d in graph.nodes.items():
        if d.get("type") not in ("file", "class", "function"):
            continue
        if kw and kw not in nid.lower() and kw not in (d.get("name", "") or "").lower():
            continue
        item = {"id": nid, "type": d.get("type"), "path": d.get("path")}
        if detail == "preview":
            item["preview"] = (d.get("code", "") or "")[:200]
        if detail == "full":
            item["code"] = d.get("code", "") or ""
        hits.append(item)
    return hits[:20]


def traverse_graph(graph, seeds: list[str], hops: int = 2,
                   edge_types=("contain", "invoke", "import", "inherit")) -> str:
    seen, lines = set(), []

    def walk(nid, depth):
        if nid in seen or depth > hops or nid not in graph.nodes:
            return
        seen.add(nid)
        d = graph.nodes[nid]
        lines.append("  " * depth + f"{nid} [{d.get('type', '?')}]")
        for _, nbr, etype in graph.out_edges(nid):
            if etype in edge_types:
                walk(nbr, depth + 1)

    for s in seeds:
        walk(s, 0)
    return "\n".join(lines) if lines else "(no matching entities)"


def retrieve_entity(graph, entity_id: str) -> dict:
    if entity_id not in graph.nodes:
        return {"id": entity_id, "error": "not found"}
    d = graph.nodes[entity_id]
    return {"id": entity_id, "path": d.get("path"), "type": d.get("type"),
            "start_line": d.get("start_line"), "end_line": d.get("end_line"),
            "code": d.get("code", "")}
