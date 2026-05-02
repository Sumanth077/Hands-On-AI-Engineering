import re
from typing import Any
import networkx as nx

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "what", "how", "when", "where", "who",
    "which", "that", "this", "these", "those", "and", "or", "but", "in",
    "on", "at", "to", "for", "of", "with", "by", "from", "about", "tell",
    "me", "explain", "describe", "give", "list", "show", "find", "get",
    "can", "please", "provide", "summarize", "discuss", "compare",
}


def _query_terms(query: str) -> list[str]:
    words = re.findall(r'\b\w+\b', query.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


def _score_node(node_name: str, terms: list[str]) -> float:
    name_lower = node_name.lower()
    score = 0.0
    for term in terms:
        if term == name_lower:
            score += 2.0
        elif term in name_lower or name_lower in term:
            score += 1.0
    return score


def retrieve_from_graph(G: nx.Graph, query: str, max_entities: int) -> dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {"entities": [], "relationships": [], "seed_nodes": []}

    terms = _query_terms(query)

    scored = [
        (node, _score_node(node, terms))
        for node in G.nodes()
    ]
    scored.sort(key=lambda x: (x[1], G.nodes[x[0]].get("mentions", 1)), reverse=True)

    seed_nodes = [n for n, s in scored if s > 0][:5]
    if not seed_nodes:
        seed_nodes = [n for n, _ in scored[:3]]

    visited_nodes: set[str] = set()
    visited_edges: set[tuple] = set()
    queue = [(n, 0) for n in seed_nodes]

    while queue and len(visited_nodes) < max_entities:
        node, depth = queue.pop(0)
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        if depth >= 2:
            continue
        for neighbor in G.neighbors(node):
            edge_key = tuple(sorted([node, neighbor]))
            visited_edges.add(edge_key)
            if neighbor not in visited_nodes:
                queue.append((neighbor, depth + 1))

    entities = []
    for node in visited_nodes:
        d = G.nodes[node]
        entities.append({
            "name": node,
            "type": d.get("type", "OTHER"),
            "description": d.get("description", ""),
            "mentions": d.get("mentions", 1),
        })
    entities.sort(key=lambda x: x["mentions"], reverse=True)

    relationships = []
    for src, tgt in visited_edges:
        if G.has_edge(src, tgt):
            ed = G[src][tgt]
            relationships.append({
                "source": src,
                "target": tgt,
                "relations": ed.get("relations", ["RELATED_TO"]),
                "description": ed.get("description", ""),
            })

    return {
        "entities": entities,
        "relationships": relationships,
        "seed_nodes": seed_nodes,
    }
