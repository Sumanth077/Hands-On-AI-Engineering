import json
import re
import time
from typing import Callable, Optional
import networkx as nx
from mistralai.client import Mistral

from config import GRAPHS_DIR, COMBINED_GRAPH_FILE

_EXTRACT_PROMPT = """Extract entities and relationships from the text below.

Return ONLY a valid JSON object with this exact structure (no extra text):
{{
  "entities": [
    {{"name": "EntityName", "type": "EntityType", "description": "one sentence"}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "relation": "RELATION_TYPE", "description": "one sentence"}}
  ]
}}

Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY, PRODUCT, OTHER
Relation types: WORKS_FOR, LOCATED_IN, PART_OF, CREATED_BY, CAUSES, RELATED_TO, LEADS, USES, OWNS, MEMBER_OF

Extract only the most significant entities and relationships. Keep names concise.

TEXT:
{text}

JSON:"""


def _parse_extraction(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"entities": [], "relationships": []}


def _extract_from_chunk(client: Mistral, text: str) -> dict:
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(text=text[:2500])}],
                temperature=0.1,
                max_tokens=800,
            )
            return _parse_extraction(resp.choices[0].message.content.strip())
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return {"entities": [], "relationships": []}


def build_doc_graph(
    chunks: list[str],
    client: Mistral,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> nx.Graph:
    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        if progress_cb:
            progress_cb(i + 1, len(chunks))

        extracted = _extract_from_chunk(client, chunk)

        for ent in extracted.get("entities", []):
            name = ent.get("name", "").strip()
            if not name or len(name) > 100:
                continue
            if G.has_node(name):
                G.nodes[name]["mentions"] = G.nodes[name].get("mentions", 1) + 1
            else:
                G.add_node(
                    name,
                    type=ent.get("type", "OTHER"),
                    description=ent.get("description", ""),
                    mentions=1,
                )

        for rel in extracted.get("relationships", []):
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            if not src or not tgt or len(src) > 100 or len(tgt) > 100:
                continue
            for node in (src, tgt):
                if not G.has_node(node):
                    G.add_node(node, type="OTHER", description="", mentions=1)
            if G.has_edge(src, tgt):
                existing = G[src][tgt].get("relations", [])
                new_rel = rel.get("relation", "RELATED_TO")
                if new_rel not in existing:
                    existing.append(new_rel)
                G[src][tgt]["relations"] = existing
            else:
                G.add_edge(
                    src, tgt,
                    relations=[rel.get("relation", "RELATED_TO")],
                    description=rel.get("description", ""),
                )
    return G


def merge_into_combined(G_new: nx.Graph) -> nx.Graph:
    G = load_combined_graph()
    for node, data in G_new.nodes(data=True):
        if G.has_node(node):
            G.nodes[node]["mentions"] = G.nodes[node].get("mentions", 1) + data.get("mentions", 1)
            if not G.nodes[node].get("description") and data.get("description"):
                G.nodes[node]["description"] = data["description"]
        else:
            G.add_node(node, **data)
    for src, tgt, data in G_new.edges(data=True):
        if G.has_edge(src, tgt):
            merged_rels = list(set(
                G[src][tgt].get("relations", []) + data.get("relations", [])
            ))
            G[src][tgt]["relations"] = merged_rels
        else:
            G.add_edge(src, tgt, **data)
    save_combined_graph(G)
    return G


def save_doc_graph(G: nx.Graph, doc_hash: str) -> None:
    path = GRAPHS_DIR / f"{doc_hash}.json"
    data = nx.node_link_data(G)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_combined_graph() -> nx.Graph:
    if COMBINED_GRAPH_FILE.exists():
        try:
            data = json.loads(COMBINED_GRAPH_FILE.read_text(encoding="utf-8"))
            return nx.node_link_graph(data)
        except Exception:
            pass
    return nx.Graph()


def save_combined_graph(G: nx.Graph) -> None:
    data = nx.node_link_data(G)
    COMBINED_GRAPH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
