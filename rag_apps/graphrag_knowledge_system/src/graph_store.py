import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx


class GraphStore:
    """
    Local knowledge graph backed by a NetworkX DiGraph.
    Persisted to disk as JSON via node_link_data so the graph survives restarts.
    """

    def __init__(self, persist_path: str = "./graph_store.json"):
        self.persist_path = Path(persist_path)
        self.G: nx.DiGraph = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> nx.DiGraph:
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return nx.node_link_graph(data, directed=True, multigraph=False)
            except Exception:
                pass
        return nx.DiGraph()

    def _save(self):
        data = nx.node_link_data(self.G)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ── Document ─────────────────────────────────────────────────────────────

    def create_document(self, doc_id: str, filename: str, file_hash: str):
        self.G.add_node(
            doc_id, node_type="document", filename=filename, file_hash=file_hash
        )
        self._save()

    def document_exists(self, file_hash: str) -> bool:
        return any(
            d.get("node_type") == "document" and d.get("file_hash") == file_hash
            for _, d in self.G.nodes(data=True)
        )

    # ── Chunks ────────────────────────────────────────────────────────────────

    def create_chunk(self, chunk_id: str, content: str, doc_id: str):
        self.G.add_node(chunk_id, node_type="chunk", content=content, doc_id=doc_id)
        if self.G.has_node(doc_id):
            self.G.add_edge(chunk_id, doc_id, rel_type="PART_OF")
        self._save()

    # ── Entities ──────────────────────────────────────────────────────────────

    def upsert_entity(
        self, name: str, entity_type: str, description: str, chunk_id: str
    ):
        if self.G.has_node(name) and self.G.nodes[name].get("node_type") == "entity":
            existing = self.G.nodes[name]
            if len(description) > len(existing.get("description", "")):
                existing["description"] = description
        else:
            self.G.add_node(
                name,
                node_type="entity",
                name=name,
                type=entity_type,
                description=description,
                community_id=None,
            )
        if self.G.has_node(chunk_id):
            self.G.add_edge(name, chunk_id, rel_type="APPEARS_IN")
        self._save()

    # ── Relationships ─────────────────────────────────────────────────────────

    def upsert_relationship(
        self, source: str, target: str, description: str, weight: float
    ):
        if not self.G.has_node(source) or not self.G.has_node(target):
            return
        if self.G.has_edge(source, target):
            ed = self.G[source][target]
            if ed.get("rel_type") == "RELATED_TO":
                ed["weight"] = (ed.get("weight", weight) + weight) / 2.0
                if len(description) > len(ed.get("description", "")):
                    ed["description"] = description
                self._save()
                return
        self.G.add_edge(
            source, target,
            rel_type="RELATED_TO",
            description=description,
            weight=weight,
        )
        self._save()

    # ── Community ─────────────────────────────────────────────────────────────

    def set_entity_community(self, entity_name: str, community_id: str):
        # No _save() here — called in a tight loop; caller saves via create_community()
        if self.G.has_node(entity_name):
            self.G.nodes[entity_name]["community_id"] = community_id

    def create_community(
        self, community_id: str, entity_names: List[str], summary: str, level: int = 0
    ):
        self.G.add_node(
            community_id,
            node_type="community",
            summary=summary,
            level=level,
            entity_count=len(entity_names),
        )
        for name in entity_names:
            if self.G.has_node(name):
                self.G.add_edge(name, community_id, rel_type="IN_COMMUNITY")
        self._save()

    def get_all_communities(self) -> List[Dict]:
        result = []
        for node_id, data in self.G.nodes(data=True):
            if data.get("node_type") != "community":
                continue
            entity_names = [
                u
                for u, v, d in self.G.edges(data=True)
                if v == node_id and d.get("rel_type") == "IN_COMMUNITY"
            ]
            result.append(
                {
                    "id": node_id,
                    "summary": data.get("summary", ""),
                    "level": data.get("level", 0),
                    "entity_count": data.get("entity_count", 0),
                    "entities": entity_names,
                }
            )
        return result

    def communities_exist(self) -> bool:
        return any(
            d.get("node_type") == "community" for _, d in self.G.nodes(data=True)
        )

    # ── Graph export for community detection ─────────────────────────────────

    def get_full_graph(self) -> Dict:
        entities = [
            {"name": n, "type": d.get("type", "")}
            for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "entity"
        ]
        relationships = [
            {"source": u, "target": v, "weight": d.get("weight", 0.5)}
            for u, v, d in self.G.edges(data=True)
            if d.get("rel_type") == "RELATED_TO"
        ]
        return {"entities": entities, "relationships": relationships}

    def get_entity_descriptions(self, names: List[str]) -> List[Dict]:
        result = []
        for name in names:
            if not self.G.has_node(name):
                continue
            d = self.G.nodes[name]
            if d.get("node_type") == "entity":
                result.append(
                    {
                        "name": name,
                        "type": d.get("type", "OTHER"),
                        "description": d.get("description", ""),
                    }
                )
        return result

    # ── Search context ────────────────────────────────────────────────────────

    def get_entity_neighborhood(self, entity_names: List[str]) -> Dict:
        seen_entities: set = set()
        seen_rels: set = set()
        entities: List[Dict] = []
        relationships: List[Dict] = []
        target_set = set(entity_names)

        def _add_entity(n: str):
            if n in seen_entities or not self.G.has_node(n):
                return
            d = self.G.nodes[n]
            if d.get("node_type") != "entity":
                return
            seen_entities.add(n)
            entities.append(
                {
                    "name": n,
                    "type": d.get("type", "OTHER"),
                    "description": d.get("description", ""),
                    "community": d.get("community_id"),
                }
            )

        # Seed with the requested entities
        for name in entity_names:
            _add_entity(name)

        # Collect all RELATED_TO edges that touch any requested entity
        for u, v, d in self.G.edges(data=True):
            if d.get("rel_type") != "RELATED_TO":
                continue
            if u not in target_set and v not in target_set:
                continue
            rkey = tuple(sorted([u, v]))
            if rkey not in seen_rels:
                seen_rels.add(rkey)
                relationships.append(
                    {
                        "source": u,
                        "target": v,
                        "description": d.get("description", ""),
                        "weight": d.get("weight", 0.5),
                    }
                )
            _add_entity(u)
            _add_entity(v)

        return {"entities": entities, "relationships": relationships}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def ping(self) -> Tuple[bool, str]:
        return True, "Connected (local NetworkX graph)"

    def close(self):
        self._save()
