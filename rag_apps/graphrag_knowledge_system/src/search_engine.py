import json
from typing import Dict, List

from src.graph_store import GraphStore
from src.llm_client import LLMClient
from src.vector_store import VectorStore

LOCAL_SEARCH_PROMPT = """You are an expert analyst answering questions using a knowledge graph.

## Knowledge Graph Context

### Entities
{entities}

### Relationships
{relationships}

### Source Text Excerpts
{chunks}

## Question
{question}

## Instructions
Answer the question using only the knowledge graph context above.
- Reference specific entities and relationships by name
- Be factual and precise
- If the context is insufficient, state what is known and what remains unclear

Answer:"""

COMMUNITY_RELEVANCE_PROMPT = """Evaluate whether this knowledge graph community summary helps answer the question.

Question: {question}

Community summary ({count} entities):
{summary}

Return JSON exactly:
{{"relevant": true_or_false, "relevance_score": 0.0_to_1.0, "key_points": ["point1", "point2"]}}"""

GLOBAL_ANSWER_PROMPT = """You are an expert analyst synthesizing a knowledge graph to answer a complex question.

## Community Knowledge (most relevant communities)
{community_contexts}

## Question
{question}

## Instructions
Provide a comprehensive answer that:
1. Synthesizes information across the relevant communities
2. Identifies overarching themes and patterns
3. Names specific entities where helpful
4. Acknowledges gaps if context is insufficient

Answer:"""


class SearchEngine:
    def __init__(self, graph_store: GraphStore, vector_store: VectorStore):
        self.graph = graph_store
        self.vector = vector_store
        self.llm = LLMClient()

    # ── Local Search ──────────────────────────────────────────────────────────

    def local_search(self, query: str, n_chunks: int = 5) -> Dict:
        if self.vector.count() == 0:
            return self._no_docs_result()

        chunk_hits = self.vector.search(query, n_results=n_chunks)
        if not chunk_hits:
            return self._no_docs_result()

        # Collect entity names from chunk metadata
        entity_names: List[str] = []
        for hit in chunk_hits:
            raw = hit.get("metadata", {}).get("entity_names", "")
            if raw:
                entity_names.extend(raw.split("|"))
        entity_names = list(set(filter(None, entity_names)))[:20]

        graph_ctx = {"entities": [], "relationships": []}
        if entity_names:
            graph_ctx = self.graph.get_entity_neighborhood(entity_names)

        entities_text = (
            "\n".join(
                f"- **{e['name']}** ({e['type']}): {e['description']}"
                for e in graph_ctx["entities"]
            )
            or "No entity context available"
        )
        rels_text = (
            "\n".join(
                f"- {r['source']} → {r['target']}: {r['description']}"
                for r in graph_ctx["relationships"]
            )
            or "No relationship context available"
        )
        chunks_text = "\n\n".join(
            f"[Excerpt {i + 1}]:\n{hit['content'][:600]}"
            for i, hit in enumerate(chunk_hits)
        )

        answer = self.llm.complete(
            LOCAL_SEARCH_PROMPT.format(
                entities=entities_text,
                relationships=rels_text,
                chunks=chunks_text,
                question=query,
            )
        )

        return {
            "answer": answer,
            "entities": graph_ctx["entities"],
            "relationships": graph_ctx["relationships"],
            "source_chunks": [h["content"][:300] for h in chunk_hits],
            "mode": "local",
        }

    # ── Global Search ─────────────────────────────────────────────────────────

    def global_search(self, query: str) -> Dict:
        communities = self.graph.get_all_communities()
        if not communities:
            return self.local_search(query)

        # Rate each community's relevance
        scored = []
        for comm in communities[:25]:
            summary = (comm.get("summary") or "").strip()
            if not summary:
                continue
            try:
                raw = self.llm.complete(
                    COMMUNITY_RELEVANCE_PROMPT.format(
                        question=query,
                        summary=summary,
                        count=comm.get("entity_count", "?"),
                    ),
                    json_mode=True,
                )
                rating = json.loads(raw)
                if rating.get("relevant") and float(rating.get("relevance_score", 0)) > 0.25:
                    scored.append(
                        {
                            **comm,
                            "relevance_score": float(rating.get("relevance_score", 0.5)),
                            "key_points": rating.get("key_points", []),
                        }
                    )
            except Exception:
                continue

        scored.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top = scored[:6]

        if not top:
            return self.local_search(query)

        community_contexts = "\n\n".join(
            f"### Community {i + 1} (relevance {c['relevance_score']:.2f})\n"
            f"Entities ({c['entity_count']}): {', '.join(c.get('entities', [])[:12])}\n"
            f"Summary: {c['summary']}\n"
            f"Key points: {', '.join(c.get('key_points', []))}"
            for i, c in enumerate(top)
        )

        answer = self.llm.complete(
            GLOBAL_ANSWER_PROMPT.format(
                community_contexts=community_contexts,
                question=query,
            )
        )

        all_entity_names = list(
            {name for c in top for name in c.get("entities", [])}
        )[:20]
        graph_ctx = {"entities": [], "relationships": []}
        if all_entity_names:
            graph_ctx = self.graph.get_entity_neighborhood(all_entity_names)

        return {
            "answer": answer,
            "entities": graph_ctx["entities"],
            "relationships": graph_ctx["relationships"],
            "communities_used": len(top),
            "source_chunks": [],
            "mode": "global",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _no_docs_result() -> Dict:
        return {
            "answer": "No documents have been indexed yet. Please upload and process documents first.",
            "entities": [],
            "relationships": [],
            "source_chunks": [],
            "mode": "none",
        }
