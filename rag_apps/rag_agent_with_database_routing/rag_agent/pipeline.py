"""
RAG Agent with Database Routing - main pipeline.

Orchestrates: route -> retrieve -> generate (or fallback if no relevant docs found).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI
from qdrant_client import QdrantClient

from .embeddings import OrqEmbeddings
from .fallback import run_fallback
from .retriever import RetrievedDoc, retrieve
from .router import RoutingDecision, build_router, route_query

MODEL = "zai/glm-5-turbo"

RAG_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "Be concise and cite specific details from the context. "
    "If the context does not contain enough information, say so clearly."
)


@dataclass
class PipelineResult:
    """The final answer plus routing and retrieval metadata for a single query."""

    answer: str
    routing: RoutingDecision
    docs: list[RetrievedDoc] = field(default_factory=list)
    used_fallback: bool = False


def build_pipeline(orq_api_key: str) -> tuple:
    """
    Initialise and return all pipeline components.
    Call once at startup and reuse across queries.
    Returns: (qdrant_client, embeddings, orq_client)
    """
    from .databases import build_databases

    client, embeddings = build_databases(orq_api_key)
    orq_client = OpenAI(base_url="https://my.orq.ai/v3/router", api_key=orq_api_key)
    return client, embeddings, orq_client


def run_pipeline(
    query: str,
    client: QdrantClient,
    embeddings: OrqEmbeddings,
    orq_client: OpenAI,
) -> PipelineResult:
    """
    Full pipeline:
      1. Route query to the most relevant database
      2. Retrieve documents from that database
      3. If relevant docs found, generate a grounded answer
      4. If no docs found, fall back to live web search
    """
    # Step 1: Route
    routing = route_query(orq_client, query)

    # Step 2: Retrieve
    docs = retrieve(query, routing.database, client, embeddings)

    # Step 3a: RAG answer
    if docs:
        context = "\n\n".join(f"[{i+1}] {doc.text}" for i, doc in enumerate(docs))
        response = orq_client.responses.create(
            model=MODEL,
            instructions=RAG_SYSTEM,
            input=f"Context:\n{context}\n\nQuestion: {query}",
        )
        answer = response.output[0].content[0].text
        return PipelineResult(answer=answer, routing=routing, docs=docs)

    # Step 3b: Fallback
    answer = run_fallback(orq_client, query)
    return PipelineResult(answer=answer, routing=routing, docs=[], used_fallback=True)
