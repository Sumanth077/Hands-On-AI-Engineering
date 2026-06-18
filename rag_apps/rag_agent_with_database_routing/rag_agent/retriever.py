"""
RAG Agent with Database Routing - Qdrant retriever.

Embeds the query and searches the routed collection.
Returns an empty list when no document clears the relevance threshold,
which signals the pipeline to trigger the fallback agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient

from .databases import SCORE_THRESHOLD
from .embeddings import OrqEmbeddings


@dataclass
class RetrievedDoc:
    """A single document retrieved from a Qdrant collection, with its similarity score."""

    text: str
    score: float
    source: str


def retrieve(
    query: str,
    collection: str,
    client: QdrantClient,
    embeddings: OrqEmbeddings,
    top_k: int = 4,
) -> list[RetrievedDoc]:
    """
    Embed the query and search the specified Qdrant collection.
    Returns only documents that meet the SCORE_THRESHOLD.
    An empty list means the fallback agent should be used.
    """
    query_vector = embeddings.embed_query(query)

    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        score_threshold=SCORE_THRESHOLD,
    )

    return [
        RetrievedDoc(
            text=hit.payload.get("text", ""),
            score=hit.score,
            source=hit.payload.get("source", collection),
        )
        for hit in response.points
    ]
