"""
RAG Agent with Database Routing - Qdrant database setup.

Three in-memory Qdrant collections are created (empty) on startup.
Documents are added by the user through the UI.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .embeddings import OrqEmbeddings

VECTOR_SIZE = 3072  # google-ai/gemini-embedding-001 output dimension via Orq.ai
COLLECTIONS = ["products", "support", "financial"]
SCORE_THRESHOLD = 0.5


def build_databases(orq_api_key: str) -> tuple[QdrantClient, OrqEmbeddings]:
    """Create empty in-memory Qdrant collections."""
    client = QdrantClient(":memory:")
    embeddings = OrqEmbeddings(orq_api_key)

    for collection_name in COLLECTIONS:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    return client, embeddings


def add_documents(
    client: QdrantClient,
    embeddings: OrqEmbeddings,
    collection_name: str,
    texts: list[str],
) -> int:
    """Embed and insert a list of text chunks into the given collection.

    Returns the number of documents successfully added.
    """
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return 0

    vectors = embeddings.embed_documents(texts)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": text, "source": collection_name},
        )
        for text, vector in zip(texts, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def doc_count(client: QdrantClient, collection_name: str) -> int:
    """Return the number of vectors stored in a collection."""
    info = client.get_collection(collection_name)
    return info.points_count or 0
