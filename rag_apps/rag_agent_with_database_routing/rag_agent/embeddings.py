"""
RAG Agent with Database Routing - Orq.ai embeddings wrapper.

Wraps the OpenAI-compatible embeddings endpoint at Orq.ai so it can be
used by both the database seeding step and the retriever.
"""

from __future__ import annotations

from openai import OpenAI

EMBEDDING_MODEL = "google-ai/gemini-embedding-001"


class OrqEmbeddings:
    """Thin wrapper around the Orq.ai embeddings endpoint."""

    def __init__(self, orq_api_key: str) -> None:
        self.client = OpenAI(
            base_url="https://my.orq.ai/v3/router",
            api_key=orq_api_key,
        )
        self.model = EMBEDDING_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (index time)."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (query time)."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
