"""
Embedding wrapper around sentence-transformers.

Uses all-MiniLM-L6-v2 (384-dim, cached in ~/.cache/huggingface/).
The model is loaded once and reused across all calls.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from customer_query_routing_agent.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self) -> None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready.")

    def embed(self, text: str) -> list[float]:
        """Return a 384-dimensional embedding for a single text string."""
        vector = self._model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts."""
        vectors = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
