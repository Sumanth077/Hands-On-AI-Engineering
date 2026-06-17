"""
Gemini Embedding 2 wrapper with HyDE vector averaging.
"""

import numpy as np
from google import genai


class GeminiEmbedder:
    """Wraps the google-genai SDK to embed text using Gemini Embedding 2.

    Also provides hyde_embed(), which embeds N hypothetical documents and
    returns their averaged vector, the core of the HyDE retrieval strategy.
    """

    MODEL = "gemini-embedding-2"

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text and return its vector."""
        result = self.client.models.embed_content(
            model=self.MODEL,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, one by one."""
        return [self.embed(text) for text in texts]

    def hyde_embed(self, hypothetical_docs: list[str]) -> list[float]:
        """Embed N hypothetical documents and average their vectors.

        The averaged vector sits closer in embedding space to real relevant
        documents than the original query would, improving retrieval precision.
        """
        embeddings = np.array([self.embed(doc) for doc in hypothetical_docs])
        return np.mean(embeddings, axis=0).tolist()
