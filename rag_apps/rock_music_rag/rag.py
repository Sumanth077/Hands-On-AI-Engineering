"""
Rock Music RAG — core pipeline.

Fetches Wikipedia pages for rock bands, indexes them with BM25,
and answers questions using Gemma 4 via the Google GenAI API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import wikipedia
from google import genai
from rank_bm25 import BM25Okapi

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID = "gemma-4-26b-a4b-it"

DEFAULT_BANDS = [
    "Audioslave",
    "Blink-182",
    "Dire Straits",
    "Evanescence",
    "Green Day",
    "Muse (band)",
    "Nirvana (band)",
    "Sum 41",
    "The Cure",
    "The Smiths",
]

PROMPT_TEMPLATE = """\
Using the information contained in the context below, give a comprehensive \
answer to the question.
If the answer is contained in the context, also report the source URL.
If the answer cannot be deduced from the context, say so clearly.

Context:
{context}

Question: {question}
"""

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content: str
    band: str
    url: str


@dataclass
class RAGResult:
    answer: str
    retrieved_chunks: list[Chunk]
    question: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_into_chunks(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """Split text into overlapping sentence chunks."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _clean_text(text: str) -> str:
    """Remove Wikipedia markup artifacts."""
    # Remove citation markers like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Main class ─────────────────────────────────────────────────────────────────

class RockRAG:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.bands: dict[str, dict] = {}   # band_name -> {url, raw_content}
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    # ── Band management ────────────────────────────────────────────────────────

    def fetch_band(self, band_name: str) -> dict:
        """
        Fetch a Wikipedia page for a band.
        Returns {"title": str, "url": str, "content": str} or raises an error.
        """
        page = wikipedia.page(title=band_name, auto_suggest=False)
        return {
            "title": page.title,
            "url": page.url,
            "content": _clean_text(page.content),
        }

    def add_band(self, band_name: str) -> str:
        """
        Fetch and add a band to the knowledge base. Rebuilds the index.
        Returns the resolved Wikipedia title.
        """
        data = self.fetch_band(band_name)
        self.bands[data["title"]] = data
        self._rebuild_index()
        return data["title"]

    def remove_band(self, band_title: str) -> None:
        """Remove a band and rebuild the index."""
        if band_title in self.bands:
            del self.bands[band_title]
            self._rebuild_index()

    def band_titles(self) -> list[str]:
        return sorted(self.bands.keys())

    # ── Indexing ───────────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """Re-chunk all band docs and rebuild the BM25 index."""
        self.chunks = []
        for band_data in self.bands.values():
            for chunk_text in _split_into_chunks(band_data["content"]):
                self.chunks.append(
                    Chunk(
                        content=chunk_text,
                        band=band_data["title"],
                        url=band_data["url"],
                    )
                )

        if not self.chunks:
            self.bm25 = None
            return

        self._tokenized = [c.content.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized)

    def index_size(self) -> int:
        return len(self.chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Return the top-k most relevant chunks for the query using BM25."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices]

    # ── Generation ─────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """Retrieve relevant chunks and generate an answer with Gemma 4."""
        retrieved = self.retrieve(question, top_k=top_k)

        if not retrieved:
            return RAGResult(
                answer="No bands have been added to the knowledge base yet.",
                retrieved_chunks=[],
                question=question,
            )

        # Build context string
        context_parts = []
        for chunk in retrieved:
            context_parts.append(f"{chunk.content}\nSource: {chunk.url}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )

        return RAGResult(
            answer=response.text.strip(),
            retrieved_chunks=retrieved,
            question=question,
        )
