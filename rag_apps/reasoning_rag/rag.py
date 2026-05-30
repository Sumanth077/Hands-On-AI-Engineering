"""URL ingestion, chunking, embedding (Mistral Embed) and ChromaDB storage.

This module is the knowledge layer for Reasoning RAG. It knows how to:

1. Fetch and clean text from a URL.
2. Split that text into overlapping chunks.
3. Embed the chunks with Mistral Embed.
4. Store / query them in a persistent ChromaDB collection.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import List

import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
try:
    # mistralai 1.x exposes the client at the top level.
    from mistralai import Mistral
except ImportError:
    # mistralai 2.x moved it into the `client` subpackage and dropped the
    # top-level re-export, so the old import path no longer resolves.
    from mistralai.client import Mistral

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBED_MODEL = "mistral-embed"
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = "reasoning_rag"

# Mistral Embed accepts batches; keep them modest to stay under request limits.
EMBED_BATCH_SIZE = 32
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

_mistral_client: Mistral | None = None
_chroma_client: chromadb.ClientAPI | None = None


@dataclass
class IngestResult:
    """Summary returned after ingesting a URL."""

    url: str
    title: str
    num_chunks: int
    num_chars: int


def _get_mistral() -> Mistral:
    """Lazily build the Mistral client so import never fails without a key."""
    global _mistral_client
    if _mistral_client is None:
        if not MISTRAL_API_KEY:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Add it to your .env file."
            )
        _mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return _mistral_client


def _get_collection():
    """Return the persistent ChromaDB collection, creating it if needed."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts with Mistral Embed, batching the requests."""
    client = _get_mistral()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def fetch_url(url: str) -> tuple[str, str]:
    """Fetch a URL and return (title, cleaned_text)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ReasoningRAG/1.0; +https://example.com/bot)"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Drop elements that never carry useful reading content.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url

    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines / whitespace.
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if not text:
        raise ValueError("No readable text could be extracted from the URL.")

    return title, text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def ingest_url(url: str) -> IngestResult:
    """Run the full pipeline: fetch -> chunk -> embed -> store in ChromaDB."""
    url = url.strip()
    if not url:
        raise ValueError("Please provide a URL.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    title, text = fetch_url(url)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("The page produced no chunks to index.")

    embeddings = embed_texts(chunks)

    collection = _get_collection()
    source_id = uuid.uuid4().hex[:8]
    ids = [f"{source_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"url": url, "title": title, "chunk_index": i} for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return IngestResult(
        url=url,
        title=title,
        num_chunks=len(chunks),
        num_chars=len(text),
    )


@dataclass
class RetrievedChunk:
    """A single retrieved passage with its provenance."""

    text: str
    url: str
    title: str
    score: float


def retrieve(query: str, k: int = 5) -> List[RetrievedChunk]:
    """Return the top-k most semantically relevant chunks for a query."""
    query = query.strip()
    if not query:
        return []

    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: List[RetrievedChunk] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        retrieved.append(
            RetrievedChunk(
                text=doc,
                url=meta.get("url", ""),
                title=meta.get("title", ""),
                # Cosine distance -> similarity score for display.
                score=round(1.0 - float(dist), 4),
            )
        )
    return retrieved


def collection_stats() -> dict:
    """Return basic stats about what is currently indexed."""
    collection = _get_collection()
    return {"count": collection.count()}


def reset_knowledge() -> None:
    """Drop all indexed content (used by the UI 'clear' control)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet; that's fine.
        pass
