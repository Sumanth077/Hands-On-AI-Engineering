"""
Actian VectorAI DB client, collection setup, and read/write helpers.

Two collections form the shared context layer that every agent reads through:

  research_documents - chunked source material. Each point carries doc_id,
                       doc_title, source_type, section and chunk_index in its
                       payload, so retrieval results can cite exactly where
                       they came from.

  research_memory    - persistent long-term memory. Findings and answered
                       questions are written back here after each run, so a
                       later session starts from accumulated context instead
                       of from zero.

No agent touches a file on disk. Everything goes through these functions.
"""

from __future__ import annotations

import time
import uuid

from actian_vectorai import (
    Distance,
    PointStruct,
    VectorAIClient,
    VectorParams,
)

from research_assistant.config import (
    COLLECTIONS,
    DOCUMENTS_COLLECTION,
    EMBEDDING_DIM,
    MEMORY_COLLECTION,
    VECTORAI_ACCESS_TOKEN,
    VECTORAI_URL,
)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client() -> VectorAIClient:
    """Return a connected VectorAI DB client.

    VectorAIClient does not connect on construction; the gRPC channel must be
    established explicitly before the collections/points namespaces work.
    """
    kwargs: dict = {"url": VECTORAI_URL}
    if VECTORAI_ACCESS_TOKEN:
        kwargs["access_token"] = VECTORAI_ACCESS_TOKEN
    client = VectorAIClient(**kwargs)
    client.connect()
    return client


# ---------------------------------------------------------------------------
# Collection setup
# ---------------------------------------------------------------------------

def setup_collections(client: VectorAIClient, dim: int = EMBEDDING_DIM) -> None:
    """Create both collections if they do not already exist. Idempotent."""
    vector_cfg = VectorParams(size=dim, distance=Distance.Cosine)
    for name in COLLECTIONS:
        if not client.collections.exists(name):
            client.collections.create(name, vectors_config=vector_cfg)
            print(f"Created collection: {name}")


def _new_id() -> int:
    """A collision-resistant positive integer point id."""
    return uuid.uuid4().int % (2**63)


# ---------------------------------------------------------------------------
# Document writes
# ---------------------------------------------------------------------------

def upsert_chunks(
    client: VectorAIClient,
    chunks: list[dict],
    embedder,
    batch_size: int = 64,
) -> int:
    """
    Embed and write document chunks into the shared documents collection.

    Each chunk dict must carry: text, doc_id, doc_title, source_type,
    section, chunk_index.
    """
    if not chunks:
        return 0

    written = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_passages([c["text"] for c in batch])
        points = [
            PointStruct(id=_new_id(), vector=vectors[i], payload=batch[i])
            for i in range(len(batch))
        ]
        client.points.upsert(DOCUMENTS_COLLECTION, points)
        written += len(points)
    return written


# ---------------------------------------------------------------------------
# Document reads
# ---------------------------------------------------------------------------

def search_documents(
    client: VectorAIClient,
    vector: list[float],
    top_k: int,
) -> list:
    """Vector search across all indexed source material."""
    if client.points.count(DOCUMENTS_COLLECTION) == 0:
        return []
    return client.points.search(DOCUMENTS_COLLECTION, vector=vector, limit=top_k)


def scroll_documents(client: VectorAIClient, limit: int = 10_000) -> list:
    """
    Read every point without a vector query. Used by get_section and the
    Library tab, which filter or aggregate payload metadata rather than
    semantic similarity. ``limit`` is the page size for each scroll request.
    """
    if client.points.count(DOCUMENTS_COLLECTION) == 0:
        return []

    points = []
    offset = None
    seen_offsets = set()
    while True:
        kwargs = {"limit": limit}
        if offset is not None:
            kwargs["offset"] = offset

        page, next_offset = client.points.scroll(DOCUMENTS_COLLECTION, **kwargs)
        points.extend(page)

        if next_offset is None:
            return points
        if next_offset in seen_offsets:
            raise RuntimeError("VectorAI DB returned a repeated scroll cursor")

        seen_offsets.add(next_offset)
        offset = next_offset


def list_documents(client: VectorAIClient) -> list[dict]:
    """Return one row per distinct ingested document, with chunk counts."""
    seen: dict[str, dict] = {}
    for point in scroll_documents(client):
        payload = getattr(point, "payload", {}) or {}
        doc_id = payload.get("doc_id")
        if not doc_id:
            continue
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "doc_title": payload.get("doc_title", doc_id),
                "source_type": payload.get("source_type", "unknown"),
                "chunks": 0,
            }
        seen[doc_id]["chunks"] += 1
    return sorted(seen.values(), key=lambda d: d["doc_title"])


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------

def search_memory(
    client: VectorAIClient,
    vector: list[float],
    top_k: int,
) -> list:
    """Search persistent memory from earlier sessions."""
    if client.points.count(MEMORY_COLLECTION) == 0:
        return []
    return client.points.search(MEMORY_COLLECTION, vector=vector, limit=top_k)


def store_memory(
    client: VectorAIClient,
    embedder,
    query: str,
    finding: str,
    session_id: str,
    score: float,
    doc_ids: list[str] | None = None,
) -> None:
    """
    Persist a finding into long-term memory.

    This is what makes the assistant improve across runs: the next session
    retrieves these alongside the raw source chunks.
    """
    payload = {
        "kind": "finding",
        "query": query,
        "finding": finding,
        "session_id": session_id,
        "critic_score": round(float(score), 2),
        "doc_ids": doc_ids or [],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    vector = embedder.embed_passage(f"{query}\n{finding}")
    client.points.upsert(
        MEMORY_COLLECTION,
        [PointStruct(id=_new_id(), vector=vector, payload=payload)],
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_collection_counts(client: VectorAIClient) -> dict[str, int]:
    """Return point counts for both collections."""
    counts = {}
    for name in COLLECTIONS:
        try:
            counts[name] = client.points.count(name)
        except Exception:
            counts[name] = 0
    return counts


def reset_memory(client: VectorAIClient) -> None:
    """Drop and recreate the memory collection. Documents are left alone."""
    if client.collections.exists(MEMORY_COLLECTION):
        client.collections.delete(MEMORY_COLLECTION)
    client.collections.create(
        MEMORY_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.Cosine),
    )
