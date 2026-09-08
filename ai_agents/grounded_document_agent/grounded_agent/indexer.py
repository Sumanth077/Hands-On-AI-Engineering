"""
Build (or load) a vector index over the parsed pages.

Embeddings run locally through Ollama. The index is persisted to disk keyed by
the same content hash as the parse cache, so re-opening the same PDF is instant
and costs nothing. Page metadata rides along on every node, so it survives all
the way to the citations.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import Document

from grounded_agent.config import SETTINGS


def build_or_load_index(documents: list[Document], doc_key: str) -> VectorStoreIndex:
    """
    Return a VectorStoreIndex for these documents. If one was already built for
    `doc_key`, load it from disk; otherwise build it and persist it.

    `configure_settings()` must have been called first so the embedding model is
    the local Ollama one.
    """
    persist_dir = SETTINGS.index_cache_dir / doc_key

    if persist_dir.exists():
        storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(storage)

    index = VectorStoreIndex.from_documents(documents)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index
