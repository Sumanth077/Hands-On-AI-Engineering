"""
Configuration and one-time wiring of LlamaIndex to local Ollama.

Everything is read from environment variables (see .env.example), so nothing
is hardcoded. `configure_settings()` points LlamaIndex at your local Ollama for
both the chat model and the embedding model; call it once at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """All runtime settings, resolved from the environment with sane defaults."""

    # LlamaParse (cloud) -- the only thing that leaves your machine.
    llama_cloud_api_key: str = os.getenv("LLAMA_CLOUD_API_KEY", "")

    # Ollama (local) -- the chat model and the embedding model both run here.
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Retrieval / citation granularity. Fewer chunks = shorter prompts = fewer
    # refine passes, which keeps a small local model responsive.
    similarity_top_k: int = int(os.getenv("SIMILARITY_TOP_K", "4"))
    citation_chunk_size: int = int(os.getenv("CITATION_CHUNK_SIZE", "512"))

    # Local model can be slow, and the citation engine may run several sequential
    # generations per question (compact-and-refine). The first call also loads the
    # model into memory. Give it plenty of room to avoid httpx ReadTimeout.
    request_timeout_s: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))

    # Context window. Keep this modest: some models (e.g. qwen3) default to a
    # huge context, and Ollama then tries to allocate a KV cache far bigger than
    # your RAM. 8192 is plenty for page-level Q&A. Lower to 4096 if still OOM.
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    # On-disk caches so we never re-parse or re-index the same file (saves credits).
    cache_dir: Path = Path(os.getenv("CACHE_DIR", ".cache"))

    @property
    def parse_cache_dir(self) -> Path:
        return self.cache_dir / "parse"

    @property
    def index_cache_dir(self) -> Path:
        return self.cache_dir / "index"


SETTINGS = Settings()


def configure_settings() -> None:
    """Wire LlamaIndex's global Settings to local Ollama. Call once at startup."""
    # Imported lazily so importing this module stays cheap (and testable).
    from llama_index.core import Settings as LlamaSettings
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    LlamaSettings.llm = Ollama(
        model=SETTINGS.ollama_model,
        base_url=SETTINGS.ollama_base_url,
        request_timeout=SETTINGS.request_timeout_s,
        context_window=SETTINGS.ollama_num_ctx,
        # Force Ollama to load with this context size (caps the KV-cache memory).
        additional_kwargs={"num_ctx": SETTINGS.ollama_num_ctx},
    )
    LlamaSettings.embed_model = OllamaEmbedding(
        model_name=SETTINGS.ollama_embed_model,
        base_url=SETTINGS.ollama_base_url,
    )


def require_api_key() -> str:
    """Return the LlamaCloud API key or raise a clear error if it is missing."""
    if not SETTINGS.llama_cloud_api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://cloud.llamaindex.ai (Settings -> API Keys)."
        )
    return SETTINGS.llama_cloud_api_key
