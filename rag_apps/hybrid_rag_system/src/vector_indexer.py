from typing import Callable, Optional
import requests
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from config import CHROMA_DIR, CHROMA_COLLECTION, OLLAMA_URL, EMBED_MODEL


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings


def check_ollama() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            has_embed = any(EMBED_MODEL in m for m in models)
            if not has_embed:
                return False, f"Model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}"
            return True, "Ollama is running"
        return False, "Ollama returned unexpected status"
    except Exception as e:
        return False, f"Ollama not reachable at {OLLAMA_URL}: {e}"


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=OllamaEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(
    chunks: list[str],
    collection: chromadb.Collection,
    doc_hash: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    batch_size = 5
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"{doc_hash}_{i + j}" for j in range(len(batch))]
        metadatas = [{"doc_hash": doc_hash, "chunk_index": i + j} for j in range(len(batch))]
        collection.add(documents=batch, ids=ids, metadatas=metadatas)
        if progress_cb:
            progress_cb(min(i + batch_size, total), total)


def collection_has_doc(collection: chromadb.Collection, doc_hash: str) -> bool:
    try:
        results = collection.get(where={"doc_hash": doc_hash}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False
