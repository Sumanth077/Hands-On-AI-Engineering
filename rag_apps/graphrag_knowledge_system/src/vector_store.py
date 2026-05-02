import os
from typing import Any, Dict, List

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB EmbeddingFunction subclass backed by LangChain's OllamaEmbeddings."""

    def __init__(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._embedder = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url,
        )
        self._base_url = base_url

    def __call__(self, input: Documents) -> Embeddings:
        return self._embedder.embed_documents(list(input))

    @staticmethod
    def name() -> str:
        return "ollama-nomic-embed-text"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction()

    def get_config(self) -> Dict[str, Any]:
        return {"base_url": self._base_url, "model": "nomic-embed-text"}


class VectorStore:
    def __init__(self, persist_path: str = "./chroma_db"):
        self._embedding_fn = OllamaEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name="text_chunks",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunk(self, chunk_id: str, text: str, metadata: Dict):
        clean_meta = {k: str(v) for k, v in metadata.items()}
        try:
            self.collection.add(
                ids=[chunk_id],
                documents=[text],
                metadatas=[clean_meta],
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        total = self.collection.count()
        if total == 0:
            return []
        k = min(n_results, total)
        results = self.collection.query(query_texts=[query], n_results=k)
        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": (
                            results["distances"][0][i]
                            if results.get("distances")
                            else None
                        ),
                    }
                )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def embed_query(self, text: str) -> List[float]:
        return self._embedding_fn([text])[0]
