"""
Protocol retriever.
Embeds clinical protocol documents with SentenceTransformers
and stores / queries them in a local Qdrant collection.
"""

import glob
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "clinical_protocols"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384


class ProtocolRetriever:
    def __init__(self, db_path: str = "./qdrant_data"):
        self.client = QdrantClient(path=db_path)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    def ingest(self, protocols_dir: str = "./protocols") -> int:
        """
        Load every .md and .txt file in protocols_dir into Qdrant.
        Wipes any existing entries first so re-ingestion is idempotent.
        """
        files = sorted(
            glob.glob(os.path.join(protocols_dir, "*.md"))
            + glob.glob(os.path.join(protocols_dir, "*.txt"))
        )
        if not files:
            return 0

        self.client.delete_collection(COLLECTION_NAME)
        self._ensure_collection()

        points = []
        for idx, filepath in enumerate(files):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            title = (
                os.path.splitext(os.path.basename(filepath))[0]
                .replace("_", " ")
                .title()
            )
            vector = self.embedder.encode(content, show_progress_bar=False).tolist()
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "title": title,
                        "content": content,
                        "source": os.path.basename(filepath),
                    },
                )
            )

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    def retrieve(self, query: str, top_k: int = 1) -> list:
        """Return the top_k most semantically similar protocols."""
        vector = self.embedder.encode(query, show_progress_bar=False).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
        )
        return result.points

    def is_populated(self) -> bool:
        return self.client.count(collection_name=COLLECTION_NAME).count > 0
