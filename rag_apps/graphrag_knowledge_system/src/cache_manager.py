import hashlib
import json
from datetime import datetime
from pathlib import Path


class CacheManager:
    def __init__(self, cache_path: str = "./cache.json"):
        self.cache_path = Path(cache_path)
        self._load()

    def _load(self):
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def _save(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def file_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def is_processed(self, file_hash: str) -> bool:
        return file_hash in self.cache

    def mark_processed(
        self, file_hash: str, filename: str, doc_id: str, chunk_count: int
    ):
        self.cache[file_hash] = {
            "filename": filename,
            "document_id": doc_id,
            "chunk_count": chunk_count,
            "processed_at": datetime.now().isoformat(),
        }
        self._save()

    def get_all(self) -> list:
        return [
            {
                "filename": v["filename"],
                "document_id": v["document_id"],
                "chunk_count": v["chunk_count"],
                "processed_at": v["processed_at"],
            }
            for v in self.cache.values()
        ]

    def remove(self, file_hash: str):
        if file_hash in self.cache:
            del self.cache[file_hash]
            self._save()
