import hashlib
import json
from pathlib import Path
from typing import Optional
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CACHE_FILE, CHUNK_SIZE, CHUNK_OVERLAP


def load_document(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    elif suffix == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF or TXT.")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def compute_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def is_cached(content_hash: str) -> Optional[dict]:
    return load_cache().get(content_hash)


def add_to_cache(content_hash: str, filename: str, num_chunks: int, num_entities: int) -> None:
    cache = load_cache()
    cache[content_hash] = {
        "filename": filename,
        "num_chunks": num_chunks,
        "num_entities": num_entities,
        "hash": content_hash,
    }
    save_cache(cache)


def list_cached_docs() -> list[dict]:
    return list(load_cache().values())
