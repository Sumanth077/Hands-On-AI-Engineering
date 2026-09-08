"""
Parse a PDF with LlamaParse into page-level LlamaIndex Documents.

LlamaParse is the one cloud call in this project. It is also the whole point:
it reads tables, figures, multi-column layouts and scans that generic parsers
mangle, and it keeps the page each chunk came from -- which is what makes the
citations trustworthy.

We cache the parsed result on disk keyed by the file's content hash, so the same
PDF is never parsed twice (that keeps LlamaParse credit use flat during
development, where you re-run constantly).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llama_index.core.schema import Document

from citation_agent.config import SETTINGS, require_api_key

# Metadata keys LlamaParse may use for the page number, in order of preference.
_PAGE_KEYS = ("page_label", "page", "page_number", "page_index")


def _file_hash(path: str | Path) -> str:
    """Content hash of the file, used as the cache key."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _normalize_page(metadata: dict, fallback_index: int) -> int:
    """Pull a page number out of whatever key LlamaParse used; fall back to order."""
    for key in _PAGE_KEYS:
        if key in metadata and metadata[key] is not None:
            try:
                return int(metadata[key])
            except (TypeError, ValueError):
                pass
    return fallback_index + 1  # 1-based


def parse_pdf(path: str | Path, file_name: str | None = None) -> list[Document]:
    """
    Parse `path` into a list of Documents, one per page, each carrying its
    `page` and `file_name` in metadata. Cached by file content hash.
    """
    path = Path(path)
    name = file_name or path.name
    cache_file = SETTINGS.parse_cache_dir / f"{_file_hash(path)}.json"

    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Document(text=d["text"], metadata=d["metadata"]) for d in raw]

    # Import here so the module imports without the SDK installed (e.g. for tests).
    from llama_cloud_services import LlamaParse

    parser = LlamaParse(
        api_key=require_api_key(),
        result_type="markdown",  # layout-aware markdown keeps tables readable
        verbose=True,
    )
    parsed = parser.load_data(str(path))

    documents: list[Document] = []
    for i, doc in enumerate(parsed):
        meta = dict(doc.metadata or {})
        meta["page"] = _normalize_page(meta, i)
        meta["file_name"] = name
        documents.append(Document(text=doc.text, metadata=meta))

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            [{"text": d.text, "metadata": d.metadata} for d in documents],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return documents
