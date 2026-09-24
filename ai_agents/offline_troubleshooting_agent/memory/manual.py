import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv

from memory.embeddings import embed_text, embed_texts
from memory.vectorai_client import ensure_collection, search as vectorai_search, upsert_point

load_dotenv()

DEFAULT_MANUAL_COLLECTION = "manual_chunks"
MANUAL_EMBEDDING_DIM = 768

HEADING_PATTERN = re.compile(r"^(##|###)[ \t]+(.*)$", re.MULTILINE)
H1_PATTERN = re.compile(r"^#[ \t]+(.*)$", re.MULTILINE)


def _configured_manual_collection() -> str:
    return os.getenv("ACTIAN_VECTORAI_MANUAL_COLLECTION", DEFAULT_MANUAL_COLLECTION)


def _first_h1_title(text: str) -> str | None:
    match = H1_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _derive_component(heading: str, text: str) -> str:
    haystack = f"{heading} {text}".lower()
    if "bearing" in haystack or "brg" in haystack:
        return "bearing"
    if "cooling" in haystack or "tmp" in haystack:
        return "cooling"
    if "pressure" in haystack or "prs" in haystack:
        return "pressure"
    return "general"


def _build_chunk(body: str, section_title: str, source_filename: str) -> dict:
    return {
        "text": body,
        "section_title": section_title,
        "source_filename": source_filename,
        "component": _derive_component(section_title, body),
    }


def chunk_markdown_file(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    source_filename = Path(path).name

    matches = list(HEADING_PATTERN.finditer(text))
    chunks: list[dict] = []

    preamble_end = matches[0].start() if matches else len(text)
    preamble = text[:preamble_end].strip()
    if preamble:
        title = _first_h1_title(preamble) or "Introduction"
        chunks.append(_build_chunk(preamble, title, source_filename))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_title = match.group(2).strip()
        body = text[start:end].strip()
        chunks.append(_build_chunk(body, section_title, source_filename))

    return chunks


def ingest_manual_documents(paths: list[str]) -> int:
    collection = _configured_manual_collection()
    ensure_collection(collection, dim=MANUAL_EMBEDDING_DIM)

    total = 0
    for path in paths:
        chunks = chunk_markdown_file(path)
        if not chunks:
            continue

        vectors = embed_texts([chunk["text"] for chunk in chunks])

        for chunk, vector in zip(chunks, vectors):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{chunk['source_filename']}:{chunk['section_title']}",
                )
            )
            upsert_point(collection, point_id, vector, chunk)

        total += len(chunks)

    return total


def search_manual_chunks(query: str, top_k: int = 3) -> list[dict]:
    collection = _configured_manual_collection()
    ensure_collection(collection, dim=MANUAL_EMBEDDING_DIM)
    vector = embed_text(query)
    raw_results = vectorai_search(collection, vector, top_k)

    results = []
    for item in raw_results:
        payload = item.get("payload") or {}
        results.append(
            {
                "id": item["id"],
                "score": item["score"],
                "text": payload.get("text"),
                "section_title": payload.get("section_title"),
                "source_filename": payload.get("source_filename"),
                "component": payload.get("component"),
            }
        )
    return results
