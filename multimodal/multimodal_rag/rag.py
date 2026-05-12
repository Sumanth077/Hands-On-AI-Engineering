"""
Multimodal RAG — core pipeline.

Ingests text, URLs, PDFs, images, audio, and video into ChromaDB.
Retrieves with Gemini Embedding 2 similarity search and generates
grounded answers with Gemini 3 Flash — passing file URIs back for
truly multimodal context on image, audio, and video sources.
"""

from __future__ import annotations

import mimetypes
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ── Constants ──────────────────────────────────────────────────────────────────

EMBED_MODEL = "gemini-embedding-2"
GEN_MODEL = "gemini-3-flash-preview"
COLLECTION_NAME = "multimodal_rag"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

MEDIA_PROMPTS = {
    "image": (
        "Describe this image in full detail. Include all visible text, objects, "
        "people, scenes, colors, spatial layout, and any other relevant information."
    ),
    "audio": (
        "Transcribe and describe this audio in full. Include all speech (verbatim), "
        "background sounds, music, tone, and any other auditory elements."
    ),
    "video": (
        "Describe and summarize this video in detail. Cover visual scenes, any "
        "on-screen text, speech (verbatim where possible), audio, and key events "
        "with approximate timestamps."
    ),
}

MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".m4a": "audio/mp4", ".flac": "audio/flac", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
}

SOURCE_ICONS = {
    "text": "📝", "url": "🌐", "pdf": "📄",
    "image": "🖼️", "audio": "🎵", "video": "🎬",
}

# ── LangChain-compatible Gemini Embeddings ─────────────────────────────────────

class GeminiEmbeddings(Embeddings):
    """Wraps Gemini Embedding 2 for use with LangChain components."""

    def __init__(self, client: genai.Client):
        self.client = client

    def _embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(model=EMBED_MODEL, contents=text)
        return list(result.embeddings[0].values)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each document individually — Gemini Embedding 2 does not support true batching."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class IndexedSource:
    label: str
    source_type: str  # text | url | pdf | image | audio | video
    chunks: int
    file_uri: str = ""
    mime_type: str = ""


@dataclass
class RAGResult:
    answer: str
    question: str
    retrieved_docs: list[Document] = field(default_factory=list)


# ── Main class ─────────────────────────────────────────────────────────────────

class MultimodalRAG:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.embeddings = GeminiEmbeddings(self.client)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        # In-memory ChromaDB collection (fresh per session)
        self._chroma = chromadb.Client()
        self._collection = self._chroma.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.sources: list[IndexedSource] = []

    # ── Private helpers ────────────────────────────────────────────────────────

    def _store_docs(self, docs: list[Document]) -> int:
        """Embed and insert LangChain Documents into ChromaDB."""
        if not docs:
            return 0
        texts = [d.page_content for d in docs]
        vectors = self.embeddings.embed_documents(texts)
        self._collection.add(
            ids=[str(uuid.uuid4()) for _ in docs],
            embeddings=vectors,
            documents=texts,
            metadatas=[d.metadata for d in docs],
        )
        return len(docs)

    def _upload_file(self, file_path: str, mime_type: str) -> str:
        """Upload a file to Gemini File API, wait for processing, return URI."""
        ref = self.client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(mime_type=mime_type),
        )
        while ref.state.name == "PROCESSING":
            time.sleep(1)
            ref = self.client.files.get(name=ref.name)
        return ref.uri

    def _describe_media(self, file_uri: str, mime_type: str, media_type: str) -> str:
        """Ask Gemini to describe/transcribe a media file via its URI."""
        prompt = MEDIA_PROMPTS.get(media_type, "Describe this content in detail.")
        response = self.client.models.generate_content(
            model=GEN_MODEL,
            contents=[
                types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime_type)),
                types.Part(text=prompt),
            ],
        )
        return response.text.strip()

    def _get_mime(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return MIME_MAP.get(ext, mimetypes.guess_type(file_path)[0] or "application/octet-stream")

    def _media_category(self, mime_type: str) -> str:
        for cat in ("image", "audio", "video"):
            if mime_type.startswith(cat):
                return cat
        return "unknown"

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_text(self, text: str, label: str = "Pasted text") -> int:
        """Index raw text input."""
        docs = self.splitter.create_documents(
            [text],
            metadatas=[{"source_type": "text", "source_label": label}],
        )
        n = self._store_docs(docs)
        self.sources.append(IndexedSource(label=label, source_type="text", chunks=n))
        return n

    def add_url(self, url: str) -> int:
        """Fetch a web page, strip HTML boilerplate, and index the content."""
        r = httpx.get(url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        docs = self.splitter.create_documents(
            [text],
            metadatas=[{"source_type": "url", "source_label": url}],
        )
        n = self._store_docs(docs)
        self.sources.append(IndexedSource(label=url, source_type="url", chunks=n))
        return n

    MAX_PDF_PAGES = 100  # cap to keep indexing fast

    def add_pdf(self, file_path: str) -> int:
        """Load a PDF with LangChain's PyPDFLoader and index up to MAX_PDF_PAGES pages."""
        label = Path(file_path).name
        pages = PyPDFLoader(file_path).load()
        if len(pages) > self.MAX_PDF_PAGES:
            pages = pages[: self.MAX_PDF_PAGES]  # take first N pages
        chunks = self.splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata.update({"source_type": "pdf", "source_label": label})
        n = self._store_docs(chunks)
        self.sources.append(IndexedSource(label=label, source_type="pdf", chunks=n))
        return n

    def add_image(self, file_path: str) -> int:
        """Upload image to Gemini File API, describe it, index the description."""
        label = Path(file_path).name
        mime_type = self._get_mime(file_path)
        file_uri = self._upload_file(file_path, mime_type)
        description = self._describe_media(file_uri, mime_type, "image")
        doc = Document(
            page_content=description,
            metadata={
                "source_type": "image", "source_label": label,
                "file_uri": file_uri, "mime_type": mime_type,
            },
        )
        n = self._store_docs([doc])
        self.sources.append(IndexedSource(label=label, source_type="image",
                                          chunks=n, file_uri=file_uri, mime_type=mime_type))
        return n

    def add_audio(self, file_path: str) -> int:
        """Upload audio to Gemini File API, transcribe it, index the transcript."""
        label = Path(file_path).name
        mime_type = self._get_mime(file_path)
        file_uri = self._upload_file(file_path, mime_type)
        transcript = self._describe_media(file_uri, mime_type, "audio")
        docs = self.splitter.create_documents(
            [transcript],
            metadatas=[{
                "source_type": "audio", "source_label": label,
                "file_uri": file_uri, "mime_type": mime_type,
            }],
        )
        n = self._store_docs(docs)
        self.sources.append(IndexedSource(label=label, source_type="audio",
                                          chunks=n, file_uri=file_uri, mime_type=mime_type))
        return n

    def add_video(self, file_path: str) -> int:
        """Upload video to Gemini File API, describe it, index the description."""
        label = Path(file_path).name
        mime_type = self._get_mime(file_path)
        file_uri = self._upload_file(file_path, mime_type)
        description = self._describe_media(file_uri, mime_type, "video")
        docs = self.splitter.create_documents(
            [description],
            metadatas=[{
                "source_type": "video", "source_label": label,
                "file_uri": file_uri, "mime_type": mime_type,
            }],
        )
        n = self._store_docs(docs)
        self.sources.append(IndexedSource(label=label, source_type="video",
                                          chunks=n, file_uri=file_uri, mime_type=mime_type))
        return n

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Retrieve relevant chunks and generate a grounded answer.
        For image/audio/video sources, the original file URI is passed to
        Gemini 3 Flash alongside the text context for truly multimodal answers.
        """
        if self._collection.count() == 0:
            return RAGResult(
                answer="No sources have been added to the knowledge base yet. "
                       "Add text, a URL, PDF, image, audio, or video in the sidebar.",
                question=question,
            )

        # Retrieval
        query_vec = self.embeddings.embed_query(question)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas"],
        )
        retrieved = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

        # Build Gemini content parts
        parts: list[types.Part] = [
            types.Part(text=(
                "You are a knowledgeable assistant. Answer the question using ONLY "
                "the provided context. Cite the source label for each piece of "
                "information you use. If the answer is not in the context, say so.\n"
            ))
        ]

        seen_uris: set[str] = set()
        context_texts: list[str] = []

        for doc in retrieved:
            meta = doc.metadata
            label = meta.get("source_label", "unknown")
            file_uri = meta.get("file_uri", "")
            mime_type = meta.get("mime_type", "")

            # Include the actual media file once per unique URI
            if file_uri and file_uri not in seen_uris:
                parts.append(
                    types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime_type))
                )
                seen_uris.add(file_uri)

            context_texts.append(f"[Source: {label}]\n{doc.page_content}")

        parts.append(types.Part(text="Context:\n\n" + "\n\n---\n\n".join(context_texts)))
        parts.append(types.Part(text=f"\nQuestion: {question}\n\nAnswer (with citations):"))

        response = self.client.models.generate_content(model=GEN_MODEL, contents=parts)

        return RAGResult(
            answer=response.text.strip(),
            question=question,
            retrieved_docs=retrieved,
        )

    # ── Stats ──────────────────────────────────────────────────────────────────

    def chunk_count(self) -> int:
        return self._collection.count()

    def source_count(self) -> int:
        return len(self.sources)
