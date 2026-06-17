"""
HyDE RAG pipeline.

Ingest: load → chunk → embed with Gemini Embedding 2 → store in ChromaDB
Query:  generate N hypothetical answers with Gemini 3 Flash
        → embed each → average vectors (HyDE embedding)
        → retrieve matching chunks from ChromaDB
        → generate final answer with Gemini 3 Flash
"""

import os
import uuid

import chromadb
from google import genai
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedder import GeminiEmbedder

# Prompts

HYPOTHETICAL_PROMPT = """\
You are a knowledgeable assistant. A user has asked the following question:

"{question}"

Write a detailed, factual hypothetical document excerpt that directly answers \
this question. Write it as if extracted from a real reference document, \
specific, informative, and self-contained. Do not explain that it is hypothetical.
"""

ANSWER_PROMPT = """\
You are a precise question-answering assistant. Use only the provided context \
to answer the question. If the context does not contain enough information, \
say so clearly. Do not invent facts.

Context:
{context}

Question: {question}

Answer:
"""


# Pipeline

class HyDERAG:
    """End-to-end HyDE retrieval-augmented generation pipeline."""

    COLLECTION = "hyde_rag"
    MODEL_ID = "gemini-3-flash-preview"

    def __init__(self, google_api_key: str):
        self.embedder = GeminiEmbedder(api_key=google_api_key)
        self.client = genai.Client(api_key=google_api_key)
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(self.COLLECTION)
        self.doc_info: dict | None = None

    # Ingest

    def ingest(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> dict:
        """Load a PDF or TXT file, chunk it, embed chunks, and store in ChromaDB.

        Replaces any previously ingested document.

        Returns a metadata dict describing what was ingested.
        """
        # Clear previous collection
        self._chroma.delete_collection(self.COLLECTION)
        self._collection = self._chroma.get_or_create_collection(self.COLLECTION)

        # Load
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        # Embed + store
        texts = [c.page_content for c in chunks]
        metadatas = [
            {k: str(v) for k, v in c.metadata.items()} for c in chunks
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = self.embedder.embed_batch(texts)

        self._collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        self.doc_info = {
            "filename": os.path.basename(file_path),
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "pages": len(docs),
        }
        return self.doc_info

    # Query

    def _generate_hypothetical_docs(self, question: str, n: int) -> list[str]:
        """Use Gemini 3 Flash to generate N hypothetical answer documents."""
        docs = []
        for _ in range(n):
            response = self.client.models.generate_content(
                model=self.MODEL_ID,
                contents=HYPOTHETICAL_PROMPT.format(question=question),
            )
            docs.append(response.text.strip())
        return docs

    def query(
        self,
        question: str,
        n_hypothetical: int = 3,
        n_results: int = 3,
    ) -> dict:
        """Run the full HyDE query pipeline.

        Returns a dict with:
          - hypothetical_docs: list of N generated hypothetical answers
          - retrieved_chunks: list of (text, metadata) tuples
          - answer: final synthesised answer
        """
        if self._collection.count() == 0:
            raise ValueError(
                "No document ingested. Please upload a document first."
            )

        n_results = min(n_results, self._collection.count())

        # Step 1: generate hypothetical documents
        hypo_docs = self._generate_hypothetical_docs(question, n_hypothetical)

        # Step 2: HyDE embedding (average of hypothetical doc embeddings)
        hyde_vector = self.embedder.hyde_embed(hypo_docs)

        # Step 3: retrieve from ChromaDB using the HyDE vector
        results = self._collection.query(
            query_embeddings=[hyde_vector],
            n_results=n_results,
        )
        retrieved_texts = results["documents"][0]
        retrieved_meta = results["metadatas"][0]

        # Step 4: generate final answer
        context = "\n\n---\n\n".join(retrieved_texts)
        response = self.client.models.generate_content(
            model=self.MODEL_ID,
            contents=ANSWER_PROMPT.format(context=context, question=question),
        )
        answer = response.text.strip()

        return {
            "hypothetical_docs": hypo_docs,
            "retrieved_chunks": list(zip(retrieved_texts, retrieved_meta)),
            "answer": answer,
        }

    def clear(self):
        """Wipe the vector store and reset document info."""
        self._chroma.delete_collection(self.COLLECTION)
        self._collection = self._chroma.get_or_create_collection(self.COLLECTION)
        self.doc_info = None
