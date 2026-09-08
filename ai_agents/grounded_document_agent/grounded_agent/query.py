"""
Answer questions over the index with inline citations.

We use LlamaIndex's CitationQueryEngine: it splits retrieved context into
numbered sources, tells the model to cite them inline as [1], [2], and returns
the source nodes so we can show the exact text (and page) behind each number.
The model only sees the retrieved pages, so the answer is grounded in the
document, not the model's memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import CitationQueryEngine

from grounded_agent.config import SETTINGS

_PAGE_KEYS = ("page", "page_label", "page_number")


@dataclass
class Citation:
    """One numbered source behind an answer."""

    number: int
    page: int | None
    file_name: str
    text: str
    score: float | None


@dataclass
class Answer:
    """An answer plus the sources it cited."""

    text: str
    citations: list[Citation]


def make_query_engine(index: VectorStoreIndex) -> CitationQueryEngine:
    """Build a citation-aware query engine over the index."""
    return CitationQueryEngine.from_args(
        index,
        similarity_top_k=SETTINGS.similarity_top_k,
        citation_chunk_size=SETTINGS.citation_chunk_size,
    )


def _page_of(metadata: dict) -> int | None:
    for key in _PAGE_KEYS:
        if metadata.get(key) is not None:
            try:
                return int(metadata[key])
            except (TypeError, ValueError):
                return None
    return None


def ask(engine: CitationQueryEngine, question: str) -> Answer:
    """Run one question and return the grounded answer with its citations."""
    response = engine.query(question)

    citations: list[Citation] = []
    for i, source in enumerate(response.source_nodes, start=1):
        meta = source.node.metadata or {}
        citations.append(
            Citation(
                number=i,
                page=_page_of(meta),
                file_name=str(meta.get("file_name", "document")),
                text=source.node.get_text(),
                score=getattr(source, "score", None),
            )
        )

    return Answer(text=str(response), citations=citations)
