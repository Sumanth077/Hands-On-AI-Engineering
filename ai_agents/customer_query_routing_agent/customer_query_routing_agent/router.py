"""
Semantic routing engine.

Searches all four VectorAI DB collections simultaneously:
  - product_faq      (clean Q&A pairs)
  - product_docs     (policy and documentation chunks)
  - resolved_tickets (historical ticket threads)
  - resolved_queries (persistent agent memory)

Results from all sources are merged, labelled by origin, and ranked by
similarity score before being passed to the LLM as unified context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from actian_vectorai import VectorAIClient

from customer_query_routing_agent.config import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from customer_query_routing_agent.embedder import Embedder
from customer_query_routing_agent.vectorstore import (
    search_docs,
    search_faq,
    search_memory,
    search_tickets,
)

# Human-readable labels shown in the UI
SOURCE_LABELS = {
    "faq": "FAQ",
    "docs": "Policy Doc",
    "tickets": "Past Ticket",
    "memory": "Agent Memory",
}


@dataclass
class RoutingResult:
    department: str
    confidence: float
    confidence_label: str
    context_docs: list[dict] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    query_vector: list[float] = field(default_factory=list)


def _label(score: float) -> str:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if score >= LOW_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


def _hits_to_docs(hits: list, source_key: str) -> list[dict]:
    """Convert ScoredPoint results into a standardised context dict."""
    docs = []
    for hit in hits:
        p = hit.payload
        docs.append({
            "source": source_key,
            "source_label": SOURCE_LABELS[source_key],
            "department": p.get("department", ""),
            "question": p.get("question") or p.get("summary") or p.get("query") or p.get("title", ""),
            "answer": p.get("answer") or p.get("content") or p.get("thread") or p.get("resolution", ""),
            "score": round(float(hit.score), 3),
            # Ticket-specific metadata
            "ticket_id": p.get("ticket_id", ""),
            "resolution_type": p.get("resolution_type", ""),
            # Doc-specific metadata
            "doc_type": p.get("doc_type", ""),
            "title": p.get("title", ""),
        })
    return docs


def route(
    query: str,
    client: VectorAIClient,
    embedder: Embedder,
) -> RoutingResult:
    """
    Search all four knowledge sources in VectorAI DB and return a unified routing result.
    """
    query_vector = embedder.embed(query)

    # Search all sources
    faq_hits = search_faq(client, query_vector, top_k=3)
    doc_hits = search_docs(client, query_vector, top_k=2)
    ticket_hits = search_tickets(client, query_vector, top_k=2)
    memory_hits = search_memory(client, query_vector, top_k=2)

    # Convert to unified context dicts
    all_docs = (
        _hits_to_docs(faq_hits, "faq")
        + _hits_to_docs(doc_hits, "docs")
        + _hits_to_docs(ticket_hits, "tickets")
        + _hits_to_docs(memory_hits, "memory")
    )

    # Sort by score descending so the LLM gets the most relevant context first
    all_docs.sort(key=lambda d: d["score"], reverse=True)

    # Department and confidence driven by the single best result across all sources
    if all_docs:
        best = all_docs[0]
        department = best["department"] or "General Inquiry"
        confidence = best["score"]
    else:
        department = "General Inquiry"
        confidence = 0.0

    return RoutingResult(
        department=department,
        confidence=confidence,
        confidence_label=_label(confidence),
        context_docs=all_docs,
        source_counts={
            "faq": len(faq_hits),
            "docs": len(doc_hits),
            "tickets": len(ticket_hits),
            "memory": len(memory_hits),
        },
        query_vector=query_vector,
    )
