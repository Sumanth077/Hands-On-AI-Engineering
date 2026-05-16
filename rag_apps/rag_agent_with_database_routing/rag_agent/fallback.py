"""
RAG Agent with Database Routing - fallback agent.

Activated when the Qdrant retriever finds no relevant documents.
Searches the web with DuckDuckGo and answers via Orq.ai zai/glm-5-turbo.
"""

from __future__ import annotations

from ddgs import DDGS
from openai import OpenAI

MODEL = "zai/glm-5-turbo"


def run_fallback(client: OpenAI, query: str) -> str:
    """Search the web and generate an answer using Orq.ai."""
    results = list(DDGS().text(query, max_results=5))
    if not results:
        context = "No web results found."
    else:
        context = "\n\n".join(
            f"{r['title']}: {r['body']}" for r in results
        )

    response = client.responses.create(
        model=MODEL,
        instructions=(
            "Answer the question using the web search results provided. "
            "Be concise and factual. Cite key details from the results."
        ),
        input=f"Question: {query}\n\nSearch results:\n{context}",
    )
    return response.output[0].content[0].text
