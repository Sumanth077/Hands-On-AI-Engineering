"""
A quick structured overview of the document, shown before the Q&A.

This uses the local Ollama model over the first few parsed pages to produce a
short title / what-it-is / key-points summary, so the user gets their bearings
before asking questions. It is intentionally simple and best-effort.

Note: LlamaExtract (from llama_cloud_services) is the production-grade way to do
this -- give it a schema and it returns validated JSON. We keep the local-LLM
version here so the app has zero extra cloud calls by default; swapping in
LlamaExtract is a small change and is noted in the README.
"""

from __future__ import annotations

from llama_index.core.schema import Document

_OVERVIEW_PROMPT = """You are summarizing a document for a reader who has not opened it yet.
Using only the text below, produce a short overview in this exact format:

Title: <the document's title, or a best guess>
What it is: <one sentence on what kind of document this is and what it covers>
Key points:
- <point 1>
- <point 2>
- <point 3>

Do not invent facts. If something is unclear, say so briefly.

DOCUMENT (first pages):
---
{context}
---
"""


def generate_overview(documents: list[Document], max_chars: int = 6000) -> str:
    """Return a short plain-text overview of the document."""
    from llama_index.core import Settings

    context = ""
    for doc in documents:
        context += doc.text.strip() + "\n\n"
        if len(context) >= max_chars:
            break
    context = context[:max_chars]

    prompt = _OVERVIEW_PROMPT.format(context=context)
    return str(Settings.llm.complete(prompt)).strip()
