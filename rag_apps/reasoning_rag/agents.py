"""Agno agents for Reasoning RAG.

Two agents collaborate:

* **Retriever agent** - turns a user question into a focused search query and
  pulls the most semantically relevant chunks out of ChromaDB.
* **Reasoning agent** - works through the retrieved evidence step by step and
  produces a grounded, cited answer.

Both agents use the OpenAI SDK pointed at the Orq.ai router.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Tuple

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv

import rag
from rag import RetrievedChunk

load_dotenv()

ORQ_API_KEY = os.getenv("ORQ_API_KEY")
ORQ_BASE_URL = "https://api.orq.ai/v3/router"
# Orq requires a `provider/model` slug. `qwen3-6b-flash` from the original spec
# is not a real Orq model; `alibaba/qwen3.6-flash` is the current Qwen "flash"
# model. Override via the ORQ_MODEL env var if your workspace enables another.
MODEL_ID = os.getenv("ORQ_MODEL", "alibaba/qwen3.6-flash")

ANSWER_MARKER = "### ANSWER"
REASONING_MARKER = "### REASONING"

_llm: OpenAIChat | None = None
_retriever_agent: Agent | None = None
_reasoning_agent: Agent | None = None


def build_llm() -> OpenAIChat:
    """Build the shared LLM (OpenAI SDK -> Orq.ai router)."""
    global _llm
    if _llm is None:
        if not ORQ_API_KEY:
            raise RuntimeError("ORQ_API_KEY is not set. Add it to your .env file.")
        _llm = OpenAIChat(
            id=MODEL_ID,
            base_url=ORQ_BASE_URL,
            api_key=ORQ_API_KEY,
        )
    return _llm


def get_retriever_agent() -> Agent:
    """The Retriever agent rewrites a question into a tight search query."""
    global _retriever_agent
    if _retriever_agent is None:
        _retriever_agent = Agent(
            name="Retriever",
            model=build_llm(),
            instructions=[
                "You are a retrieval specialist for a semantic search system.",
                "Given a user's question, produce the single best search query "
                "to find relevant passages in a vector database.",
                "Expand abbreviations and include key entities and synonyms.",
                "Respond with ONLY the search query text — no quotes, no labels, "
                "no explanation.",
            ],
            markdown=False,
        )
    return _retriever_agent


def get_reasoning_agent() -> Agent:
    """The Reasoning agent thinks step by step and writes the cited answer."""
    global _reasoning_agent
    if _reasoning_agent is None:
        _reasoning_agent = Agent(
            name="Reasoning",
            model=build_llm(),
            instructions=[
                "You are a careful analyst that answers strictly from the "
                "provided SOURCES.",
                "Always respond in exactly two sections using these headers:",
                f"{REASONING_MARKER}",
                "  - Think step by step. Walk through which sources are relevant, "
                "what they say, and how they combine to answer the question. "
                "Refer to sources as [1], [2], etc.",
                f"{ANSWER_MARKER}",
                "  - Give a concise, well-structured final answer grounded only in "
                "the sources, with inline citations like [1] or [2].",
                "If the sources do not contain the answer, say so explicitly in the "
                "ANSWER section instead of guessing.",
                "Never invent citations for sources that were not provided.",
            ],
            markdown=True,
        )
    return _reasoning_agent


def _agent_text(question: str, agent: Agent) -> str:
    """Run an agent once (no streaming) and return its text content."""
    response = agent.run(question)
    return (getattr(response, "content", "") or "").strip()


def run_retriever(question: str, k: int = 5) -> List[RetrievedChunk]:
    """Refine the query with the Retriever agent, then pull chunks from ChromaDB."""
    refined = question
    try:
        candidate = _agent_text(question, get_retriever_agent())
        # Guard against a chatty model; keep it to a single line.
        candidate = candidate.splitlines()[0].strip() if candidate else ""
        if candidate:
            refined = candidate
    except Exception:
        # If query refinement fails, fall back to the raw question.
        refined = question

    chunks = rag.retrieve(refined, k=k)
    # Safety net: if the refined query returned nothing, retry with the original.
    if not chunks and refined != question:
        chunks = rag.retrieve(question, k=k)
    return chunks


def build_sources_block(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered SOURCES block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.title or chunk.url}".strip()
        lines.append(f"{header}\n{chunk.text}")
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """Assemble the full reasoning-agent prompt."""
    sources = build_sources_block(chunks)
    return (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{sources}\n\n"
        "Answer the question using the format described in your instructions."
    )


def split_trace(buffer: str) -> Tuple[str, str]:
    """Split a streamed buffer into (reasoning, answer) for the two panels."""
    text = buffer.replace("<think>", "").replace("</think>", "")

    answer = ""
    reasoning = text

    if ANSWER_MARKER in text:
        reasoning, answer = text.split(ANSWER_MARKER, 1)

    reasoning = reasoning.replace(REASONING_MARKER, "").strip()
    answer = answer.strip()
    return reasoning, answer


def stream_answer(
    question: str, chunks: List[RetrievedChunk]
) -> Iterator[Tuple[str, str]]:
    """Stream the Reasoning agent, yielding (reasoning, answer) snapshots."""
    if not chunks:
        yield (
            "No knowledge has been indexed yet, so there is nothing to reason over.",
            "I don't have any sources to answer from. Please add a URL as a "
            "knowledge source first.",
        )
        return

    agent = get_reasoning_agent()
    prompt = build_prompt(question, chunks)

    buffer = ""
    streamed_any = False
    try:
        for event in agent.run(prompt, stream=True):
            delta = getattr(event, "content", None)
            if not delta:
                continue
            streamed_any = True
            buffer += delta
            yield split_trace(buffer)
    except TypeError:
        # Some Agno versions don't accept stream kwarg on run(); fall back.
        streamed_any = False

    if not streamed_any:
        response = agent.run(prompt)
        buffer = getattr(response, "content", "") or ""
        yield split_trace(buffer)
