"""Reasoning RAG - Gradio UI.

Two parallel panels:
* Left  - the live, step-by-step reasoning trace.
* Right - the grounded final answer with numbered citations.
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

import rag
from agents import run_retriever, stream_answer
from rag import RetrievedChunk

load_dotenv()

TOP_K = 5

CUSTOM_CSS = """
#reasoning-panel, #answer-panel {
    min-height: 360px;
    border-radius: 12px;
    padding: 8px 16px;
}
#reasoning-panel { background: #0f172a0a; border: 1px solid #6366f133; }
#answer-panel { background: #05966908; border: 1px solid #05966933; }
.panel-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; }
footer { visibility: hidden; }
"""


def _missing_keys() -> list[str]:
    """Return a list of required environment variable names that are not currently set."""
    missing = []
    if not os.getenv("ORQ_API_KEY"):
        missing.append("ORQ_API_KEY")
    if not os.getenv("MISTRAL_API_KEY"):
        missing.append("MISTRAL_API_KEY")
    return missing


def add_source(url: str) -> str:
    """Ingest a URL into ChromaDB and report what happened."""
    if not url or not url.strip():
        return "Please paste a URL first."
    try:
        result = rag.ingest_url(url)
    except Exception as exc:  # surfaced to the user in the UI
        return f"Could not ingest the URL: {exc}"

    stats = rag.collection_stats()
    return (
        f"Added **{result.title}**\n\n"
        f"- Source: {result.url}\n"
        f"- Chunks indexed: {result.num_chunks}\n"
        f"- Characters: {result.num_chars:,}\n"
        f"- Total chunks in knowledge base: {stats['count']}"
    )


def clear_knowledge() -> str:
    """Wipe the knowledge base."""
    rag.reset_knowledge()
    return "Knowledge base cleared. Add a URL to start again."


def format_citations(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered citation list."""
    if not chunks:
        return "_No citations yet._"
    lines = ["### Citations"]
    for i, chunk in enumerate(chunks, start=1):
        snippet = chunk.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "…"
        title = chunk.title or chunk.url or "source"
        link = f"[{title}]({chunk.url})" if chunk.url else title
        lines.append(
            f"**[{i}]** {link}  \n"
            f"<sub>relevance {chunk.score}</sub>  \n"
            f"> {snippet}"
        )
    return "\n\n".join(lines)


def ask(question: str):
    """Retrieve, then stream reasoning + answer into the two panels."""
    question = (question or "").strip()
    if not question:
        yield "", "Please enter a question.", "_No citations yet._"
        return

    yield "Retrieving the most relevant passages…", "", "_No citations yet._"

    try:
        chunks = run_retriever(question, k=TOP_K)
    except Exception as exc:
        yield "", f"Retrieval failed: {exc}", "_No citations yet._"
        return

    citations = format_citations(chunks)

    if not chunks:
        yield (
            "No knowledge has been indexed yet.",
            "Please add a knowledge source URL above before asking a question.",
            "_No citations yet._",
        )
        return

    yield "Reasoning over the retrieved sources…", "", citations

    try:
        for reasoning, answer in stream_answer(question, chunks):
            yield (reasoning or "_Thinking…_", answer or "", citations)
    except Exception as exc:
        yield (f"Reasoning failed: {exc}", "", citations)


def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface with the two-panel reasoning UI."""
    with gr.Blocks(title="Reasoning RAG", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        gr.Markdown(
            "# Reasoning RAG\n"
            "Answer questions from your own knowledge sources with a **visible "
            "reasoning trace** and **cited answers**."
        )

        missing = _missing_keys()
        if missing:
            gr.Markdown(
                f"> ⚠️ Missing environment variable(s): **{', '.join(missing)}**. "
                "Copy `.env.example` to `.env` and fill them in."
            )

        with gr.Group():
            gr.Markdown("### 1. Add a knowledge source")
            with gr.Row():
                url_input = gr.Textbox(
                    label="Source URL",
                    placeholder="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                    scale=4,
                )
                add_btn = gr.Button("Add source", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
            source_status = gr.Markdown()

        with gr.Group():
            gr.Markdown("### 2. Ask a question")
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="What is retrieval-augmented generation?",
                    scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

        gr.Markdown("### 3. Reasoning & answer")
        with gr.Row():
            with gr.Column():
                gr.Markdown("<div class='panel-title'>🧠 Reasoning trace</div>")
                reasoning_panel = gr.Markdown(elem_id="reasoning-panel")
            with gr.Column():
                gr.Markdown("<div class='panel-title'>✅ Grounded answer</div>")
                answer_panel = gr.Markdown(elem_id="answer-panel")
                citations_panel = gr.Markdown()

        add_btn.click(add_source, inputs=url_input, outputs=source_status)
        url_input.submit(add_source, inputs=url_input, outputs=source_status)
        clear_btn.click(clear_knowledge, inputs=None, outputs=source_status)

        ask_outputs = [reasoning_panel, answer_panel, citations_panel]
        ask_btn.click(ask, inputs=question_input, outputs=ask_outputs)
        question_input.submit(ask, inputs=question_input, outputs=ask_outputs)

        gr.Examples(
            examples=[
                "What is the main idea described in the source?",
                "Summarize the key points with citations.",
            ],
            inputs=question_input,
        )

    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
