"""
HyDE RAG, Gradio UI
"""

import os

import gradio as gr
from dotenv import load_dotenv

from rag import HyDERAG

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────

pipeline: HyDERAG | None = None


def get_or_create_pipeline(google_key: str) -> HyDERAG:
    """Return the existing pipeline, creating one with the given key if none exists yet."""
    global pipeline
    if pipeline is None:
        pipeline = HyDERAG(google_api_key=google_key)
    return pipeline


# ── Handlers ───────────────────────────────────────────────────────────────────

def validate_keys(google_key: str) -> tuple[bool, str]:
    """Check that a Google API key was provided, returning a success flag and error message."""
    if not google_key.strip():
        return False, "Google API key is required."
    return True, ""


def ingest_document(file, chunk_size, chunk_overlap, google_key):
    """Validate inputs and run the uploaded file through the ingest pipeline, updating the status panel."""
    if file is None:
        return gr.update(value="No file uploaded.", visible=True), gr.update(value=None, visible=False)

    ok, msg = validate_keys(google_key)
    if not ok:
        return gr.update(value=msg, visible=True), gr.update(value=None, visible=False)

    try:
        rag = get_or_create_pipeline(google_key)
        info = rag.ingest(file.name, int(chunk_size), int(chunk_overlap))
        status_msg = (
            f"Document processed successfully.\n"
            f"File: {info['filename']}\n"
            f"Pages: {info['pages']}  |  Chunks: {info['total_chunks']}\n"
            f"Chunk size: {info['chunk_size']}  |  Overlap: {info['chunk_overlap']}"
        )
        return gr.update(value=status_msg, visible=True), gr.update(value=info, visible=True)
    except Exception as e:
        return gr.update(value=f"Error: {str(e)}", visible=True), gr.update(value=None, visible=False)


def run_query(question, n_hypothetical, n_results, google_key):
    """Validate inputs, run the HyDE query pipeline, and format the hypothetical docs, chunks, and answer for display."""
    if not question.strip():
        return "", "", ""

    ok, msg = validate_keys(google_key)
    if not ok:
        return msg, "", ""

    try:
        rag = get_or_create_pipeline(google_key)
        result = rag.query(
            question=question,
            n_hypothetical=int(n_hypothetical),
            n_results=int(n_results),
        )

        # Format hypothetical documents
        hypo_text = ""
        for i, doc in enumerate(result["hypothetical_docs"], 1):
            hypo_text += f"Hypothetical Document {i}\n"
            hypo_text += "-" * 50 + "\n"
            hypo_text += doc + "\n\n"

        # Format retrieved chunks
        chunks_text = ""
        for i, (text, meta) in enumerate(result["retrieved_chunks"], 1):
            source = meta.get("source", "")
            page = meta.get("page", "")
            label = f"Chunk {i}"
            if source:
                label += f"  |  {os.path.basename(str(source))}"
            if page:
                label += f"  |  Page {int(float(str(page))) + 1}"
            chunks_text += f"{label}\n"
            chunks_text += "-" * 50 + "\n"
            chunks_text += text.strip() + "\n\n"

        return hypo_text.strip(), chunks_text.strip(), result["answer"]

    except ValueError as e:
        return str(e), "", ""
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def clear_all():
    """Reset the pipeline and clear all UI fields back to their initial state."""
    global pipeline
    if pipeline is not None:
        pipeline.clear()
        pipeline = None
    return (
        None,
        "",
        "",
        "",
        "",
        gr.update(value="Document cleared.", visible=True),
        gr.update(value=None, visible=False),
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

CSS = """
.panel-header {
    font-size: 13px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.status-box textarea {
    font-size: 13px !important;
    color: #374151 !important;
    background: #f9fafb !important;
    border-color: #e5e7eb !important;
}
.answer-box textarea {
    font-size: 15px !important;
    line-height: 1.7 !important;
}
.hypo-box textarea, .chunks-box textarea {
    font-family: ui-monospace, SFMono-Regular, monospace !important;
    font-size: 12.5px !important;
    line-height: 1.6 !important;
    color: #111827 !important;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="HyDE RAG",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.Markdown(
        """
# HyDE RAG
**Hypothetical Document Embeddings** for higher-precision retrieval.
Instead of embedding your query directly, the system generates hypothetical answers,
embeds them, averages the vectors, and uses that richer signal to retrieve the most
relevant chunks from your document.

**Models:** Gemini 3 Flash (hypothesis generation + answer synthesis) · Gemini Embedding 2 (vectors) · ChromaDB (vector store)
        """
    )
    gr.Markdown("---")

    with gr.Row(equal_height=False):

        # ── Left column — configuration ───────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.Markdown("### Document")
            file_input = gr.File(
                label="Upload PDF or TXT",
                file_types=[".pdf", ".txt"],
            )

            with gr.Row():
                chunk_size_slider = gr.Slider(
                    minimum=100,
                    maximum=1500,
                    value=500,
                    step=50,
                    label="Chunk Size (tokens)",
                )
                chunk_overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=50,
                    step=10,
                    label="Chunk Overlap",
                )

            ingest_btn = gr.Button("Process Document", variant="primary", size="lg")

            ingest_status = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False,
                lines=4,
                elem_classes=["status-box"],
            )

            doc_info_json = gr.JSON(label="Document Info", visible=False)

            gr.Markdown("---")
            gr.Markdown("### Retrieval Settings")

            n_hypothetical_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Hypothetical Documents to Generate",
                info="More docs = smoother averaged vector, more LLM calls",
            )
            n_results_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=3,
                step=1,
                label="Chunks to Retrieve",
            )

            gr.Markdown("---")
            gr.Markdown("### API Keys")

            google_key_input = gr.Textbox(
                label="Google API Key",
                type="password",
                value=os.getenv("GOOGLE_API_KEY", ""),
                placeholder="AIza...",
                info="Get one free at aistudio.google.com",
            )

            gr.Markdown("---")
            clear_btn = gr.Button("Clear Document & Reset", variant="secondary")

        # ── Right column — query interface ────────────────────────────────
        with gr.Column(scale=2):

            gr.Markdown("### Ask a Question")
            question_input = gr.Textbox(
                label="",
                placeholder="e.g. What are the main findings of this study?",
                lines=3,
            )

            query_btn = gr.Button("Ask", variant="primary", size="lg")

            gr.Markdown("---")

            with gr.Accordion("Hypothetical Documents Generated", open=True):
                gr.Markdown(
                    '<p class="panel-header">What the model imagined as an answer — '
                    "these embeddings guide retrieval</p>"
                )
                hypo_output = gr.Textbox(
                    label="",
                    lines=12,
                    interactive=False,
                    elem_classes=["hypo-box"],
                    placeholder="Hypothetical documents will appear here after you ask a question.",
                )

            with gr.Accordion("Retrieved Context Chunks", open=True):
                gr.Markdown(
                    '<p class="panel-header">Chunks retrieved from your document '
                    "using the averaged HyDE embedding</p>"
                )
                chunks_output = gr.Textbox(
                    label="",
                    lines=10,
                    interactive=False,
                    elem_classes=["chunks-box"],
                    placeholder="Retrieved chunks will appear here.",
                )

            gr.Markdown("### Final Answer")
            answer_output = gr.Textbox(
                label="",
                lines=7,
                interactive=False,
                elem_classes=["answer-box"],
                placeholder="The synthesised answer will appear here.",
            )

    # ── Event bindings ─────────────────────────────────────────────────────

    ingest_btn.click(
        fn=ingest_document,
        inputs=[file_input, chunk_size_slider, chunk_overlap_slider, google_key_input],
        outputs=[ingest_status, doc_info_json],
    )

    query_btn.click(
        fn=run_query,
        inputs=[question_input, n_hypothetical_slider, n_results_slider, google_key_input],
        outputs=[hypo_output, chunks_output, answer_output],
    )

    question_input.submit(
        fn=run_query,
        inputs=[question_input, n_hypothetical_slider, n_results_slider, google_key_input],
        outputs=[hypo_output, chunks_output, answer_output],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            file_input,
            question_input,
            hypo_output,
            chunks_output,
            answer_output,
            ingest_status,
            doc_info_json,
        ],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
