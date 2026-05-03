"""
Research Team — Gradio UI
Seek + Scout powered by MiniMax M2.5 via OpenRouter.

Usage:
    uv run python app.py
"""

from dotenv import load_dotenv

load_dotenv()

import shutil
from pathlib import Path

import gradio as gr
from team import research_team

# ── Paths ──────────────────────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
KNOWLEDGE_DIR.mkdir(exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────────

CSS = """
#title     { text-align: center; }
#subtitle  { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }
#report    { min-height: 560px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1.2rem; }
"""

EXAMPLES = [
    ["Research Anthropic — their products, key people, and recent developments"],
    ["Research OpenAI and summarize their products, key people, and recent enterprise moves"],
    ["Research Mistral AI — what do we know about their models, funding, and team?"],
    ["Research Perplexity AI — business model, funding history, and competitive position"],
    ["Research xAI — Grok models, team background, and market positioning"],
]

PLACEHOLDER = "_Your research report will appear here once you submit a query._"


# ── File Upload ────────────────────────────────────────────────────────────────

def upload_docs(files) -> str:
    """Save uploaded files into the knowledge directory for Scout to read."""
    if not files:
        return "No files uploaded."
    saved = []
    for f in files:
        src = Path(f.name)
        dest = KNOWLEDGE_DIR / src.name
        shutil.copy(src, dest)
        saved.append(src.name)
    return f"Uploaded: {', '.join(saved)}"


def list_docs() -> str:
    """Show all files currently in the knowledge directory."""
    files = [f.name for f in KNOWLEDGE_DIR.iterdir() if not f.name.startswith(".")]
    return "Knowledge docs: " + (", ".join(files) if files else "none")


# ── Research Function ──────────────────────────────────────────────────────────

def run_research(query: str):
    query = query.strip()
    if not query:
        yield PLACEHOLDER, ""
        return

    yield "_Researching — this may take a minute..._", "Running..."

    try:
        response = research_team.run(query, stream=False)
        report = response.content if response.content else PLACEHOLDER
    except Exception as exc:
        yield PLACEHOLDER + f"\n\n---\n\n**Error:** {exc}", f"Error: {exc}"
        return

    yield report, "Done."


def clear_all():
    return "", PLACEHOLDER, ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Research Team") as demo:

    gr.Markdown("# 🔍 Research Team", elem_id="title")
    gr.Markdown(
        "Deep research powered by **MiniMax M2.5** via OpenRouter. "
        "Seek searches the web while Scout checks internal documents — "
        "the team leader synthesises both into a structured report.",
        elem_id="subtitle",
    )

    # ── Internal Docs Upload ───────────────────────────────────────────────────
    with gr.Accordion("📂 Internal Knowledge (optional)", open=False):
        gr.Markdown(
            "Upload any `.txt`, `.md`, or `.pdf` files you want Scout to read. "
            "These become part of the internal knowledge base for every query."
        )
        with gr.Row():
            file_upload = gr.File(
                label="Upload documents",
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf"],
            )
            upload_status = gr.Textbox(
                label="Upload status",
                interactive=False,
                lines=2,
            )
        with gr.Row():
            upload_btn = gr.Button("📤 Save to Knowledge Base", size="sm")
            list_btn = gr.Button("📋 List Uploaded Docs", size="sm")

        upload_btn.click(fn=upload_docs, inputs=file_upload, outputs=upload_status)
        list_btn.click(fn=list_docs, outputs=upload_status)

    # ── Query ──────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                placeholder='e.g. "Research Anthropic — their products, key people, and recent developments"',
                label="Research Query",
                lines=2,
            )
        with gr.Column(scale=1, min_width=140):
            submit_btn = gr.Button("🔍 Research", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", size="lg")

    gr.Examples(
        examples=EXAMPLES,
        inputs=query_input,
        label="Example Queries",
    )

    with gr.Row():
        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1, scale=1)

    report_output = gr.Markdown(value=PLACEHOLDER, elem_id="report")

    submit_btn.click(fn=run_research, inputs=query_input, outputs=[report_output, status_box])
    query_input.submit(fn=run_research, inputs=query_input, outputs=[report_output, status_box])
    clear_btn.click(fn=clear_all, outputs=[query_input, report_output, status_box])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="indigo"))
