"""Gradio app for extracting structured clinical profiles from medical PDFs and images."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from document_processor import process_upload
from llm_extractor import _build_client, extract_from_page
from merger import merge_profiles

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

CUSTOM_CSS = """
.flagged-box textarea {
    color: #dc3545 !important;
    font-weight: 600 !important;
    background: #fff5f5 !important;
    border-color: #f5c2c7 !important;
}
"""


def _format_flagged_text(flagged_items: list[str]) -> str:
    """Format the list of flagged abnormal values into a human-readable string."""
    if not flagged_items:
        return "No abnormal or critical values flagged."
    return "\n".join(f"• {item}" for item in flagged_items)


def parse_document(
    upload: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[dict, str]:
    """Process the uploaded medical document and return the structured profile and flagged values."""
    if upload is None:
        raise gr.Error("Please upload a medical PDF or image.")

    file_path = Path(upload)
    if not file_path.exists():
        raise gr.Error("Uploaded file could not be found.")

    progress(0, desc="Preparing document...")
    pages = process_upload(file_path)
    if not pages:
        raise gr.Error("No pages were found in the uploaded document.")

    client = _build_client()
    profiles = []
    total = len(pages)

    for index, page in enumerate(pages, start=1):
        progress(
            index / total,
            desc=f"Analyzing page {index}/{total} ({page.kind}) with Gemma 4...",
        )
        profiles.append(extract_from_page(client, page))

    progress(1.0, desc="Merging results...")
    merged = merge_profiles(profiles)
    payload = merged.model_dump()
    return payload, _format_flagged_text(payload["flagged_items"])


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface for the medical document parser."""
    with gr.Blocks(title="Medical Document Parser") as demo:
        gr.Markdown(
            """
            # Medical Document Parser
            Upload a medical **PDF** or **image** (lab report, prescription, imaging result, or clinical notes).
            The app routes each page through text or vision extraction, then uses **Gemma 4** to build a unified clinical profile.
            """
        )

        with gr.Row():
            upload = gr.File(
                label="Medical document",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"],
                type="filepath",
            )

        parse_button = gr.Button("Extract Clinical Profile", variant="primary")

        with gr.Row():
            json_output = gr.JSON(label="Structured Clinical Profile")
            flagged_output = gr.Textbox(
                label="Flagged Abnormal Values",
                lines=10,
                elem_classes=["flagged-box"],
            )

        parse_button.click(
            fn=parse_document,
            inputs=[upload],
            outputs=[json_output, flagged_output],
        )

        gr.Markdown(
            """
            **How it works**
            1. PyMuPDF classifies each PDF page as text (>50 characters) or vision.
            2. Text pages are sent directly to Gemma 4; vision pages are rendered at 150 DPI.
            3. Per-page extractions are merged into one JSON profile with abnormal values highlighted.
            """
        )

    return demo


if __name__ == "__main__":
    build_app().launch(css=CUSTOM_CSS, theme=gr.themes.Soft())
