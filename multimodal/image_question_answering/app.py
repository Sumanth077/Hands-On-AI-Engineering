"""Image Question Answering: upload a PDF, select a page, and ask visual questions answered by Gemma 4 via the Gemini API."""

import io
import os

import fitz  # PyMuPDF
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# ── environment ───────────────────────────────────────────────────────────────

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_ID = "gemma-4-31b-it"

client = genai.Client(api_key=GOOGLE_API_KEY or None)

# ── PDF → images ──────────────────────────────────────────────────────────────

_page_images: list[Image.Image] = []


def pdf_to_images(path: str, dpi: int = 150) -> list[Image.Image]:
    """Render every page of the PDF at the given DPI and return them as a list of PIL images."""
    doc = fitz.open(path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return images


# ── Gradio callbacks ──────────────────────────────────────────────────────────


def on_pdf_upload(pdf_file):
    """Handle a PDF upload by rendering all pages and initialising the page slider."""
    global _page_images

    if pdf_file is None:
        _page_images = []
        return None, gr.update(visible=False, value=1), ""

    _page_images = pdf_to_images(pdf_file.name)
    n = len(_page_images)

    return (
        _page_images[0],
        gr.update(
            minimum=1,
            maximum=n,
            value=1,
            step=1,
            label=f"Page (1–{n})",
            visible=True,
        ),
        f"Loaded {n} page{'s' if n != 1 else ''}.",
    )


def on_page_change(page_num: int):
    """Return the PIL image for the selected 1-based page number."""
    idx = int(page_num) - 1
    if _page_images and 0 <= idx < len(_page_images):
        return _page_images[idx]
    return None


def on_submit(page_num: int, question: str):
    """Send the selected page image and question to Gemma 4 and return the answer text."""
    if not _page_images:
        return "Please upload a PDF first."
    if not question.strip():
        return "Please type a question."

    idx = int(page_num) - 1
    if not (0 <= idx < len(_page_images)):
        return f"Page {int(page_num)} is out of range (PDF has {len(_page_images)} pages)."

    buf = io.BytesIO()
    _page_images[idx].save(buf, format="PNG")
    image_bytes = buf.getvalue()

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                question,
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high")
            ),
        )
        return response.text or "(no answer returned)"
    except Exception as exc:
        return f"Error: {exc}"


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Image Question Answering") as demo:
    gr.Markdown(
        "# Image Question Answering\n"
        "Upload a PDF, select a page, and ask any visual question — "
        "answered by **Gemma 4** with thinking mode via the Gemini API."
    )

    with gr.Row(equal_height=False):
        # ── left: PDF viewer ──────────────────────────────────────────────
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                file_count="single",
            )
            status_box = gr.Textbox(
                show_label=False,
                interactive=False,
                max_lines=1,
                container=False,
            )
            page_slider = gr.Slider(
                minimum=1,
                maximum=2,
                value=1,
                step=1,
                label="Page",
                visible=False,
            )
            page_image = gr.Image(
                label="Page Preview",
                type="pil",
                interactive=False,
                height=640,
            )

        # ── right: Q&A ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            question_box = gr.Textbox(
                label="Visual Question",
                placeholder=(
                    "What does the chart on this page show?\n"
                    "What is the title of the document?\n"
                    "List the bullet points in section 2."
                ),
                lines=4,
            )
            submit_btn = gr.Button(
                "Answer Question",
                variant="primary",
                size="lg",
            )
            answer_box = gr.Textbox(
                label="Answer",
                lines=14,
                interactive=False,
            )

    # ── event wiring ──────────────────────────────────────────────────────
    pdf_input.change(
        fn=on_pdf_upload,
        inputs=[pdf_input],
        outputs=[page_image, page_slider, status_box],
    )

    page_slider.change(
        fn=on_page_change,
        inputs=[page_slider],
        outputs=[page_image],
    )

    submit_btn.click(
        fn=on_submit,
        inputs=[page_slider, question_box],
        outputs=[answer_box],
        show_progress="minimal",
    )

    question_box.submit(
        fn=on_submit,
        inputs=[page_slider, question_box],
        outputs=[answer_box],
        show_progress="minimal",
    )


if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print(
            "[WARNING] GOOGLE_API_KEY is not set.\n"
            "Create a .env file with GOOGLE_API_KEY=<your_key> before running queries."
        )
    demo.launch(theme=gr.themes.Soft())
