from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz
from PIL import Image

TEXT_THRESHOLD = 50
RENDER_DPI = 150

PageKind = Literal["text", "vision"]


@dataclass
class ProcessedPage:
    """Holds the content of a single document page classified as text or vision."""

    page_number: int
    kind: PageKind
    text: str | None = None
    image_bytes: bytes | None = None


def _text_page(page: fitz.Page) -> ProcessedPage:
    """Extract plain text from a PDF page and return it as a text ProcessedPage."""
    return ProcessedPage(
        page_number=page.number + 1,
        kind="text",
        text=page.get_text("text").strip(),
    )


def _vision_page(page: fitz.Page) -> ProcessedPage:
    """Render a PDF page to a PNG image at RENDER_DPI and return it as a vision ProcessedPage."""
    pixmap = page.get_pixmap(dpi=RENDER_DPI)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return ProcessedPage(
        page_number=page.number + 1,
        kind="vision",
        image_bytes=buffer.getvalue(),
    )


def _classify_pdf_page(page: fitz.Page) -> ProcessedPage:
    """Route a PDF page to text or vision extraction based on the extracted character count."""
    text = page.get_text("text").strip()
    if len(text) > TEXT_THRESHOLD:
        return ProcessedPage(
            page_number=page.number + 1,
            kind="text",
            text=text,
        )
    return _vision_page(page)


def _image_to_page(image_bytes: bytes, page_number: int = 1) -> ProcessedPage:
    """Convert raw image bytes to a vision ProcessedPage, normalizing to RGB PNG."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return ProcessedPage(
        page_number=page_number,
        kind="vision",
        image_bytes=buffer.getvalue(),
    )


def process_upload(file_path: str | Path) -> list[ProcessedPage]:
    """Dispatch a PDF or image file to the appropriate page-processing path and return all pages."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages: list[ProcessedPage] = []
        with fitz.open(path) as document:
            for page in document:
                pages.append(_classify_pdf_page(page))
        return pages

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return [_image_to_page(path.read_bytes())]

    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
