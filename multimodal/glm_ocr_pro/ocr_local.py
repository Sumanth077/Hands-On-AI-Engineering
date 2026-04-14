import json
from io import BytesIO
import urllib.request
from typing import Generator

import pypdfium2 as pdfium
from PIL import Image

from utils import image_bytes_to_base64


OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama_status() -> tuple[bool, str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, "Ollama is running."
        return False, "Ollama is not responding as expected."
    except Exception as exc:
        return False, f"Ollama not reachable: {exc}"


def _iter_pdf_image_bytes(pdf_bytes: bytes, max_pages: int, render_scale: float = 2.0) -> Generator[bytes, None, None]:
    doc = pdfium.PdfDocument(pdf_bytes)
    page_count = min(len(doc), max_pages)
    for i in range(page_count):
        page = doc[i]
        bitmap = page.render(scale=render_scale)
        pil_image = bitmap.to_pil()
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        yield buf.getvalue()


def _optimize_image_bytes(image_bytes: bytes, max_side: int = 1400, quality: int = 75, grayscale: bool = False) -> bytes:
    """Optimize image for local LLM consumption."""
    image = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB if needed, or Grayscale (L) for speed/memory savings
    if grayscale:
        image = image.convert("L")
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    
    w, h = image.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    
    buf = BytesIO()
    # JPEG format for smaller size, or PNG if quality is paramount
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def warmup_local_model(timeout_seconds: int = 120) -> str:
    payload = {"model": "glm-ocr", "prompt": "ready", "stream": False, "keep_alive": "10m"}
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            _ = resp.read()
        return "Warm-up completed. Model is loaded."
    except Exception as exc:
        return f"Warm-up failed: {exc}"


def _run_single_image_local_stream(image_bytes: bytes, timeout_seconds: int, num_predict: int = 2048) -> Generator[str, None, None]:
    """Calls Ollama chat endpoint with streaming enabled."""
    prompt = (
        "Extract all readable text from this image and return structured Markdown. "
        "Maintain layout accuracy: preserve headings, lists, tables, and formulas. "
        "Do not apologize or explain. Just return the extraction."
    )
    payload = {
        "model": "glm-ocr",
        "stream": True,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes_to_base64(image_bytes)],
            }
        ],
    }

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )

    try:
        # We manually iterate over response lines for streaming
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            for line in resp:
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def extract_markdown_local_stream(
    file_bytes: bytes,
    mime_type: str,
    max_pdf_pages: int = 5,
    timeout_seconds: int = 600,
    optimize_images: bool = True,
    max_side: int = 1400,
    grayscale: bool = False,
    pdf_render_scale: float = 2.0,
    num_predict: int = 2048,
) -> Generator[str, None, None]:
    """Main entry point for local extraction, yielding chunks of text."""
    if mime_type == "application/pdf":
        for idx, page_bytes in enumerate(
            _iter_pdf_image_bytes(file_bytes, max_pages=max_pdf_pages, render_scale=pdf_render_scale),
            start=1,
        ):
            yield f"## Page {idx}\n\n"
            if optimize_images:
                page_bytes = _optimize_image_bytes(page_bytes, max_side=max_side, grayscale=grayscale)
            yield from _run_single_image_local_stream(file_bytes, timeout_seconds=timeout_seconds, num_predict=num_predict)
            yield "\n\n---\n\n"
    else:
        if optimize_images:
            file_bytes = _optimize_image_bytes(file_bytes, max_side=max_side, grayscale=grayscale)
        yield from _run_single_image_local_stream(page_bytes, timeout_seconds=timeout_seconds, num_predict=num_predict)
