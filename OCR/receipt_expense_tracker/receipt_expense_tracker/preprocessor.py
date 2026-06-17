"""
Image preprocessor for receipt and invoice photos.
Handles crumpled, low-light, and thermal-printed receipts
using Pillow enhancements before passing to the vision model.
"""

import io
from PIL import Image, ImageEnhance, ImageOps

MAX_DIMENSION = 1280


def preprocess(image: Image.Image) -> Image.Image:
    """
    Prepare a receipt image for vision model extraction.

    Steps:
    1. Auto-rotate from EXIF metadata (fixes phone photos taken sideways)
    2. Convert to RGB (handles RGBA, greyscale, palette modes)
    3. Resize to fit within MAX_DIMENSION while preserving aspect ratio
    4. Boost contrast to recover faded thermal print
    5. Sharpen to improve text readability on crumpled receipts
    """
    # Auto-orient from EXIF
    image = ImageOps.exif_transpose(image)

    # Normalise to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize: keep aspect ratio, cap longest side at MAX_DIMENSION
    image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    # Contrast boost (1.0 = original, 1.5 = moderately enhanced)
    image = ImageEnhance.Contrast(image).enhance(1.5)

    # Sharpness boost (1.0 = original, 2.0 = sharper)
    image = ImageEnhance.Sharpness(image).enhance(2.0)

    return image


def to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert a PIL image to raw bytes for embedding in a prompt."""
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=90)
    return buf.getvalue()
