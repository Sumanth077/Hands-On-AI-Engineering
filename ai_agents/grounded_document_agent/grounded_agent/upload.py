"""Helpers for handling uploaded PDFs without retaining the raw upload."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar

T = TypeVar("T")


def parse_uploaded_pdf(
    file_bytes: bytes,
    file_name: str,
    parse_pdf: Callable[..., T],
) -> T:
    """Parse upload bytes from a temporary file and always remove that file."""
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(file_bytes)

        assert tmp_path is not None
        return parse_pdf(tmp_path, file_name=file_name)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
