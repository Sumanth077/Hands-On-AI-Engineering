"""Helpers for embedding generated strings in inline JavaScript."""

import json


def serialize_inline_script_data(value: str) -> str:
    """Serialize a string without leaving HTML-significant characters raw."""
    return (
        json.dumps(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
