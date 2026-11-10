"""HTML rendering helpers for externally sourced Rock Music RAG content."""

from html import escape
from typing import Mapping


def escape_html(value: object) -> str:
    """Escape a dynamic value before inserting it into HTML text or attributes."""
    return escape(str(value), quote=True)


def render_chunk_card_html(chunk: Mapping[str, object]) -> str:
    """Build a chunk card while escaping all Wikipedia sourced fields."""
    content = escape_html(chunk["content"])
    band = escape_html(chunk["band"])
    url = escape_html(chunk["url"])
    return (
        f'<div class="chunk-card">{content}'
        f'<div class="chunk-source">🎵 {band} · '
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'style="color:#64748b;">{url}</a></div></div>'
    )
