"""Build Gemini request contents for the Music Explorer chat."""

from __future__ import annotations


def _extract_text(content) -> str:
    """Gradio may store message content as a string or content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def build_contents(types, gemini_file, history: list[dict], new_prompt: str) -> list:
    """Build a stateless Gemini request with prior text and the current audio."""
    contents = []
    for msg in history:
        role = msg["role"]
        text = _extract_text(msg["content"])
        contents.append(
            types.Content(
                role="user" if role == "user" else "model",
                parts=[types.Part.from_text(text=text)],
            )
        )

    audio_part = types.Part.from_uri(
        file_uri=gemini_file.uri,
        mime_type=gemini_file.mime_type,
    )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_prompt), audio_part],
        )
    )
    return contents


def reset_audio_context() -> tuple[list, str]:
    """Start a fresh conversation whenever the selected audio source changes."""
    return [], ""
