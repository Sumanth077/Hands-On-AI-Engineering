"""Small OpenAI-compatible client for GLM 5.3 repository reasoning."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


class LLMError(RuntimeError):
    """The configured reasoning model could not complete a request."""


def is_configured() -> bool:
    return bool(_api_key())


def _api_key() -> str | None:
    """Return the default endpoint key, with a custom-endpoint fallback."""
    return os.getenv("AI_GATEWAY_API_KEY") or os.getenv("LLM_API_KEY")


def chat(messages: list[dict[str, str]], *, temperature: float = 0.1) -> str:
    api_key = _api_key()
    if not api_key:
        raise LLMError("Add AI_GATEWAY_API_KEY to .env before asking repository questions.")
    base_url = os.getenv("LLM_BASE_URL", "https://ai-gateway.vercel.sh/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "zai/glm-5.3")
    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=90.0,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise LLMError(f"The repository model request failed: {error}") from error


def json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("The model did not return the requested JSON object.")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise LLMError("The model returned invalid JSON.") from error
    if not isinstance(value, dict):
        raise LLMError("The model response was not a JSON object.")
    return value

