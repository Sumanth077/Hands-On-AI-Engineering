"""Small OpenAI-compatible client for the intake agent."""

from __future__ import annotations

import json
import os

from openai import OpenAI, OpenAIError


GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"


class IntakeModelError(RuntimeError):
    """A safe, user-readable model error."""


def is_configured() -> bool:
    """Return whether a model key is available without exposing it."""

    return bool(os.getenv("AI_GATEWAY_API_KEY"))


def _client() -> OpenAI:
    key = os.getenv("AI_GATEWAY_API_KEY")
    if not key:
        raise IntakeModelError("Add AI_GATEWAY_API_KEY to .env before starting an intake.")
    return OpenAI(base_url=GATEWAY_URL, api_key=key)


def model_name() -> str:
    """Return the exact model ID while keeping the default reproducible."""

    return os.getenv("LLM_MODEL", "zai/glm-5.3")


def structured_response(system_prompt: str, user_prompt: str) -> dict:
    """Request JSON only and validate that the gateway returned an object."""

    try:
        response = _client().chat.completions.create(
            model=model_name(),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as error:
        raise IntakeModelError(f"The intake model request failed: {error}") from error

    content = response.choices[0].message.content or "{}"
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise IntakeModelError("The intake model returned an invalid structured response. Please try again.") from error
    if not isinstance(value, dict):
        raise IntakeModelError("The intake model returned an unexpected response. Please try again.")
    return value
