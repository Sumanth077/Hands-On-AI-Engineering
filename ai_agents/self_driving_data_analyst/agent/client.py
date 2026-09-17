"""
Minimal client for the Liner Model API.

Liner's Model API is OpenAI Chat Completions-compatible: same request/response
shape, same tool-calling flow. This wrapper only implements the pieces this
project actually uses (chat completions, function calling, reasoning_effort,
usage tracking) rather than pulling in the full OpenAI SDK for one endpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import requests

LINER_BASE_URL = "https://platform.liner.com/api/v1"
LINER_MODEL = "liner-mark-1.0"


class LinerAPIError(RuntimeError):
    pass


@dataclass
class LinerResponse:
    raw: dict[str, Any]

    @property
    def message(self) -> dict[str, Any]:
        return self.raw["choices"][0]["message"]

    @property
    def content(self) -> Optional[str]:
        return self.message.get("content")

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.message.get("tool_calls") or []

    @property
    def usage(self) -> dict[str, Any]:
        return self.raw.get("usage", {})

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def cached_tokens(self) -> int:
        details = self.usage.get("prompt_tokens_details") or {}
        return details.get("cached_tokens", 0)


class LinerClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 120):
        self.api_key = api_key or os.environ.get("LINER_API_KEY")
        if not self.api_key:
            raise LinerAPIError(
                "LINER_API_KEY is not set. Add it to your .env file or pass it explicitly."
            )
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
        reasoning_effort: str = "medium",
        response_format: Optional[dict[str, Any]] = None,
        parallel_tool_calls: bool = True,
    ) -> LinerResponse:
        payload: dict[str, Any] = {
            "model": LINER_MODEL,
            "messages": messages,
            "reasoning_effort": reasoning_effort,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            payload["parallel_tool_calls"] = parallel_tool_calls
        if response_format:
            payload["response_format"] = response_format

        try:
            resp = requests.post(
                f"{LINER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LinerAPIError(f"Request to Liner Model API failed: {exc}") from exc

        if resp.status_code >= 400:
            raise LinerAPIError(f"Liner Model API returned {resp.status_code}: {resp.text}")

        return LinerResponse(raw=resp.json())
