"""
NVIDIA NIM client with 202 async polling support.

LangChain's OpenAI wrapper does not handle NVIDIA's 202 + NVCF-REQID polling,
which causes requests to hang indefinitely on cold starts.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_ID = "deepseek-ai/deepseek-v4-flash"
HTTPX_TIMEOUT = httpx.Timeout(240.0, connect=30.0)
POLL_INTERVAL_SECONDS = 2
MAX_POLL_SECONDS = 280
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3.0

# Transient errors when NVIDIA closes connections during warm-up (common on Windows).
TRANSIENT_ERRORS = (
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.PoolTimeout,
    ConnectionError,
    TimeoutError,
)


class NvidiaAPIError(Exception):
    """Raised when the NVIDIA API returns an error response."""


def get_api_key() -> str:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key or api_key == "your_nvidia_api_key_here":
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add a valid key to your .env file "
            "(see .env.example). Get one at https://build.nvidia.com/"
        )
    return api_key


def create_http_client() -> httpx.Client:
    """Fresh client per request — avoids reusing a connection NVIDIA already closed."""
    return httpx.Client(
        timeout=HTTPX_TIMEOUT,
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
    )


def messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, str]]:
    openai_messages: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            openai_messages.append({"role": "system", "content": str(message.content)})
        elif isinstance(message, HumanMessage):
            openai_messages.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage):
            openai_messages.append({"role": "assistant", "content": str(message.content)})
        else:
            openai_messages.append({"role": "user", "content": str(message.content)})
    return openai_messages


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content:
            return str(content)
    raise NvidiaAPIError(f"Unexpected NVIDIA response shape: {data!r}")


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise NvidiaAPIError(
            "NVIDIA API unauthorized (401). Check that NVIDIA_API_KEY is valid."
        )
    if response.status_code == 403:
        raise NvidiaAPIError(
            "NVIDIA API forbidden (403). Your key may lack access to this model."
        )
    if response.status_code == 404:
        raise NvidiaAPIError(
            f"NVIDIA API not found (404). Model '{MODEL_ID}' may be unavailable."
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500]
        raise NvidiaAPIError(
            f"NVIDIA API error {response.status_code}: {detail}"
        ) from exc


def _request_with_retries(request_fn, label: str) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        client = create_http_client()
        try:
            response = request_fn(client)
            response.read()
            return response
        except TRANSIENT_ERRORS as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"[NVIDIA] {label} failed ({exc!r}) — "
                    f"retry {attempt}/{MAX_RETRIES - 1} in {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise NvidiaAPIError(
                    f"NVIDIA API connection failed after {MAX_RETRIES} attempts. "
                    f"Last error: {exc}. This often happens when the free-tier "
                    f"model is warming up — wait a minute and try again."
                ) from exc
        finally:
            client.close()
    raise NvidiaAPIError(f"NVIDIA API failed: {last_error}")


def _poll_for_completion(api_key: str, request_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + MAX_POLL_SECONDS
    poll_url = f"{NVIDIA_BASE_URL}/status/{request_id}"

    print(
        f"[NVIDIA] Model warming up (request {request_id[:8]}...) — polling",
        flush=True,
    )

    while time.monotonic() < deadline:
        def poll_once(client: httpx.Client) -> httpx.Response:
            return client.get(poll_url, headers=_headers(api_key))

        response = _request_with_retries(poll_once, "status poll")

        if response.status_code == 200:
            return response.json()

        if response.status_code == 202:
            status = response.headers.get("NVCF-STATUS", "pending")
            print(f"[NVIDIA] Still pending ({status})...", flush=True)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        _raise_for_status(response)

    raise TimeoutError(
        f"NVIDIA API did not become ready within {MAX_POLL_SECONDS}s "
        f"(request {request_id})"
    )


def _post_chat_completion(
    api_key: str, body: dict[str, Any]
) -> httpx.Response:
    def do_post(client: httpx.Client) -> httpx.Response:
        return client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=_headers(api_key),
            json=body,
        )

    return _request_with_retries(do_post, "chat completion")


def chat_completion(
    messages: list[dict[str, str]] | list[BaseMessage],
    *,
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> str:
    api_key = get_api_key()
    if messages and isinstance(messages[0], BaseMessage):
        payload_messages = messages_to_openai(messages)  # type: ignore[arg-type]
    else:
        payload_messages = messages  # type: ignore[assignment]

    body = {
        "model": MODEL_ID,
        "messages": payload_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    print(f"[NVIDIA] Calling {MODEL_ID}...", flush=True)

    response = _post_chat_completion(api_key, body)

    if response.status_code == 200:
        content = _extract_content(response.json())
        print("[NVIDIA] Response received.", flush=True)
        return content

    if response.status_code == 202:
        request_id = response.headers.get("NVCF-REQID") or response.headers.get(
            "nvcf-reqid"
        )
        if not request_id:
            raise NvidiaAPIError(
                "NVIDIA returned 202 but no NVCF-REQID header for polling."
            )
        result = _poll_for_completion(api_key, request_id)
        content = _extract_content(result)
        print("[NVIDIA] Response received after polling.", flush=True)
        return content

    _raise_for_status(response)
    raise NvidiaAPIError("Unreachable")


def verify_connection() -> None:
    """Quick API check; raises on failure."""
    chat_completion(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
    )
