"""Fish Audio ASR and Drama 3 timestamped narration client."""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Iterable

import httpx

from codebase_guide.models import TimestampSegment, TranscriptResult, VoiceResult


API_ROOT = "https://api.fish.audio/v1"
DEFAULT_REFERENCE_ID = "c70cfb98867a4bfb9ae20d2401257b92"


class FishAudioError(RuntimeError):
    """A user-readable Fish Audio API error."""


def _headers(model: str) -> dict[str, str]:
    key = os.getenv("FISH_API_KEY")
    if not key:
        raise FishAudioError("Add FISH_API_KEY to .env to use speech input or narration.")
    return {"Authorization": f"Bearer {key}", "model": model}


def _api_error(operation: str, error: httpx.HTTPError) -> FishAudioError:
    """Keep Fish's response message visible without exposing the API key."""
    if not isinstance(error, httpx.HTTPStatusError) or error.response is None:
        return FishAudioError(f"Fish Audio {operation} failed: {error}")

    response = error.response
    detail = response.text.strip()[:800]
    suffix = f" Fish Audio said: {detail}" if detail else ""
    return FishAudioError(
        f"Fish Audio {operation} failed with HTTP {response.status_code}.{suffix}"
    )


def _segments(items: Iterable[dict]) -> list[TimestampSegment]:
    return [
        TimestampSegment(str(item.get("text", "")), float(item.get("start", 0)), float(item.get("end", 0)))
        for item in items
    ]


def transcribe(audio: bytes, filename: str) -> TranscriptResult:
    try:
        response = httpx.post(
            f"{API_ROOT}/asr",
            headers=_headers("transcribe-1-pro"),
            files={"audio": (filename, audio, "audio/wav")},
            data={"ignore_timestamps": "false", "language": "en"},
            timeout=90.0,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        raise _api_error("transcription", error) from error
    return TranscriptResult(str(payload.get("text", "")).strip(), _segments(payload.get("segments", [])))


def narration_script(answer: str) -> str:
    """Turn the written Markdown answer into natural narration for Drama 3."""
    spoken = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
    spoken = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", spoken)
    spoken = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", spoken)
    spoken = re.sub(r"\[[^\]]+?:\d+\]", "", spoken)
    spoken = re.sub(r"\[\d+\]", "", spoken)

    cleaned_lines: list[str] = []
    for line in spoken.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*(?:[-*+] |\d+[.)] )", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        cleaned_lines.append(line.strip())

    spoken = " ".join(line for line in cleaned_lines if line)
    spoken = re.sub(r"`([^`]+)`", r"\1", spoken)
    spoken = re.sub(r"(?<!\w)[*_]{1,3}|[*_]{1,3}(?!\w)", "", spoken)
    spoken = spoken.replace("#", "")
    spoken = re.sub(r"\s+", " ", spoken).strip()
    spoken = re.sub(r"\s+([,.!?;:])", r"\1", spoken)
    return f"[clear, precise, and helpful to a developer] {spoken}"


def generate_voice(text: str) -> VoiceResult:
    payload = {
        "text": text,
        "reference_id": os.getenv("FISH_REFERENCE_ID") or DEFAULT_REFERENCE_ID,
        "format": "mp3",
        "mp3_bitrate": 128,
        "normalize": True,
        "temperature": 0.7,
        "top_p": 0.7,
        "chunk_length": 200,
    }
    chunks: list[bytes] = []
    segments: list[TimestampSegment] = []
    try:
        with httpx.stream(
            "POST",
            f"{API_ROOT}/tts/stream/with-timestamp",
            headers={**_headers("drama-3"), "Content-Type": "application/json"},
            json=payload,
            timeout=90.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                event = json.loads(raw)
                if encoded := event.get("audio_base64"):
                    chunks.append(base64.b64decode(encoded))
                segments.extend(_segments((event.get("alignment") or {}).get("segments", [])))
    except httpx.HTTPError as error:
        raise _api_error("narration", error) from error
    except (ValueError, json.JSONDecodeError) as error:
        raise FishAudioError(f"Fish Audio narration failed: {error}") from error
    if not chunks:
        raise FishAudioError("Fish Audio returned no playable narration.")
    return VoiceResult(audio=b"".join(chunks), segments=segments)

