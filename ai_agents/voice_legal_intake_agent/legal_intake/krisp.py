"""Krisp session-token and recorded-audio transcription helpers."""

from __future__ import annotations

import asyncio
import io
import json
import os
import wave

import httpx
import websockets


TOKEN_URL = "https://api.developers.krisp.ai/v2/sdk/tokens/session"
STT_URL = "wss://streaming.krisp.ai/stt"
CHUNK_MS = 80


class KrispError(RuntimeError):
    """A safe, user-readable Krisp error."""


def is_configured() -> bool:
    """Return whether the server-only Krisp key is present."""

    return bool(os.getenv("KRISP_API_KEY"))


def _session_token() -> str:
    key = os.getenv("KRISP_API_KEY")
    if not key:
        raise KrispError("Add KRISP_API_KEY to .env before recording a voice intake.")
    try:
        response = httpx.post(
            TOKEN_URL,
            headers={"Authorization": f"api-key {key}", "Content-Type": "application/json"},
            json={"usage_type": "stt", "expiration_ttl": 5},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise KrispError(f"Krisp session token request failed: {error}") from error
    token = response.json().get("data", {}).get("session_key")
    if not token:
        raise KrispError("Krisp did not return a session token. Check your API key and STT access.")
    return str(token)


def _wav_frames(audio: bytes) -> bytes:
    """Validate the browser recording against Krisp's required PCM format."""

    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getframerate() != 16_000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise KrispError("Record at 16 kHz, mono, 16-bit PCM WAV. The app sets this automatically for new recordings.")
            return wav.readframes(wav.getnframes())
    except wave.Error as error:
        raise KrispError("The recording was not a valid WAV file. Please record again in the browser.") from error


async def _transcribe_async(audio: bytes, vocabulary: list[str]) -> str:
    token = _session_token()
    frames = _wav_frames(audio)
    chunk_size = 16_000 * CHUNK_MS // 1000 * 2
    finals: list[str] = []
    handshake = {
        "model": "krisp-realtime-stt-agents-v1",
        "vocabulary": vocabulary,
        "sample_rate": 16_000,
        "channels": 1,
        "audio_format": "pcm_s16le",
        "turn": {"prediction": True, "sensitivity": 0.9, "max_silence": 2000, "silence_validation": 100},
        "language_hints": ["en"],
    }
    try:
        async with websockets.connect(STT_URL, additional_headers={"Authorization": f"api-key {token}"}) as socket:
            await socket.send(json.dumps(handshake))
            ready = json.loads(await socket.recv())
            if ready.get("type") != "ready":
                raise KrispError(f"Krisp could not start the STT session: {ready}")
            for start in range(0, len(frames), chunk_size):
                await socket.send(frames[start : start + chunk_size])
                # Krisp expects audio frames at their natural pace. Sending a saved
                # browser recording as fast as possible can harm recognition quality.
                await asyncio.sleep(CHUNK_MS / 1000)
            await socket.send(json.dumps({"type": "end_of_audio"}))
            async for raw_message in socket:
                message = json.loads(raw_message)
                if message.get("type") == "transcript" and message.get("transcript", {}).get("isFinal"):
                    finals.append("".join(word.get("text", "") for word in message["transcript"].get("words", [])))
                if message.get("type") == "error":
                    raise KrispError(message.get("error", {}).get("message", "Krisp returned an STT error."))
                if message.get("type") == "end":
                    break
    except websockets.WebSocketException as error:
        raise KrispError(f"Krisp STT connection failed: {error}") from error
    return "".join(finals).strip()


def transcribe_recording(audio: bytes, vocabulary: list[str] | None = None) -> str:
    """Transcribe a browser WAV recording through Krisp's STT-for-agents model."""

    text = asyncio.run(_transcribe_async(audio, vocabulary or []))
    if not text:
        raise KrispError("Krisp returned an empty transcript. Please record again and speak a little closer to the microphone.")
    return text
