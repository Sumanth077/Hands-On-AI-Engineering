"""Small server-side client for the Kombucha-1 streaming endpoint."""

from __future__ import annotations

import json
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import websocket

ENDPOINT = (
    "wss://api.meetjamie.ai/v1/platform/diarize"
    "?encoding=pcm_s16le&sample_rate=16000&channels=1"
)
FRAME_MS = 100
FRAME_BYTES = 3_200


class KombuchaError(RuntimeError):
    """Raised when the sample cannot be streamed safely."""


def _validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as audio:
        if audio.getcomptype() != "NONE":
            raise KombuchaError("The demo audio must be an uncompressed PCM WAV file.")
        if audio.getsampwidth() != 2:
            raise KombuchaError("The demo audio must use 16-bit samples.")
        if audio.getframerate() != 16_000:
            raise KombuchaError("The demo audio must be sampled at 16 kHz.")
        if audio.getnchannels() != 1:
            raise KombuchaError("The demo audio must be mono.")


def _receive_message(socket: websocket.WebSocket) -> dict[str, Any] | None:
    try:
        raw = socket.recv()
    except websocket.WebSocketTimeoutException:
        return None

    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def stream_wav(
    wav_path: Path,
    token: str,
    on_event: Callable[[dict[str, Any]], None],
) -> None:
    """Send one PCM WAV in real time and forward Kombucha-1 events to ``on_event``."""

    _validate_wav(wav_path)
    if not token.startswith("jp_"):
        raise KombuchaError("KOMBUCHA_API_KEY must be a Platform token starting with jp_.")

    socket = websocket.create_connection(
        ENDPOINT,
        header={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    try:
        first_message = _receive_message(socket)
        if first_message is None or first_message.get("type") != "session":
            raise KombuchaError("Kombucha-1 did not return a session message.")
        on_event(first_message)

        socket.settimeout(0.01)
        next_frame_at = time.monotonic()
        with wave.open(str(wav_path), "rb") as audio:
            while frame := audio.readframes(1_600):
                delay = next_frame_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

                socket.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
                next_frame_at += FRAME_MS / 1_000

                while message := _receive_message(socket):
                    on_event(message)

        socket.settimeout(15)
        socket.send(json.dumps({"type": "finalize"}))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            message = _receive_message(socket)
            if message is None:
                continue
            on_event(message)
            if message.get("type") == "finished":
                return

        raise KombuchaError("Kombucha-1 did not finish the session within 60 seconds.")
    finally:
        socket.close()
