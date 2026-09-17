"""Thin wrapper around AssemblyAI's Sync API.

One HTTP call in, finished transcript back in the same response — no
polling. Runs on the flagship Universal-3.5 Pro model.

Docs: https://www.assemblyai.com/docs/sync-stt/getting-started/transcribe-a-short-audio-file
"""
import os

import requests

SYNC_URL = "https://sync.assemblyai.com/transcribe"
MODEL = "universal-3-5-pro"


def transcribe_audio(file_path: str) -> dict:
    """Sends a WAV file to the Sync API and returns the parsed JSON result
    (includes 'text', 'words', and 'session_id')."""
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set. Check your .env file.")

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    response = requests.post(
        SYNC_URL,
        headers={
            "Authorization": api_key,
            "X-AAI-Model": MODEL,
        },
        files={"audio": (os.path.basename(file_path), audio_bytes, "audio/wav")},
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Sync API request failed ({response.status_code}): {response.text}"
        )

    result = response.json()
    if "session_id" in result:
        # Worth keeping in logs — it's the first thing AssemblyAI support asks for.
        print(f"[AssemblyAI session_id: {result['session_id']}]")
    return result
