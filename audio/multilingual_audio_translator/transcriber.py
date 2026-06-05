"""Speech-to-text transcription and language detection using faster-whisper."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")

_model: WhisperModel | None = None


def _get_device_config() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        device, compute_type = _get_device_config()
        _model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
    return _model


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float


def transcribe(audio_path: str) -> TranscriptionResult:
    """Transcribe audio and detect the spoken language."""
    if not audio_path:
        raise ValueError("No audio file provided.")

    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise ValueError("No speech detected in the audio.")

    return TranscriptionResult(
        text=text,
        language=info.language,
        language_probability=info.language_probability,
    )
