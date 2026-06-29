"""Text-to-speech synthesis using Kokoro TTS."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24_000

# Kokoro lang_code and default voice per target language.
LANG_CONFIG: dict[str, tuple[str, str]] = {
    "en": ("a", "af_heart"),
    "es": ("e", "af_heart"),
    "fr": ("f", "af_heart"),
    "de": ("a", "af_heart"),
    "it": ("i", "af_heart"),
    "pt": ("p", "af_heart"),
    "hi": ("h", "af_heart"),
    "ja": ("j", "af_heart"),
    "zh": ("z", "af_heart"),
    "ko": ("a", "af_heart"),
    "ar": ("a", "af_heart"),
    "ru": ("a", "af_heart"),
    "nl": ("a", "af_heart"),
    "pl": ("a", "af_heart"),
    "tr": ("a", "af_heart"),
    "vi": ("a", "af_heart"),
    "th": ("a", "af_heart"),
    "id": ("a", "af_heart"),
    "sv": ("a", "af_heart"),
    "uk": ("a", "af_heart"),
}

_pipelines: dict[str, KPipeline] = {}


def _get_pipeline(lang_code: str) -> KPipeline:
    """Return the cached KPipeline for the given Kokoro language code, creating it if needed."""
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


def synthesize(text: str, target_language: str) -> str:
    """Synthesize speech from text and return the path to a WAV file."""
    if not text.strip():
        raise ValueError("Nothing to synthesize.")

    lang_code, voice = LANG_CONFIG.get(target_language, ("a", "af_heart"))
    pipeline = _get_pipeline(lang_code)

    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    if not chunks:
        raise ValueError("Speech synthesis produced no audio.")

    audio = np.concatenate(chunks)

    output = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(str(output), audio, SAMPLE_RATE)
    return str(output)
