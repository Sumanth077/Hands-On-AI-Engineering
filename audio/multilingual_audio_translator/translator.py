"""LLM-based translation via Orq.ai."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ORQ_BASE_URL = "https://api.orq.ai/v3/router"
DEFAULT_MODEL = os.getenv("TRANSLATION_MODEL", "gemini-3-flash-preview")

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "sv": "Swedish",
    "uk": "Ukrainian",
}


def _get_client() -> OpenAI:
    """Build and return an OpenAI client pointed at the Orq.ai router."""
    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        raise ValueError(
            "ORQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(base_url=ORQ_BASE_URL, api_key=api_key)


def translate(text: str, source_language: str, target_language: str) -> str:
    """Translate transcript text to the target language."""
    if not text.strip():
        raise ValueError("Nothing to translate.")

    source_name = LANGUAGE_NAMES.get(source_language, source_language)
    target_name = LANGUAGE_NAMES.get(target_language, target_language)

    client = _get_client()
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Translate the user's text "
                    f"from {source_name} to {target_name}. Preserve meaning and tone. "
                    "Return only the translated text with no commentary."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )

    translated = response.choices[0].message.content
    if not translated or not translated.strip():
        raise ValueError("Translation model returned an empty response.")

    return translated.strip()
