from __future__ import annotations

import json
import os
import re

from google import genai
from google.genai import types

from document_processor import ProcessedPage
from schemas import ClinicalProfile

MODEL_ID = "gemma-4-31b-it"

SYSTEM_INSTRUCTION = """You are a clinical document extraction assistant.
Extract structured medical information from the provided page content.
Return only factual information present in the document.
Use empty strings or empty lists when information is missing.
For lab findings, set status to:
- normal: within reference range
- abnormal: outside reference range but not life-threatening
- critical: dangerously out of range or marked critical
Populate flagged_items with concise descriptions of abnormal or critical values."""

EXTRACTION_PROMPT = """Extract a structured clinical profile from this medical document page.
Include patient demographics, laboratory results, imaging findings, clinical signals,
and any abnormal or critical values in flagged_items."""


def _build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=api_key)


def _generation_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_json_schema=ClinicalProfile.model_json_schema(),
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH,
        ),
    )


def _parse_response_text(text: str) -> ClinicalProfile:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    return ClinicalProfile.model_validate(data)


def extract_from_page(client: genai.Client, page: ProcessedPage) -> ClinicalProfile:
    config = _generation_config()
    page_context = f"{EXTRACTION_PROMPT}\n\nPage {page.page_number}."

    if page.kind == "text" and page.text:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"{page_context}\n\nDocument text:\n{page.text}",
            config=config,
        )
    elif page.kind == "vision" and page.image_bytes:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(data=page.image_bytes, mime_type="image/png"),
                page_context,
            ],
            config=config,
        )
    else:
        return ClinicalProfile()

    if not response.text:
        return ClinicalProfile()

    return _parse_response_text(response.text)
