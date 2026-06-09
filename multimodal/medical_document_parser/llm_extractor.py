from __future__ import annotations

import base64
import json
import os
import re

from openai import OpenAI

from document_processor import ProcessedPage
from schemas import ClinicalProfile

MODEL_ID = "gemma-4-31b-it"
ORQ_BASE_URL = "https://api.orq.ai/v3/router"

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


def _build_client() -> OpenAI:
    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        raise ValueError("ORQ_API_KEY is not set. Add it to your .env file.")
    return OpenAI(
        base_url=ORQ_BASE_URL,
        api_key=api_key,
    )


def _system_message() -> str:
    schema = json.dumps(ClinicalProfile.model_json_schema(), indent=2)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        "Respond with valid JSON only, matching this schema:\n"
        f"{schema}"
    )


def _parse_response_text(text: str) -> ClinicalProfile:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    return ClinicalProfile.model_validate(data)


def extract_from_page(client: OpenAI, page: ProcessedPage) -> ClinicalProfile:
    page_context = f"{EXTRACTION_PROMPT}\n\nPage {page.page_number}."
    messages: list[dict] = [{"role": "system", "content": _system_message()}]

    if page.kind == "text" and page.text:
        messages.append(
            {
                "role": "user",
                "content": f"{page_context}\n\nDocument text:\n{page.text}",
            }
        )
    elif page.kind == "vision" and page.image_bytes:
        image_b64 = base64.b64encode(page.image_bytes).decode("utf-8")
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": page_context},
                ],
            }
        )
    else:
        return ClinicalProfile()

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        return ClinicalProfile()

    return _parse_response_text(content)
