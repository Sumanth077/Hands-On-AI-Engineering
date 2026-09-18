"""
Post-call intelligence, powered by Telnyx Inference.

After a call ends, we ask a Telnyx-hosted model to write a short, personalised
follow-up message from the structured call record. The endpoint is
OpenAI-compatible, so the official `openai` client works with just the base URL
and API key swapped. Optionally the follow-up is texted back to the caller
through the Telnyx Messaging API.
"""

from __future__ import annotations

import httpx
from openai import OpenAI

from appointment_agent.config import SETTINGS, require_api_key

_SYSTEM_PROMPT = (
    "You write short, warm SMS follow-ups for a business front desk. "
    "Two sentences maximum. No emojis. Confirm the key detail and sign off "
    "with the business name. Never invent details that are not provided."
)


def _client() -> OpenAI:
    return OpenAI(base_url=SETTINGS.inference_base_url, api_key=require_api_key())


def generate_follow_up(context: str) -> str:
    """Draft a personalised follow-up message from a plain-text call context."""
    response = _client().chat.completions.create(
        model=SETTINGS.inference_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Business: {SETTINGS.business_name}\n"
                    f"Call context:\n{context}\n\n"
                    "Write the follow-up message."
                ),
            },
        ],
        temperature=0.4,
        max_tokens=160,
    )
    return (response.choices[0].message.content or "").strip()


def send_sms(to_number: str, text: str) -> dict:
    """Send the follow-up via the Telnyx Messaging API. Only called when SEND_SMS=true."""
    if not SETTINGS.sms_from:
        raise RuntimeError("SMS_FROM is not set. Add a Telnyx number to send SMS.")
    resp = httpx.post(
        "https://api.telnyx.com/v2/messages",
        headers={"Authorization": f"Bearer {require_api_key()}"},
        json={"from": SETTINGS.sms_from, "to": to_number, "text": text},
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()
