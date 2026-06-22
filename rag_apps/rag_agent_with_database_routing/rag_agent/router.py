"""
RAG Agent with Database Routing - router.

Uses Orq.ai with zai/glm-5-turbo to classify each query into one of
three databases and return a structured RoutingDecision.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

MODEL = "zai/glm-5-turbo"

_DB_OPTIONS = ("products", "support", "financial")

# Keywords used as a fallback when the model does not return valid JSON
_KEYWORDS: dict[str, list[str]] = {
    "financial": [
        "price", "pricing", "plan", "billing", "invoice", "payment",
        "revenue", "contract", "discount", "tax", "vat", "cost", "fee",
        "subscription", "refund", "charge",
    ],
    "support": [
        "reset", "password", "account", "locked", "cancel", "troubleshoot",
        "slow", "error", "help", "contact", "return", "policy", "issue",
        "invite", "2fa", "two-factor", "authentication", "export",
    ],
}


class RoutingDecision(BaseModel):
    """Holds which database a query was routed to and why."""

    database: Literal["products", "support", "financial"]
    reasoning: str


ROUTING_INSTRUCTIONS = """\
You are a query routing agent. Classify the user query into exactly one of three databases:

- products: Questions about product features, specifications, availability,
  hardware, software tools, or what a product does.

- support: Questions about account issues, troubleshooting, how-to guides,
  policies (returns, refunds, cancellation), contacting support, or fixing a problem.

- financial: Questions about pricing, plans, billing, invoices, payments,
  revenue reports, contracts, discounts, or taxes.

You MUST respond with ONLY a JSON object and nothing else. No explanation, no markdown.
Example: {"database": "support", "reasoning": "User asks about password reset"}
"""


def _keyword_fallback(query: str) -> RoutingDecision:
    """Classify by keyword matching when the model returns non-JSON."""
    lower = query.lower()
    for db in ("financial", "support"):  # products is the default
        if any(kw in lower for kw in _KEYWORDS[db]):
            return RoutingDecision(
                database=db,
                reasoning=f"Keyword match fallback (model did not return JSON).",
            )
    return RoutingDecision(
        database="products",
        reasoning="Keyword fallback default.",
    )


def build_router(orq_api_key: str) -> OpenAI:
    """Return an Orq.ai OpenAI client for routing."""
    return OpenAI(base_url="https://my.orq.ai/v3/router", api_key=orq_api_key)


def route_query(client: OpenAI, query: str) -> RoutingDecision:
    """Classify a query and return the routing decision."""
    response = client.responses.create(
        model=MODEL,
        instructions=ROUTING_INSTRUCTIONS,
        input=query,
    )
    text = response.output[0].content[0].text.strip()

    # Strip markdown code fences if the model wraps output anyway
    text = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL).strip()

    # Try to extract a JSON object anywhere in the text
    if not text:
        return _keyword_fallback(query)

    try:
        data = json.loads(text)
        if data.get("database") in _DB_OPTIONS:
            return RoutingDecision(**data)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try to find a JSON object embedded in prose
    match = re.search(r'\{[^{}]*"database"\s*:[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if data.get("database") in _DB_OPTIONS:
                return RoutingDecision(**data)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return _keyword_fallback(query)
