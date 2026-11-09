from __future__ import annotations

import os
from typing import Any

import requests

from freshness_checker.models import Citation, FreshnessResult, FreshnessStatus, TechnicalClaim

DECISIONS_URL = "https://api.parallel.ai/v1/decisions"


def _status_from_answer(answer: dict[str, Any]) -> tuple[FreshnessStatus, float | None]:
    if answer.get("type") != "choice":
        return FreshnessStatus.NEEDS_REVIEW, None

    selected = answer.get("choice")
    probabilities = answer.get("probabilities", {})
    confidence = probabilities.get(selected) if isinstance(probabilities, dict) else None
    if selected == "current":
        return FreshnessStatus.CURRENT, confidence
    if selected == "outdated":
        return FreshnessStatus.OUTDATED, confidence
    return FreshnessStatus.NEEDS_REVIEW, confidence


def _citation_models(raw_citations: list[dict[str, Any]]) -> list[Citation]:
    return [
        Citation(
            url=item["url"],
            title=item.get("title"),
            excerpts=item.get("excerpts") or [],
        )
        for item in raw_citations
        if item.get("url")
    ]


def check_claim(claim: TechnicalClaim) -> FreshnessResult:
    """Ask one bounded choice question so citations belong to this one claim."""
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        raise RuntimeError("PARALLEL_API_KEY is not configured.")

    request_body = {
        "state": {
            "technical_claim": claim.claim,
            "document_evidence": claim.source_quote,
            "document_context": claim.context,
            "claim_category": claim.category.value,
        },
        "questions": {
            "freshness": {
                "type": "choice",
                "instructions": (
                    "Check current public documentation and authoritative sources when available. "
                    "Classify whether the technical claim from the supplied document is still accurate. "
                    "Choose needs_review when current evidence is insufficient, conflicting, or does not "
                    "clearly address the claim."
                ),
                "criteria": {
                    "current": "Current authoritative documentation supports the claim or instruction.",
                    "outdated": (
                        "Current authoritative documentation contradicts the claim, replaces it, "
                        "or marks the referenced feature, version, endpoint, or instruction as deprecated."
                    ),
                    "needs_review": (
                        "Available evidence is insufficient or conflicting, so a human should review it."
                    ),
                },
            }
        },
    }
    try:
        response = requests.post(
            DECISIONS_URL,
            headers={"x-api-key": api_key},
            json=request_body,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        return FreshnessResult(
            claim=claim,
            status=FreshnessStatus.NEEDS_REVIEW,
            error=f"Parallel Decisions request failed: {error}",
        )

    answer = payload.get("answers", {}).get("freshness", {})
    status, confidence = _status_from_answer(answer)
    citations = _citation_models(payload.get("citations", []))
    if not citations:
        # An uncited answer is not auditable, even when the classifier selected CURRENT.
        status = FreshnessStatus.NEEDS_REVIEW
    return FreshnessResult(
        claim=claim,
        status=status,
        confidence=confidence,
        citations=citations,
        raw_answer=answer,
    )


def check_claims(claims: list[TechnicalClaim]) -> list[FreshnessResult]:
    return [check_claim(claim) for claim in claims]
