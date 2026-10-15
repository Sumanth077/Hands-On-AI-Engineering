from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .models import ReviewResult, Source


def _answer_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("value", "answer", "choice", "result", "probability"):
            if key in value:
                return value[key]
    return value


def parse_review_response(payload: dict[str, Any]) -> ReviewResult:
    answers = payload.get("answers", payload.get("data", {}).get("answers", {}))
    if not isinstance(answers, dict):
        answers = {}

    coverage = _answer_value(answers.get("evidence_coverage"))
    grounding = _answer_value(answers.get("claim_grounding"))
    action = _answer_value(answers.get("action"))
    coverage_probability = (
        float(coverage)
        if isinstance(coverage, (int, float)) and not isinstance(coverage, bool)
        else None
    )
    grounding_probability = (
        float(grounding)
        if isinstance(grounding, (int, float)) and not isinstance(grounding, bool)
        else None
    )
    coverage_value = coverage if isinstance(coverage, bool) else None
    grounding_value = grounding if isinstance(grounding, bool) else None
    if coverage_probability is not None:
        coverage_value = coverage_probability >= 0.7
    if grounding_probability is not None:
        grounding_value = grounding_probability >= 0.7

    explanation = "JEV approved the evidence review."
    if action == "research_again" or coverage_value is False or grounding_value is False:
        explanation = "JEV found that the report needs more evidence or stronger claim grounding."

    return ReviewResult(
        status="completed",
        evidence_coverage=coverage_value,
        claim_grounding=grounding_value,
        evidence_coverage_probability=coverage_probability,
        claim_grounding_probability=grounding_probability,
        action=action if isinstance(action, str) else None,
        explanation=explanation,
        raw=payload,
    )


class JevReviewer:
    """Uses JEV as a decision layer, never as the report writer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def review(
        self, question: str, answer: str, sources: list[Source], citation_issues: list[str]
    ) -> ReviewResult:
        if not self.settings.ai_gateway_api_key:
            return ReviewResult(
                status="unavailable",
                evidence_coverage=None,
                claim_grounding=None,
                evidence_coverage_probability=None,
                claim_grounding_probability=None,
                action=None,
                explanation="AI_GATEWAY_API_KEY is not set, so JEV review was skipped.",
            )

        evidence = [
            {
                "number": source.number,
                "title": source.title,
                "url": source.url,
                "snippet": source.snippet[:1200],
            }
            for source in sources
        ]
        payload = {
            "model": self.settings.jev_model,
            "state": {
                "question": question,
                "report": answer,
                "sources": evidence,
                "deterministic_citation_issues": citation_issues,
            },
            "questions": {
                "evidence_coverage": {
                    "type": "boolean",
                    "instructions": "Is the retrieved evidence sufficient to answer the user's question without major gaps?",
                },
                "claim_grounding": {
                    "type": "boolean",
                    "instructions": "Are the report's material factual claims supported by the numbered sources attached to it?",
                },
                "action": {
                    "type": "choice",
                    "instructions": "Choose research_again if evidence coverage or claim grounding is inadequate. Otherwise choose approve.",
                    "criteria": {
                        "approve": "The evidence covers the question and the report's material claims are grounded in attached sources.",
                        "research_again": "Evidence is incomplete, weak, or does not support important report claims.",
                    },
                },
            },
        }
        try:
            response = httpx.post(
                f"{self.settings.jev_base_url}/evaluate",
                headers={
                    "Authorization": f"Bearer {self.settings.ai_gateway_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=45.0,
            )
            response.raise_for_status()
            return parse_review_response(response.json())
        except (httpx.HTTPError, ValueError) as error:
            return ReviewResult(
                status="unavailable",
                evidence_coverage=None,
                claim_grounding=None,
                evidence_coverage_probability=None,
                claim_grounding_probability=None,
                action=None,
                explanation=f"JEV review could not run: {error}",
            )

