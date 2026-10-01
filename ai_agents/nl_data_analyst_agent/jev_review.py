"""Small, typed Jev review client for the Gradio-side checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_JEV_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
DEFAULT_JEV_MODEL = "typesafe-ai/jev"
DEFAULT_JEV_MIN_CONFIDENCE = 0.70


def jev_url() -> str:
    return os.getenv("JEV_URL", DEFAULT_JEV_URL)


def jev_model() -> str:
    return os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL)


def jev_min_confidence() -> float:
    return float(os.getenv("JEV_MIN_CONFIDENCE", str(DEFAULT_JEV_MIN_CONFIDENCE)))


@dataclass(frozen=True)
class Review:
    check: str
    verdict: str
    confidence: float | None
    issue: str
    actionable: bool
    status: str = "ok"

    def message(self) -> str:
        confidence = "unknown" if self.confidence is None else f"{self.confidence:.0%}"
        return f"Jev {self.check}: {self.verdict} ({confidence} confidence). {self.issue}"


def _post_jev(payload: dict[str, Any]) -> dict[str, Any]:
    key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("AI_GATEWAY_API_KEY is not configured")
    response = httpx.post(
        jev_url(),
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=float(os.getenv("JEV_TIMEOUT_SECONDS", "10")),
    )
    response.raise_for_status()
    return response.json()


def _choice_review(
    *,
    check: str,
    state: dict[str, Any],
    verdicts: dict[str, str],
    issues: dict[str, str],
    actionable_verdict: str,
) -> Review:
    questions = {
        "verdict": {
            "type": "choice",
            "instructions": f"Make the focused {check} judgment described by the criteria.",
            "criteria": verdicts,
        },
        "issue": {
            "type": "choice",
            "instructions": "Select the single most important issue, or none.",
            "criteria": issues,
        },
    }
    try:
        body = _post_jev({"model": jev_model(), "state": state, "questions": questions})
        answers = body["answers"]
        verdict_answer = answers["verdict"]
        verdict = str(verdict_answer["choice"])
        confidence = float(verdict_answer["confidence"])
        issue_key = str(answers["issue"]["choice"])
        issue = issues.get(issue_key, issue_key.replace("_", " "))
        threshold = jev_min_confidence()
        inconsistent = (verdict == actionable_verdict) == (issue_key == "none")
        if inconsistent and verdict == actionable_verdict:
            issue = "Jev requested review without identifying a specific issue; treated as inconclusive."
        elif inconsistent:
            issue = issues["none"]
        actionable = verdict == actionable_verdict and confidence >= threshold and not inconsistent
        status = "inconsistent" if inconsistent else ("ok" if confidence >= threshold else "low_confidence")
        return Review(check, verdict, confidence, issue, actionable, status)
    except (KeyError, TypeError, ValueError, RuntimeError, httpx.HTTPError) as error:
        return Review(check, "unavailable", None, f"Review unavailable: {error}", False, "error")


def review_question(question: str, schema: str, metric_definitions: list[str]) -> Review:
    return _choice_review(
        check="question clarity",
        state={"question": question, "database_schema": schema, "known_metric_definitions": metric_definitions},
        verdicts={
            "clear": "The question has enough information to answer from this schema and known definitions.",
            "clarify": "An essential definition is missing; answering would require a material assumption.",
        },
        issues={
            "none": "No essential definition is missing.",
            "metric_definition": "Clarify how the requested metric should be calculated.",
            "time_range": "Clarify the required time range.",
            "population": "Clarify which records or population should be included.",
            "comparison": "Clarify the comparison or grouping requested.",
        },
        actionable_verdict="clarify",
    )


def review_answer(question: str, sql: str, rows: list[dict[str, Any]], answer: str) -> Review:
    return _choice_review(
        check="answer grounding",
        state={"question": question, "executed_sql": sql, "returned_rows": rows, "answer": answer},
        verdicts={
            "grounded": "Every material factual claim in the answer is supported by the SQL result rows.",
            "mismatch": "At least one material factual claim is not supported by the SQL result rows.",
        },
        issues={
            "none": "The answer is supported by the returned rows.",
            "unsupported_value": "A reported value is not supported by the returned rows.",
            "unsupported_comparison": "A comparison or ranking is not supported by the returned rows.",
            "unsupported_scope": "The answer claims a broader scope than the returned rows support.",
            "missing_limitation": "The answer omits a limitation that materially changes its meaning.",
        },
        actionable_verdict="mismatch",
    )
