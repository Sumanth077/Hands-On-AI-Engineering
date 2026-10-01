"""Offline Jev contract tests. No model calls are made."""

from __future__ import annotations

import jev_review


def response(verdict: str, confidence: float, issue: str) -> dict:
    return {
        "answers": {
            "verdict": {"type": "choice", "choice": verdict, "confidence": confidence},
            "issue": {"type": "choice", "choice": issue, "confidence": 0.9},
        }
    }


def test_clear_question_is_not_blocked(monkeypatch) -> None:
    monkeypatch.setattr(jev_review, "_post_jev", lambda _: response("clear", 0.93, "none"))
    review = jev_review.review_question("Revenue by region?", "orders: region TEXT", ["revenue = quantity * unit_price"])
    assert review.verdict == "clear"
    assert review.confidence == 0.93
    assert not review.actionable


def test_unclear_question_requests_clarification(monkeypatch) -> None:
    monkeypatch.setattr(jev_review, "_post_jev", lambda _: response("clarify", 0.91, "metric_definition"))
    review = jev_review.review_question("Which customer is best?", "customers: id INTEGER", [])
    assert review.actionable
    assert "metric" in review.issue.lower()


def test_answer_grounding_mismatch_provides_review_path(monkeypatch) -> None:
    monkeypatch.setattr(jev_review, "_post_jev", lambda _: response("mismatch", 0.88, "unsupported_value"))
    review = jev_review.review_answer("How many?", "SELECT count(*) AS n FROM orders", [{"n": 4}], "There are 9.")
    assert review.actionable
    assert review.verdict == "mismatch"
    assert "value" in review.issue.lower()


def test_low_confidence_never_blocks(monkeypatch) -> None:
    monkeypatch.setattr(jev_review, "_post_jev", lambda _: response("clarify", 0.42, "time_range"))
    review = jev_review.review_question("Revenue?", "orders: created_at TEXT", [])
    assert review.status == "low_confidence"
    assert not review.actionable


def test_api_failure_never_blocks(monkeypatch) -> None:
    def fail(_):
        raise RuntimeError("offline")

    monkeypatch.setattr(jev_review, "_post_jev", fail)
    review = jev_review.review_answer("How many?", "SELECT 1", [{"1": 1}], "One")
    assert review.status == "error"
    assert review.verdict == "unavailable"
    assert not review.actionable


def test_settings_are_read_at_request_time(monkeypatch) -> None:
    captured = {}

    def fake_post(payload):
        captured["payload"] = payload
        captured["url"] = jev_review.jev_url()
        return response("clear", 0.75, "none")

    monkeypatch.setenv("JEV_MODEL", "typesafe-ai/custom-jev")
    monkeypatch.setenv("JEV_URL", "https://example.test/v1/evaluate")
    monkeypatch.setenv("JEV_MIN_CONFIDENCE", "0.80")
    monkeypatch.setattr(jev_review, "_post_jev", fake_post)

    review = jev_review.review_question("How many orders?", "orders: id INTEGER", [])

    assert captured["payload"]["model"] == "typesafe-ai/custom-jev"
    assert captured["url"] == "https://example.test/v1/evaluate"
    assert review.status == "low_confidence"


def test_actionable_verdict_without_issue_is_inconclusive(monkeypatch) -> None:
    monkeypatch.setattr(jev_review, "_post_jev", lambda _: response("clarify", 0.95, "none"))
    review = jev_review.review_question("Best customer?", "customers: id INTEGER", [])
    assert review.status == "inconsistent"
    assert not review.actionable
    assert "inconclusive" in review.issue


def test_non_actionable_verdict_ignores_conflicting_issue(monkeypatch) -> None:
    monkeypatch.setattr(jev_review, "_post_jev", lambda _: response("grounded", 0.95, "unsupported_value"))
    review = jev_review.review_answer("How many?", "SELECT 4", [{"n": 4}], "There are four.")
    assert review.status == "inconsistent"
    assert not review.actionable
    assert review.issue == "The answer is supported by the returned rows."
