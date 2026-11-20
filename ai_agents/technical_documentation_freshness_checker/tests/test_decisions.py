from freshness_checker import decisions
from freshness_checker.models import ClaimCategory, FreshnessStatus, TechnicalClaim


def make_claim() -> TechnicalClaim:
    return TechnicalClaim(
        claim="Use API v1 for this request.",
        category=ClaimCategory.API,
        source_quote="Use API v1 for this request.",
        context="A setup guide.",
        why_current_context_matters="API versions can change.",
    )


class StubResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


def test_uncited_current_decision_needs_review(monkeypatch) -> None:
    monkeypatch.setenv("PARALLEL_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(
        decisions.requests,
        "post",
        lambda *args, **kwargs: StubResponse(
            {
                "answers": {
                    "freshness": {
                        "type": "choice",
                        "choice": "current",
                        "probabilities": {"current": 0.9},
                    }
                },
                "citations": [],
            }
        ),
    )

    result = decisions.check_claim(make_claim())

    assert result.status is FreshnessStatus.NEEDS_REVIEW
    assert result.citations == []


def test_cited_current_decision_remains_current(monkeypatch) -> None:
    monkeypatch.setenv("PARALLEL_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(
        decisions.requests,
        "post",
        lambda *args, **kwargs: StubResponse(
            {
                "answers": {
                    "freshness": {
                        "type": "choice",
                        "choice": "current",
                        "probabilities": {"current": 0.9},
                    }
                },
                "citations": [
                    {
                        "url": "https://example.com/api",
                        "title": "API docs",
                        "excerpts": ["API v1 is supported."],
                    }
                ],
            }
        ),
    )

    result = decisions.check_claim(make_claim())

    assert result.status is FreshnessStatus.CURRENT
    assert len(result.citations) == 1
