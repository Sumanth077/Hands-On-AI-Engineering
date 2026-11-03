from freshness_checker.claims import select_claims
from freshness_checker.models import ClaimCategory, TechnicalClaim


def claim(text: str) -> TechnicalClaim:
    return TechnicalClaim(
        claim=text,
        category=ClaimCategory.API,
        source_quote=text,
        context="Context from the document.",
        why_current_context_matters="API contracts can change.",
    )


def test_select_claims_removes_duplicates_and_applies_limit() -> None:
    selected = select_claims(
        [claim("Use API v1."), claim("  use api v1. "), claim("Install package 2.0.")],
        limit=1,
    )

    assert len(selected) == 1
    assert selected[0].claim == "Use API v1."
