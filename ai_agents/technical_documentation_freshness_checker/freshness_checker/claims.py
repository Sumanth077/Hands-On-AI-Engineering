from __future__ import annotations

from freshness_checker.models import TechnicalClaim


def select_claims(claims: list[TechnicalClaim], limit: int) -> list[TechnicalClaim]:
    """Keep a small, distinct set of claims before any live-web request is made."""
    selected: list[TechnicalClaim] = []
    seen: set[str] = set()

    for claim in claims:
        normalized = " ".join(claim.claim.lower().split())
        if len(normalized) < 8 or normalized in seen:
            continue
        selected.append(claim)
        seen.add(normalized)
        if len(selected) >= limit:
            break

    return selected
