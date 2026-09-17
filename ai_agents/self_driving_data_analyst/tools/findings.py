"""
save_finding(...) - the agent's way of writing something down for good.

Kept separate from the model's free-text output on purpose: findings saved
this way persist in structured state and survive into the final report,
observations mentioned only in passing in the model's reasoning do not.
"""

from __future__ import annotations

from agent.state import InvestigationState


def save_finding(state: InvestigationState, finding: str, importance: str = "medium", evidence: list[str] | None = None) -> dict:
    if importance not in ("low", "medium", "high"):
        importance = "medium"
    state.add_finding(finding=finding, importance=importance, evidence=evidence or [])
    return {"success": True, "saved": finding}
