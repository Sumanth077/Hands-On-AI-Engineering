"""Typed models used by the intake agent and local case store."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class IntakeFacts(BaseModel):
    """Facts collected from the user without legal interpretation."""

    matter_type: str | None = None
    narrative: str = ""
    people_or_organisations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    dates_and_timeline: list[str] = Field(default_factory=list)
    evidence_mentioned: list[str] = Field(default_factory=list)
    impact_or_loss: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class IntakeDecision(BaseModel):
    """The agent's next safe intake action."""

    facts: IntakeFacts
    next_question: str | None = None
    status: Literal["collecting", "ready_for_review"]
    reason: str


class TranscriptTurn(BaseModel):
    """A final speech-to-text result preserved for the current intake."""

    text: str
    source: Literal["voice", "text"]


class IntakeGraphState(TypedDict):
    """LangGraph state for a single guided intake."""

    history: list[dict[str, str]]
    facts: dict
    decision: dict


class CaseRecord(BaseModel):
    """The local review record saved after a human approves the draft."""

    case_id: str
    created_at: str
    summary_markdown: str
    facts: IntakeFacts
