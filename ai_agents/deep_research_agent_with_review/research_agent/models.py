from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    number: int
    title: str
    url: str
    snippet: str
    published_date: str | None = None


@dataclass(frozen=True)
class ReviewResult:
    status: str
    evidence_coverage: bool | None
    claim_grounding: bool | None
    evidence_coverage_probability: float | None
    claim_grounding_probability: float | None
    action: str | None
    explanation: str
    raw: dict = field(default_factory=dict)

    @property
    def needs_revision(self) -> bool:
        return self.status == "completed" and (
            self.action == "research_again"
            or self.evidence_coverage is False
            or self.claim_grounding is False
        )


@dataclass(frozen=True)
class ResearchResult:
    question: str
    answer: str
    sources: list[Source]
    plan: list[str]
    searches: list[str]
    review: ReviewResult
    revision_count: int
    citation_issues: list[str]

