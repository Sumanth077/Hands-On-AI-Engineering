"""Provider-neutral decisions, evidence, and results. No executable model tools."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["Matches", "Does not match", "Needs review"]
Role = Literal["repository", "readme", "licence", "documentation", "release", "activity"]


class Structured(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Limits(Structured):
    candidate_limit: int = Field(default=5, ge=1, le=20)
    max_web_requests: int = Field(default=30, ge=1, le=200)
    max_model_calls: int = Field(default=30, ge=1, le=100)
    max_iterations: int = Field(default=3, ge=1, le=5)
    concurrency: int = Field(default=3, ge=1, le=20)
    max_input_chars: int = Field(default=80000, ge=4000, le=250000)
    max_output_chars: int = Field(default=24000, ge=1000, le=100000)
    max_output_tokens: int = Field(default=4000, ge=256, le=16000)
    max_source_chars: int = Field(default=12000, ge=500, le=30000)


class Criterion(Structured):
    id: str = Field(min_length=1, max_length=60)
    label: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)
    required: bool
    preference_quote: str = Field(default="", max_length=1000)


class Interpretation(Structured):
    summary: str = Field(min_length=1, max_length=1000)
    criteria: list[Criterion] = Field(min_length=1, max_length=10)
    queries: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        min_length=1, max_length=3
    )


class Discovered(Structured):
    repo_url: str
    title: str
    snippet: str | None
    search_urls: list[str]


class Candidate(Structured):
    repo_url: str
    reason: str = Field(min_length=1, max_length=1000)


class SelectionDecision(Structured):
    repo_url: str
    action: Literal["inspect", "exclude", "defer"]
    reason: str = Field(min_length=1, max_length=500)


class Selection(Structured):
    candidates: list[Candidate] = Field(max_length=20)
    decisions: list[SelectionDecision] = Field(default_factory=list, max_length=60)


class SourceRequest(Structured):
    repo_url: str
    url: str
    role: Role
    parent_id: str
    reason: str = Field(min_length=1, max_length=500)


class ResearchPlan(Structured):
    requests: list[SourceRequest] = Field(max_length=100)


class Citation(Structured):
    source_id: str
    excerpt: str = Field(default="", max_length=1000)
    passage_id: str = ""


class Assessment(Structured):
    criterion_id: str
    verdict: Verdict
    explanation: str = Field(min_length=1, max_length=1500)
    citations: list[Citation] = Field(max_length=10)
    basis: Literal["documented", "unknown"]
    processing_note: str = ""


class Fact(Structured):
    kind: Literal[
        "licence",
        "separate_terms",
        "self_hosting",
        "capability",
        "latest_release",
        "repository_activity",
        "conflict",
    ]
    value: str = Field(min_length=1, max_length=1500)
    citations: list[Citation] = Field(min_length=1, max_length=10)


class Extraction(Structured):
    project_name: str = Field(min_length=1, max_length=200)
    facts: list[Fact] = Field(max_length=30)
    assessments: list[Assessment] = Field(max_length=10)
    licence_issues: list["LicenceIssue"] = Field(default_factory=list, max_length=10)


class ProjectExtraction(Structured):
    repo_url: str
    extraction: Extraction


class BatchExtraction(Structured):
    items: list[ProjectExtraction] = Field(max_length=2)


class LicenceIssue(Structured):
    component: str
    scope: Literal["evaluated_path", "optional_unrelated", "unknown"]
    conflicting_terms: bool
    explanation: str
    citations: list[Citation] = Field(min_length=1, max_length=10)


class Source(Structured):
    id: str
    repo_url: str
    url: str
    loaded_url: str | None = None
    role: Role
    parent_id: str | None = None
    retrieval_timestamp: datetime
    outcome: str
    error: str | None = None
    markdown: str = ""
    links: list[str] = Field(default_factory=list)
    truncated: bool = False


class Project(Structured):
    candidate: Candidate
    name: str
    facts: list[Fact] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)


class Recommendation(Structured):
    repo_url: str
    explanation: str = Field(min_length=1, max_length=1500)
    tradeoffs: str = Field(min_length=1, max_length=1500)
    citations: list[Citation] = Field(min_length=1, max_length=10)


class Synthesis(Structured):
    shortlist: list[Recommendation] = Field(max_length=20)


class ResearchResult(Structured):
    requirement: str
    offline: bool
    created_at: datetime
    interpretation: Interpretation | None = None
    discovered: list[Discovered] = Field(default_factory=list)
    selection_decisions: list[SelectionDecision] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    shortlist: list[Recommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    status: Literal["complete", "partial", "failed"] = "partial"
