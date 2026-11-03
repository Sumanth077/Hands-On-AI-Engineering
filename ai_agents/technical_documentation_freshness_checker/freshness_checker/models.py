from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClaimCategory(str, Enum):
    API = "api_version_or_endpoint"
    PACKAGE = "package_or_library_version"
    MODEL = "model_name_or_availability"
    DEPENDENCY = "dependency_or_runtime_requirement"
    CONFIGURATION = "configuration_or_setup_instruction"
    DEPRECATION = "deprecation_or_support_status"
    INTEGRATION = "external_integration"
    OTHER = "other_time_sensitive_claim"


class TechnicalClaim(BaseModel):
    claim: str = Field(description="The technical claim or instruction, stated concisely.")
    category: ClaimCategory
    source_quote: str = Field(
        description="A short, verbatim quote from the supplied Markdown that supports the claim."
    )
    context: str = Field(
        description="Short surrounding context that helps a reviewer understand the claim."
    )
    why_current_context_matters: str = Field(
        description="Why this statement could become stale and needs a live-web check."
    )


class TechnicalClaimSet(BaseModel):
    claims: list[TechnicalClaim] = Field(
        default_factory=list,
        description=(
            "Only externally verifiable technical statements or instructions that can change "
            "over time. Exclude timeless explanations, internal prose, and general background."
        ),
    )


class FreshnessStatus(str, Enum):
    CURRENT = "CURRENT"
    OUTDATED = "OUTDATED"
    NEEDS_REVIEW = "NEEDS REVIEW"


class Citation(BaseModel):
    url: str
    title: str | None = None
    excerpts: list[str] = Field(default_factory=list)


class FreshnessResult(BaseModel):
    claim: TechnicalClaim
    status: FreshnessStatus
    confidence: float | None = None
    citations: list[Citation] = Field(default_factory=list)
    raw_answer: dict = Field(default_factory=dict)
    error: str | None = None
