from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Basic patient demographics extracted from the document."""

    name: str = ""
    age: str = ""
    date: str = ""


class LabFinding(BaseModel):
    """A single laboratory test result with its value, reference range, and status."""

    test: str = ""
    value: str = ""
    normal_range: str = ""
    status: Literal["normal", "abnormal", "critical"] = "normal"


class ImagingResult(BaseModel):
    """Findings and impression from a single imaging study."""

    study: str = ""
    findings: str = ""
    impression: str = ""


class ClinicalProfile(BaseModel):
    """Unified clinical profile combining patient info, labs, imaging, and flagged values."""

    patient: PatientInfo = Field(default_factory=PatientInfo)
    lab_findings: list[LabFinding] = Field(default_factory=list)
    imaging_results: list[ImagingResult] = Field(default_factory=list)
    clinical_signals: list[str] = Field(default_factory=list)
    flagged_items: list[str] = Field(default_factory=list)
