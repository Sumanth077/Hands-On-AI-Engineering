from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    name: str = ""
    age: str = ""
    date: str = ""


class LabFinding(BaseModel):
    test: str = ""
    value: str = ""
    normal_range: str = ""
    status: Literal["normal", "abnormal", "critical"] = "normal"


class ImagingResult(BaseModel):
    study: str = ""
    findings: str = ""
    impression: str = ""


class ClinicalProfile(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    lab_findings: list[LabFinding] = Field(default_factory=list)
    imaging_results: list[ImagingResult] = Field(default_factory=list)
    clinical_signals: list[str] = Field(default_factory=list)
    flagged_items: list[str] = Field(default_factory=list)
