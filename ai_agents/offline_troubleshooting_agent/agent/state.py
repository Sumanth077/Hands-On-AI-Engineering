from pydantic import BaseModel, Field


class InvestigationState(BaseModel):
    investigation_id: str
    objective: str

    current_readings: dict = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)

    retrieved_incidents: list[dict] = Field(default_factory=list)
    retrieved_manual_chunks: list[dict] = Field(default_factory=list)

    tool_calls: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    final_result: dict | None = None


class InvestigationResult(BaseModel):
    likely_cause: str
    confidence: float
    recommendation: str
    supporting_evidence: list[str]
    retrieved_incident_ids: list[str]
    manual_sources: list[str]
