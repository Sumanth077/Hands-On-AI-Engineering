"""
Persistent investigation state for the Self-Driving Data Analyst.

Raw chat history alone isn't enough for a long-running investigation: the
agent needs an explicit, structured record of what it has found, what it
still suspects, and what it has already tried, so it can reason about the
investigation as a whole rather than just the last message.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    hypothesis: str
    status: Literal["untested", "supported", "rejected"] = "untested"
    confidence: Optional[float] = None
    evidence: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    finding: str
    importance: Literal["low", "medium", "high"] = "medium"
    evidence: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    call_index: int
    model_call_index: int
    tool_name: str
    arguments: dict
    result_summary: str
    is_error: bool = False
    timestamp: float = Field(default_factory=time.time)


class UsageRecord(BaseModel):
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def estimated_cost_usd(self) -> float:
        # Liner Model API list pricing for liner-mark-1.0, regardless of which
        # underlying model the orchestrator actually routed the request to.
        uncached_input = max(self.prompt_tokens - self.cached_tokens, 0)
        cost = (
            uncached_input * (1.0 / 1_000_000)
            + self.cached_tokens * (0.10 / 1_000_000)
            + self.completion_tokens * (6.0 / 1_000_000)
        )
        return round(cost, 4)


class InvestigationState(BaseModel):
    objective: str
    reasoning_effort: str = "medium"

    findings: list[Finding] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    step_count: int = 0
    is_finished: bool = False
    stop_reason: Optional[str] = None
    final_report: Optional[str] = None
    root_cause: Optional[str] = None
    confidence: Optional[float] = None
    recommended_actions: list[str] = Field(default_factory=list)

    usage: UsageRecord = Field(default_factory=UsageRecord)
    started_at: float = Field(default_factory=time.time)

    # ---- helpers used by the harness ----

    def recent_tool_calls(self, n: int = 6) -> list[ToolCallRecord]:
        return self.tool_calls[-n:]

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result_summary: str,
        model_call_index: int = 0,
        is_error: bool = False,
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            call_index=len(self.tool_calls),
            model_call_index=model_call_index,
            tool_name=tool_name,
            arguments=arguments,
            result_summary=result_summary[:800],
            is_error=is_error,
        )
        self.tool_calls.append(record)
        return record

    def add_finding(self, finding: str, importance: str = "medium", evidence: Optional[list[str]] = None) -> None:
        self.findings.append(Finding(finding=finding, importance=importance, evidence=evidence or []))

    def upsert_hypothesis(self, hypothesis: str, status: str = "untested", confidence: Optional[float] = None, evidence: Optional[list[str]] = None) -> None:
        for h in self.hypotheses:
            if h.hypothesis.strip().lower() == hypothesis.strip().lower():
                h.status = status
                h.confidence = confidence
                if evidence:
                    h.evidence.extend(evidence)
                return
        self.hypotheses.append(
            Hypothesis(hypothesis=hypothesis, status=status, confidence=confidence, evidence=evidence or [])
        )

    def recent_sql_queries(self, n: int = 4) -> list[str]:
        queries = [
            tc.arguments.get("query", "")
            for tc in self.tool_calls
            if tc.tool_name == "run_sql" and not tc.is_error
        ]
        return queries[-n:]

    def elapsed_seconds(self) -> float:
        return round(time.time() - self.started_at, 1)

    def summary_for_prompt(self) -> str:
        """A compact, human-readable snapshot of the investigation so far,
        used to keep the model grounded without re-sending the full raw
        tool-call history on every turn."""
        lines = [f"Objective: {self.objective}", f"Step: {self.step_count}"]

        if self.findings:
            lines.append("Findings so far:")
            for f in self.findings:
                lines.append(f"  - [{f.importance}] {f.finding}")

        if self.hypotheses:
            lines.append("Hypotheses:")
            for h in self.hypotheses:
                conf = f" (confidence {h.confidence})" if h.confidence is not None else ""
                lines.append(f"  - [{h.status}]{conf} {h.hypothesis}")

        if self.open_questions:
            lines.append("Open questions:")
            for q in self.open_questions:
                lines.append(f"  - {q}")

        return "\n".join(lines)
