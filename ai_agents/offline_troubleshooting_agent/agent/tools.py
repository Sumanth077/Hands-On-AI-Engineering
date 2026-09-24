from collections.abc import Callable
from typing import Any

from agent.state import InvestigationResult, InvestigationState
from memory.incident_store import search_incidents
from memory.manual import search_manual_chunks
from simulator.history import History
from simulator.machine import Machine

RESULT_SUMMARY_MAX_LENGTH = 200


class AgentTools:
    """Bound tool methods exposed to the local LLM for a single investigation.

    Wraps a running Machine simulator, its reading history, and a shared
    InvestigationState so every call both returns a plain result to the model
    and records its effect on the investigation state.
    """

    def __init__(self, machine: Machine, history: History, state: InvestigationState):
        self.machine = machine
        self.history = history
        self.state = state

    def _log_call(self, name: str, args: dict, result_summary: str) -> None:
        summary = result_summary
        if len(summary) > RESULT_SUMMARY_MAX_LENGTH:
            summary = summary[: RESULT_SUMMARY_MAX_LENGTH - 3] + "..."
        self.state.tool_calls.append({"name": name, "args": args, "result_summary": summary})

    def get_current_readings(self) -> dict:
        """Get the machine's current sensor readings.

        Returns the most recent temperature, vibration, pressure, status, and
        error code from the simulated machine. Use this first to see the
        present state before looking at history or searching memory.

        Returns:
            dict: The current reading with keys temperature_c, vibration_mm_s,
            pressure_bar, status, error_code, and timestamp.
        """
        reading = self.machine.get_current_readings()
        result = reading.model_dump(mode="json")
        self._log_call("get_current_readings", {}, str(result))
        return result

    def get_recent_history(self, limit: int = 20) -> list[dict]:
        """Get recent sensor readings leading up to now.

        Returns up to `limit` of the most recent readings in chronological
        order, useful for seeing how temperature, vibration, and pressure
        have trended over the last several steps of a developing fault.

        Args:
            limit: Maximum number of recent readings to return. Defaults to 20.

        Returns:
            list[dict]: Readings in chronological order (oldest first), each
            with keys temperature_c, vibration_mm_s, pressure_bar, status,
            error_code, and timestamp.
        """
        history_df = self.history.get_recent_history(limit=limit)
        result = history_df.to_dict(orient="records")
        self._log_call("get_recent_history", {"limit": limit}, f"{len(result)} readings")
        return result

    def search_manual(self, query: str, top_k: int = 3) -> list[dict]:
        """Search the local equipment manual and troubleshooting guide.

        Embeds the query locally and searches manual chunks for relevant
        guidance on normal operating ranges, error codes, or inspection
        steps.

        Args:
            query: A natural-language description of what you're looking
                for, e.g. "bearing vibration warning" or "cooling fan
                inspection".
            top_k: Maximum number of manual chunks to return. Defaults to 3.

        Returns:
            list[dict]: Matching manual chunks, each with id, score, text,
            section_title, source_filename, and component.
        """
        results = search_manual_chunks(query, top_k=top_k)
        self.state.retrieved_manual_chunks.extend(results)
        self._log_call(
            "search_manual", {"query": query, "top_k": top_k}, f"{len(results)} chunks"
        )
        return results

    def search_past_incidents(self, query: str, top_k: int = 3) -> list[dict]:
        """Search previously resolved incidents for similar past cases.

        Embeds the query locally and searches confirmed, human-verified past
        incidents for cases with similar symptoms. Retrieved incidents are
        evidence to weigh against current readings, not guaranteed truth —
        they may not apply if the current evidence doesn't actually match.

        Args:
            query: A natural-language description of the current symptoms,
                e.g. "high temperature and high vibration".
            top_k: Maximum number of past incidents to return. Defaults to 3.

        Returns:
            list[dict]: Matching past incidents, each with id, score, and the
            incident's stored fields (confirmed_cause, fix_applied, outcome,
            readings, error_code, etc).
        """
        results = search_incidents(query, top_k=top_k)
        self.state.retrieved_incidents.extend(results)
        self._log_call(
            "search_past_incidents",
            {"query": query, "top_k": top_k},
            f"{len(results)} incidents",
        )
        return results

    def save_finding(self, finding: str, evidence: str) -> str:
        """Record an important observation made during the investigation.

        Use this to note a conclusion you've drawn from the evidence so far
        (e.g. "vibration pattern matches bearing wear") along with what
        supports it. This does not save anything permanently — it only
        builds up the visible investigation trail for this session.

        Args:
            finding: A short statement of what you observed or concluded.
            evidence: What specific evidence supports this finding.

        Returns:
            str: A short confirmation that the finding was recorded.
        """
        entry = f"{finding} (evidence: {evidence})"
        self.state.findings.append(entry)
        self._log_call("save_finding", {"finding": finding, "evidence": evidence}, entry)
        return "Finding recorded."

    def finish_investigation(
        self,
        likely_cause: str,
        confidence: float,
        recommendation: str,
        supporting_evidence: list[str],
        retrieved_incident_ids: list[str],
        manual_sources: list[str],
    ) -> dict:
        """Finish the investigation and produce a structured recommendation.

        Call this once you have enough evidence to give a reasonable
        conclusion. This does NOT save anything to permanent memory — a
        human must separately confirm the real cause and fix before
        anything is written to durable incident memory.

        Args:
            likely_cause: The most likely root cause based on the evidence.
            confidence: A confidence score between 0.0 and 1.0.
            recommendation: The recommended next action (an inspection or
                normal maintenance step, never an unsafe control action).
            supporting_evidence: Short statements of the evidence supporting
                this conclusion.
            retrieved_incident_ids: IDs of past incidents that informed this
                conclusion, if any.
            manual_sources: Section titles from the manual that informed
                this conclusion, if any.

        Returns:
            dict: The structured investigation result.
        """
        result = InvestigationResult(
            likely_cause=likely_cause,
            confidence=confidence,
            recommendation=recommendation,
            supporting_evidence=supporting_evidence,
            retrieved_incident_ids=retrieved_incident_ids,
            manual_sources=manual_sources,
        )
        result_dict = result.model_dump()
        self.state.final_result = result_dict
        self._log_call(
            "finish_investigation",
            {
                "likely_cause": likely_cause,
                "confidence": confidence,
                "recommendation": recommendation,
                "supporting_evidence": supporting_evidence,
                "retrieved_incident_ids": retrieved_incident_ids,
                "manual_sources": manual_sources,
            },
            str(result_dict),
        )
        return result_dict

    def tool_list(self) -> list[Callable[..., Any]]:
        """Return the bound tool methods this agent is allowed to call."""
        return [
            self.get_current_readings,
            self.get_recent_history,
            self.search_manual,
            self.search_past_incidents,
            self.save_finding,
            self.finish_investigation,
        ]
