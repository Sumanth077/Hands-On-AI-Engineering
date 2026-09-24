import uuid
from datetime import datetime

from agent.state import InvestigationState
from models.schemas import Incident
from simulator.machine import PRESSURE_RANGE, TEMP_RANGE, VIBRATION_RANGE

DEFAULT_MACHINE_ID = "Machine A"


def _derive_symptoms(current_readings: dict) -> list[str]:
    """Deterministically label out-of-range readings against the simulator's
    baseline ranges — not something to ask the model for."""
    symptoms: list[str] = []

    temperature_c = current_readings.get("temperature_c")
    if temperature_c is not None:
        if temperature_c > TEMP_RANGE[1]:
            symptoms.append("elevated temperature")
        elif temperature_c < TEMP_RANGE[0]:
            symptoms.append("low temperature")

    vibration_mm_s = current_readings.get("vibration_mm_s")
    if vibration_mm_s is not None:
        if vibration_mm_s > VIBRATION_RANGE[1]:
            symptoms.append("elevated vibration")
        elif vibration_mm_s < VIBRATION_RANGE[0]:
            symptoms.append("low vibration")

    pressure_bar = current_readings.get("pressure_bar")
    if pressure_bar is not None:
        if pressure_bar > PRESSURE_RANGE[1]:
            symptoms.append("elevated pressure")
        elif pressure_bar < PRESSURE_RANGE[0]:
            symptoms.append("low pressure")

    return symptoms


def build_default_review_fields(state: InvestigationState, current_readings: dict) -> dict:
    """Pre-fill values for the human review form.

    These are a starting point for a technician to edit, never something
    auto-submitted — the model's likely_cause only becomes confirmed_cause
    if a human leaves it unchanged and explicitly clicks save.
    """
    result = state.final_result or {}
    return {
        "confirmed_cause": result.get("likely_cause", ""),
        "fix_applied": result.get("recommendation", ""),
        "outcome": "resolved",
        "notes": "\n".join(result.get("supporting_evidence") or []),
        "symptoms": _derive_symptoms(current_readings),
    }


def build_incident_from_review(
    state: InvestigationState,
    current_readings: dict,
    confirmed_cause: str,
    fix_applied: str,
    outcome: str,
    notes: str,
    symptoms: list[str],
) -> Incident:
    """Build the Incident that will actually be written to durable memory.

    `diagnosis` is preserved as the model's original likely_cause for the
    record; `confirmed_cause`/`fix_applied`/`outcome`/`notes` come from the
    human-edited form fields, which may differ from what the model
    suggested — that distinction is the whole point of the review step.
    """
    result = state.final_result or {}
    return Incident(
        incident_id=str(uuid.uuid4()),
        created_at=datetime.now(),
        machine_id=DEFAULT_MACHINE_ID,
        symptoms=symptoms,
        temperature_c=current_readings.get("temperature_c"),
        vibration_mm_s=current_readings.get("vibration_mm_s"),
        pressure_bar=current_readings.get("pressure_bar"),
        error_code=current_readings.get("error_code"),
        diagnosis=result.get("likely_cause", ""),
        confirmed_cause=confirmed_cause,
        fix_applied=fix_applied,
        outcome=outcome,
        notes=notes,
    )
