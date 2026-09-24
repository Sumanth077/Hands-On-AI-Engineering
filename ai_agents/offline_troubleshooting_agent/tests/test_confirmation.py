from agent.confirmation import build_default_review_fields, build_incident_from_review
from agent.state import InvestigationState


def _state_with_result(**result_overrides) -> InvestigationState:
    result = dict(
        likely_cause="Worn bearing",
        confidence=0.9,
        recommendation="Inspect the bearing assembly.",
        supporting_evidence=["vibration elevated", "temperature elevated"],
        retrieved_incident_ids=[],
        manual_sources=[],
    )
    result.update(result_overrides)
    return InvestigationState(
        investigation_id="inv-1",
        objective="Investigate bearing fault",
        final_result=result,
    )


def test_build_default_review_fields_flags_bearing_fault_shape():
    state = _state_with_result()
    current_readings = {
        "temperature_c": 92.0,
        "vibration_mm_s": 8.1,
        "pressure_bar": 5.0,
        "error_code": "BRG-02",
    }

    fields = build_default_review_fields(state, current_readings)

    assert fields["confirmed_cause"] == "Worn bearing"
    assert fields["fix_applied"] == "Inspect the bearing assembly."
    assert fields["outcome"] == "resolved"
    assert "vibration elevated" in fields["notes"]
    assert "temperature elevated" in fields["notes"]
    assert set(fields["symptoms"]) == {"elevated temperature", "elevated vibration"}
    assert "low pressure" not in fields["symptoms"]
    assert "elevated pressure" not in fields["symptoms"]


def test_build_default_review_fields_flags_pressure_fault_shape():
    state = _state_with_result(likely_cause="Pressure leak")
    current_readings = {
        "temperature_c": 77.0,
        "vibration_mm_s": 2.4,
        "pressure_bar": 2.8,
        "error_code": "PRS-03",
    }

    fields = build_default_review_fields(state, current_readings)

    # 77.0C is mildly above the 65-75C baseline too, per the pressure-loss
    # scenario in the project reference — both symptoms are expected.
    assert set(fields["symptoms"]) == {"low pressure", "elevated temperature"}


def test_build_default_review_fields_normal_readings_have_no_symptoms():
    state = _state_with_result()
    current_readings = {
        "temperature_c": 70.0,
        "vibration_mm_s": 2.0,
        "pressure_bar": 5.0,
        "error_code": None,
    }

    fields = build_default_review_fields(state, current_readings)

    assert fields["symptoms"] == []


def test_build_incident_from_review_preserves_model_diagnosis_and_human_correction():
    # This is the critical safety guarantee: a human correcting the model's
    # guess must never overwrite the record of what the model actually said.
    state = _state_with_result(likely_cause="Worn bearing")
    current_readings = {
        "temperature_c": 92.0,
        "vibration_mm_s": 8.1,
        "pressure_bar": 5.0,
        "error_code": "BRG-02",
    }

    incident = build_incident_from_review(
        state,
        current_readings,
        confirmed_cause="Loose access panel (not actually a bearing issue)",
        fix_applied="Refastened the access panel.",
        outcome="resolved",
        notes="Technician found the vibration was from a loose panel, not the bearing.",
        symptoms=["elevated vibration"],
    )

    assert incident.diagnosis == "Worn bearing"
    assert incident.confirmed_cause == "Loose access panel (not actually a bearing issue)"
    assert incident.diagnosis != incident.confirmed_cause
    assert incident.fix_applied == "Refastened the access panel."
    assert incident.temperature_c == 92.0
    assert incident.vibration_mm_s == 8.1
    assert incident.pressure_bar == 5.0
    assert incident.error_code == "BRG-02"
    assert incident.machine_id == "Machine A"


def test_build_incident_from_review_keeps_confirmed_cause_when_human_agrees():
    state = _state_with_result(likely_cause="Worn bearing")
    current_readings = {"temperature_c": 92.0, "vibration_mm_s": 8.1, "pressure_bar": 5.0}

    incident = build_incident_from_review(
        state,
        current_readings,
        confirmed_cause="Worn bearing",
        fix_applied="Bearing replaced.",
        outcome="resolved",
        notes="",
        symptoms=["elevated vibration", "elevated temperature"],
    )

    assert incident.diagnosis == "Worn bearing"
    assert incident.confirmed_cause == "Worn bearing"
