import uuid

import pytest

from agent.state import InvestigationState
from agent.tools import AgentTools
from memory.embeddings import embedding_health_check
from memory.vectorai_client import health_check as vectorai_health_check
from simulator.machine import Machine


def _services_reachable() -> bool:
    try:
        embedding_status = embedding_health_check()
        vectorai_status = vectorai_health_check()
    except Exception:
        return False

    return bool(
        embedding_status.get("reachable")
        and embedding_status.get("model_available")
        and vectorai_status.get("reachable")
    )


SERVICES_REACHABLE = _services_reachable()


def _build_tools(seed: int = 1) -> AgentTools:
    machine = Machine(seed=seed)
    state = InvestigationState(
        investigation_id=str(uuid.uuid4()), objective="Investigate test fault"
    )
    return AgentTools(machine=machine, history=machine.history, state=state)


def test_get_current_readings_returns_dict_and_logs_call():
    tools = _build_tools()

    result = tools.get_current_readings()

    assert isinstance(result, dict)
    expected_keys = {
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "status",
        "error_code",
        "timestamp",
    }
    assert expected_keys <= set(result.keys())
    assert len(tools.state.tool_calls) == 1
    assert tools.state.tool_calls[0]["name"] == "get_current_readings"


def test_get_recent_history_returns_list_of_dicts_and_logs_call():
    tools = _build_tools()
    tools.machine.advance_simulation(steps=5)

    result = tools.get_recent_history(limit=3)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(item, dict) for item in result)
    assert len(tools.state.tool_calls) == 1
    assert tools.state.tool_calls[0]["name"] == "get_recent_history"
    assert tools.state.tool_calls[0]["args"] == {"limit": 3}


def test_save_finding_appends_to_findings_and_logs_call():
    tools = _build_tools()

    confirmation = tools.save_finding("vibration is elevated", "vibration reading of 8 mm/s")

    assert confirmation == "Finding recorded."
    assert tools.state.findings == [
        "vibration is elevated (evidence: vibration reading of 8 mm/s)"
    ]
    assert len(tools.state.tool_calls) == 1
    assert tools.state.tool_calls[0]["name"] == "save_finding"


def test_finish_investigation_stores_final_result_and_logs_call():
    tools = _build_tools()

    result = tools.finish_investigation(
        likely_cause="worn bearing",
        confidence=0.8,
        recommendation="Inspect the bearing assembly.",
        supporting_evidence=["vibration is elevated"],
        retrieved_incident_ids=[],
        manual_sources=["BRG-02 — Bearing vibration warning"],
    )

    assert result["likely_cause"] == "worn bearing"
    assert result["confidence"] == 0.8
    assert tools.state.final_result == result
    assert len(tools.state.tool_calls) == 1
    assert tools.state.tool_calls[0]["name"] == "finish_investigation"


def test_tool_list_does_not_include_save_incident():
    tools = _build_tools()

    names = [tool.__name__ for tool in tools.tool_list()]

    assert "save_incident" not in names
    assert names == [
        "get_current_readings",
        "get_recent_history",
        "search_manual",
        "search_past_incidents",
        "save_finding",
        "finish_investigation",
    ]


@pytest.mark.skipif(
    not SERVICES_REACHABLE,
    reason="Requires a live local embedding model and a live Actian VectorAI instance.",
)
def test_search_manual_populates_retrieved_manual_chunks():
    tools = _build_tools()

    results = tools.search_manual("bearing vibration overheating", top_k=3)

    assert results
    assert tools.state.retrieved_manual_chunks == results
    assert len(tools.state.tool_calls) == 1
    assert tools.state.tool_calls[0]["name"] == "search_manual"


@pytest.mark.skipif(
    not SERVICES_REACHABLE,
    reason="Requires a live local embedding model and a live Actian VectorAI instance.",
)
def test_search_past_incidents_populates_retrieved_incidents():
    tools = _build_tools()

    results = tools.search_past_incidents("high temperature and high vibration", top_k=3)

    assert tools.state.retrieved_incidents == results
    assert len(tools.state.tool_calls) == 1
    assert tools.state.tool_calls[0]["name"] == "search_past_incidents"
