import inspect

from agent.prompts import SYSTEM_PROMPT
from agent.state import InvestigationState
from agent.tools import AgentTools
from memory.incident_store import save_incident
from simulator.machine import Machine

# Every public method on Machine that changes its operating state. If a tool
# method's source references any of these, it has actuation capability it
# shouldn't have (per section 25: the agent must never directly control
# machinery).
FORBIDDEN_MACHINE_MUTATORS = {"reset_machine", "start_fault", "advance_simulation", "trigger_fault"}


def _build_tools() -> AgentTools:
    machine = Machine(seed=1)
    state = InvestigationState(
        investigation_id="safety-test",
        objective="test",
        current_readings=machine.get_current_readings().model_dump(mode="json"),
    )
    return AgentTools(machine=machine, history=machine.history, state=state)


def test_tool_methods_never_reference_machine_mutating_operations():
    tools = _build_tools()
    for fn in tools.tool_list():
        source = inspect.getsource(fn)
        for forbidden in FORBIDDEN_MACHINE_MUTATORS:
            assert forbidden not in source, (
                f"{fn.__name__} references '{forbidden}', a machine-mutating operation — "
                "the tool surface must have zero actuation capability, by design."
            )


def test_save_incident_is_never_in_tool_list():
    tools = _build_tools()
    tool_names = {fn.__name__ for fn in tools.tool_list()}
    assert "save_incident" not in tool_names
    assert save_incident not in tools.tool_list()


def test_system_prompt_contains_required_safety_language():
    prompt = SYSTEM_PROMPT.lower()

    assert "simulation" in prompt or "demo" in prompt, (
        "prompt must frame this as a simulation/demo assistant"
    )
    assert "not a certified" in prompt, (
        "prompt must explicitly disclaim being a certified safety system"
    )

    assert "bypass" in prompt and "interlock" in prompt, (
        "prompt must instruct the model to never bypass interlocks"
    )
    assert "bypass" in prompt and ("safety mechanism" in prompt or "alarm" in prompt), (
        "prompt must instruct the model to never disable alarms / bypass safety mechanisms"
    )

    assert "directly control" in prompt and "machin" in prompt, (
        "prompt must state the agent never directly controls machinery"
    )

    assert "inspect" in prompt, "prompt must recommend inspection as the safe alternative"
    assert "unsafe" in prompt, "prompt must warn against unsafe control actions"
