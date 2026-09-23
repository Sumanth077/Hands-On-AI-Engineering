import json
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.state import InvestigationState  # noqa: E402
from agent.tools import AgentTools  # noqa: E402
from simulator.machine import Machine  # noqa: E402
from simulator.scenarios import FAULT_TARGETS, trigger_fault  # noqa: E402


def main() -> None:
    print("Setting up simulated bearing fault...")
    machine = Machine(seed=42)
    trigger_fault(machine, "bearing")
    machine.advance_simulation(steps=FAULT_TARGETS["bearing"].total_steps + 2)

    state = InvestigationState(
        investigation_id=str(uuid.uuid4()),
        objective="Investigate bearing fault",
    )
    tools = AgentTools(machine=machine, history=machine.history, state=state)

    print("\n--- get_current_readings ---")
    readings = tools.get_current_readings()
    print(readings)

    print("\n--- get_recent_history(limit=10) ---")
    history = tools.get_recent_history(limit=10)
    print(f"{len(history)} readings returned")

    print("\n--- search_manual('bearing') ---")
    manual_results = tools.search_manual("bearing")
    for result in manual_results:
        print(f"  score={result['score']:.4f}  section_title={result['section_title']!r}")

    print("\n--- search_past_incidents('bearing overheating') ---")
    incident_results = tools.search_past_incidents("bearing overheating")
    for result in incident_results:
        print(f"  score={result['score']:.4f}  confirmed_cause={result.get('confirmed_cause')!r}")

    print("\n--- save_finding(...) ---")
    finding_confirmation = tools.save_finding(
        "vibration and temperature both elevated, pressure normal",
        f"current readings: {readings}",
    )
    print(finding_confirmation)

    print("\n--- finish_investigation(...) ---")
    final_result = tools.finish_investigation(
        likely_cause="worn bearing",
        confidence=0.75,
        recommendation="Inspect the bearing assembly and lubrication condition.",
        supporting_evidence=[
            "vibration and temperature both elevated, pressure normal",
        ],
        retrieved_incident_ids=[r["id"] for r in incident_results],
        manual_sources=[r["section_title"] for r in manual_results],
    )
    print(final_result)

    print("\n--- final investigation state ---")
    print(json.dumps(state.model_dump(mode="json"), indent=2, default=str))

    ok = (
        len(state.tool_calls) == 6
        and len(state.findings) == 1
        and state.final_result is not None
    )
    print("\nPASS: all six tools ran and state accumulated correctly." if ok else "\nFAIL: state did not accumulate as expected.")


if __name__ == "__main__":
    main()
