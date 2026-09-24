import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.harness import run_investigation  # noqa: E402
from agent.llm_client import health_check  # noqa: E402
from simulator.machine import Machine  # noqa: E402
from simulator.scenarios import FAULT_TARGETS, trigger_fault  # noqa: E402


def main() -> None:
    print("Checking Ollama availability before running a real investigation...")
    status = health_check()
    print(f"health_check result: {status}")

    if not status["reachable"]:
        print("\nFAIL: Ollama is not reachable. Is it running?")
        return

    if not status["model_available"]:
        print(
            "\nFAIL: the configured OLLAMA_MODEL is not pulled locally yet. "
            "Run `ollama pull qwen3:4b-instruct` (or your configured model) and try again."
        )
        return

    print("\nSetting up simulated bearing fault...")
    machine = Machine(seed=7)
    trigger_fault(machine, "bearing")
    machine.advance_simulation(steps=FAULT_TARGETS["bearing"].total_steps + 2)

    print("Running investigation...\n")
    final_state = None
    for state in run_investigation(
        machine, machine.history, "Investigate bearing fault on Machine A"
    ):
        final_state = state
        last_call = state.tool_calls[-1]["name"] if state.tool_calls else "(none yet)"
        print(f"  tool_calls so far: {len(state.tool_calls)}  |  most recent: {last_call}")

    print("\n--- final_result ---")
    print(final_state.final_result)

    print("\n--- errors ---")
    print(final_state.errors)

    ok = final_state.final_result is not None and not any(
        error in ("model_ended_without_finish_investigation", "max_rounds_reached")
        for error in final_state.errors
    )
    print(
        "\nPASS: investigation reached a conclusion via finish_investigation."
        if ok
        else "\nFAIL: investigation did not cleanly reach finish_investigation "
        "(see errors above — a fallback result was used)."
    )


if __name__ == "__main__":
    main()
