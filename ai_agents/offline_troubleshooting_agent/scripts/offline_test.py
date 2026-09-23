"""Section 19 Part 3 — THE OFFLINE TEST.

Run this AFTER disconnecting the network, in its own terminal, with no
further involvement needed. It is fully self-contained:

    python scripts/offline_test.py

It first guards against accidentally running "offline" while still online
(a real network connection would silently invalidate the whole test), then
triggers a bearing fault, runs a real investigation, and checks whether the
reference incident saved during the online setup phase
(data/offline_test_baseline.json) is recalled from local memory with zero
internet access. Results are written to a timestamped file under docs/ so
they survive even if the terminal closes before reconnecting.
"""

import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.network_status import NETWORK_CHECK_HOST, NETWORK_CHECK_PORT, is_online  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(PROJECT_ROOT, "data", "offline_test_baseline.json")

NETWORK_GUARD_TIMEOUT = 2.0

SEARCH_QUERY = "bearing fault high vibration high temperature BRG-02"


class _Log:
    """Prints each line immediately and buffers it for the markdown report."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)


log = _Log()


def _write_log_file(status: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = os.path.join(PROJECT_ROOT, "docs", f"OFFLINE_TEST_LOG_{timestamp}.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Offline Test Log — {status}\n\n")
        fh.write(f"Run at: {datetime.now().isoformat()}\n\n")
        fh.write("```\n")
        fh.write("\n".join(log.lines))
        fh.write("\n```\n")
    return log_path


def main() -> None:
    overall_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Network-reachability guard
    # ------------------------------------------------------------------
    log("=" * 70)
    log("OFFLINE TEST — network guard")
    log("=" * 70)
    log(f"Checking whether {NETWORK_CHECK_HOST}:{NETWORK_CHECK_PORT} is reachable...")

    if is_online(timeout=NETWORK_GUARD_TIMEOUT):
        log("")
        log("!" * 70)
        log("STILL ONLINE — this does not look disconnected.")
        log(f"A TCP connection to {NETWORK_CHECK_HOST}:{NETWORK_CHECK_PORT} succeeded, ")
        log("which means the network is still up. The offline test's whole point is")
        log("proving this works with ZERO internet access, so running it now would")
        log("prove nothing. Disconnect Wi-Fi / unplug the network, then re-run this")
        log("script.")
        log("!" * 70)
        log_path = _write_log_file("ABORTED — still online")
        log(f"\n(Full log written to {log_path})")
        sys.exit(1)

    log(f"OK — {NETWORK_CHECK_HOST}:{NETWORK_CHECK_PORT} unreachable within "
        f"{NETWORK_GUARD_TIMEOUT}s. Proceeding as offline.\n")

    # These are imported only after the guard passes, so a failed guard exits
    # fast without needing local services to be initialized.
    from agent.harness import run_investigation
    from agent.llm_client import health_check as llm_health_check
    from memory.embeddings import embedding_health_check
    from memory.incident_store import search_incidents
    from memory.vectorai_client import health_check as vectorai_health_check
    from simulator.machine import Machine
    from simulator.scenarios import FAULT_TARGETS, trigger_fault

    # ------------------------------------------------------------------
    # 2. Local service health checks
    # ------------------------------------------------------------------
    log("=" * 70)
    log("Local service health checks")
    log("=" * 70)

    llm_status = llm_health_check()
    log(f"Ollama / LLM:        {llm_status}")

    embed_status = embedding_health_check()
    log(f"Embedding model:     {embed_status}")

    vectorai_status = vectorai_health_check()
    log(f"Actian VectorAI DB:  {vectorai_status}")
    log("")

    # ------------------------------------------------------------------
    # 3. Trigger the same bearing fault
    # ------------------------------------------------------------------
    log("=" * 70)
    log("Triggering bearing fault")
    log("=" * 70)
    machine = Machine(seed=7)
    machine.reset_machine()
    trigger_fault(machine, "bearing")
    machine.advance_simulation(steps=FAULT_TARGETS["bearing"].total_steps + 2)
    log(f"Current readings: {machine.get_current_readings()}\n")

    # ------------------------------------------------------------------
    # 4. Real investigation
    # ------------------------------------------------------------------
    log("=" * 70)
    log("Running investigation")
    log("=" * 70)

    investigation_start = time.monotonic()
    final_state = None
    round_num = 0
    last_count = 0
    for state in run_investigation(machine, machine.history, "Investigate bearing fault on Machine A"):
        final_state = state
        if len(state.tool_calls) != last_count:
            last_count = len(state.tool_calls)
            round_num += 1
            last_call = state.tool_calls[-1]["name"] if state.tool_calls else "(none yet)"
            log(f"  [{round_num}] tool_calls so far: {last_count}  |  most recent: {last_call}")
    investigation_elapsed = time.monotonic() - investigation_start

    reached_finish = final_state is not None and final_state.final_result is not None and not any(
        error in ("model_ended_without_finish_investigation", "max_rounds_reached")
        for error in final_state.errors
    )
    log(f"\nfinal_result: {final_state.final_result if final_state else None}")
    log(f"errors: {final_state.errors if final_state else 'N/A'}")
    log(f"\nReached finish_investigation: {'Y' if reached_finish else 'N'}")
    log(f"Investigation wall-clock time: {investigation_elapsed:.2f}s ({investigation_elapsed / 60:.1f} min)\n")

    # ------------------------------------------------------------------
    # 5. Check whether the offline-test reference incident is recalled
    # ------------------------------------------------------------------
    log("=" * 70)
    log("Checking recall of the offline-test reference incident")
    log("=" * 70)

    baseline_id = None
    baseline_cause = None
    if not os.path.isfile(BASELINE_PATH):
        log(f"FAIL: baseline file not found at {BASELINE_PATH}.")
        log("(Part A's online setup must run first, while still connected.)")
        recalled = False
        found_ids: list[str] = []
    else:
        with open(BASELINE_PATH, encoding="utf-8") as fh:
            baseline = json.load(fh)
        baseline_id = baseline["incident_id"]
        baseline_cause = baseline.get("confirmed_cause")
        log(f"Baseline reference incident_id: {baseline_id}")
        log(f"Baseline confirmed_cause: {baseline_cause!r}")

        search_results = search_incidents(SEARCH_QUERY, top_k=5)
        found_ids = [item.get("id") for item in search_results]
        log(f"search_incidents({SEARCH_QUERY!r}) returned ids: {found_ids}")

        recalled = baseline_id in found_ids
        if recalled:
            log(f"\nPASS: reference incident {baseline_id} was recalled by exact id match.")
        else:
            log(f"\nFAIL: reference incident {baseline_id} was NOT among the returned ids.")

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.monotonic() - overall_start

    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Reached finish_investigation:      {'Y' if reached_finish else 'N'}")
    log(
        f"Correct prior incident recalled:   {'Y' if recalled else 'N'}"
        + (f" (id={baseline_id})" if baseline_id else " (no baseline to compare)")
    )
    log(f"state.errors:                      {final_state.errors if final_state else 'N/A'}")
    log(f"Total wall-clock time:              {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)")

    overall_pass = reached_finish and recalled
    log(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    log_path = _write_log_file("PASS" if overall_pass else "FAIL")
    log(f"\n(Full log written to {log_path})")


if __name__ == "__main__":
    main()
