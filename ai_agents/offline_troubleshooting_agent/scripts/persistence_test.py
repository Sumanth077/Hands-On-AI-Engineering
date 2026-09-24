"""Persistence test (section 22): prove a genuinely NEW incident survives a
real Actian VectorAI DB container restart, verified from a fresh process.

Usage:
    python scripts/persistence_test.py --mode save
    # ... restart the offline-troubleshooting-vectorai container ...
    python scripts/persistence_test.py --mode verify
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.incident_store import get_incident, save_incident, search_incidents  # noqa: E402
from models.schemas import Incident  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(PROJECT_ROOT, "data", "persistence_test_baseline.json")

MARKER_INCIDENT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "persistence-test-marker"))
MARKER_CONFIRMED_CAUSE = "PERSISTENCE TEST MARKER - safe to ignore/delete"
SEARCH_QUERY = "persistence test marker safe to ignore"


def _build_marker_incident() -> Incident:
    return Incident(
        incident_id=MARKER_INCIDENT_ID,
        created_at=datetime.now(),
        machine_id="Machine A",
        symptoms=["persistence test marker"],
        temperature_c=999.9,
        vibration_mm_s=999.9,
        pressure_bar=999.9,
        error_code="TEST-00",
        diagnosis="Not a real diagnosis — written by scripts/persistence_test.py.",
        confirmed_cause=MARKER_CONFIRMED_CAUSE,
        fix_applied="N/A — persistence test only, no real fix.",
        outcome="persistence-test",
        notes=(
            "Written by scripts/persistence_test.py --mode save to verify Actian "
            "VectorAI DB data survives a container restart (section 22)."
        ),
    )


def run_save() -> None:
    incident = _build_marker_incident()
    save_incident(incident)

    baseline = incident.model_dump(mode="json")
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)

    print(f"Saved persistence-test marker incident: {incident.incident_id}")
    print("Field values written:")
    for key, value in baseline.items():
        print(f"  {key}: {value!r}")
    print(f"\nBaseline saved to: {BASELINE_PATH}")
    print(
        "\nNow restart the Actian VectorAI DB container, e.g.:\n"
        "  docker compose restart offline-troubleshooting-vectorai\n"
        "then run: python scripts/persistence_test.py --mode verify"
    )


def _diff_fields(baseline: dict, retrieved: dict) -> list[str]:
    mismatches = []
    all_keys = sorted(set(baseline.keys()) | set(retrieved.keys()))
    for key in all_keys:
        base_val = baseline.get(key, "<missing>")
        retr_val = retrieved.get(key, "<missing>")
        if base_val != retr_val:
            mismatches.append(f"  {key}: expected {base_val!r}, got {retr_val!r}")
    return mismatches


def run_verify() -> None:
    if not os.path.isfile(BASELINE_PATH):
        print(f"FAIL: no baseline file found at {BASELINE_PATH}.")
        print("Run `python scripts/persistence_test.py --mode save` first.")
        sys.exit(1)

    with open(BASELINE_PATH, encoding="utf-8") as fh:
        baseline = json.load(fh)

    print(f"Loaded baseline for incident_id={baseline['incident_id']}")

    failures: list[str] = []

    print("\n--- get_incident() ---")
    retrieved = get_incident(baseline["incident_id"])
    if retrieved is None:
        print("FAIL: get_incident() returned None — record not found.")
        failures.append("get_incident() returned None")
    else:
        retrieved_dump = retrieved.model_dump(mode="json")
        mismatches = _diff_fields(baseline, retrieved_dump)
        if mismatches:
            print("FAIL: field mismatch(es) between baseline and retrieved record:")
            for line in mismatches:
                print(line)
            failures.append(f"{len(mismatches)} field mismatch(es) via get_incident()")
        else:
            print("PASS: every field matches the saved baseline exactly.")

    print("\n--- search_incidents() ---")
    search_results = search_incidents(SEARCH_QUERY, top_k=5)
    found_ids = [item.get("id") for item in search_results]
    if baseline["incident_id"] in found_ids:
        print(f"PASS: marker incident found via search_incidents() (query={SEARCH_QUERY!r}).")
        print(f"  matched ids returned: {found_ids}")
    else:
        print(f"FAIL: marker incident NOT found via search_incidents() (query={SEARCH_QUERY!r}).")
        print(f"  ids returned instead: {found_ids}")
        failures.append("marker incident not found via search_incidents()")

    print("\n" + "=" * 60)
    if not failures:
        print("PASS: persistence test succeeded — the marker incident survived")
        print("the container restart, verified from this fresh process, via")
        print("both get_incident() and search_incidents().")
    else:
        print("FAIL: persistence test did not fully pass. Issues:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["save", "verify"], required=True)
    args = parser.parse_args()

    if args.mode == "save":
        run_save()
    else:
        run_verify()


if __name__ == "__main__":
    main()
