import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memory.incident_store import _configured_incident_collection, save_incident  # noqa: E402
from memory.manual import ingest_manual_documents  # noqa: E402
from memory.vectorai_client import delete_collection, ensure_collection  # noqa: E402
from models.schemas import Incident  # noqa: E402

MANUAL_PATHS = [
    os.path.join(PROJECT_ROOT, "documents", "machine_manual.md"),
    os.path.join(PROJECT_ROOT, "documents", "troubleshooting_guide.md"),
]
SEED_INCIDENTS_PATH = os.path.join(PROJECT_ROOT, "data", "seed_incidents.json")
INCIDENT_EMBEDDING_DIM = 768


def load_seed_incidents(path: str = SEED_INCIDENTS_PATH) -> list[Incident]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Incident.model_validate(item) for item in raw]


def clear_incidents_collection(collection: str | None = None) -> None:
    target_collection = collection or _configured_incident_collection()
    delete_collection(target_collection)
    ensure_collection(target_collection, dim=INCIDENT_EMBEDDING_DIM)


def seed_incidents(collection: str | None = None) -> int:
    incidents = load_seed_incidents()
    for incident in incidents:
        save_incident(incident, collection=collection)
    return len(incidents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed local memory (manual chunks + optionally demo incidents)."
    )
    parser.add_argument(
        "--mode",
        choices=["fresh", "seeded"],
        default="fresh",
        help="'fresh': manual only, incidents collection cleared to empty. "
        "'seeded': manual plus demo incidents from data/seed_incidents.json.",
    )
    args = parser.parse_args()

    print(f"Seeding manual memory (mode={args.mode})...")
    manual_count = ingest_manual_documents(MANUAL_PATHS)
    print(f"manual chunks ingested: {manual_count}")

    if args.mode == "fresh":
        print("\nClearing incidents collection for a fresh (zero-incident) state...")
        clear_incidents_collection()
        print("incidents collection cleared.")

        if manual_count > 0:
            print("\nPASS: manual documents ingested, incidents collection is empty.")
        else:
            print("\nFAIL: no manual chunks were ingested.")
        return

    print("\nSeeding demo incidents...")
    incident_count = seed_incidents()
    print(f"incidents seeded: {incident_count}")

    if manual_count > 0 and incident_count > 0:
        print("\nPASS: manual documents and demo incidents seeded successfully.")
    else:
        print("\nFAIL: seeding did not complete as expected.")


if __name__ == "__main__":
    main()
