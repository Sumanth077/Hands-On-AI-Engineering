import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

from seed_memory import clear_incidents_collection, seed_incidents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset the local Actian memory layer (incidents collection only)."
    )
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="After clearing, reload data/seed_incidents.json into the incidents collection.",
    )
    args = parser.parse_args()

    print("Clearing incidents collection...")
    clear_incidents_collection()
    print("incidents collection cleared.")

    if args.reseed:
        print("Reseeding demo incidents...")
        count = seed_incidents()
        print(f"incidents seeded: {count}")
    else:
        print("No reseed requested — incidents collection left empty.")

    print(
        "\nNote: this only resets the Actian memory layer. It does not touch the running "
        "simulator's in-memory machine state — use the Gradio app's 'Reset Machine' button for that."
    )


if __name__ == "__main__":
    main()
