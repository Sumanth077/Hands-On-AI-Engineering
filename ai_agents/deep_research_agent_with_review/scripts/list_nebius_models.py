"""List models available to the Token Factory key in .env without invoking a model."""

import sys
from pathlib import Path

# Running a script directly puts its own directory, not the project root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.config import load_settings
from research_agent.nebius import ModelDiscoveryError, list_nebius_models


def main() -> int:
    settings = load_settings()
    try:
        models = list_nebius_models(settings)
    except ModelDiscoveryError as error:
        print(f"Model discovery failed: {error}")
        return 1

    nemotron_models = [model for model in models if "nemotron" in model.lower()]
    print("Models available to this API key:")
    for model in models:
        print(f"- {model}")
    print("\nNemotron matches:")
    if nemotron_models:
        for model in nemotron_models:
            print(f"- {model}")
    else:
        print("- None returned. Ask the teammate who created the key to confirm its model access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

