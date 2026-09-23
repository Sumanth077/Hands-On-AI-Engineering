import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm_client import generate_reply, health_check  # noqa: E402


def main() -> None:
    print("Checking local Ollama setup...")
    result = health_check()

    print(f"reachable:       {result['reachable']}")
    print(f"model_available: {result['model_available']}")
    print(f"local_models:    {result['local_models']}")

    if not result["reachable"]:
        print("\nFAIL: could not reach the local Ollama server.")
        return

    if not result["model_available"]:
        print("\nFAIL: configured OLLAMA_MODEL is not pulled locally yet.")
        return

    reply = generate_reply("Say OK if you can read this.")
    print(f"\nmodel reply: {reply!r}")
    print("\nPASS: Ollama is reachable, the model is available, and it responded.")


if __name__ == "__main__":
    main()
