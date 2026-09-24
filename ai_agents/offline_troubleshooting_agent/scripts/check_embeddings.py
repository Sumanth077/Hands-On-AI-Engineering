import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.embeddings import embed_text, embedding_health_check  # noqa: E402


def main() -> None:
    print("Checking local embedding model setup...")
    result = embedding_health_check()

    print(f"reachable:       {result['reachable']}")
    print(f"model_available: {result['model_available']}")
    print(f"local_models:    {result['local_models']}")

    if not result["reachable"]:
        print("\nFAIL: could not reach the local Ollama server.")
        return

    if not result["model_available"]:
        print("\nFAIL: configured EMBEDDING_MODEL is not pulled locally yet.")
        return

    vector = embed_text("bearing overheating vibration")
    print(f"\nvector length: {len(vector)}")
    print(f"first 5 values: {vector[:5]}")
    print("\nPASS: embedding model is reachable, available, and generated a vector.")


if __name__ == "__main__":
    main()
