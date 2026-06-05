"""Run: python scripts/test_nvidia.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvidia_client import MODEL_ID, chat_completion, get_api_key, verify_connection


def main() -> None:
    key = get_api_key()
    print(f"API key loaded: {key[:8]}...{key[-4:]}")
    print(f"Model: {MODEL_ID}")
    print("Testing connection...")
    verify_connection()
    print("verify_connection() OK")
    reply = chat_completion(
        [{"role": "user", "content": "Say hello in 5 words or fewer."}],
        max_tokens=32,
    )
    print(f"Reply: {reply}")


if __name__ == "__main__":
    main()
