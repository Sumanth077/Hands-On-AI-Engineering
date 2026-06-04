"""Quick check that Orq.ai is reachable and all three debate models respond.

Run from the project folder (with venv activated):

    py test_orq.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

MODELS = [
    ("Debater A", os.getenv("DEBATER_A_MODEL", "google-ai/gemini-3-flash-preview")),
    ("Debater B", os.getenv("DEBATER_B_MODEL", "mistral/mistral-small-latest")),
    ("Judge", os.getenv("JUDGE_MODEL", "moonshotai/kimi-k2.6")),
]


def main() -> int:
    api_key = os.getenv("ORQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-orq"):
        print("FAIL: Set a real ORQ_API_KEY in .env (copy from .env.example if needed).")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: Install dependencies: pip install -r requirements.txt")
        return 1

    client = OpenAI(
        base_url="https://api.orq.ai/v3/router",
        api_key=api_key,
    )

    print("Testing Orq.ai router at https://api.orq.ai/v3/router\n")
    failures = 0

    for label, model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                max_tokens=10,
            )
            text = (response.choices[0].message.content or "").strip()
            print(f"OK   {label}: {model}")
            print(f"     -> {text[:80]}\n")
        except Exception as exc:
            failures += 1
            detail = str(exc).strip() or type(exc).__name__
            if exc.__cause__ and str(exc.__cause__) not in detail:
                detail = f"{detail} — {exc.__cause__}"
            print(f"FAIL {label}: {model}")
            print(f"     -> {type(exc).__name__}: {detail}\n")

    if failures:
        print(f"{failures} model(s) failed. Check ORQ_API_KEY and AI Router → Providers in Orq.")
        return 1

    print("All three models responded. You can run: py app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
