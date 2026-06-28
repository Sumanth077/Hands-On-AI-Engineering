import os
from agno.models.openai.like import OpenAILike

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "minimax/minimax-m2.5:free"


def get_model() -> OpenAILike:
<<<<<<< HEAD
=======
    """Return a configured OpenAILike model instance pointed at MiniMax M2.5 via OpenRouter."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    return OpenAILike(
        id=MODEL_ID,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
    )
