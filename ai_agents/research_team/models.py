import os
from agno.models.openai.like import OpenAILike

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "minimax/minimax-m2.5:free"


def get_model() -> OpenAILike:
    return OpenAILike(
        id=MODEL_ID,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
    )
