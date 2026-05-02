import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage


class LLMClient:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in environment")
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=api_key,
            max_tokens=2048,
        )
        self.llm_json = ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=api_key,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

    def complete(self, prompt: str, json_mode: bool = False, max_retries: int = 3) -> str:
        client = self.llm_json if json_mode else self.llm
        messages = [HumanMessage(content=prompt)]
        for attempt in range(max_retries):
            try:
                response = client.invoke(messages)
                return response.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Mistral API error after {max_retries} attempts: {e}")

    def complete_with_system(
        self, system: str, user: str, json_mode: bool = False
    ) -> str:
        client = self.llm_json if json_mode else self.llm
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = client.invoke(messages)
        return response.content
