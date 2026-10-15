from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str
    nebius_base_url: str
    nebius_model: str
    tavily_api_key: str
    tavily_search_depth: str
    langsmith_api_key: str
    langsmith_project: str
    ai_gateway_api_key: str
    jev_base_url: str
    jev_model: str

    @property
    def missing_required(self) -> list[str]:
        values = {
            "NEBIUS_API_KEY": self.nebius_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            "AI_GATEWAY_API_KEY": self.ai_gateway_api_key,
        }
        return [name for name, value in values.items() if not value]


def load_settings() -> Settings:
    load_dotenv(override=False)
    return Settings(
        nebius_api_key=os.getenv("NEBIUS_API_KEY", "").strip(),
        nebius_base_url=os.getenv(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
        ).rstrip("/"),
        nebius_model=os.getenv(
            "NEBIUS_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b"
        ).strip(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        tavily_search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "advanced").strip(),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
        langsmith_project=os.getenv(
            "LANGSMITH_PROJECT", "deep-research-agent-with-review"
        ).strip(),
        ai_gateway_api_key=os.getenv("AI_GATEWAY_API_KEY", "").strip(),
        jev_base_url=os.getenv("JEV_BASE_URL", "https://ai-gateway.vercel.sh/v1").rstrip(
            "/"
        ),
        jev_model=os.getenv("JEV_MODEL", "typesafe-ai/jev").strip(),
    )


def configure_tracing(settings: Settings) -> None:
    """Set LangSmith's standard environment variables without exposing secrets."""
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

