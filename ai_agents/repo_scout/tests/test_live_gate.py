"""The default gate protects both the legacy client and the optional model backend."""

import pytest
from pydantic import SecretStr

from repo_scout.apify_client import ApifyClient
from repo_scout.config import Settings
from repo_scout.llm import OpenAICompatibleBackend
from repo_scout.models import FetchInput
from repo_scout.research_models import Limits


async def test_clients_disabled_without_explicit_switch(monkeypatch):
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "false")
    async with ApifyClient(Settings(token=SecretStr("synthetic"))) as api:
        with pytest.raises(ValueError, match="disabled"):
            await api.fetch(FetchInput(url="https://example.com"))
    backend = OpenAICompatibleBackend(
        "https://example.com/completions", "explicit-test", SecretStr("synthetic")
    )
    with pytest.raises(ValueError, match="disabled"):
        await backend.generate("interpret", {}, {}, Limits())
