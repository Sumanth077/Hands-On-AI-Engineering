"""Offline tests must never load real credentials or open real HTTP transports."""

import httpx
import pytest


@pytest.fixture(autouse=True)
def isolate_live_credentials_and_network(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline tests attempted a real network transport")

    async def async_forbidden(*args, **kwargs):
        raise AssertionError("Offline tests attempted a real network transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_forbidden)
