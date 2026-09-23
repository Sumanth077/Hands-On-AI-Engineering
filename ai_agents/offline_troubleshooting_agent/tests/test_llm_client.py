from types import SimpleNamespace

import pytest

import agent.llm_client as llm_client_module
from agent.llm_client import assert_local_only


def test_assert_local_only_raises_for_cloud_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        assert_local_only("https://ollama.com", "qwen3:4b-instruct")


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "gpt-oss-120b-cloud"])
def test_assert_local_only_raises_for_cloud_model(model, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        assert_local_only("http://localhost:11434", model)


def test_assert_local_only_raises_when_api_key_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "some-cloud-key")
    with pytest.raises(RuntimeError):
        assert_local_only("http://localhost:11434", "qwen3:4b-instruct")


def test_assert_local_only_passes_for_normal_local_config(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert_local_only("http://localhost:11434", "qwen3:4b-instruct")


def test_health_check_matches_untagged_model_against_latest_tag(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3")

    fake_models = [SimpleNamespace(model="qwen3:latest")]
    fake_client = SimpleNamespace(list=lambda: SimpleNamespace(models=fake_models))
    monkeypatch.setattr(llm_client_module, "get_client", lambda: fake_client)

    result = llm_client_module.health_check()

    assert result["reachable"] is True
    assert result["model_available"] is True
