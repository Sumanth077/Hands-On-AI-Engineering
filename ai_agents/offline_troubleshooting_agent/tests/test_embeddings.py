from types import SimpleNamespace

import pytest

import memory.embeddings as embeddings_module


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def embed(self, model, input):
        self.calls.append({"model": model, "input": input})
        return self._response


def test_embed_text_returns_plain_list_of_floats(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    fake_client = FakeClient(SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]]))
    monkeypatch.setattr(embeddings_module, "get_client", lambda: fake_client)

    result = embeddings_module.embed_text("bearing overheating vibration")

    assert result == [0.1, 0.2, 0.3]
    assert isinstance(result, list)
    assert all(isinstance(value, float) for value in result)


def test_embed_texts_returns_batch_in_input_order(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    fake_client = FakeClient(SimpleNamespace(embeddings=[[1.0, 2.0], [3.0, 4.0]]))
    monkeypatch.setattr(embeddings_module, "get_client", lambda: fake_client)

    result = embeddings_module.embed_texts(["first symptom", "second symptom"])

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert fake_client.calls[0]["input"] == ["first symptom", "second symptom"]


def test_embed_text_raises_via_assert_local_only_for_cloud_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text:cloud")

    with pytest.raises(RuntimeError):
        embeddings_module.embed_text("anything")


def test_embed_texts_raises_via_assert_local_only_for_cloud_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text:cloud")

    with pytest.raises(RuntimeError):
        embeddings_module.embed_texts(["anything"])
