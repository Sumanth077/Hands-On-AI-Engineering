import os

from dotenv import load_dotenv

from agent.llm_client import _normalize_tag, assert_local_only, get_client

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"


def _configured_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)


def _configured_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    base_url = _configured_base_url()
    model = _configured_embedding_model()
    assert_local_only(base_url, model)

    client = get_client()
    response = client.embed(model=model, input=text)
    return [float(value) for value in response.embeddings[0]]


def embed_texts(texts: list[str]) -> list[list[float]]:
    base_url = _configured_base_url()
    model = _configured_embedding_model()
    assert_local_only(base_url, model)

    client = get_client()
    response = client.embed(model=model, input=texts)
    return [[float(value) for value in vector] for vector in response.embeddings]


def embedding_health_check() -> dict:
    base_url = _configured_base_url()
    model = _configured_embedding_model()
    assert_local_only(base_url, model)

    client = get_client()
    try:
        response = client.list()
        local_models = [m.model for m in response.models]
        reachable = True
    except Exception:
        local_models = []
        reachable = False

    normalized_model = _normalize_tag(model)
    model_available = any(
        _normalize_tag(available) == normalized_model for available in local_models
    )

    return {
        "reachable": reachable,
        "model_available": model_available,
        "local_models": local_models,
    }
