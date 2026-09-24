import os

import ollama
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:4b-instruct"


def _configured_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)


def _configured_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def _normalize_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def assert_local_only(base_url: str, model: str) -> None:
    if "ollama.com" in base_url.lower():
        raise RuntimeError(
            f"OLLAMA_BASE_URL '{base_url}' points at Ollama's cloud API. "
            "This project must run against a local-only Ollama server."
        )

    if "cloud" in model.lower():
        raise RuntimeError(
            f"OLLAMA_MODEL '{model}' looks like a cloud-offload model tag. "
            "This project must use a fully local model, not a cloud-backed one."
        )

    if os.getenv("OLLAMA_API_KEY"):
        raise RuntimeError(
            "OLLAMA_API_KEY is set in the environment. Cloud auth has no purpose "
            "in this local-only setup — unset it before continuing."
        )


def get_client() -> ollama.Client:
    return ollama.Client(host=_configured_base_url())


def health_check() -> dict:
    base_url = _configured_base_url()
    model = _configured_model()
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


def generate_reply(prompt: str, system: str | None = None) -> str:
    base_url = _configured_base_url()
    model = _configured_model()
    assert_local_only(base_url, model)

    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(model=model, messages=messages)
    return response.message.content
