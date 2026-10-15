from __future__ import annotations

import httpx

from .config import Settings


class ModelDiscoveryError(RuntimeError):
    pass


def list_nebius_models(settings: Settings) -> list[str]:
    """Return model IDs exposed to the supplied Token Factory API key."""
    if not settings.nebius_api_key:
        raise ModelDiscoveryError("NEBIUS_API_KEY is not set.")

    try:
        response = httpx.get(
            f"{settings.nebius_base_url}/models",
            headers={"Authorization": f"Bearer {settings.nebius_api_key}"},
            timeout=20.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ModelDiscoveryError(
            f"Could not list Token Factory models: {error}"
        ) from error

    payload = response.json()
    entries = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ModelDiscoveryError("The models endpoint returned an unexpected response.")

    model_ids = [item.get("id", "") for item in entries if isinstance(item, dict)]
    return sorted(model_id for model_id in model_ids if model_id)

