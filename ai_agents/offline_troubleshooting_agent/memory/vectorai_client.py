import os

from actian_vectorai import Distance, PointStruct, VectorAIClient, VectorParams
from dotenv import load_dotenv

load_dotenv()

DEFAULT_HOST = "localhost"
DEFAULT_PORT = "6574"


def _configured_host() -> str:
    return os.getenv("ACTIAN_VECTORAI_HOST", DEFAULT_HOST)


def _configured_port() -> str:
    return os.getenv("ACTIAN_VECTORAI_PORT", DEFAULT_PORT)


def get_client() -> VectorAIClient:
    url = f"{_configured_host()}:{_configured_port()}"
    client = VectorAIClient(url)
    client.connect()
    return client


def health_check() -> dict:
    try:
        client = get_client()
        result = client.health_check()
        return {"reachable": True, **result}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def open_collection(name: str) -> None:
    """Open a collection for read/write, reattaching to its on-disk data.

    `client.collections.list()`/`exists()` reflect what's on disk, but a
    collection isn't actually queryable after a server restart until it's
    explicitly reopened here — the metadata and query-serving paths are
    separate concerns in this engine. Verified empirically against the
    installed SDK: calling this on a collection that's already open is a
    no-op that returns True and raises nothing; only a genuinely
    nonexistent collection raises CollectionNotFoundError.
    """
    client = get_client()
    client.vde.open_collection(name)


def ensure_collection(name: str, dim: int = 768) -> None:
    client = get_client()

    if client.collections.exists(name):
        open_collection(name)
        return

    vectors_config = VectorParams(size=dim, distance=Distance.Cosine)
    client.collections.create(name, vectors_config=vectors_config)
    open_collection(name)


def upsert_point(collection: str, point_id: int | str, vector: list[float], payload: dict) -> None:
    client = get_client()
    point = PointStruct(id=point_id, vector=vector, payload=payload)
    client.points.upsert(collection, [point])


def search(collection: str, vector: list[float], top_k: int = 5) -> list[dict]:
    client = get_client()
    raw_results = client.points.search(collection, vector=vector, limit=top_k)
    return [
        {"id": result.id, "score": result.score, "payload": result.payload}
        for result in raw_results
    ]


def retrieve_points(collection: str, ids: list[str]) -> list[dict]:
    client = get_client()
    raw_results = client.points.get(collection, ids)
    return [
        {"id": result.id, "score": None, "payload": result.payload}
        for result in raw_results
    ]


def delete_collection(name: str) -> None:
    client = get_client()
    client.collections.delete(name, strict=False)
