import os

import pytest

from memory.embeddings import embedding_health_check
from memory.manual import ingest_manual_documents, search_manual_chunks
from memory.vectorai_client import health_check as vectorai_health_check

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL_PATHS = [
    os.path.join(PROJECT_ROOT, "documents", "machine_manual.md"),
    os.path.join(PROJECT_ROOT, "documents", "troubleshooting_guide.md"),
]


def _services_reachable() -> bool:
    try:
        embedding_status = embedding_health_check()
        vectorai_status = vectorai_health_check()
    except Exception:
        return False

    return bool(
        embedding_status.get("reachable")
        and embedding_status.get("model_available")
        and vectorai_status.get("reachable")
    )


SERVICES_REACHABLE = _services_reachable()

pytestmark = pytest.mark.skipif(
    not SERVICES_REACHABLE,
    reason="Requires a live local embedding model and a live Actian VectorAI instance.",
)


@pytest.fixture(scope="module", autouse=True)
def _seed_manual_collection():
    ingest_manual_documents(MANUAL_PATHS)


def test_bearing_query_retrieves_bearing_component():
    results = search_manual_chunks("bearing vibration overheating", top_k=3)
    components = [result["component"] for result in results]
    assert "bearing" in components


def test_cooling_query_retrieves_cooling_component():
    results = search_manual_chunks("cooling fan airflow overheating", top_k=3)
    components = [result["component"] for result in results]
    assert "cooling" in components


def test_pressure_query_retrieves_pressure_component():
    results = search_manual_chunks("pressure drop leak seal", top_k=3)
    components = [result["component"] for result in results]
    assert "pressure" in components
