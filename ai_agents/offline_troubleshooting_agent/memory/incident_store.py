import os

from dotenv import load_dotenv

from memory.embeddings import embed_text
from memory.vectorai_client import (
    ensure_collection,
    retrieve_points,
    search as vectorai_search,
    upsert_point,
)
from models.schemas import Incident

load_dotenv()

DEFAULT_INCIDENT_COLLECTION = "incidents"
INCIDENT_EMBEDDING_DIM = 768


def _configured_incident_collection() -> str:
    return os.getenv("ACTIAN_VECTORAI_INCIDENT_COLLECTION", DEFAULT_INCIDENT_COLLECTION)


def build_embedding_text(incident: Incident) -> str:
    lines = [f"Machine: {incident.machine_id}"]

    if incident.symptoms:
        lines.append(f"Symptoms: {', '.join(incident.symptoms)}")

    reading_parts = []
    if incident.temperature_c is not None:
        reading_parts.append(f"temperature {incident.temperature_c} C")
    if incident.vibration_mm_s is not None:
        reading_parts.append(f"vibration {incident.vibration_mm_s} mm/s")
    if incident.pressure_bar is not None:
        reading_parts.append(f"pressure {incident.pressure_bar} bar")
    lines.append(
        f"Readings: {', '.join(reading_parts)}" if reading_parts else "Readings: not recorded"
    )

    lines.append(f"Error code: {incident.error_code or 'none'}")
    lines.append(f"Confirmed cause: {incident.confirmed_cause}")
    lines.append(f"Fix applied: {incident.fix_applied}")
    lines.append(f"Outcome: {incident.outcome}")

    return "\n".join(lines)


def save_incident(incident: Incident, collection: str | None = None) -> None:
    target_collection = collection or _configured_incident_collection()

    text = build_embedding_text(incident)
    vector = embed_text(text)

    ensure_collection(target_collection, dim=INCIDENT_EMBEDDING_DIM)
    payload = incident.model_dump(mode="json")
    upsert_point(target_collection, incident.incident_id, vector, payload)


def get_incident(incident_id: str, collection: str | None = None) -> Incident | None:
    target_collection = collection or _configured_incident_collection()
    ensure_collection(target_collection, dim=INCIDENT_EMBEDDING_DIM)

    results = retrieve_points(target_collection, [incident_id])
    if not results:
        return None

    payload = results[0].get("payload")
    if not payload:
        return None

    return Incident.model_validate(payload)


def search_incidents(query: str, top_k: int = 3, collection: str | None = None) -> list[dict]:
    target_collection = collection or _configured_incident_collection()
    ensure_collection(target_collection, dim=INCIDENT_EMBEDDING_DIM)

    vector = embed_text(query)
    raw_results = vectorai_search(target_collection, vector, top_k)

    results = []
    for item in raw_results:
        payload = item.get("payload") or {}
        results.append({**payload, "id": item["id"], "score": item["score"]})
    return results
