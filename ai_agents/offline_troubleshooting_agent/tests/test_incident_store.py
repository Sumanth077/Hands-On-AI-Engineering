import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from memory.embeddings import embedding_health_check
from memory.incident_store import get_incident, save_incident, search_incidents
from memory.vectorai_client import delete_collection, health_check as vectorai_health_check
from models.schemas import Incident

TEST_COLLECTION = "incidents_test_scratch"


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


def _build_incident(**overrides) -> Incident:
    base = dict(
        incident_id=str(uuid.uuid4()),
        created_at=datetime(2026, 9, 20, 10, 0, 0),
        machine_id="Machine A",
        symptoms=["high temperature", "high vibration"],
        temperature_c=92.0,
        vibration_mm_s=8.1,
        pressure_bar=5.0,
        error_code="BRG-02",
        diagnosis="Likely worn bearing.",
        confirmed_cause="worn bearing",
        fix_applied="bearing replacement",
        outcome="resolved",
        notes=None,
    )
    base.update(overrides)
    return Incident(**base)


def test_incident_missing_required_field_raises_validation_error():
    with pytest.raises(ValidationError):
        Incident(
            incident_id=str(uuid.uuid4()),
            created_at=datetime(2026, 9, 20, 10, 0, 0),
            machine_id="Machine A",
            symptoms=["high vibration"],
            temperature_c=92.0,
            vibration_mm_s=8.1,
            pressure_bar=5.0,
            error_code="BRG-02",
            # diagnosis intentionally omitted
            confirmed_cause="worn bearing",
            fix_applied="bearing replacement",
            outcome="resolved",
        )


def test_incident_round_trips_through_dump_and_validate():
    incident = _build_incident()
    dumped = incident.model_dump(mode="json")
    restored = Incident.model_validate(dumped)
    assert restored == incident


@pytest.mark.skipif(
    not SERVICES_REACHABLE,
    reason="Requires a live local embedding model and a live Actian VectorAI instance.",
)
class TestIncidentStoreIntegration:
    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def _cleanup_test_collection(cls):
        yield
        delete_collection(TEST_COLLECTION)

    def test_save_and_get_incident_round_trips(self):
        incident = _build_incident()
        save_incident(incident, collection=TEST_COLLECTION)

        fetched = get_incident(incident.incident_id, collection=TEST_COLLECTION)

        assert fetched is not None
        assert fetched.incident_id == incident.incident_id
        assert fetched.confirmed_cause == incident.confirmed_cause
        assert fetched.diagnosis == incident.diagnosis
        assert fetched.temperature_c == incident.temperature_c
        assert fetched.vibration_mm_s == incident.vibration_mm_s

    def test_search_incidents_ranks_bearing_above_pressure_for_similar_wording(self):
        bearing_incident = _build_incident(
            incident_id=str(uuid.uuid4()),
            symptoms=["high temperature", "high vibration"],
            temperature_c=92.0,
            vibration_mm_s=8.1,
            pressure_bar=5.0,
            error_code="BRG-02",
            diagnosis="Likely worn bearing.",
            confirmed_cause="worn bearing",
            fix_applied="bearing replacement",
        )
        pressure_incident = _build_incident(
            incident_id=str(uuid.uuid4()),
            symptoms=["low pressure", "mildly elevated temperature"],
            temperature_c=77.0,
            vibration_mm_s=2.4,
            pressure_bar=2.8,
            error_code="PRS-03",
            diagnosis="Likely pressure system leak.",
            confirmed_cause="seal leak",
            fix_applied="replaced seal",
        )

        save_incident(bearing_incident, collection=TEST_COLLECTION)
        save_incident(pressure_incident, collection=TEST_COLLECTION)

        # top_k is wider than 2 deliberately: this collection is shared across
        # the tests in this class (per the spec's single dedicated test
        # collection), so an earlier test's saved incident can also be
        # present here. We only care that bearing outranks pressure, not that
        # pressure is literally the #2 result.
        results = search_incidents(
            "machine running hot and shaking more than normal",
            top_k=10,
            collection=TEST_COLLECTION,
        )

        ids_in_order = [result["incident_id"] for result in results]
        assert bearing_incident.incident_id in ids_in_order
        assert pressure_incident.incident_id in ids_in_order
        assert ids_in_order.index(bearing_incident.incident_id) < ids_in_order.index(
            pressure_incident.incident_id
        )
