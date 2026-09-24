from types import SimpleNamespace

from actian_vectorai import Distance, PointStruct

import memory.vectorai_client as vectorai_module


class FakeCollections:
    def __init__(self, existing_names=()):
        self.create_calls: list[dict] = []
        self.exists_calls: list[str] = []
        self._existing = set(existing_names)

    def exists(self, name):
        self.exists_calls.append(name)
        return name in self._existing

    def create(self, name, *, vectors_config=None, **kwargs):
        self.create_calls.append({"name": name, "vectors_config": vectors_config})
        self._existing.add(name)


class FakeVde:
    def __init__(self):
        self.open_calls: list[str] = []

    def open_collection(self, name):
        self.open_calls.append(name)
        return True


class FakePoints:
    def __init__(self, search_results=None):
        self.upsert_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self._search_results = search_results or []

    def upsert(self, collection_name, points):
        self.upsert_calls.append({"collection_name": collection_name, "points": points})

    def search(self, collection_name, vector, *, limit=10, **kwargs):
        self.search_calls.append(
            {"collection_name": collection_name, "vector": vector, "limit": limit}
        )
        return self._search_results


class FakeClientThatFailsHealthCheck:
    def health_check(self):
        raise ConnectionError("connection refused")


def test_ensure_collection_creates_and_opens_when_missing(monkeypatch):
    fake_collections = FakeCollections(existing_names=())
    fake_vde = FakeVde()
    fake_client = SimpleNamespace(collections=fake_collections, vde=fake_vde)
    monkeypatch.setattr(vectorai_module, "get_client", lambda: fake_client)

    vectorai_module.ensure_collection("manual_chunks", dim=768)

    assert fake_collections.exists_calls == ["manual_chunks"]
    assert len(fake_collections.create_calls) == 1
    call = fake_collections.create_calls[0]
    assert call["name"] == "manual_chunks"
    assert call["vectors_config"].size == 768
    assert call["vectors_config"].distance == Distance.Cosine
    assert fake_vde.open_calls == ["manual_chunks"]


def test_ensure_collection_reopens_without_creating_when_already_exists(monkeypatch):
    # This is the specific behavior that prevents a restart-time reopen from
    # shadowing real on-disk data with an empty freshly-created collection.
    fake_collections = FakeCollections(existing_names=("manual_chunks",))
    fake_vde = FakeVde()
    fake_client = SimpleNamespace(collections=fake_collections, vde=fake_vde)
    monkeypatch.setattr(vectorai_module, "get_client", lambda: fake_client)

    vectorai_module.ensure_collection("manual_chunks", dim=768)

    assert fake_collections.exists_calls == ["manual_chunks"]
    assert fake_collections.create_calls == []
    assert fake_vde.open_calls == ["manual_chunks"]


def test_open_collection_calls_vde_open_collection(monkeypatch):
    fake_vde = FakeVde()
    fake_client = SimpleNamespace(vde=fake_vde)
    monkeypatch.setattr(vectorai_module, "get_client", lambda: fake_client)

    vectorai_module.open_collection("incidents")

    assert fake_vde.open_calls == ["incidents"]


def test_upsert_point_builds_point_struct_and_calls_upsert(monkeypatch):
    # PointStruct.id only accepts a non-negative int or a valid UUID string
    # (confirmed against the real actian-vectorai-client SDK), so this uses
    # a UUID rather than an arbitrary string like "inc-1".
    point_id = "12345678-1234-5678-1234-567812345678"

    fake_points = FakePoints()
    fake_client = SimpleNamespace(points=fake_points)
    monkeypatch.setattr(vectorai_module, "get_client", lambda: fake_client)

    vectorai_module.upsert_point(
        "incidents", point_id, [0.1, 0.2, 0.3], {"cause": "worn bearing"}
    )

    assert len(fake_points.upsert_calls) == 1
    call = fake_points.upsert_calls[0]
    assert call["collection_name"] == "incidents"
    assert len(call["points"]) == 1

    point = call["points"][0]
    assert isinstance(point, PointStruct)
    assert point.id == point_id
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {"cause": "worn bearing"}


def test_search_calls_client_with_right_args_and_normalizes_result(monkeypatch):
    fake_result = SimpleNamespace(id="inc-1", score=0.87, payload={"cause": "worn bearing"})
    fake_points = FakePoints(search_results=[fake_result])
    fake_client = SimpleNamespace(points=fake_points)
    monkeypatch.setattr(vectorai_module, "get_client", lambda: fake_client)

    results = vectorai_module.search("incidents", [0.1, 0.2, 0.3], top_k=3)

    assert fake_points.search_calls == [
        {"collection_name": "incidents", "vector": [0.1, 0.2, 0.3], "limit": 3}
    ]
    assert results == [{"id": "inc-1", "score": 0.87, "payload": {"cause": "worn bearing"}}]


def test_health_check_returns_unreachable_without_raising(monkeypatch):
    monkeypatch.setattr(
        vectorai_module, "get_client", lambda: FakeClientThatFailsHealthCheck()
    )

    result = vectorai_module.health_check()

    assert result["reachable"] is False
    assert "connection refused" in result["error"]
