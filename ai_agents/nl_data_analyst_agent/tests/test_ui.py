"""Offline checks for Gradio database selection and Eve event projection."""

import json
from pathlib import Path

import pytest

import main
from seed_data import seed_database


def test_connect_local_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "my-data.sqlite"
    seed_database(database)
    monkeypatch.setattr(main, "REGISTRY", tmp_path / "connections.json")
    selected, status, schema, chat, sql, results, pending, session, approve, reject, suggestions = (
        main.connect_database(None, str(database))
    )
    assert len(selected) == 32
    assert "my-data.sqlite" in status
    assert "orders:" in schema
    assert chat == [] and sql == "" and results.empty and pending is None and session is None
    assert not approve["visible"] and not reject["visible"]
    assert selected in json.loads(main.REGISTRY.read_text())
    assert any("orders" in question for question in suggestions["choices"])


def test_uploaded_file_takes_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "upload.db"
    seed_database(database)
    monkeypatch.setattr(main, "REGISTRY", tmp_path / "connections.json")
    main.connect_database(str(database), "C:\\missing.sqlite")
    assert str(database.resolve()) in json.loads(main.REGISTRY.read_text()).values()


def test_demo_button_creates_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "demo.sqlite"
    monkeypatch.setattr(main, "DEMO_DATABASE", database)
    monkeypatch.setattr(main, "REGISTRY", tmp_path / "connections.json")
    selected, status, *_ = main.use_demo_database()
    assert database.is_file()
    assert selected == "demo" and "demo.sqlite" in status


def test_question_requires_connection() -> None:
    with pytest.raises(Exception, match="Connect a SQLite database"):
        main.ask("How many orders?", [], None, None, None)
