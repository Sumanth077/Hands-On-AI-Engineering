"""Offline checks for Gradio database selection and Eve event projection."""

import json
from pathlib import Path

import pytest

import main
from jev_review import Review
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


def test_clarification_happens_before_eve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "demo.sqlite"
    seed_database(database)
    registry = tmp_path / "connections.json"
    registry.write_text(json.dumps({"demo": str(database)}))
    monkeypatch.setattr(main, "REGISTRY", registry)
    monkeypatch.setattr(main, "review_question", lambda *_: Review(
        "question clarity", "clarify", 0.94, "Clarify how the requested metric should be calculated.", True
    ))
    monkeypatch.setattr(main, "eve_post", lambda *_: pytest.fail("Eve must not receive an unclear question"))

    chat, retained_question, sql, results, pending, session, approve, reject = main.ask(
        "Who is best?", [], None, "demo", None
    )

    assert retained_question == "Who is best?"
    assert "Please clarify" in chat[-1]["content"]
    assert sql == "" and results.empty and pending is None and session is None
    assert not approve["visible"] and not reject["visible"]


def test_grounding_mismatch_is_clearly_unverified() -> None:
    review = Review(
        "answer grounding", "mismatch", 0.96,
        "A reported value is not supported by the returned rows.", True,
    )
    displayed, note = main.present_grounding_review("Revenue was $900.", review)
    assert displayed.startswith("⚠️ **Unverified draft — awaiting user review.**")
    assert "not treated as final" in displayed
    assert "Revise the previous answer" in displayed
    assert "**Eve draft (unverified):**" in displayed
    assert note is None


def test_batched_sql_calls_do_not_mix_completed_rows_with_pending_approval(monkeypatch) -> None:
    revenue_call = "call_revenue"
    metadata_call = "call_metadata"
    revenue_sql = "SELECT region, SUM(revenue) AS revenue FROM orders WHERE year = 2025 GROUP BY region"
    metadata_sql = "SELECT MIN(order_date), MAX(order_date), COUNT(*) FROM orders"
    events = [
        {"type": "actions.requested", "data": {"actions": [
            {"callId": revenue_call, "toolName": "run_sql", "input": {"databaseId": "demo", "sql": revenue_sql}},
        ]}},
        {"type": "actions.requested", "data": {"actions": [
            {"callId": metadata_call, "toolName": "run_sql", "input": {"databaseId": "demo", "sql": metadata_sql}},
        ]}},
        {"type": "action.result", "data": {"result": {
            "callId": revenue_call,
            "toolName": "run_sql",
            "output": {
                "sql": revenue_sql,
                "rows": [{"region": "South", "revenue": 98664}],
                "review": {
                    "check": "SQL relevance", "verdict": "relevant", "confidence": 0.95,
                    "issue": "The SQL addresses the question.", "actionable": False, "status": "ok",
                },
                "execution": "completed",
            },
        }}},
        {"type": "input.requested", "data": {"requests": [{
            "requestId": "request_metadata",
            "kind": "tool-approval",
            "action": {
                "callId": metadata_call,
                "toolName": "run_sql",
                "input": {"databaseId": "demo", "sql": metadata_sql},
            },
        }]}},
        {"type": "session.waiting", "data": {}},
    ]

    monkeypatch.setattr(main, "review_answer", lambda *_: pytest.fail("Pending calls are not grounded"))
    answer, displayed_sql, displayed_rows, pending = main.project_eve_events(
        events, "Which region had the most completed-order revenue in 2025?",
    )

    assert answer == ""
    assert displayed_sql == metadata_sql
    assert displayed_rows.empty
    assert pending == {
        "requestId": "request_metadata",
        "callId": metadata_call,
        "sql": metadata_sql,
        "priorCompleted": {
            "callId": revenue_call,
            "sql": revenue_sql,
            "rows": [{"region": "South", "revenue": 98664}],
        },
    }


def test_approved_call_selects_only_its_matching_result(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "review_answer",
        lambda *_: Review("answer grounding", "grounded", 0.95, "Supported.", False),
    )
    events = [
        {"type": "action.result", "data": {"result": {
            "callId": "call_other", "toolName": "run_sql",
            "output": {"sql": "SELECT 1 AS n", "rows": [{"n": 1}]},
        }}},
        {"type": "action.result", "data": {"result": {
            "callId": "call_approved", "toolName": "run_sql",
            "output": {"sql": "SELECT 2 AS n", "rows": [{"n": 2}]},
        }}},
        {"type": "message.completed", "data": {"message": "The approved query returned 2.", "finishReason": "stop"}},
    ]

    _, displayed_sql, displayed_rows, pending = main.project_eve_events(
        events, "Return the approved value", focus_call_id="call_approved",
    )

    assert displayed_sql == "SELECT 2 AS n"
    assert displayed_rows.to_dict(orient="records") == [{"n": 2}]
    assert pending is None


def test_rejection_restores_prior_result_with_a_clear_label(monkeypatch) -> None:
    sent = {}
    prior = {"callId": "call_revenue", "sql": "SELECT revenue FROM totals", "rows": [{"revenue": 98664}]}
    pending = {
        "requestId": "request_metadata",
        "callId": "call_metadata",
        "sql": "SELECT COUNT(*) FROM orders",
        "priorCompleted": prior,
    }
    monkeypatch.setattr(main, "eve_post", lambda path, payload: sent.update(path=path, payload=payload))
    monkeypatch.setattr(
        main,
        "consume_events",
        lambda session, question, focus_call_id=None: ("Query rejected.", "", main.pd.DataFrame(), None),
    )

    chat, sql, rows, next_pending, *_ = main.decide(
        False, [], pending, {"id": "session_1", "cursor": 4, "question": "Revenue?"},
    )

    assert sent["payload"]["inputResponses"][0]["optionId"] == "cancel"
    assert sql == prior["sql"]
    assert rows.to_dict(orient="records") == prior["rows"]
    assert next_pending is None
    assert "earlier completed query" in chat[-1]["content"]
    assert "not the rejected query" in chat[-1]["content"]
