"""Authoritative complete-input counts, ledger migration and generic requirement policy."""

import json
import sqlite3

import httpx
import pytest
from pydantic import SecretStr

from repo_scout.llm import ModelBudgetError, ModelQuotaError
from repo_scout.openai_backend import SELECTED_OPENAI_MODEL, CostLedger, OpenAIBackend
from repo_scout.research import enforce_required_conditions
from repo_scout.research_models import Criterion, Interpretation, Limits


async def test_same_input_is_counted_before_atomic_reservation(tmp_path):
    from test_openai_backend import response

    ledger = CostLedger(tmp_path / "ledger.sqlite3")
    counted = None

    def handler(request):
        nonlocal counted
        body = json.loads(request.content)
        if request.url.path.endswith("/input_tokens"):
            assert ledger.summary()["attempts"] == 0
            counted = body
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 100}
            )
        assert counted and all(body[k] == v for k, v in counted.items())
        assert ledger.summary()["committed_upper_bound_usd"] == 0.00302
        return httpx.Response(200, json=response())

    backend = OpenAIBackend(
        SELECTED_OPENAI_MODEL,
        SecretStr("synthetic"),
        ledger=ledger,
        transport=httpx.MockTransport(handler),
    )
    await backend.generate(
        "test", {"unicode": "表é"}, {"type": "object", "properties": {}}, Limits()
    )
    assert (
        backend.count_records[0]["input_payload_sha256"]
        == backend.usage_records[0]["input_payload_sha256"]
    )
    assert ledger.summary()["unsettled_attempts"] == 0


async def test_count_failure_never_sends_generation_or_reserves(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.sqlite3")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(429, json={"error": {"message": "quota"}})

    backend = OpenAIBackend(
        SELECTED_OPENAI_MODEL,
        SecretStr("synthetic"),
        ledger=ledger,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelQuotaError):
        await backend.generate("test", {}, {}, Limits())
    assert calls == ["/v1/responses/input_tokens"]
    assert ledger.summary()["attempts"] == 0


def test_migration_preserves_old_settled_costs_and_uncertain_holds(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE attempts (id TEXT PRIMARY KEY, stage TEXT, reserved INTEGER NOT NULL, charged INTEGER, status TEXT NOT NULL)"
        )
        db.execute("INSERT INTO attempts VALUES ('old','extract',213000000,39953550,'settled')")
        db.execute("INSERT INTO attempts VALUES ('uncertain','extract',10000000,NULL,'reserved')")
    ledger = CostLedger(path)
    assert ledger.summary()["committed_upper_bound_usd"] == 0.04995355
    assert ledger.summary()["unsettled_attempts"] == 1
    attempt, reserved = ledger.reserve("new", 4000, input_tokens=9000)
    assert reserved == 4800000
    ledger.settle(attempt, 8000, 1000, 4000)
    assert CostLedger(path).summary()["unsettled_attempts"] == 1


def test_usage_over_count_retains_hold_and_stops(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.sqlite3")
    attempt, _ = ledger.reserve("new", 4000, input_tokens=100)
    with pytest.raises(ModelBudgetError, match="counted"):
        ledger.settle(attempt, 101, 50, 4000)
    assert ledger.summary()["unsettled_attempts"] == 1


@pytest.mark.parametrize(
    "text,quote,required",
    [
        ("Find tools that support encryption and show recent activity.", "", True),
        ("Find fast tools; maintenance is optional.", "maintenance is optional", False),
        ("Find tools with audits; prefer a web UI.", "prefer audits", True),
        ("Find tools with audits; prefer a web UI.", "", True),
    ],
)
def test_preference_requires_explicit_verbatim_user_wording(text, quote, required):
    original = Interpretation(
        summary="Test",
        queries=["tools"],
        criteria=[
            Criterion(
                id="condition",
                label="Condition",
                description="Preference: condition",
                required=False,
                preference_quote=quote,
            )
        ],
    )
    result = enforce_required_conditions(text, original)
    assert result.criteria[0].required is required
    assert original.criteria[0].required is False
