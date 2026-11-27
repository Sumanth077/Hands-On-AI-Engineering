"""Official OpenAI SDK and cost ledger tests; only synthetic HTTP responses."""

import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from pydantic import SecretStr

from repo_scout.demo import DATA, SAMPLE_REQUIREMENT, DemoModel, DemoWeb
from repo_scout.llm import (
    ModelAuthenticationError,
    ModelBudgetError,
    ModelError,
    ModelQuotaError,
    StructuredModel,
)
from repo_scout.openai_backend import (
    SELECTED_OPENAI_MODEL,
    CostLedger,
    OpenAIBackend,
    strict_schema,
)
from repo_scout.research import Researcher
from repo_scout.research_models import Interpretation, Limits
from repo_scout.service import live_configuration_ready, live_setup_issues, run_research


def response(text=None, *, status="completed", usage=True):
    body = {
        "id": "resp_synthetic",
        "object": "response",
        "created_at": 0,
        "model": SELECTED_OPENAI_MODEL,
        "status": status,
        "error": None,
        "output": [
            {
                "id": "msg_synthetic",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text or json.dumps(DATA["interpretation"]),
                        "annotations": [],
                    }
                ],
            }
        ],
    }
    if usage:
        body["usage"] = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 10},
        }
    return body


def with_counts(handler):
    def routed(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 100}
            )
        return handler(request)

    return routed


def backend(tmp_path, handler):
    return OpenAIBackend(
        SELECTED_OPENAI_MODEL,
        SecretStr("synthetic-openai-key"),
        ledger=CostLedger(tmp_path / "budget.sqlite3"),
        transport=httpx.MockTransport(with_counts(handler)),
    )


async def test_sdk_responses_schema_auth_tools_and_pre_request_reservation(tmp_path):
    calls = []
    ledger = CostLedger(tmp_path / "budget.sqlite3")

    def handler(request):
        calls.append(request)
        assert ledger.summary()["committed_upper_bound_usd"] == 0.00302
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer synthetic-openai-key"
        payload = json.loads(request.content)
        assert payload["text"]["format"]["strict"]
        assert payload["text"]["format"]["schema"] == strict_schema(
            Interpretation.model_json_schema()
        )
        assert payload["tools"] == [] and payload["tool_choice"] == "none"
        assert payload["store"] is False and payload["background"] is False
        assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
        assert payload["service_tier"] == "default"
        assert "previous_response_id" not in payload
        assert payload["max_output_tokens"] == 4000
        return httpx.Response(200, json=response())

    provider = backend(tmp_path, handler)
    output = await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert output.criteria
    assert len(calls) == 1
    assert provider.usage_records[0]["usage"]["total_tokens"] == 150
    assert provider.usage_records[0]["estimated_cost_usd"] == 0.000035
    assert ledger.summary()["committed_upper_bound_usd"] == 0.0000575


@pytest.mark.parametrize(
    "status,error_type",
    [
        (401, ModelAuthenticationError),
        (403, ModelAuthenticationError),
        (402, ModelQuotaError),
        (429, ModelQuotaError),
        (500, ModelError),
    ],
)
async def test_no_sdk_retries_or_fallback_on_errors(tmp_path, status, error_type):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": "synthetic-openai-key private error",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
        )

    provider = backend(tmp_path, handler)
    with pytest.raises(error_type) as error:
        await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert "synthetic-openai-key" not in str(error.value)
    assert len(calls) == 1
    assert provider.ledger.summary()["unsettled_attempts"] == 1
    assert provider.ledger.summary()["committed_upper_bound_usd"] == 0.00302
    # Even manual retry in another backend/process must count the uncertain attempt.
    reloaded = CostLedger(tmp_path / "budget.sqlite3")
    reloaded.reserve("retry", 4000, input_tokens=100)
    assert reloaded.summary()["unsettled_attempts"] == 2


async def test_timeout_retains_full_reservation(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("private timeout", request=request)

    provider = backend(tmp_path, handler)
    with pytest.raises(ModelError, match="uncertain"):
        await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert provider.ledger.summary()["committed_upper_bound_usd"] == 0.00302


@pytest.mark.parametrize(
    "text,status,has_usage",
    [
        ("not JSON", "completed", True),
        ('{"wrong":true}', "completed", True),
        (None, "incomplete", True),
        (None, "incomplete", False),
    ],
)
async def test_invalid_or_incomplete_output_still_accounts_for_usage(
    tmp_path, text, status, has_usage
):
    provider = backend(
        tmp_path,
        lambda request: httpx.Response(200, json=response(text, status=status, usage=has_usage)),
    )
    with pytest.raises(ModelError):
        await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert provider.ledger.summary()["unsettled_attempts"] == (0 if has_usage else 1)


def test_ledger_cap_is_atomic_and_cannot_be_raised_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BUDGET_USD", "1000")
    path = tmp_path / "budget.sqlite3"
    CostLedger(path)

    def reserve(_):
        try:
            CostLedger(path).reserve("concurrent", 4000)
            return True
        except ModelBudgetError:
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(reserve, range(4))) == 2
    assert CostLedger(path).summary()["cap_usd"] == 0.50


def test_usage_bounds_and_unknown_model_fail_closed(tmp_path):
    ledger = CostLedger(tmp_path / "budget.sqlite3")
    attempt, _ = ledger.reserve("test", 4000)
    with pytest.raises(ModelBudgetError):
        ledger.settle(attempt, 1_100_000, 10, 4000)
    with pytest.raises(ValueError, match="verified pricing"):
        OpenAIBackend("more-expensive-model", SecretStr("synthetic"), ledger=ledger)
    assert ledger.summary()["committed_upper_bound_usd"] == 0.213


def test_smaller_output_allowance_fits_remaining_cap_without_reset(tmp_path, monkeypatch):
    monkeypatch.setattr("repo_scout.openai_backend.CAP_NANO", 250_000_000)
    ledger = CostLedger(tmp_path / "budget.sqlite3")
    first, _ = ledger.reserve("first", 4000)
    ledger.settle(first, 180000, 3100, 4000)
    _, reservation = ledger.reserve("shortlist", 4000)
    allowance = (reservation - 210_000_000) // 750
    assert 256 <= allowance < 4000
    assert ledger.summary()["committed_upper_bound_usd"] <= 0.25


async def test_entire_pipeline_uses_official_openai_sdk_offline(tmp_path):
    scripted = DemoModel()
    calls = []

    async def handler(request):
        prompt = json.loads(json.loads(request.content)["input"])
        calls.append(prompt["stage"])
        text = await scripted.generate(prompt["stage"], prompt["data"], {}, Limits())
        return httpx.Response(200, json=response(text))

    provider = backend(tmp_path, handler)
    limits = Limits(candidate_limit=2)
    async with DemoWeb().client() as web:
        result = await Researcher(web, StructuredModel(provider, limits), limits).run(
            SAMPLE_REQUIREMENT
        )
    assert {"interpret", "select", "plan", "extract", "shortlist"} <= set(calls)
    assert result.shortlist
    assert all(item["provider"] == "openai_responses" for item in result.diagnostics["model_usage"])
    assert result.diagnostics["model_budget"]["committed_upper_bound_usd"] < 0.25


async def test_openai_missing_key_does_not_use_existing_gemini_key(monkeypatch):
    monkeypatch.setenv("REPOSCOUT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-available-gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not live_configuration_ready()
    assert "OPENAI_API_KEY missing" in live_setup_issues()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await run_research(SAMPLE_REQUIREMENT, Limits(), offline=False)


@pytest.mark.parametrize("unexpected", ["model", "tier", "cache_write"])
async def test_unexpected_billable_settings_fail_closed(tmp_path, unexpected):
    body = response()
    if unexpected == "model":
        body["model"] = "more-expensive-model"
    elif unexpected == "tier":
        body["service_tier"] = "priority"
    else:
        body["usage"]["input_tokens_details"]["cache_write_tokens"] = 100
    provider = backend(tmp_path, lambda request: httpx.Response(200, json=body))
    with pytest.raises(ModelBudgetError):
        await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert provider.ledger.summary()["unsettled_attempts"] == 1
