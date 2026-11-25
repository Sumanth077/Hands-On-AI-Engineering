"""Official SDK exercised with synthetic HTTP responses; never a live Google call."""

import json

import httpx
import pytest
from pydantic import SecretStr

from repo_scout.demo import DATA, SAMPLE_REQUIREMENT, DemoModel, DemoWeb
from repo_scout.gemini import SELECTED_MODEL, GeminiBackend
from repo_scout.llm import ModelError, ModelQuotaError, StructuredModel
from repo_scout.research import Researcher
from repo_scout.research_models import Extraction, Interpretation, Limits, ResearchPlan
from repo_scout.service import live_configuration_ready, run_research


def response(text=None, *, reason="STOP", usage=True):
    body = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": text or json.dumps(DATA["interpretation"])}],
                },
                "finishReason": reason,
            }
        ],
        "modelVersion": SELECTED_MODEL,
    }
    if usage:
        body["usageMetadata"] = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 10,
            "totalTokenCount": 160,
        }
    return body


def backend(handler):
    return GeminiBackend(
        SELECTED_MODEL, SecretStr("synthetic-google-key"), transport=httpx.MockTransport(handler)
    )


async def test_official_sdk_auth_exact_model_json_schema_no_web_tools(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-be-used")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://must-not-be-used.example.com")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == f"/v1beta/models/{SELECTED_MODEL}:generateContent"
        assert not request.url.query
        assert request.headers["x-goog-api-key"] == "synthetic-google-key"
        payload = json.loads(request.content)
        config = payload["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"] == Interpretation.model_json_schema()
        assert config["maxOutputTokens"] == 4000
        assert config["candidateCount"] == 1
        assert not payload.get("tools")
        assert "googleSearch" not in request.content.decode()
        assert "urlContext" not in request.content.decode()
        assert "UNTRUSTED DATA" in json.dumps(payload["systemInstruction"])
        return httpx.Response(200, json=response())

    provider = backend(handler)
    model = StructuredModel(provider, Limits())
    interpreted = await model.ask("interpret", {"requirement": "synthetic"}, Interpretation)
    assert interpreted.criteria[0].required
    assert len(requests) == 1
    assert provider.usage_records[0]["usage"]["total_token_count"] == 160
    assert provider.usage_records[0]["cost_usd"] is None
    assert provider.usage_records[0]["account_tier"] == "unknown"
    assert "synthetic-google-key" not in json.dumps(provider.usage_records)


@pytest.mark.parametrize("status", [429, 503, 400, 403])
async def test_errors_have_one_attempt_no_secret_echo_or_provider_fallback(status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={
                "error": {
                    "code": status,
                    "status": "RESOURCE_EXHAUSTED" if status == 429 else "UNKNOWN",
                    "message": "Never expose synthetic-google-key or private provider body",
                }
            },
        )

    provider = backend(handler)
    model = StructuredModel(provider, Limits())
    error_type = ModelQuotaError if status == 429 else ModelError
    with pytest.raises(error_type) as error:
        await model.ask("interpret", {}, Interpretation)
    assert len(requests) == 1
    assert "synthetic-google-key" not in str(error.value)
    assert "private provider body" not in str(error.value)
    assert provider.usage_records[0]["usage"] is None
    if status == 429:
        with pytest.raises(ModelQuotaError):
            await model.ask("interpret", {}, Interpretation)
        assert len(requests) == 1
        assert model.calls == 1
        assert model.traces[0]["outcome"] == "quota_exhausted"


@pytest.mark.parametrize("reason", ["MAX_TOKENS", "SAFETY", "CONTINUATION"])
async def test_incomplete_output_no_sdk_continuation_and_usage_survives(reason):
    requests = []

    def handler(request):
        requests.append(request)
        body = response(reason=reason)
        if reason == "CONTINUATION":
            body["continuationToken"] = "c3ludGhldGlj"
        return httpx.Response(200, json=body)

    provider = backend(handler)
    with pytest.raises(ModelError, match="incomplete"):
        await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert len(requests) == 1
    assert provider.usage_records[0]["usage"]["total_token_count"] == 160


@pytest.mark.parametrize("text", ["not JSON", '{"wrong":true}'])
async def test_invalid_json_and_assessment_schema_are_not_trusted(text):
    provider = backend(lambda request: httpx.Response(200, json=response(text)))
    with pytest.raises(ModelError, match="invalid structured output"):
        await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert provider.usage_records[0]["usage"]["prompt_token_count"] == 100


async def test_quota_mid_research_preserves_completed_evidence_and_stops_calls():
    fixture_model = DemoModel()
    calls = []

    async def handler(request):
        payload = json.loads(request.content)
        prompt = json.loads(payload["contents"][0]["parts"][0]["text"])
        stage = prompt["stage"]
        calls.append(stage)
        # Complete root assessment and first evidence round; stop at the next batch.
        if stage == "extract" and calls.count("extract") == 3:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "synthetic quota",
                    }
                },
            )
        output = await fixture_model.generate(stage, prompt["data"], {}, Limits())
        return httpx.Response(200, json=response(output))

    provider = backend(handler)
    limits = Limits(candidate_limit=2)
    async with DemoWeb().client() as web:
        result = await Researcher(web, StructuredModel(provider, limits), limits).run(
            SAMPLE_REQUIREMENT
        )
    assert calls == ["interpret", "select", "extract", "plan", "extract", "plan", "extract"]
    assert result.status == "partial"
    assert result.sources
    assert result.projects[0].facts
    assert any(a.verdict == "Matches" for a in result.projects[0].assessments)
    assert result.projects[
        1
    ].facts  # Both members of the completed batch survive quota interruption.
    assert any("quota exhausted" in warning for warning in result.warnings)
    assert len(result.diagnostics["model_usage"]) == 7
    assert result.diagnostics["model_usage"][-1]["outcome"] == "quota_exhausted"
    # Gemini metadata never appears inside the Apify measurement/usage records.
    assert all(
        "gemini" not in raw["measurement"]["actor"] for raw in result.diagnostics["retrievals"]
    )


async def test_live_gate_and_unverified_paid_model_rejected_before_network(monkeypatch):
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "false")
    provider = GeminiBackend(SELECTED_MODEL, SecretStr("synthetic-google-key"))
    with pytest.raises(ValueError, match="disabled"):
        await provider.generate("interpret", {}, {}, Limits())
    with pytest.raises(ValueError, match="verified free-tier"):
        GeminiBackend("gemini-4-carbon", SecretStr("synthetic-google-key"))
    assert not provider.usage_records


async def test_service_gemini_config_and_environment_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOSCOUT_LLM_PROVIDER", "gemini")
    for name, value in {
        "REPOSCOUT_ENABLE_LIVE": "true",
        "GEMINI_MODEL": SELECTED_MODEL,
        "GEMINI_API_KEY": "synthetic-google-key",
        "APIFY_TOKEN": "synthetic-apify-key",
    }.items():
        monkeypatch.setenv(name, value)
    assert live_configuration_ready()
    # Unsupported legacy settings cannot silently select another provider.
    monkeypatch.setenv("REPOSCOUT_LLM_PROVIDER", "openai-compatible")
    assert not live_configuration_ready()
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        await run_research(SAMPLE_REQUIREMENT, Limits(), offline=False)
    monkeypatch.setenv("REPOSCOUT_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "unverified-model")
    assert not live_configuration_ready()
    with pytest.raises(ValueError, match="verified free-tier"):
        await run_research(SAMPLE_REQUIREMENT, Limits(), offline=False)
    monkeypatch.delenv("GEMINI_MODEL")
    with pytest.raises(ValueError, match="GEMINI_API_KEY and GEMINI_MODEL"):
        await run_research(SAMPLE_REQUIREMENT, Limits(), offline=False)


async def test_complete_workflow_uses_sdk_structured_plans_and_assessments():
    scripted = DemoModel()
    stages = []

    async def handler(request):
        payload = json.loads(request.content)
        prompt = json.loads(payload["contents"][0]["parts"][0]["text"])
        stages.append(prompt["stage"])
        if prompt["stage"] == "plan":
            assert (
                payload["generationConfig"]["responseJsonSchema"]
                == ResearchPlan.model_json_schema()
            )
        if prompt["stage"] == "extract":
            assert (
                payload["generationConfig"]["responseJsonSchema"] == Extraction.model_json_schema()
            )
        text = await scripted.generate(prompt["stage"], prompt["data"], {}, Limits())
        return httpx.Response(200, json=response(text))

    provider = backend(handler)
    limits = Limits(candidate_limit=1)
    async with DemoWeb().client() as web:
        result = await Researcher(web, StructuredModel(provider, limits), limits).run(
            SAMPLE_REQUIREMENT
        )
    assert {"interpret", "select", "plan", "extract", "shortlist"} <= set(stages)
    assert result.shortlist
    assert all(a.citations and a.verdict == "Matches" for a in result.projects[0].assessments)
    assert len(result.diagnostics["model_usage"]) == len(stages)
    assert all(item["model"] == SELECTED_MODEL for item in result.diagnostics["model_usage"])


async def test_missing_usage_is_unknown_and_unexpected_tools_are_never_executed():
    calls = []

    def handler(request):
        calls.append(request)
        body = response(usage=False)
        body["candidates"][0]["content"]["parts"] = [
            {"functionCall": {"name": "google_search", "args": {"query": "never execute this"}}}
        ]
        return httpx.Response(200, json=body)

    provider = backend(handler)
    with pytest.raises(ModelError, match="tool call rejected"):
        await StructuredModel(provider, Limits()).ask("interpret", {}, Interpretation)
    assert len(calls) == 1
    assert provider.usage_records[0]["usage"] is None


def test_dotenv_loading_never_overwrites_existing_environment_or_file(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    from dotenv import load_dotenv

    from repo_scout import service

    env_file = tmp_path / ".env"
    original = "GEMINI_API_KEY=synthetic-file-key\nGEMINI_MODEL=gemini-3.5-flash-lite\nAPIFY_TOKEN=synthetic-apify\nREPOSCOUT_ENABLE_LIVE=false\n"
    env_file.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        service,
        "load_dotenv",
        lambda **kwargs: load_dotenv(env_file, override=kwargs.get("override", False)),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-existing-env-key")
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "false")
    service.configured_limits()
    assert not service.live_configuration_ready()
    import os

    assert os.environ["GEMINI_API_KEY"] == "synthetic-existing-env-key"
    assert env_file.read_text(encoding="utf-8") == original
