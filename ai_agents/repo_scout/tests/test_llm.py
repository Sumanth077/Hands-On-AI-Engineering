"""Provider transport tests use mock responses only; no selected live provider."""

import json

import httpx
import pytest
from pydantic import SecretStr

from repo_scout.demo import DATA
from repo_scout.llm import ModelError, OpenAICompatibleBackend, StructuredModel
from repo_scout.research_models import Interpretation, Limits


async def test_explicit_provider_transport_contract_and_schema_validation():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.headers["Authorization"] == "Bearer synthetic-key"
        assert "synthetic-key" not in str(request.url)
        assert payload["model"] == "operator-selected-test-model"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert "UNTRUSTED DATA" in payload["messages"][0]["content"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(DATA["interpretation"])}}]}
        )

    backend = OpenAICompatibleBackend(
        "https://provider.example.com/v1/chat/completions",
        "operator-selected-test-model",
        SecretStr("synthetic-key"),
        httpx.MockTransport(handler),
    )
    model = StructuredModel(backend, Limits())
    result = await model.ask("interpret", {"requirement": "test"}, Interpretation)
    assert result.criteria[0].required
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body,status",
    [
        ({"choices": []}, 200),
        ({"choices": [{"message": {"refusal": "Test refusal"}}]}, 200),
        ({"error": "synthetic-key"}, 429),
    ],
)
async def test_provider_failure_does_not_echo_body_or_retry(body, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=body)

    backend = OpenAICompatibleBackend(
        "https://provider.example.com/completions",
        "explicit-test-model",
        SecretStr("synthetic-key"),
        httpx.MockTransport(handler),
    )
    with pytest.raises(ModelError) as error:
        await StructuredModel(backend, Limits()).ask("interpret", {}, Interpretation)
    assert "synthetic-key" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["input", "output", "calls", "invalid", "timeout"])
async def test_model_limits(kind):
    class Backend:
        calls = 0

        async def generate(self, *args):
            self.calls += 1
            if kind == "timeout":
                raise httpx.ReadTimeout("synthetic private error")
            if kind == "output":
                return "x" * 30000
            if kind == "invalid":
                return "not json"
            return json.dumps(DATA["interpretation"])

    backend = Backend()
    model = StructuredModel(backend, Limits(max_model_calls=1))
    context = {"large": "a" * 100000} if kind == "input" else {}
    if kind == "calls":
        await model.ask("interpret", context, Interpretation)
    with pytest.raises(ModelError):
        await model.ask("interpret", context, Interpretation)
    assert backend.calls <= 1


def test_no_default_provider_or_insecure_secret_endpoint():
    with pytest.raises(ValueError):
        OpenAICompatibleBackend("http://example.com?token=bad", "", SecretStr("synthetic"))
