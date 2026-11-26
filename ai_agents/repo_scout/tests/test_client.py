"""All responses are synthetic test data, never live evidence or benchmark results."""

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from repo_scout.apify_client import ApifyClient, compare_content, redact
from repo_scout.config import Settings
from repo_scout.models import FetchInput, SearchInput

TOKEN = "synthetic-test-token"
PAGE = {
    "url": "https://example.com",
    "fetch": {"httpStatusCode": 200, "loadedUrl": "https://example.com"},
    "metadata": {},
    "markdown": "# Test data",
}
REQUEST = FetchInput(url="https://example.com")


def client(handler, **kwargs):
    return ApifyClient(
        Settings(token=SecretStr(TOKEN)), transport=httpx.MockTransport(handler), **kwargs
    )


@pytest.mark.parametrize(
    "body,outcome,status",
    [
        (PAGE, "success", 200),
        ({**PAGE, "markdown": "  "}, "empty", 200),
        ({**PAGE, "markdown": None}, "failed", 200),
        ({**PAGE, "markdown": ["bad"]}, "failed", 200),
        ({**PAGE, "fetch": {**PAGE["fetch"], "httpStatusCode": 404}}, "failed", 404),
        ({**PAGE, "fetch": {**PAGE["fetch"], "httpStatusCode": 500}}, "failed", 500),
        ({"markdown": "missing envelope"}, "failed", None),
        ({"code": "UNSUPPORTED_CONTENT_TYPE", "error": "test"}, "failed", None),
    ],
)
async def test_fetch_validation(body, outcome, status):
    async with client(lambda r: httpx.Response(200, json=body)) as api:
        out = await api.fetch(REQUEST)
    assert out.measurement.outcome == outcome
    assert out.measurement.target_status == status


@pytest.mark.parametrize(
    "results,expected",
    [
        ([], "empty"),
        (
            [
                {
                    "position": 1,
                    "url": "https://github.com/example/test",
                    "title": "Synthetic",
                    "snippet": "Test data",
                }
            ],
            "success",
        ),
        ([{"title": "malformed"}], "failed"),
    ],
)
async def test_search(results, expected):
    def handler(r):
        assert r.url == "https://web-search.apify.actor/"
        assert json.loads(r.content)["allowedDomains"] == ["github.com"]
        return httpx.Response(200, json={"query": "test", "search": {}, "results": results})

    async with client(handler) as api:
        out = await api.search(SearchInput(query="test"))
    assert out.measurement.outcome == expected


async def test_concurrent_partial_failures_are_ordered_and_bounded():
    active = peak = 0

    async def handler(r):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        if json.loads(r.content)["url"].endswith("/bad"):
            return httpx.Response(429, json={"error": "synthetic rate limit"})
        return httpx.Response(200, json=PAGE)

    async with client(handler) as api:
        out = await api.fetch_many(
            [REQUEST, FetchInput(url="https://example.com/bad"), REQUEST], concurrency=2
        )
    assert peak == 2
    assert api.peak_concurrency == 2
    assert api.active_requests == 0
    assert all(item.measurement.request_started_at for item in out)
    assert all(item.measurement.request_finished_at for item in out)
    assert max(item.measurement.concurrent_requests_at_start for item in out) == 2
    assert [o.measurement.outcome for o in out] == ["success", "failed", "success"]


async def test_classic_wait_dataset_and_matching_inputs():
    calls = []
    payloads = []

    def handler(r):
        calls.append((r.method, r.url.path))
        assert TOKEN not in str(r.url)
        assert r.headers["Authorization"] == f"Bearer {TOKEN}"
        if r.method == "POST":
            payloads.append(json.loads(r.content))
        if r.url.host == "web-fetch.apify.actor":
            return httpx.Response(200, json={**PAGE, "markdown": "#  Test\n data"})
        if r.method == "POST":
            return httpx.Response(201, json={"data": {"id": "test-run", "status": "RUNNING"}})
        if "actor-runs" in r.url.path:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "test-run",
                        "status": "SUCCEEDED",
                        "defaultDatasetId": "test-dataset",
                        "usageTotalUsd": 0.002,
                    }
                },
            )
        assert r.url.path == "/v2/datasets/test-dataset/items"
        return httpx.Response(200, json=[PAGE])

    async with client(handler) as api:
        outcomes = await asyncio.gather(api.fetch(REQUEST), api.fetch(REQUEST, classic=True))
    assert payloads[0] == payloads[1] == REQUEST.model_dump()
    assert len(calls) == 4
    assert outcomes[1].measurement.run_id == "test-run"
    assert outcomes[1].measurement.usage == {"usageTotalUsd": 0.002}
    assert compare_content(outcomes)["normalized_content_equal"] is True


@pytest.mark.parametrize("status", ["FAILED", "ABORTED", "TIMED-OUT"])
async def test_classic_failure_no_dataset_or_retries(status):
    calls = []

    def handler(r):
        calls.append(r)
        return httpx.Response(201, json={"data": {"id": "test", "status": status}})

    async with client(handler) as api:
        out = await api.fetch(REQUEST, classic=True)
    assert out.measurement.outcome == "failed"
    assert status in out.measurement.error
    assert len(calls) == 1


async def test_classic_exhausted_wait_retains_run_id():
    calls = []

    def handler(r):
        calls.append(r)
        return httpx.Response(200, json={"data": {"id": "test", "status": "RUNNING"}})

    async with client(handler) as api:
        out = await api.fetch(REQUEST, classic=True)
    assert len(calls) == 3
    assert out.measurement.run_id == "test"
    assert out.measurement.outcome == "uncertain"


@pytest.mark.parametrize("classic", [False, True])
async def test_transport_timeout_no_retry(classic):
    calls = []

    def handler(r):
        calls.append(r)
        raise httpx.ReadTimeout(f"Do not expose {TOKEN}", request=r)

    async with client(handler) as api:
        out = await api.fetch(REQUEST, classic=classic)
    assert len(calls) == 1
    assert out.measurement.outcome == "uncertain"
    assert TOKEN not in out.model_dump_json()


async def test_deadline_and_incomplete_comparison():
    async def handler(r):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json=PAGE)

    async with client(handler, deadline=0.001) as api:
        out = await api.fetch(REQUEST)
    assert out.measurement.outcome == "uncertain"
    assert compare_content([out, out])["normalized_content_equal"] is None


async def test_redaction_in_persistable_outcome():
    body = {**PAGE, "markdown": f"echo {TOKEN}", "Authorization": "Bearer another-secret"}
    async with client(lambda r: httpx.Response(200, json=body)) as api:
        out = await api.fetch(REQUEST)
    assert TOKEN not in out.model_dump_json()
    assert "another-secret" not in out.model_dump_json()
    redacted = redact(
        {"nested": ["https://example.com?token=other&x=1"], "cookie": "private"}, TOKEN
    )
    assert "other" not in json.dumps(redacted)
    assert "private" not in json.dumps(redacted)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/a",
        "http://localhost",
        "http://127.0.0.1",
        "https://user:pass@example.com",
        "https://example.com?token=secret",
    ],
)
def test_invalid_public_url(url):
    with pytest.raises(ValidationError):
        FetchInput(url=url)


@pytest.mark.parametrize("items", [[], [PAGE, PAGE], {"bad": "shape"}])
async def test_classic_missing_or_extra_dataset(items):
    def handler(r):
        if r.method == "POST":
            return httpx.Response(
                201,
                json={
                    "data": {"id": "test", "status": "SUCCEEDED", "defaultDatasetId": "test-data"}
                },
            )
        return httpx.Response(200, json=items)

    async with client(handler) as api:
        out = await api.fetch(REQUEST, classic=True)
    assert out.measurement.outcome == "failed"


async def test_non_json_response():
    async with client(lambda r: httpx.Response(200, text="not JSON")) as api:
        out = await api.fetch(REQUEST)
    assert out.measurement.outcome == "failed"
    assert out.raw[0]["body"]["non_json_body"] == "not JSON"
