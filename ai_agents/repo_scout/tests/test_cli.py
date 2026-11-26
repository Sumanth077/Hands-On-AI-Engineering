"""Local CLI checks; never creates a real HTTP transport."""

import json

import httpx
import pytest
from test_client import PAGE

from repo_scout import cli
from repo_scout.apify_client import ApifyClient


@pytest.mark.parametrize(
    "command", [["fetch", "https://example.com"], ["research", "Find projects"]]
)
def test_billable_guard(monkeypatch, command):
    monkeypatch.setattr("sys.argv", ["repo-scout", *command])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


async def test_cli_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "synthetic-test-token")
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "true")

    def handler(r):
        if r.url.host == "web-fetch.apify.actor":
            return httpx.Response(200, json=PAGE)
        if r.method == "POST":
            return httpx.Response(
                201,
                json={"data": {"id": "test", "status": "SUCCEEDED", "defaultDatasetId": "test"}},
            )
        return httpx.Response(200, json=[PAGE])

    monkeypatch.setattr(
        cli,
        "ApifyClient",
        lambda settings, **kwargs: ApifyClient(
            settings, transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    args = cli.parser().parse_args(
        ["--allow-billable", "--output-dir", str(tmp_path), "compare", "https://example.com"]
    )
    assert await cli.run(args) == 0
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 3
    record = json.loads(next(tmp_path.rglob("measurements.json")).read_text())
    assert record["comparison"]["normalized_content_equal"] is True
    assert all(m["model_generation_seconds"] is None for m in record["measurements"])
