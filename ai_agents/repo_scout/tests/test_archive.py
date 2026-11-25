"""Archive and terminal-auth regression checks use synthetic data only."""

import json

import httpx

from repo_scout.archive import RunArchive
from repo_scout.demo import SAMPLE_REQUIREMENT, DemoModel, DemoWeb
from repo_scout.llm import ModelAuthenticationError, StructuredModel
from repo_scout.research import Researcher
from repo_scout.research_models import Limits
from repo_scout.service import run_research


async def test_archive_sanitizes_and_persists_complete_result(tmp_path):
    result = await run_research(SAMPLE_REQUIREMENT, Limits(candidate_limit=1), offline=True)
    archive = RunArchive(["synthetic-secret"], root=tmp_path)
    archive.record("synthetic-test", {"value": "synthetic-secret", "api_key": "never expose"})
    archive.record(
        "usage-test",
        {
            "prompt_token_count": 123,
            "totalTokenCount": 321,
            "max_output_tokens": 4000,
            "token": 123456,
        },
    )
    saved = archive.save(result)
    journal = (archive.folder / "responses.jsonl").read_text(encoding="utf-8")
    assert "synthetic-secret" not in journal and "never expose" not in journal
    usage = json.loads(journal.splitlines()[1])["data"]
    assert usage["prompt_token_count"] == 123
    assert usage["totalTokenCount"] == 321
    assert usage["max_output_tokens"] == 4000
    assert usage["token"] == "[REDACTED]"
    assert json.loads((archive.folder / "result.json").read_text(encoding="utf-8"))["projects"]
    assert (archive.folder / "comparison.csv").is_file()
    assert saved.diagnostics["saved_result_dir"] == str(archive.folder)


async def test_authentication_failure_stops_before_repeated_model_calls():
    class AuthFailure(DemoModel):
        stages = []

        async def generate(self, stage, context, schema, limits):
            self.stages.append(stage)
            if stage == "extract":
                raise ModelAuthenticationError("Synthetic authentication failure")
            return await super().generate(stage, context, schema, limits)

    backend = AuthFailure()
    limits = Limits(candidate_limit=2)
    async with DemoWeb().client() as web:
        result = await Researcher(web, StructuredModel(backend, limits), limits).run(
            SAMPLE_REQUIREMENT
        )
    assert backend.stages.count("extract") == 1
    assert backend.stages[-1] == "extract"
    assert result.sources
    assert any("authentication failure" in warning for warning in result.warnings)


async def test_apify_authentication_failure_retains_outcomes_and_stops_discovery():
    fixture = DemoWeb()
    async with fixture.client() as web:
        await web.http.aclose()
        web.http = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(401, json={"error": "synthetic"})
            )
        )
        model = StructuredModel(DemoModel(), Limits())
        result = await Researcher(web, model, Limits()).run(SAMPLE_REQUIREMENT)
    assert model.calls == 1
    assert result.diagnostics["retrievals"]
    assert any("Apify authentication" in warning for warning in result.warnings)


async def test_pending_fetches_are_not_sent_after_auth_failure():
    from repo_scout.models import FetchInput

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": "synthetic authentication failure"})

    async with DemoWeb().client() as web:
        await web.http.aclose()
        web.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        outcomes = await web.fetch_many(
            [FetchInput(url="https://example.com") for _ in range(3)],
            concurrency=1,
            stop_on_auth_quota=True,
        )
    assert len(calls) == 1
    assert outcomes[0].measurement.actor_http_status == 401
    assert all(out.measurement.error.startswith("Not requested:") for out in outcomes[1:])
