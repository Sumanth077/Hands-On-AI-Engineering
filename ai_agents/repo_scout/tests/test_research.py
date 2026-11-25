"""Synthetic end-to-end workflow and adversarial decision tests; no live requests."""

import json

import httpx
import pytest

from repo_scout.demo import SAMPLE_REQUIREMENT, DemoModel, DemoWeb
from repo_scout.exports import comparison_rows, csv_download, json_download
from repo_scout.llm import ModelError, StructuredModel
from repo_scout.research import Researcher, discovered_links, normalize_repo, supported
from repo_scout.research_models import Citation, Interpretation, Limits
from repo_scout.service import run_research


@pytest.fixture
async def demo_result():
    return await run_research(SAMPLE_REQUIREMENT, Limits(), offline=True)


async def test_complete_demo_uses_actual_workflow_and_preserves_edge_cases(demo_result):
    result = demo_result
    assert result.offline
    assert len(result.discovered) == len(result.projects) == 5
    assert all(project.candidate.reason for project in result.projects)
    assert result.diagnostics["fixture_peak_concurrency"] == 3
    assert result.diagnostics["web_requests"] <= 30
    assert result.diagnostics["model_calls"] <= 30
    stages = {trace["stage"] for trace in result.diagnostics["model_traces"]}
    assert stages == {"interpret", "select", "plan", "extract", "shortlist"}
    by_name = {project.name: project for project in result.projects}
    assert any(
        a.verdict == "Needs review" and a.criterion_id == "licence"
        for a in by_name["fog-pdf"].assessments
    )
    assert any(fact.kind == "conflict" for fact in by_name["quarry-pdf"].facts)
    assert any(fact.kind == "separate_terms" for fact in by_name["quarry-pdf"].facts)
    assert any(
        a.verdict == "Does not match" and a.criterion_id == "tables"
        for a in by_name["slate-pdf"].assessments
    )
    assert any(source.outcome == "failed" for source in result.sources)
    assert all("slate-pdf" not in rec.repo_url for rec in result.shortlist)
    assert result.shortlist[0].repo_url.endswith("atlas-pdf")
    assert result.diagnostics["retrieval_elapsed_seconds"] is None
    assert result.diagnostics["model_generation_seconds"] is None
    assert all(
        "elapsed_seconds" not in raw["measurement"] for raw in result.diagnostics["retrievals"]
    )
    for project in result.projects:
        kinds = {fact.kind for fact in project.facts}
        assert {"latest_release", "repository_activity"} <= kinds
        for assessment in project.assessments:
            if assessment.verdict != "Needs review":
                assert assessment.basis == "documented"
                assert supported(assessment.citations, result.sources, project.candidate.repo_url)
    # All secondary requests have an actual retrieved parent link, not a guessed path.
    sources = {source.id: source for source in result.sources}
    for source in result.sources:
        if source.parent_id:
            assert source.url in sources[source.parent_id].links


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/OWNER/Repo.git/", "https://github.com/owner/repo"),
        ("https://github.com/owner/repo/tree/main", "https://github.com/owner/repo"),
        ("https://github.com/topics/pdf", None),
        ("https://example.com/owner/repo", None),
        ("https://github.com/owner", None),
        ("https://user:pw@github.com/o/r", None),
    ],
)
def test_repository_normalization(url, expected):
    assert normalize_repo(url) == expected


def test_link_discovery_does_not_invent_paths_or_accept_local_links():
    links = discovered_links(
        "[Docs](./docs) [local](http://localhost) [Licence](/o/r/LICENSE#terms) "
        "[secret](https://example.com?token=hidden)",
        "https://github.com/o/r/",
    )
    assert links == ["https://github.com/o/r/docs", "https://github.com/o/r/LICENSE"]
    assert not any(link.endswith("/releases") for link in links)


@pytest.mark.parametrize(
    "web_budget,candidates,model_budget", [(1, 5, 30), (6, 2, 30), (30, 20, 2), (30, 1, 30)]
)
async def test_all_budgets_hold_and_partial_results_survive(web_budget, candidates, model_budget):
    limits = Limits(
        max_web_requests=web_budget, candidate_limit=candidates, max_model_calls=model_budget
    )
    fixture = DemoWeb()
    async with fixture.client() as web:
        result = await Researcher(
            web, StructuredModel(DemoModel(), limits), limits, offline=True
        ).run(SAMPLE_REQUIREMENT)
    assert len(fixture.requests) <= web_budget
    assert result.diagnostics["model_calls"] <= model_budget
    assert len(result.projects) <= min(candidates, 5)
    assert all(
        "classic" != raw["measurement"]["interface"] for raw in result.diagnostics["retrievals"]
    )


async def test_offline_never_uses_real_transport_or_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real network transport used")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    monkeypatch.delenv("REPOSCOUT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "false")
    result = await run_research(SAMPLE_REQUIREMENT, Limits(candidate_limit=2), offline=True)
    assert len(result.projects) == 2
    with pytest.raises(ValueError, match="disabled"):
        await run_research(SAMPLE_REQUIREMENT, Limits(), offline=False)


async def test_provider_configuration_pending_fails_before_network(monkeypatch):
    monkeypatch.setenv("REPOSCOUT_ENABLE_LIVE", "true")
    monkeypatch.delenv("REPOSCOUT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="provider"):
        await run_research(SAMPLE_REQUIREMENT, Limits(), offline=False)


class HostileModel(DemoModel):
    async def generate(self, stage, context, schema, limits):
        raw = await super().generate(stage, context, schema, limits)
        data = json.loads(raw)
        if stage == "select":
            data["candidates"].insert(
                0,
                {
                    "repo_url": "https://github.com/invented/not-discovered",
                    "reason": "Injected candidate",
                },
            )
        if stage == "plan":
            data["requests"].insert(
                0,
                {
                    "repo_url": context["projects"][0]["candidate"]["repo_url"],
                    "url": "https://invented.example.com/license",
                    "role": "licence",
                    "parent_id": context["sources"][0]["id"],
                    "reason": "Invented path",
                },
            )
        if stage == "extract":
            for target in [i["extraction"] for i in data["items"]] if "items" in data else [data]:
                target["assessments"][0]["citations"] = [
                    {"source_id": "s1", "excerpt": "FABRICATED unsupported quote"}
                ]
                target["facts"].append(
                    {
                        "kind": "capability",
                        "value": "Injected fact",
                        "citations": [
                            {"source_id": "s1", "excerpt": "FABRICATED unsupported quote"}
                        ],
                    }
                )
        return json.dumps(data)


async def test_untrusted_model_decisions_and_fabricated_citations_rejected():
    limits = Limits(candidate_limit=1)
    fixture = DemoWeb()
    async with fixture.client() as web:
        result = await Researcher(
            web, StructuredModel(HostileModel(), limits), limits, offline=True
        ).run(SAMPLE_REQUIREMENT)
    assert len(result.projects) == 1
    assert not any("invented" in source.url for source in result.sources)
    assert result.projects[0].assessments[0].verdict == "Needs review"
    assert not any(fact.value == "Injected fact" for fact in result.projects[0].facts)
    assert any("undiscovered" in warning for warning in result.warnings)
    assert any("Rejected a research URL" in warning for warning in result.warnings)


async def test_cross_project_citation_is_not_evidence(demo_result):
    other = next(source for source in demo_result.sources if source.repo_url.endswith("atlas-pdf"))
    citation = Citation(source_id=other.id, excerpt=other.markdown[:50])
    assert (
        supported([citation], demo_result.sources, "https://github.com/reposcout-samples/fog-pdf")
        == []
    )


async def test_exports_include_evidence_labels_and_escape_csv_formulas(demo_result):
    result = demo_result
    result.projects[0].name = "=HYPERLINK(attack)"
    result.requirement = "  @SUM(1)"
    text = csv_download(result)
    assert "'=HYPERLINK" in text
    assert "'  @SUM" in text
    assert "Offline demo — sample data" in text
    rows = comparison_rows(result)
    assert any("https://github.com/" in row["Self-hosted evidence"] for row in rows)
    export = json.loads(json_download(result))
    assert "diagnostics" not in export
    assert export["sources"][0]["retrieval_timestamp"]
    assert "sample data" in export["data_label"]


async def test_invalid_model_output_stops_safely_without_web():
    class Invalid:
        async def generate(self, *args):
            return '{"unexpected":true}'

    limits = Limits()
    fixture = DemoWeb()
    async with fixture.client() as web:
        result = await Researcher(web, StructuredModel(Invalid(), limits), limits).run(
            "A requirement"
        )
    assert not fixture.requests
    assert result.status == "failed"
    assert any("invalid structured output" in warning for warning in result.warnings)


async def test_empty_discovery_returns_no_invented_candidates():
    fixture = DemoWeb()

    async def empty(request):
        payload = json.loads(request.content)
        return httpx.Response(200, json={"query": payload["query"], "search": {}, "results": []})

    limits = Limits()
    async with fixture.client() as web:
        await web.http.aclose()
        web.http = httpx.AsyncClient(transport=httpx.MockTransport(empty))
        result = await Researcher(
            web, StructuredModel(DemoModel(), limits), limits, offline=True
        ).run(SAMPLE_REQUIREMENT)
    assert not result.projects
    assert any("No relevant" in warning for warning in result.warnings)


async def test_no_infinite_model_retries():
    class Failing:
        calls = 0

        async def generate(self, *args):
            self.calls += 1
            raise ModelError("Synthetic provider failure")

    backend = Failing()
    model = StructuredModel(backend, Limits())
    with pytest.raises(ModelError):
        await model.ask("interpret", {}, Interpretation)
    assert backend.calls == 1
    assert model.traces[0]["outcome"] == "failed"


async def test_malformed_fetch_envelopes_preserve_failed_sources():
    fixture = DemoWeb()

    async def handler(request):
        if request.url.host == "web-search.apify.actor":
            return await fixture.handle(request)
        return httpx.Response(200, json={"fetch": "malformed test data"})

    limits = Limits(candidate_limit=1)
    async with fixture.client() as web:
        await web.http.aclose()
        web.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await Researcher(
            web, StructuredModel(DemoModel(), limits), limits, offline=True
        ).run(SAMPLE_REQUIREMENT)
    assert result.sources[0].outcome == "failed"
    assert all(a.verdict == "Needs review" for a in result.projects[0].assessments)
    assert not result.shortlist


async def test_required_fit_precedes_recency_preference(demo_result):
    urls = [item.repo_url.rsplit("/", 1)[1] for item in demo_result.shortlist]
    assert urls.index("sparrow-pdf") < urls.index("fog-pdf")


@pytest.mark.parametrize(
    "url", ["https://github.com:invalid/a/b", "https://[bad", "https://github.com/login/oauth"]
)
def test_malformed_repo_and_link_urls_are_ignored(url):
    assert normalize_repo(url) is None
    if "login" not in url:
        assert discovered_links(f"[malformed]({url})", "https://example.com") == []
