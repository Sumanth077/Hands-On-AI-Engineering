"""Minimized reproductions of the empty-selection contract and Actor schema gaps."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from repo_scout.apify_client import ApifyClient
from repo_scout.config import Settings
from repo_scout.demo import DATA, SAMPLE_REQUIREMENT, DemoModel, DemoWeb
from repo_scout.llm import ModelError, StructuredModel
from repo_scout.models import FetchInput, SearchInput
from repo_scout.research import Researcher
from repo_scout.research_models import Discovered, Interpretation, Limits, ResearchResult, Selection


def checkpoint():
    return ResearchResult(
        requirement=SAMPLE_REQUIREMENT,
        offline=False,
        created_at=datetime.now(timezone.utc),
        interpretation=Interpretation.model_validate(DATA["interpretation"]),
        discovered=[
            Discovered(
                repo_url="https://github.com/reposcout-samples/atlas-pdf",
                title="Synthetic PDF library",
                snippet="PDF parsing; inspect unknown licence and dates.",
                search_urls=["https://github.com/reposcout-samples/atlas-pdf"],
            )
        ],
        diagnostics={
            "web_requests": 3,
            "model_calls": 2,
            "model_traces": [],
            "retrievals": [],
            "retrieval_elapsed_seconds": 70,
        },
    )


async def test_original_empty_response_requires_reasons_and_does_not_force_candidates():
    class Backend:
        async def generate(self, *args):
            return '{"candidates":[]}'

    result = checkpoint()
    with pytest.raises(ModelError, match="every discovered"):
        await Researcher(None, StructuredModel(Backend(), Limits()), Limits())._select_candidates(
            result
        )
    assert result.projects == []


async def test_reasoned_empty_selection_is_valid_and_keeps_exclusion_reason():
    result = checkpoint()
    repo = result.discovered[0].repo_url

    class Backend:
        async def generate(self, *args):
            return json.dumps(
                {
                    "candidates": [],
                    "decisions": [
                        {
                            "repo_url": repo,
                            "action": "exclude",
                            "reason": "Supplied title is unrelated to the requested subject",
                        }
                    ],
                }
            )

    await Researcher(None, StructuredModel(Backend(), Limits()), Limits())._select_candidates(
        result
    )
    assert not result.projects
    assert result.selection_decisions[0].reason


async def test_unknown_fields_do_not_prevent_inspection_and_resume_never_searches():
    result = checkpoint()
    fixture = DemoWeb()
    async with fixture.client() as web:
        resumed = await Researcher(
            web, StructuredModel(DemoModel(), Limits()), Limits()
        ).resume_discovery(result)
    assert resumed.projects
    assert resumed.diagnostics["previous_discovery_requests_reused"] == 3
    assert all("query" not in request for request in fixture.requests)
    assert resumed.diagnostics["web_requests"] == 3 + len(fixture.requests)
    assert resumed.diagnostics["model_calls"] > 2
    assert len({request["url"] for request in fixture.requests}) == len(fixture.requests)


async def test_null_search_snippet_keeps_entire_results_page_usable():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "query": "PDF",
                "search": {},
                "results": [
                    {
                        "position": 1,
                        "url": "https://github.com/example/one",
                        "title": "Parser",
                        "snippet": None,
                    },
                    {
                        "position": 2,
                        "url": "https://github.com/example/two",
                        "title": "Tables",
                        "snippet": "PDF tables",
                    },
                ],
            },
        )

    async with ApifyClient(
        Settings(token="synthetic"), transport=httpx.MockTransport(handler)
    ) as web:
        outcome = await web.search(SearchInput(query="PDF"))
    assert outcome.measurement.outcome == "success"
    assert outcome.raw[-1]["body"]["results"][0]["snippet"] is None


async def test_requested_text_and_structured_links_survive_retrieval():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "url": "https://example.com/plain",
                "fetch": {"httpStatusCode": 200, "loadedUrl": "https://example.com/plain"},
                "metadata": {},
                "markdown": None,
                "text": "MIT License\nLocal table parser.",
                "links": ["https://example.com/usage", "http://localhost/private"],
            },
        )

    limits = Limits(candidate_limit=1)
    r = checkpoint()
    r.projects = []
    from repo_scout.research_models import Candidate, Project, SourceRequest

    repo = r.discovered[0].repo_url
    r.projects = [Project(name="one", candidate=Candidate(repo_url=repo, reason="Plausible"))]
    async with ApifyClient(
        Settings(token="synthetic"), transport=httpx.MockTransport(handler)
    ) as web:
        await Researcher(web, None, limits)._retrieve(
            r,
            [
                SourceRequest(
                    repo_url=repo,
                    url="https://example.com/plain",
                    role="licence",
                    parent_id="discovery",
                    reason="Verify",
                )
            ],
        )
    assert r.sources[0].markdown.startswith("MIT License")
    assert r.sources[0].links == ["https://example.com/usage"]


async def test_actor_success_does_not_accept_target_404_even_with_text():
    async with ApifyClient(
        Settings(token="synthetic"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "url": "https://example.com/missing",
                    "fetch": {"httpStatusCode": 404, "loadedUrl": "https://example.com/missing"},
                    "metadata": {},
                    "text": "Not found",
                    "links": [],
                },
            )
        ),
    ) as web:
        outcome = await web.fetch(
            FetchInput(url="https://example.com/missing", formats=["markdown", "text", "links"])
        )
    assert outcome.measurement.target_status == 404
    assert outcome.measurement.outcome == "failed" and outcome.content is None


async def test_saved_explicit_candidate_reason_fills_only_redundant_missing_inspect_decision():
    result = checkpoint()
    selection = Selection.model_validate(
        {
            "candidates": [
                {
                    "repo_url": result.discovered[0].repo_url,
                    "reason": "Plausible lead; all qualifications need retrieval",
                }
            ],
            "decisions": [],
        }
    )

    class NoCall:
        async def ask(self, *args):
            raise AssertionError("Saved selection must not generate again")

    await Researcher(None, NoCall(), Limits())._select_candidates(result, selection_cache=selection)
    assert len(result.projects) == 1
    assert result.selection_decisions[0].action == "inspect"
    assert result.selection_decisions[0].reason == selection.candidates[0].reason


async def test_selected_candidate_cannot_override_explicit_exclusion():
    result = checkpoint()
    repo = result.discovered[0].repo_url
    selection = Selection.model_validate(
        {
            "candidates": [{"repo_url": repo, "reason": "Inspect"}],
            "decisions": [
                {"repo_url": repo, "action": "exclude", "reason": "Conflicting decision"}
            ],
        }
    )
    with pytest.raises(ModelError, match="disagree"):
        await Researcher(None, None, Limits())._select_candidates(result, selection_cache=selection)
    assert result.projects == []
