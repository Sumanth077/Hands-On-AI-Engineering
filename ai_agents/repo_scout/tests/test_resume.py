"""Saved live-shaped state is synthetic here; resume must not rediscover or refetch it."""

import json

from repo_scout.demo import SAMPLE_REQUIREMENT, DemoModel, DemoWeb
from repo_scout.llm import StructuredModel
from repo_scout.research import Researcher
from repo_scout.research_models import Limits


async def test_planning_context_is_compact_and_resume_reuses_successes():
    limits = Limits(candidate_limit=2, max_iterations=1)
    fixture = DemoWeb()
    async with fixture.client() as web:
        previous = await Researcher(web, StructuredModel(DemoModel(), limits), limits).run(
            SAMPLE_REQUIREMENT
        )
    old_urls = {source.url for source in previous.sources}
    prior_calls = previous.diagnostics["model_calls"]
    resumed_fixture = DemoWeb()
    resumed_limits = limits.model_copy(update={"max_iterations": 2})
    async with resumed_fixture.client() as web:
        researcher = Researcher(web, StructuredModel(DemoModel(), resumed_limits), resumed_limits)
        context = researcher._planning_sources(previous)
        assert len(json.dumps(context)) < 15000
        result = await researcher.resume(previous)
    assert all("query" not in request for request in resumed_fixture.requests)
    assert all(request["url"] not in old_urls for request in resumed_fixture.requests)
    assert result.diagnostics["model_calls"] >= prior_calls
    assert result.diagnostics["web_requests"] <= resumed_limits.max_web_requests
    assert len(result.projects) == 2
