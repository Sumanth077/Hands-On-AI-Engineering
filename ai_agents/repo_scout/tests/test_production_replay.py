"""Synthetic control-flow tests; never substitutes fixtures for a live replay result."""

import pytest

from repo_scout.demo import SAMPLE_REQUIREMENT, DemoModel
from repo_scout.exports import csv_download, json_download, result_label
from repo_scout.llm import ModelBudgetError, StructuredModel
from repo_scout.research import Researcher
from repo_scout.research_models import Limits
from repo_scout.service import run_research


async def test_replay_discards_previous_assessments_and_never_calls_web():
    limits = Limits(candidate_limit=1)
    original = await run_research(SAMPLE_REQUIREMENT, limits, offline=True)
    original.offline = False  # Synthetic live-shaped state, only inside this offline test.
    original.projects[0].assessments[0].explanation = "MANUAL SENTINEL"

    class Inspect(DemoModel):
        async def generate(self, stage, context, schema, limits):
            assert "MANUAL SENTINEL" not in str(context)
            assert stage not in {"select", "plan"}
            return await super().generate(stage, context, schema, limits)

    result = await Researcher(None, StructuredModel(Inspect(), limits), limits).reassess_saved(
        original
    )
    assert result.diagnostics["web_requests"] == 0
    assert {t["stage"] for t in result.diagnostics["model_traces"]} == {
        "interpret",
        "extract",
        "shortlist",
    }
    assert "MANUAL SENTINEL" not in str(result.projects)


async def test_replay_budget_stop_keeps_sources_and_no_old_verdicts():
    limits = Limits(candidate_limit=1)
    original = await run_research(SAMPLE_REQUIREMENT, limits, offline=True)
    original.offline = False

    class Stop:
        async def generate(self, *args):
            raise ModelBudgetError("Budget stopped")

    result = await Researcher(None, StructuredModel(Stop(), limits), limits).reassess_saved(
        original
    )
    assert result.status == "failed"
    assert result.sources == original.sources
    assert not result.projects[0].assessments and not result.shortlist
    assert result.diagnostics["web_requests"] == 0


async def test_demo_evidence_cannot_be_mistaken_for_production_replay():
    original = await run_research(SAMPLE_REQUIREMENT, Limits(candidate_limit=1), offline=True)
    with pytest.raises(ValueError, match="live evidence"):
        await Researcher(None, None, Limits()).reassess_saved(original)


async def test_recovery_keeps_limits_history_and_clears_resolved_omission():
    limits = Limits(candidate_limit=1)
    original = await run_research(SAMPLE_REQUIREMENT, limits, offline=True)
    original.offline = False
    old_calls = original.diagnostics["model_calls"]
    old_web = original.diagnostics["web_requests"]
    for trace in original.diagnostics["model_traces"]:
        trace["elapsed_seconds"] = 0  # Synthetic live-shaped timing only.
    warning = f"Model omitted assessment for {original.projects[0].name}; retained as unresolved"
    original.warnings.append(warning)

    class Inspect(DemoModel):
        async def generate(self, stage, context, schema, limits):
            assert stage in {"extract", "shortlist"}
            return await super().generate(stage, context, schema, limits)

    result = await Researcher(None, StructuredModel(Inspect(), limits), limits).recover_assessments(
        original
    )
    assert result.interpretation == original.interpretation
    assert result.sources == original.sources
    assert result.diagnostics["web_requests"] == old_web
    assert result.diagnostics["new_web_requests"] == 0
    assert result.diagnostics["model_calls"] == old_calls + 2
    assert warning not in result.warnings
    assert warning in original.warnings


async def test_export_origin_distinguishes_review_and_automated_replay():
    result = await run_research(SAMPLE_REQUIREMENT, Limits(candidate_limit=1), offline=True)
    result.offline = False
    result.diagnostics["offline_evidence_review"] = {"external_calls": 0}
    assert "Manually reviewed" in result_label(result)
    result.diagnostics["execution_mode"] = "automated_saved_evidence_replay"
    assert "Automated production replay" in json_download(result)
    assert "Automated production replay" in csv_download(result)
