"""Synthetic minimized reproductions of defects traced in archived live responses."""

from datetime import datetime, timezone

import pytest

from repo_scout.llm import StructuredModel
from repo_scout.openai_backend import strict_schema
from repo_scout.reprocess import apply_review
from repo_scout.research import Researcher, discovered_links, evidence_window, supported
from repo_scout.research_models import (
    Assessment,
    Candidate,
    Citation,
    Criterion,
    Extraction,
    Interpretation,
    Limits,
    Project,
    ResearchResult,
    Source,
)

REPO = "https://github.com/example/parser"


def state(markdown):
    source = Source(
        id="s1",
        repo_url=REPO,
        url=REPO,
        role="readme",
        retrieval_timestamp=datetime.now(timezone.utc),
        outcome="success",
        markdown=markdown,
        links=[REPO, REPO + "/docs/usage.md"],
    )
    return ResearchResult(
        requirement="A permissively licensed local PDF table parser",
        offline=False,
        created_at=datetime.now(timezone.utc),
        interpretation=Interpretation(
            summary="Project-code licence",
            queries=["parser"],
            criteria=[
                Criterion(
                    id="licence",
                    label="Licence",
                    description="Permissive project code",
                    required=True,
                )
            ],
        ),
        sources=[source],
        projects=[
            Project(
                name="parser",
                candidate=Candidate(repo_url=REPO, reason="Discovered"),
                assessments=[
                    Assessment(
                        criterion_id="licence",
                        verdict="Needs review",
                        explanation="Old",
                        citations=[],
                        basis="unknown",
                    )
                ],
            )
        ],
    )


def test_markdown_escapes_and_bold_are_equivalent_but_link_targets_are_not():
    result = state(
        "pypdf\\_table\\_extraction only works with text-based PDFs.\n**Tables?** — Yes."
    )
    citations = [
        Citation(source_id="s1", excerpt="pypdf_table_extraction only works with text-based PDFs."),
        Citation(source_id="s1", excerpt="Tables? — Yes."),
    ]
    assert supported(citations, result.sources, REPO) == citations
    result.sources[0].markdown += "\n**PDF Parser for AI-ready data. Open-source.**"
    partial_bold = Citation(source_id="s1", excerpt="**PDF Parser for AI-ready data.")
    assert supported([partial_bold], result.sources, REPO) == [partial_bold]
    assert not supported(
        [Citation(source_id="s1", excerpt="works with scanned PDFs")], result.sources, REPO
    )
    result.sources[0].markdown = "[MIT](https://example.com/license)"
    assert not supported(
        [Citation(source_id="s1", excerpt="[MIT](https://attacker.com/license)")],
        result.sources,
        REPO,
    )
    assert not supported(citations, result.sources, REPO + "-other")


def test_late_local_example_and_table_path_survive_bounded_context():
    markdown = (
        "Navigation\n\n"
        + ("noise\n\n" * 2000)
        + "pip install parser\n\n"
        + "from parser import find_tables\n\n| Table extraction (simple borders) | Yes | Free |"
    )
    window = evidence_window(markdown, 1200)
    assert len(window) <= 1200
    assert "pip install parser" in window and "find_tables" in window
    assert "Table extraction (simple borders)" in window
    result = state(markdown)
    researcher = Researcher(None, None, Limits(max_source_chars=1200))
    assert "find_tables" in "\n".join(
        p["text"] for p in researcher._source_context(result, REPO)[0]["passages"]
    )


def test_planner_keeps_unseen_links_and_documentation_survives_navigation_cap():
    result = state("README")
    context = Researcher(None, None, Limits())._planning_sources(result)
    assert context[0]["links"] == [REPO + "/docs/usage.md"]
    markdown = "\n".join(f"[nav](https://example.com/{i})" for i in range(110))
    markdown += "\n[Usage](https://example.com/docs/usage.md)"
    assert "https://example.com/docs/usage.md" in discovered_links(markdown, REPO)


@pytest.mark.parametrize(
    "scope,conflicting,expected",
    [
        ("optional_unrelated", True, "Matches"),
        ("evaluated_path", True, "Needs review"),
        ("unknown", True, "Needs review"),
        ("evaluated_path", False, "Matches"),
    ],
)
async def test_conflict_requires_applicability_and_actual_conflicting_terms(
    scope, conflicting, expected
):
    result = state("MIT licence. Optional weights have separate terms.")

    class Backend:
        async def generate(self, *args):
            return Extraction(
                project_name="parser",
                facts=[
                    {
                        "kind": "conflict",
                        "value": "No conflicting code licence is documented.",
                        "citations": [{"source_id": "s1", "excerpt": "MIT licence."}],
                    }
                ],
                assessments=[
                    {
                        "criterion_id": "licence",
                        "verdict": "Matches",
                        "explanation": "Permissive code licence",
                        "basis": "documented",
                        "citations": [{"source_id": "s1", "excerpt": "MIT licence."}],
                    }
                ],
                licence_issues=[
                    {
                        "component": "weights",
                        "scope": scope,
                        "conflicting_terms": conflicting,
                        "explanation": "Terms for evaluated path",
                        "citations": [
                            {"source_id": "s1", "excerpt": "Optional weights have separate terms."}
                        ],
                    }
                ],
            ).model_dump_json()

    limits = Limits()
    await Researcher(None, StructuredModel(Backend(), limits), limits)._extract(
        result, result.projects[0]
    )
    assert result.projects[0].assessments[0].verdict == expected


def test_offline_review_rejects_unsupported_evidence_and_unresolved_shortlist():
    result = state("MIT licence.")
    change = {
        "repo_url": REPO,
        "assessment": {
            "criterion_id": "licence",
            "verdict": "Matches",
            "explanation": "Review",
            "basis": "documented",
            "citations": [{"source_id": "s1", "excerpt": "Apache licence."}],
        },
    }
    with pytest.raises(ValueError, match="unsupported"):
        apply_review(result, {"changes": [change]})
    with pytest.raises(ValueError, match="every required"):
        apply_review(
            result,
            {
                "changes": [],
                "shortlist": [
                    {
                        "repo_url": REPO,
                        "explanation": "Fit",
                        "tradeoffs": "None",
                        "citations": [{"source_id": "s1", "excerpt": "MIT licence."}],
                    }
                ],
            },
        )


def test_new_scoped_issues_remain_strict_on_openai_wire():
    schema = strict_schema(Extraction.model_json_schema())
    assert set(schema["required"]) == set(schema["properties"])
    assert "default" not in schema["properties"]["licence_issues"]
    assert set(schema["$defs"]["LicenceIssue"]["required"]) == set(
        schema["$defs"]["LicenceIssue"]["properties"]
    )
