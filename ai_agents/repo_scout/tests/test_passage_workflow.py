"""Meaningful evidence-loss, quotation and candidate-capacity regressions."""

from datetime import datetime, timezone

from repo_scout.evidence import passages, select_passages
from repo_scout.research import supported
from repo_scout.research_models import Citation, Source


async def test_twenty_candidate_assessments_fit_model_call_budget_without_cross_citations():
    import json

    from repo_scout.llm import StructuredModel
    from repo_scout.research import Researcher
    from repo_scout.research_models import (
        Candidate,
        Criterion,
        Interpretation,
        Limits,
        Project,
        ResearchResult,
    )

    result = ResearchResult(
        requirement="Locally runnable projects",
        offline=False,
        created_at=datetime.now(timezone.utc),
        interpretation=Interpretation(
            summary="Local use",
            queries=["local software"],
            criteria=[
                Criterion(id="local", label="Local use", description="Runs locally", required=True)
            ],
        ),
    )
    for index in range(20):
        repo = f"https://github.com/example/project-{index}"
        result.projects.append(
            Project(name=f"project-{index}", candidate=Candidate(repo_url=repo, reason="Plausible"))
        )
        result.sources.append(
            source(f"Project {index} runs locally.").model_copy(
                update={"id": f"s{index}", "repo_url": repo, "url": repo}
            )
        )

    class Backend:
        async def generate(self, stage, context, schema, limits):
            items = []
            for project in context["projects"]:
                page = project["sources"][0]
                items.append(
                    {
                        "repo_url": project["repo_url"],
                        "extraction": {
                            "project_name": "Ignored",
                            "facts": [],
                            "licence_issues": [],
                            "assessments": [
                                {
                                    "criterion_id": "local",
                                    "verdict": "Matches",
                                    "explanation": "Documented local use",
                                    "basis": "documented",
                                    "citations": [
                                        {
                                            "source_id": page["id"],
                                            "passage_id": page["passages"][0]["id"],
                                            "excerpt": "",
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                )
            return json.dumps({"items": items})

    limits = Limits(candidate_limit=20)
    model = StructuredModel(Backend(), limits)
    await Researcher(None, model, limits)._extract_projects(result, result.projects)
    assert model.calls == 10
    for index, project in enumerate(result.projects):
        assert project.assessments[0].citations[0].excerpt == f"Project {index} runs locally."


def source(content):
    return Source(
        id="s1",
        repo_url="https://github.com/example/parser",
        url="https://github.com/example/parser",
        role="repository",
        retrieval_timestamp=datetime.now(timezone.utc),
        outcome="success",
        markdown=content,
    )


def test_large_retrieved_tail_reaches_context_without_destroying_archive():
    content = (
        "Navigation and project files\n" * 1500
        + "\nMIT License: permission to use and distribute.\n\n"
        + "Quick start: pip install parser\nparser local-input.pdf\nRuns on the local machine.\n\n"
        + "Table extraction returns rows and columns.\n\nCommits on Nov 25, 2026\n"
    )
    original = source(content)
    context, coverage = select_passages([original], 5000)
    text = "\n".join(p["text"] for s in context for p in s["passages"])
    assert "local-input.pdf" in text and "Table extraction" in text and "MIT License" in text
    assert original.markdown == content and len(original.markdown) > 12000
    assert coverage["characters"] <= 5000 and coverage["context_limited"]


def test_passage_citation_resolves_original_instead_of_generated_quote():
    page = source(
        "A **local** PDF parser uses `open('file.pdf')`.\nTable extraction is documented."
    )
    passage = passages(page.id, page.markdown)[0]
    citation = Citation(source_id="s1", passage_id=passage["id"], excerpt="Rephrased by model")
    valid = supported([citation], [page], page.repo_url)
    assert valid[0].excerpt == page.markdown
    assert not supported([citation], [page], "https://github.com/other/repo")
    assert not supported([Citation(source_id="s1", passage_id="s1:p999")], [page], page.repo_url)


def test_legacy_visible_formatting_matches_but_changed_claim_does_not():
    page = source(
        "The `DefaultReader` uses [ReaderLib](https://example.com/reader).\n**MIT License**"
    )
    cite = Citation(source_id="s1", excerpt="The DefaultReader uses ReaderLib.")
    assert supported([cite], [page], page.repo_url)
    assert not supported(
        [Citation(source_id="s1", excerpt="The DefaultReader uses a proprietary cloud service.")],
        [page],
        page.repo_url,
    )
    assert not supported(
        [Citation(source_id="s1", excerpt="[ReaderLib](https://attacker.com/reader)")],
        [page],
        page.repo_url,
    )
