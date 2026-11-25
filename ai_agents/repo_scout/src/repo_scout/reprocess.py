"""Offline, explicitly reviewed corrections over archived live evidence; no providers."""

from .research import discovered_links, supported
from .research_models import Assessment, Interpretation, Recommendation, ResearchResult


def restore_saved_sources(result: ResearchResult) -> ResearchResult:
    result = result.model_copy(deep=True)
    responses = {
        item["measurement"]["sanitized_input"].get("url"): item
        for item in result.diagnostics.get("retrievals", [])
        if item["measurement"]["outcome"] == "success"
    }
    for source in result.sources:
        response = responses.get(source.url)
        if source.outcome != "success" or not response or not response.get("content"):
            continue
        # Restore only already archived Actor content; no URL retrieval or inference.
        source.markdown = response["content"]
        source.links = discovered_links(source.markdown, source.loaded_url or source.url)
        source.truncated = False
    return result


def apply_review(result: ResearchResult, review: dict) -> ResearchResult:
    """Apply an auditable review manifest, not an automatic positive-match heuristic.

    All revised findings must pass the normal candidate/excerpt provenance guard.
    Semantic entailment is the reviewer's responsibility and remains documented.
    """
    result = restore_saved_sources(result)
    original_interpretation = result.interpretation.model_dump()
    if "interpretation" in review:
        corrected = Interpretation.model_validate(review["interpretation"])
        if {(c.id, c.required) for c in corrected.criteria} != {
            (c.id, c.required) for c in result.interpretation.criteria
        }:
            raise ValueError("Review cannot remove or downgrade required criteria")
        result.interpretation = corrected
    projects = {p.candidate.repo_url: p for p in result.projects}
    for change in review["changes"]:
        project = projects[change["repo_url"]]
        assessment = Assessment.model_validate(change["assessment"])
        if not assessment.citations or len(
            supported(assessment.citations, result.sources, project.candidate.repo_url)
        ) != len(assessment.citations):
            raise ValueError("Review contains unsupported or cross-candidate evidence")
        old = next(a for a in project.assessments if a.criterion_id == assessment.criterion_id)
        project.assessments[project.assessments.index(old)] = assessment
    # Delete only explicitly reviewed false-conflict statements; preserve original archive.
    for removal in review.get("remove_false_conflicts", []):
        project = projects[removal["repo_url"]]
        project.facts = [
            f for f in project.facts if not (f.kind == "conflict" and f.value == removal["value"])
        ]
    required = {c.id for c in result.interpretation.criteria if c.required}
    result.shortlist = []
    for item in review.get("shortlist", []):
        recommendation = Recommendation.model_validate(item)
        project = projects[recommendation.repo_url]
        if any(a.verdict != "Matches" for a in project.assessments if a.criterion_id in required):
            raise ValueError("Reviewed shortlist requires every required criterion to match")
        if len(supported(recommendation.citations, result.sources, recommendation.repo_url)) != len(
            recommendation.citations
        ):
            raise ValueError("Unsupported reviewed shortlist")
        result.shortlist.append(recommendation)
    result.diagnostics["offline_evidence_review"] = {
        "external_calls": 0,
        "origin": "Archived live Apify evidence; offline agent review, no LLM generation or fixtures",
        "review": review,
        "original_interpretation": original_interpretation,
        "historical_warnings": list(result.warnings),
    }
    result.warnings = [
        "Offline review of saved live evidence; original reports and usage preserved."
    ]
    result.status = (
        "partial"
        if any(a.verdict == "Needs review" for p in result.projects for a in p.assessments)
        else "complete"
    )
    return result
