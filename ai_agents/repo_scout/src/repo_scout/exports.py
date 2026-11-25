"""Evidence-aware export records; CSV cells cannot execute spreadsheet formulas."""

import csv
import io
import json

from .research_models import ResearchResult


def comparison_rows(result: ResearchResult) -> list[dict]:
    sources = {source.id: source for source in result.sources}
    criteria = (
        {criterion.id: criterion.label for criterion in result.interpretation.criteria}
        if result.interpretation
        else {}
    )
    rows = []
    for project in result.projects:
        row = {"Project": project.name, "Repository": project.candidate.repo_url}
        required = (
            {c.id for c in result.interpretation.criteria if c.required}
            if result.interpretation
            else set()
        )
        verdicts = {a.criterion_id: a.verdict for a in project.assessments}
        row["Overall"] = (
            "Excluded by required condition"
            if any(verdicts.get(c) == "Does not match" for c in required)
            else "Documented match"
            if required and all(verdicts.get(c) == "Matches" for c in required)
            else "Needs review"
        )
        for assessment in project.assessments:
            label = criteria.get(assessment.criterion_id, assessment.criterion_id)
            row[label] = assessment.verdict
            row[label + " evidence"] = " | ".join(
                dict.fromkeys(
                    sources[cite.source_id].url
                    for cite in assessment.citations
                    if cite.source_id in sources
                )
            )
        rows.append(row)
    return rows


def result_label(result: ResearchResult) -> str:
    if result.offline:
        return "Offline demo — sample data"
    if result.diagnostics.get("execution_mode") == "live_assessment_recovery":
        return "Live discovery and retrieval — automated assessment recovery from saved evidence"
    if result.diagnostics.get("execution_mode") == "automated_saved_evidence_replay":
        return "Automated production replay — saved live evidence; no manual assessment overrides"
    if result.diagnostics.get("offline_evidence_review"):
        return "Manually reviewed report — saved live evidence; not automated model output"
    return "Live research — documented evidence, not execution tests"


def json_download(result: ResearchResult) -> str:
    # Main export excludes raw diagnostic response bodies; evidence excerpts and provenance remain.
    data = result.model_dump(mode="json", exclude={"diagnostics"})
    data["data_label"] = result_label(result)
    return json.dumps(data, indent=2, ensure_ascii=False)


def csv_download(result: ResearchResult) -> str:
    rows = comparison_rows(result)
    for row in rows:
        row["Data mode"] = result_label(result)
        row["Requirement"] = result.requirement
        row["Result created at"] = result.created_at.isoformat()
    output = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["Project", "Data mode"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: "'" + value
                if isinstance(value, str)
                and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                else value
                for key, value in row.items()
            }
        )
    return output.getvalue()
