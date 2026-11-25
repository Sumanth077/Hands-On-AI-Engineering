"""Run with: uv run streamlit run streamlit_app.py"""

import asyncio
import os
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from repo_scout.evidence_display import evidence_preview
from repo_scout.exports import comparison_rows, csv_download, json_download, result_label
from repo_scout.research_models import Limits, ResearchResult
from repo_scout.service import (
    active_provider_label,
    configured_limits,
    live_configuration_ready,
    live_setup_issues,
    run_research,
)


def show_citations(citations, sources):
    for cite in citations:
        source = sources.get(cite.source_id)
        if source:
            st.link_button(f"Evidence {source.id} · {source.role}", source.url)
            st.markdown(
                evidence_preview(cite.excerpt, source.markdown, source.loaded_url or source.url),
                unsafe_allow_html=False,
            )
            st.caption(f"Retrieved {source.retrieval_timestamp.isoformat()} · documented evidence")
            with st.expander("View raw source", expanded=False):
                st.caption("Original cited excerpt · unchanged")
                st.code(cite.excerpt, language="markdown", wrap_lines=True)
                st.caption("Full retrieved source · unchanged")
                st.code(source.markdown, language="markdown", wrap_lines=True)


OUTPUTS = Path(__file__).resolve().parent / "outputs"


@st.cache_data(show_spinner=False)
def load_saved_session(path: str, modified_ns: int) -> dict:
    """Read real local results; the timestamp invalidates the read-only cache."""
    saved_path = Path(path).resolve()
    if not saved_path.is_relative_to(OUTPUTS.resolve()):
        raise ValueError("Saved session must be inside outputs")
    saved = ResearchResult.model_validate_json(saved_path.read_text(encoding="utf-8"))
    if saved.offline:
        raise ValueError("Saved session must contain real research")
    return saved.model_dump(mode="json")


st.set_page_config(page_title="RepoScout", page_icon="🔎", layout="wide")
st.title("RepoScout")
st.write("Find open-source projects that fit your requirements, with evidence you can inspect.")
st.caption("Reasoning provider: " + active_provider_label())
live_ready = live_configuration_ready()
if not live_ready:
    st.info(
        "Live research is unavailable: "
        + "; ".join(live_setup_issues())
        + ". Credentials stay local."
    )
else:
    st.warning(
        "Live research is enabled. Starting a new run will use billable web and model APIs within the configured limits."
    )

with st.form("research_form"):
    requirement = st.text_area(
        "What do you need?",
        placeholder="Describe the projects and capabilities you need.",
        height=130,
        max_chars=4000,
        key="requirement",
    )
    candidate_limit = st.number_input(
        "Candidate limit", min_value=1, max_value=20, value=5, step=1, key="candidate_limit"
    )
    start = st.form_submit_button("Start research", type="primary", disabled=not live_ready)

if start:
    progress = st.progress(0.0, text="Preparing research")
    try:
        limits = configured_limits(int(candidate_limit))
        result = asyncio.run(
            run_research(
                requirement,
                limits,
                offline=False,
                progress=lambda label, fraction: progress.progress(fraction, text=label),
            )
        )
        st.session_state["research_result"] = result.model_dump(mode="json")
    except (ValueError, ValidationError):
        st.error(
            "Check the requirement and configured limits. Any previously saved result is retained."
        )
    except Exception:
        # Do not echo provider exceptions, prompts, tokens, or raw responses in the task UI.
        st.error("Research stopped unexpectedly. A new run requires pressing Start research again.")
    finally:
        progress.empty()

# Loading saved real sessions is independent of credentials and never calls APIs.
with st.expander("Saved research sessions"):
    saved_paths = []
    for saved_path in sorted(OUTPUTS.glob("live-*/result.json"), reverse=True):
        try:
            load_saved_session(str(saved_path), saved_path.stat().st_mtime_ns)
            saved_paths.append(str(saved_path))
        except (OSError, ValueError):
            continue
    selected_session = st.selectbox(
        "Saved session",
        saved_paths,
        index=None,
        placeholder="Choose a saved research session",
        format_func=lambda path: Path(path).parent.name,
    )
    if st.button("Open saved session", disabled=selected_session is None):
        try:
            st.session_state["research_result"] = load_saved_session(
                selected_session, Path(selected_session).stat().st_mtime_ns
            )
        except (OSError, ValueError):
            st.error("The saved session could not be loaded. Existing results are retained.")
    if not saved_paths:
        st.caption("No saved research sessions are available yet.")

# Ignore legacy fictional in-memory state without changing any archived files.
if st.session_state.get("research_result", {}).get("offline", False):
    del st.session_state["research_result"]

# Optional local recovery hook for an existing saved real result.
restore = os.getenv("REPOSCOUT_RESTORE_RESULT")
if restore and "research_result" not in st.session_state:
    try:
        restore_path = Path(restore).resolve()
        st.session_state["research_result"] = load_saved_session(
            str(restore_path), restore_path.stat().st_mtime_ns
        )
    except (OSError, ValueError):
        st.info("The configured saved research session is unavailable.")

if "research_result" in st.session_state:
    result = ResearchResult.model_validate(st.session_state["research_result"])
    if not result.offline and result.status == "partial" and result.projects:
        if st.button("Continue unresolved research", disabled=not live_configuration_ready()):
            saved_limits = Limits.model_validate(result.diagnostics["limits"])
            progress = st.progress(0.4, text="Resuming saved evidence")
            result = asyncio.run(
                run_research(
                    result.requirement,
                    saved_limits,
                    offline=False,
                    resume_result=result,
                    progress=lambda label, fraction: progress.progress(fraction, text=label),
                )
            )
            st.session_state["research_result"] = result.model_dump(mode="json")
            progress.empty()
    st.divider()
    st.subheader("Research results")
    st.caption(f"Saved result · {result.created_at.isoformat()} · {result.status}")
    st.caption(result_label(result))
    st.text(result.requirement)
    if result.interpretation:
        st.text(result.interpretation.summary)
        with st.expander("Comparison criteria and search queries"):
            for criterion in result.interpretation.criteria:
                st.text(
                    f"{'Required' if criterion.required else 'Preference'} · {criterion.label}: {criterion.description}"
                )
            for query in result.interpretation.queries:
                st.text(query)
    for warning in dict.fromkeys(result.warnings):
        st.info(warning)
    if result.selection_decisions:
        with st.expander("Discovery selection and exclusion reasons"):
            for decision in result.selection_decisions:
                st.text(f"{decision.action} · {decision.repo_url}: {decision.reason}")
    if result.projects:
        st.dataframe(
            comparison_rows(result),
            hide_index=True,
            width="stretch",
            column_config={"Repository": st.column_config.LinkColumn("Repository")},
        )
    else:
        st.info("No candidates to compare. No missing evidence was treated as a negative finding.")
    st.caption(
        "Matches and failures describe documented evidence. No project capabilities were execution-tested. Each assessment's source links also appear in expandable evidence below."
    )

    sources = {source.id: source for source in result.sources}
    st.subheader("Shortlist")
    for index, item in enumerate(result.shortlist, 1):
        st.link_button(f"{index}. {item.repo_url.rsplit('/', 1)[1]}", item.repo_url)
        st.text(item.explanation)
        st.text(item.tradeoffs)
        project = next(p for p in result.projects if p.candidate.repo_url == item.repo_url)
        unresolved = [a.criterion_id for a in project.assessments if a.verdict == "Needs review"]
        if unresolved:
            st.caption("Provisional fit · manual review required: " + ", ".join(unresolved))
        with st.expander(f"Shortlist evidence · {item.repo_url.rsplit('/', 1)[1]}"):
            show_citations(item.citations, sources)
    if not result.shortlist:
        st.write(
            "No supported shortlist is available. Review the evidence and unresolved criteria."
        )

    st.subheader("Project evidence")
    for project in result.projects:
        with st.expander(project.name):
            st.link_button("Repository", project.candidate.repo_url)
            st.text("Selected because: " + project.candidate.reason)
            for assessment in project.assessments:
                criterion = next(
                    c for c in result.interpretation.criteria if c.id == assessment.criterion_id
                )
                st.text(f"{criterion.label} · {assessment.verdict} · {assessment.basis}")
                st.text(assessment.explanation)
                if assessment.processing_note:
                    st.caption(
                        {
                            "context_limited": "Selected passages were assessed; additional retrieved content is retained in the report.",
                            "citation_rejected": "An evidence reference was rejected. The page is retained; this is not evidence that the capability is absent.",
                            "unresolved": "Available evidence does not settle this condition.",
                        }.get(assessment.processing_note, assessment.processing_note)
                    )
                show_citations(assessment.citations, sources)
            for fact in project.facts:
                st.text(f"{fact.kind.replace('_', ' ').title()}: {fact.value}")
                show_citations(fact.citations, sources)
            for source in result.sources:
                if source.repo_url == project.candidate.repo_url and source.outcome != "success":
                    st.text(f"Unavailable {source.role} evidence: {source.error or source.outcome}")
                    st.link_button(f"Failed source {source.id}", source.url)

    left, right = st.columns(2)
    left.download_button(
        "Download JSON",
        json_download(result),
        "reposcout-research.json",
        "application/json",
        key="download_json",
    )
    right.download_button(
        "Download CSV",
        csv_download(result),
        "reposcout-comparison.csv",
        "text/csv",
        key="download_csv",
    )
    with st.expander("Diagnostics · retrieval traces and benchmark tooling"):
        st.caption(
            "Retrieval and model timings are recorded separately. These research timings are not a classic/server benchmark."
        )
        st.json(result.diagnostics)
        st.code(
            "uv run repo-scout --allow-billable compare https://example.com", language="powershell"
        )
        st.caption(
            "Separate classic/server tooling is retained for later approved live tests. It is never invoked by normal research. REPOSCOUT_ENABLE_LIVE defaults to false. The earlier smoke test is deferred."
        )
