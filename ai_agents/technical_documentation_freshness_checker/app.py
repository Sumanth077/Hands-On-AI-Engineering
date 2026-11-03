from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from freshness_checker.ade import parse_and_extract
from freshness_checker.claims import select_claims
from freshness_checker.decisions import check_claims
from freshness_checker.models import FreshnessResult, FreshnessStatus, TechnicalClaim

load_dotenv()

st.set_page_config(page_title="Documentation Freshness Checker", page_icon="🧭", layout="wide")


def _missing_keys() -> list[str]:
    return [
        key
        for key in ("VISION_AGENT_API_KEY", "PARALLEL_API_KEY")
        if not os.getenv(key)
    ]


def _reset_results() -> None:
    for key in ("metadata", "results", "document_name", "checked_claim_count"):
        st.session_state.pop(key, None)


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.write(uploaded_file.getbuffer())
    finally:
        temp_file.close()
    return Path(temp_file.name)


@st.cache_data(ttl="20m", max_entries=3, show_spinner=False)
def _analyze_document(file_bytes: bytes, file_name: str) -> tuple[list[TechnicalClaim], dict]:
    """Cache a parse/extract result briefly to avoid reprocessing the same PDF."""
    class Upload:
        name = file_name

        def getbuffer(self) -> bytes:
            return file_bytes

    document_path = _save_upload(Upload())
    try:
        _, extracted_claims, metadata = parse_and_extract(document_path)
        return extracted_claims, metadata
    finally:
        document_path.unlink(missing_ok=True)


def _status_color(status: FreshnessStatus) -> str:
    return {
        FreshnessStatus.CURRENT: "green",
        FreshnessStatus.OUTDATED: "red",
        FreshnessStatus.NEEDS_REVIEW: "orange",
    }[status]


def _render_result(result: FreshnessResult, number: int) -> None:
    with st.container(border=True):
        heading, badge = st.columns([5, 1])
        heading.markdown(f"**{number}. {result.claim.claim}**")
        badge.markdown(
            f":{_status_color(result.status)}[**{result.status.value}**]"
        )
        st.caption(f"{result.claim.category.value.replace('_', ' ').title()} · {result.claim.why_current_context_matters}")

        if result.confidence is not None:
            st.caption(f"Decision confidence: {result.confidence:.0%}")
        if result.error:
            st.error(result.error)

        left, right = st.columns(2)
        with left:
            with st.expander("Document evidence", expanded=False):
                st.markdown(result.claim.source_quote)
                st.caption(result.claim.context)
        with right:
            with st.expander("Live web evidence", expanded=False):
                if not result.citations:
                    st.caption("No web citations were returned. Treat this result as needing review.")
                for citation in result.citations:
                    title = citation.title or citation.url
                    st.markdown(f"[{title}]({citation.url})")
                    for excerpt in citation.excerpts:
                        st.caption(excerpt)


st.title("Technical Documentation Freshness Checker")
st.write(
    "Find technical statements in a PDF that can go stale, then verify only those claims "
    "against the live web."
)

with st.sidebar:
    st.header("Check a document")
    uploaded = st.file_uploader("Technical PDF", type=["pdf"], on_change=_reset_results)
    max_claims = int(os.getenv("MAX_CLAIMS", "25"))
    st.caption(
        "LandingAI ADE identifies stale-prone claims. Parallel automatically checks up to "
        f"{max_claims} distinct claims against the live web."
    )

missing = _missing_keys()
if missing:
    st.warning(f"Add {', '.join(missing)} to `.env`, then restart the app.")

if not uploaded:
    st.info(
        "Upload an API guide, migration document, architecture document, or implementation PDF. "
        "The checker ignores ordinary prose and focuses on versions, endpoints, dependencies, "
        "configuration, deprecations, models, and external integrations."
    )
    st.stop()

if st.button("Check document freshness", type="primary", disabled=bool(missing)):
    try:
        with st.status("Checking the document against current technical information...", expanded=True) as status:
            st.write("LandingAI ADE is converting the PDF into layout-aware Markdown.")
            extracted_claims, metadata = _analyze_document(uploaded.getvalue(), uploaded.name)
            claims_to_check = select_claims(extracted_claims, max_claims)
            st.write(f"ADE identified {len(claims_to_check)} stale-prone technical claims.")
            st.write("Parallel Decisions is checking each claim against current web evidence.")
            st.session_state.results = check_claims(claims_to_check)
            status.update(label="Freshness decisions ready", state="complete")
        st.session_state.metadata = metadata
        st.session_state.document_name = uploaded.name
        st.session_state.checked_claim_count = len(claims_to_check)
    except Exception as error:
        st.error(f"Could not process this PDF: {error}")

results: list[FreshnessResult] | None = st.session_state.get("results")
if results:
    st.subheader("Freshness decisions")
    counts = {status: sum(item.status == status for item in results) for status in FreshnessStatus}
    metric_columns = st.columns(3)
    metric_columns[0].metric("Current", counts[FreshnessStatus.CURRENT])
    metric_columns[1].metric("Outdated", counts[FreshnessStatus.OUTDATED])
    metric_columns[2].metric("Needs review", counts[FreshnessStatus.NEEDS_REVIEW])
    st.caption(
        "LandingAI ADE selected the claims automatically. Parallel Decisions checked each one against "
        "the live web. The items below need attention."
    )
    exceptions = [result for result in results if result.status != FreshnessStatus.CURRENT]
    if not exceptions:
        st.success("No stale or uncertain claims were found in the checked set.")
    for index, result in enumerate(exceptions, start=1):
        _render_result(result, index)

    current_results = [result for result in results if result.status == FreshnessStatus.CURRENT]
    if current_results:
        with st.expander(f"Checked and still current ({len(current_results)})"):
            st.caption("These claims were verified automatically and did not need attention.")
            for index, result in enumerate(current_results, start=1):
                _render_result(result, index)

    st.divider()
    st.caption("ADE extraction metadata is preserved for audit and troubleshooting.")
    with st.expander("View ADE extraction metadata"):
        st.json(st.session_state.get("metadata", {}))
