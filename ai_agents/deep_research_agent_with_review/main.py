from __future__ import annotations

import streamlit as st

from research_agent.config import load_settings
from research_agent.nebius import ModelDiscoveryError, list_nebius_models
from research_agent.workflow import DeepResearchWorkflow

st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="wide")

if "chat" not in st.session_state:
    st.session_state.chat = []

settings = load_settings()

SUGGESTIONS = {
    "Compare current approaches to reducing hallucinations in production AI agents.": (
        "What are the latest approaches to reducing hallucinations in production AI agents? "
        "Compare the evidence and trade-offs."
    ),
    "Research open-weight reasoning models and their practical limits.": (
        "Research the current state of open-weight reasoning models. What has changed "
        "recently, and what are the practical limits?"
    ),
    "Assess AI customer support in regulated industries.": (
        "What are the strongest arguments for and against using AI agents for customer "
        "support in regulated industries?"
    ),
}


def _review_label(value: bool | None) -> str:
    if value is True:
        return "Pass"
    if value is False:
        return "Needs work"
    return "Unavailable"


def _probability_text(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "Not returned"


def render_result_details(result) -> None:
    """Keep supporting evidence available without competing with the report."""
    decision = result.review.action or "unavailable"
    st.caption(
        f":material/fact_check: JEV review: **{decision}** · "
        f"evidence coverage: **{_review_label(result.review.evidence_coverage)}** · "
        f"claim grounding: **{_review_label(result.review.claim_grounding)}**"
    )
    with st.expander("View JEV evidence review"):
        if result.review.status != "completed":
            st.warning(result.review.explanation)
            st.caption("The research report is still shown, but it was not reviewed by JEV.")
        else:
            st.caption(
                "JEV evaluates the report against the Tavily sources. It does not write the report."
            )
            coverage, grounding, action = st.columns(3)
            coverage.metric(
                "Evidence coverage",
                _review_label(result.review.evidence_coverage),
                _probability_text(
                    getattr(result.review, "evidence_coverage_probability", None)
                ),
            )
            grounding.metric(
                "Claim grounding",
                _review_label(result.review.claim_grounding),
                _probability_text(
                    getattr(result.review, "claim_grounding_probability", None)
                ),
            )
            action.metric("Decision", result.review.action or "Unavailable")
            st.write(result.review.explanation)
            st.caption(f"Review status: {result.review.status}. Automatic research revisions: {result.revision_count}.")

    if result.citation_issues:
        st.warning("Citation check: " + " ".join(result.citation_issues))
    with st.expander("Sources"):
        for source in result.sources:
            st.markdown(f"**[{source.number}] [{source.title}]({source.url})**")
            st.caption(source.snippet)
    with st.expander("Research activity"):
        if result.plan:
            st.markdown("**Plan**")
            for item in result.plan:
                st.markdown(f"- {item}")
        st.markdown("**Tavily searches**")
        for search in result.searches:
            st.markdown(f"- {search}")

st.title("Deep Research Agent with Review Layer")
st.caption(
    "Plans live-web research, writes a cited report, and asks JEV to check the evidence before it is shown."
)

with st.sidebar:
    st.header("Run status")
    missing = settings.missing_required
    if missing:
        st.warning("Add these values to `.env`: " + ", ".join(missing))
    else:
        st.success("Nebius, Tavily, and JEV keys are configured.")
    st.caption(f"Research model: `{settings.nebius_model}`")
    st.caption(f"JEV decision model: `{settings.jev_model}`")

    if st.button("Check available Nebius models", width="stretch"):
        try:
            with st.spinner("Checking the Token Factory models endpoint..."):
                models = list_nebius_models(settings)
            nemotron = [model for model in models if "nemotron" in model.lower()]
            if nemotron:
                st.success("Nemotron models available to this key:")
                st.code("\n".join(nemotron), language=None)
            else:
                st.warning("The key worked, but no Nemotron models were returned.")
            with st.expander("All available models"):
                st.code("\n".join(models), language=None)
        except ModelDiscoveryError as error:
            st.error(str(error))

    st.divider()
    tracing = "enabled" if settings.langsmith_api_key else "not configured"
    st.caption(f"LangSmith tracing: {tracing}")
    if settings.langsmith_api_key:
        st.link_button("Open LangSmith", "https://smith.langchain.com", width="stretch")

    if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.chat = []
        st.rerun()

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "result" in message:
            render_result_details(message["result"])

selected_suggestion = None
if not st.session_state.chat:
    with st.container(border=True):
        st.subheader("Try a research question")
        st.caption("Use one of these prompts, or ask about any topic that benefits from current, source-backed research.")
        selected_suggestion = st.pills(
            "Research ideas",
            list(SUGGESTIONS),
            label_visibility="collapsed",
        )

question = st.chat_input("Ask a research question")
if selected_suggestion:
    question = SUGGESTIONS[selected_suggestion]
if question:
    if settings.missing_required:
        st.error("Add the required API keys to `.env`, then restart Streamlit.")
        st.stop()

    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Starting research...", expanded=True) as status:
            def progress(label: str, detail: str) -> None:
                st.write(f"**{label}:** {detail}")

            try:
                result = DeepResearchWorkflow(settings).run(question, progress)
                status.update(label="Research complete", state="complete", expanded=False)
            except Exception as error:  # Present provider errors clearly in the UI.
                status.update(label="Research stopped", state="error", expanded=True)
                st.error(f"The research run could not finish: {error}")
                st.stop()

    st.session_state.chat.append({"role": "assistant", "content": result.answer, "result": result})
    # Render the finished response from persistent chat state on a fresh pass.
    # This keeps it separate from Streamlit's collapsible live-status container.
    st.rerun()

