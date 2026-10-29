"""Streamlit interface for the Voice Legal Intake Agent."""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from legal_intake.agent import draft_summary, process_turn
from legal_intake.krisp import KrispError, is_configured as krisp_is_configured, transcribe_recording
from legal_intake.llm import IntakeModelError, is_configured as model_is_configured, model_name
from legal_intake.models import IntakeFacts
from legal_intake.storage import load_recent_intakes, save_reviewed_intake


load_dotenv()

st.set_page_config(page_title="Voice legal intake agent", page_icon=":material/gavel:", layout="wide")


def initialize_state() -> None:
    """Initialize an intake that exists only in this browser session until review."""

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("facts", IntakeFacts())
    st.session_state.setdefault("next_question", "Tell me what happened, in your own words.")
    st.session_state.setdefault("ready_for_review", False)
    st.session_state.setdefault("last_transcript", "")
    st.session_state.setdefault("processed_recording_hash", "")


def start_new_intake() -> None:
    """Clear unreviewed case material from the active browser session."""

    st.session_state.history = []
    st.session_state.facts = IntakeFacts()
    st.session_state.next_question = "Tell me what happened, in your own words."
    st.session_state.ready_for_review = False
    st.session_state.last_transcript = ""
    st.session_state.processed_recording_hash = ""


def render_facts(facts: IntakeFacts) -> None:
    """Show only factual, editable-looking intake categories to the user."""

    fields = {
        "People and organisations": facts.people_or_organisations,
        "Timeline": facts.dates_and_timeline,
        "Evidence mentioned": facts.evidence_mentioned,
        "Impact mentioned": facts.impact_or_loss,
        "Still to confirm": facts.missing_information,
    }
    if facts.matter_type:
        st.caption(f"Matter type: {facts.matter_type}")
    for label, items in fields.items():
        with st.expander(label, expanded=label == "Still to confirm"):
            if items:
                st.markdown("\n".join(f"- {item}" for item in items))
            else:
                st.caption("Not provided yet.")


def has_minimum_review_information(history: list[dict[str, str]], facts: IntakeFacts) -> bool:
    """Avoid ending an intake after a single broad first account."""

    user_turns = sum(1 for item in history if item["role"] == "user")
    required_groups = (
        facts.narrative,
        facts.people_or_organisations,
        facts.dates_and_timeline,
        facts.evidence_mentioned,
        facts.impact_or_loss,
    )
    return user_turns >= 2 and all(required_groups)


def fallback_follow_up(facts: IntakeFacts) -> str:
    """Keep the intake moving if a model ends its question with an early review decision."""

    if not facts.people_or_organisations:
        return "Who was involved, and what is each person's connection to the situation?"
    if not facts.dates_and_timeline:
        return "When did this happen, and what is the sequence of events as you remember it?"
    if not facts.evidence_mentioned:
        return "Is there any evidence you are aware of, such as messages, photographs, documents, or witnesses?"
    if not facts.impact_or_loss:
        return "What impact or loss has resulted from the situation?"
    return "Is there any other factual detail you think the reviewer should know?"


def run_turn(text: str, source: str) -> None:
    """Run one guided intake turn while preserving the visible conversation."""

    st.session_state.history.append({"role": "user", "content": text, "source": source})
    with st.status(":shimmer[Reviewing the intake details]", type="compact") as status:
        decision = process_turn(
            [{"role": item["role"], "content": item["content"]} for item in st.session_state.history],
            st.session_state.facts,
        )
        status.update(label="Intake details updated", state="complete")
    st.session_state.facts = decision.facts
    ready_for_review = (
        decision.status == "ready_for_review"
        and has_minimum_review_information(st.session_state.history, decision.facts)
    )
    suggested_question = decision.next_question or ""
    if not ready_for_review and (not suggested_question or "ready for your review" in suggested_question.lower()):
        suggested_question = fallback_follow_up(decision.facts)
    st.session_state.next_question = suggested_question or fallback_follow_up(decision.facts)
    st.session_state.ready_for_review = ready_for_review
    if st.session_state.ready_for_review:
        assistant_response = "I have captured the essential factual details. The intake is ready for your review."
    else:
        assistant_response = (
            "I have recorded the details you shared for the intake.\n\n"
            f"{st.session_state.next_question}"
        )
    st.session_state.history.append({"role": "assistant", "content": assistant_response, "source": "agent"})


initialize_state()

with st.sidebar:
    st.header("Intake controls", icon=":material/tune:")
    st.caption(f"Model: {model_name()}")
    st.caption("Speech-to-text: Krisp `krisp-realtime-stt-agents-v1`")
    if st.button("Start a new intake", icon=":material/restart_alt:", width="stretch"):
        start_new_intake()
        st.rerun()
    st.divider()
    st.caption("The agent documents facts for human review. It does not provide legal advice or assess a case.")
    st.badge("Krisp configured" if krisp_is_configured() else "Add KRISP_API_KEY", icon=":material/mic:", color="green" if krisp_is_configured() else "orange")
    st.badge("Intake model configured" if model_is_configured() else "Add AI_GATEWAY_API_KEY", icon=":material/psychology:", color="green" if model_is_configured() else "orange")

st.title("Voice legal intake agent", icon=":material/gavel:")
st.write("Describe a situation by voice or text. The agent collects facts, asks focused follow-up questions, and creates a reviewable intake summary.")
st.warning(
    "This project is for documentation and intake support only. It does not provide legal advice, legal representation, or a prediction about an outcome.",
    icon=":material/info:",
)

left, right = st.columns((1.1, 0.9), gap="large")

with left:
    st.subheader("Guided intake", icon=":material/forum:")
    for item in st.session_state.history:
        with st.chat_message(item["role"], avatar=":material/gavel:" if item["role"] == "assistant" else None):
            if item.get("source") == "voice":
                st.caption("Krisp transcript")
            st.markdown(item["content"])

    if not st.session_state.history:
        with st.chat_message("assistant", avatar=":material/gavel:"):
            st.markdown(st.session_state.next_question)

    if not st.session_state.ready_for_review:
        recording = st.audio_input(
            "Record your response",
            sample_rate=16_000,
            key=f"recording_{len(st.session_state.history)}",
        )
        typed_response = st.chat_input("Or type your response")
        recording_hash = hashlib.sha256(recording.getvalue()).hexdigest() if recording is not None else ""
        if recording is not None and recording_hash != st.session_state.processed_recording_hash:
            try:
                with st.status(":shimmer[Transcribing your recording with Krisp]", type="compact") as status:
                    transcript = transcribe_recording(recording.getvalue(), ["affidavit", "injunction", "mediation", "subpoena"])
                    status.update(label="Voice response transcribed", state="complete")
                st.session_state.last_transcript = transcript
                st.session_state.processed_recording_hash = recording_hash
                run_turn(transcript, "voice")
                st.rerun()
            except (KrispError, IntakeModelError) as error:
                st.error(str(error))
        elif typed_response:
            try:
                run_turn(typed_response, "text")
                st.rerun()
            except IntakeModelError as error:
                st.error(str(error))
    else:
        st.success("The agent has enough information for a draft. Review and edit it before saving.", icon=":material/check_circle:")

with right:
    st.subheader("Intake progress", icon=":material/fact_check:")
    if not st.session_state.history:
        st.info(
            "Your transcript and the agent's next question will appear in the guided intake. "
            "Captured details stay here for reference.",
            icon=":material/forum:",
        )
    else:
        st.caption("The agent updates these factual details after each response.")
        with st.expander("Details captured so far", expanded=False):
            render_facts(st.session_state.facts)

    if st.session_state.ready_for_review:
        draft = draft_summary(st.session_state.facts)
        edited_draft = st.text_area("Review the draft summary", draft, height=420)
        first, second = st.columns(2)
        with first:
            st.download_button(
                "Download Markdown",
                edited_draft,
                file_name="intake_summary.md",
                mime="text/markdown",
                icon=":material/download:",
                width="stretch",
            )
        with second:
            if st.button("Save reviewed intake", icon=":material/save:", width="stretch"):
                record = save_reviewed_intake(edited_draft, st.session_state.facts)
                st.success(f"Saved locally with case ID {record.case_id[:8]}.")
    else:
        st.info(
            "The editable intake summary appears once the agent has collected the essential facts.",
            icon=":material/edit_note:",
        )

with st.expander("Recent locally saved intakes", icon=":material/history:"):
    records = load_recent_intakes()
    if not records:
        st.caption("No reviewed summaries have been saved on this machine yet.")
    else:
        st.dataframe(
            [{"Case ID": record.case_id[:8], "Saved": record.created_at, "Matter type": record.facts.matter_type or "Not specified"} for record in records],
            hide_index=True,
            alt="Locally saved reviewed intake summaries",
        )
