"""Streamlit interface for the Voice Codebase Onboarding Agent."""

from __future__ import annotations

import os
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from codebase_guide.fish_audio import FishAudioError, generate_voice, narration_script, transcribe
from codebase_guide.llm import LLMError, is_configured
from codebase_guide.repository import GitHubRepository, LocalRepository, RepositoryError
from codebase_guide.workflow import investigate


load_dotenv()

st.set_page_config(
    page_title="Voice codebase onboarding agent",
    page_icon=":material/account_tree:",
    layout="wide",
)

SUGGESTIONS = {
    "Find the entry point": "What is the main entry point of this repository?",
    "Trace authentication": "Where does authentication happen? Trace the relevant flow.",
    "Change the UI": "Where should I start if I want to change the user interface?",
}
CITATION = re.compile(r"\[([^\]]+):(\d+)\]")


def initialize_state() -> None:
    st.session_state.setdefault("snapshot", None)
    st.session_state.setdefault("reader", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_files", [])
    st.session_state.setdefault("last_answer", "")


def reset_repository() -> None:
    st.session_state.snapshot = None
    st.session_state.reader = None
    st.session_state.messages = []
    st.session_state.last_files = []
    st.session_state.last_answer = ""


def citation_focus(answer: str, files: list[Any]) -> tuple[str, int] | None:
    available = {file.path: file for file in files}
    for path, raw_line in CITATION.findall(answer):
        if path in available and 1 <= int(raw_line) <= available[path].line_count:
            return path, int(raw_line)
    return None


def source_window(content: str, line_number: int | None) -> tuple[str, int]:
    lines = content.splitlines()
    if not lines:
        return "(This file is empty.)", 1
    if line_number is None:
        start = 1
    else:
        start = max(1, line_number - 8)
    end = min(len(lines), start + 34)
    excerpt = "\n".join(f"{number:>4} | {lines[number - 1]}" for number in range(start, end + 1))
    return excerpt, start


def render_message(message: dict[str, Any]) -> None:
    if message.get("audio"):
        st.audio(
            message["audio"],
            format="audio/mpeg",
            autoplay=bool(message.get("autoplay")),
            alt="Spoken explanation of the inspected codebase files",
        )
        with st.expander("Drama 3 narration script and timestamps", icon=":material/graphic_eq:"):
            st.code(message.get("narration", ""), language=None)
            segments = message.get("segments", [])
            if segments:
                st.dataframe(
                    [
                        {"Word": item["text"], "Start": f"{item['start']:.2f}s", "End": f"{item['end']:.2f}s"}
                        for item in segments
                    ],
                    hide_index=True,
                    alt="Word timestamps for the Drama 3 narration",
                )
    elif message.get("voice_error"):
        st.caption(message["voice_error"])

    with st.expander("Read the full codebase explanation", expanded=not bool(message.get("audio"))):
        st.markdown(message["answer"])
        if message.get("plan"):
            st.caption(f"Read-only investigation: {message['plan']}")


def run_question(question: str) -> None:
    snapshot = st.session_state.snapshot
    reader = st.session_state.reader
    if snapshot is None or reader is None:
        st.warning("Load a repository before asking a question.")
        return
    if not is_configured():
        st.error("Add AI_GATEWAY_API_KEY to .env, then restart the app before asking repository questions.")
        return

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar=":material/account_tree:"):
        try:
            with st.status(":shimmer[Planning a read-only repository investigation]", type="compact") as status:
                result = investigate(reader, snapshot, question)
                status.update(label="Repository evidence collected", state="complete")
        except (LLMError, RepositoryError) as error:
            st.error(str(error))
            st.session_state.messages.pop()
            return

        narration = narration_script(result.spoken_answer or result.answer)
        voice = None
        voice_error = None
        if os.getenv("FISH_API_KEY"):
            try:
                with st.status(":shimmer[Generating Drama 3 narration]", type="compact") as status:
                    voice = generate_voice(narration)
                    status.update(label="Drama 3 narration ready", state="complete")
            except FishAudioError as error:
                voice_error = f"Written guide is ready. Audio could not be generated: {error}"
        else:
            voice_error = "Add FISH_API_KEY to enable spoken explanations and voice questions."

        message = {
            "role": "assistant",
            "answer": result.answer,
            "plan": result.plan,
            "audio": voice.audio if voice else None,
            "segments": [segment.__dict__ for segment in voice.segments] if voice else [],
            "narration": narration,
            "voice_error": voice_error,
        }
        render_message({**message, "autoplay": True})

    st.session_state.messages.append(message)
    st.session_state.last_files = result.files
    st.session_state.last_answer = result.answer


initialize_state()

with st.sidebar:
    st.header("Repository", icon=":material/folder_open:")
    source_type = st.segmented_control(
        "Source", ["Public GitHub URL", "Local repository"], default="Public GitHub URL"
    )
    if source_type == "Public GitHub URL":
        source_value = st.text_input(
            "Repository URL",
            placeholder="https://github.com/owner/repository",
            key="github_url",
        )
    else:
        source_value = st.text_input(
            "Local repository path",
            placeholder=r"C:\Users\USER\Projects\my_repository",
            key="local_path",
        )

    if st.button("Load repository", icon=":material/download:", width="stretch"):
        try:
            reader = GitHubRepository(source_value) if source_type == "Public GitHub URL" else LocalRepository(source_value)
            with st.status(":shimmer[Reading repository metadata and file tree]", type="compact") as status:
                snapshot = reader.snapshot()
                status.update(label="Repository ready for read-only exploration", state="complete")
            reset_repository()
            st.session_state.reader = reader
            st.session_state.snapshot = snapshot
            st.rerun()
        except RepositoryError as error:
            st.error(str(error))

    if st.button("Clear repository", icon=":material/delete_sweep:", width="stretch"):
        reset_repository()
        st.rerun()

    st.divider()
    st.caption("Model: GLM 5.3 (`zai/glm-5.3`)")
    st.caption("Voice: Fish Audio Drama 3. Speech input: transcribe-1-pro.")
    if is_configured():
        st.badge("Repository model configured", icon=":material/check:", color="green")
    else:
        st.badge("Add AI_GATEWAY_API_KEY to ask questions", icon=":material/key:", color="orange")

st.title("Voice codebase onboarding agent", icon=":material/account_tree:")
st.write(
    "Connect a real repository, ask a focused question, and follow a spoken explanation grounded in the files the agent inspected."
)

snapshot = st.session_state.snapshot
if snapshot is None:
    st.info(
        "Start by loading a public GitHub repository URL or a local Git repository. "
        "The agent only reads repository metadata and selected source files. It does not run commands or modify the target project.",
        icon=":material/info:",
    )
    st.stop()

left, right = st.columns((1.05, 0.95), gap="large")
suggested_question: str | None = None
with left:
    st.subheader("Conversation", icon=":material/forum:")
    if not st.session_state.messages:
        suggestions = st.empty()
        picked = suggestions.pills(
            "Try a repository question", list(SUGGESTIONS), selection_mode="single", label_visibility="collapsed"
        )
        if picked:
            suggested_question = SUGGESTIONS[picked]
            suggestions.empty()
    for message in st.session_state.messages:
        avatar = ":material/account_tree:" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                render_message(message)
            else:
                st.markdown(message["content"])
    new_response = st.empty()

with right:
    st.subheader(snapshot.name, icon=":material/folder:")
    st.caption(snapshot.origin)
    st.write(snapshot.description)
    metrics = st.columns(2)
    metrics[0].metric("Inspectable files", len(snapshot.file_paths))
    metrics[1].metric("Revision", snapshot.default_branch)

    files = st.session_state.last_files
    if files:
        focus = citation_focus(st.session_state.last_answer, files)
        default_index = next((index for index, item in enumerate(files) if focus and item.path == focus[0]), 0)
        selected_path = st.selectbox("Evidence inspected for the latest answer", [item.path for item in files], index=default_index)
        selected = next(item for item in files if item.path == selected_path)
        line = focus[1] if focus and focus[0] == selected.path else None
        excerpt, start = source_window(selected.content, line)
        st.caption(f"Read-only source preview, beginning at line {start}")
        st.code(excerpt, language=None, height=520)
    elif snapshot.readme:
        st.caption("README preview")
        st.code(snapshot.readme[:6_000], language="markdown", height=520)
    else:
        st.caption("Ask a question to inspect the relevant files.")

submission = st.chat_input(
    "Ask where a feature lives, how a request flows, or where to start a change",
    accept_audio=True,
    audio_sample_rate=16000,
    submit_mode="disable",
)

if suggested_question or submission:
    question = suggested_question or submission.text.strip()
    if submission and submission.audio is not None:
        try:
            with st.status(":shimmer[Transcribing your question]", type="compact") as status:
                transcript = transcribe(submission.audio.getvalue(), submission.audio.name)
                status.update(label="Speech-to-text complete", state="complete")
            question = transcript.text
        except FishAudioError as error:
            st.error(str(error))
            st.stop()
    if not question:
        st.warning("Type or record a repository question before sending.")
        st.stop()
    with new_response.container():
        run_question(question)

