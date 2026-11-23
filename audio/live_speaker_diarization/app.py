"""Stream a bundled public debate to Kombucha-1 and inspect live speaker turns."""

from __future__ import annotations

import json
import os
import tempfile
from html import escape
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from kombucha_client import KombuchaError, stream_wav

ROOT = Path(__file__).parent
SAMPLE_AUDIO = ROOT / "assets" / "european_parliament_debate.wav"
SAMPLE_TRANSCRIPT = ROOT / "assets" / "plain_transcript.json"
SPEAKER_COLOURS = [
    "#0F766E",
    "#2563EB",
    "#7C3AED",
    "#C2410C",
    "#BE123C",
    "#4D7C0F",
    "#0369A1",
    "#A16207",
    "#9333EA",
    "#047857",
]

load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Live Speaker Diarization", page_icon="🎙️", layout="wide")


def _reset_run() -> None:
    st.session_state.events = []
    st.session_state.segments = []
    st.session_state.open_turn = None
    st.session_state.status = "Waiting to start"
    st.session_state.session_id = None


def _handle_event(event: dict[str, Any]) -> None:
    st.session_state.events.append(event)
    event_type = event.get("type")
    if event_type == "session":
        st.session_state.session_id = event.get("session_id")
        st.session_state.status = "Connected — streaming the discussion"
    elif event_type == "speakers":
        st.session_state.segments.extend(event.get("segments", []))
        st.session_state.open_turn = event.get("open")
        st.session_state.status = "Receiving live speaker labels"
    elif event_type == "final":
        st.session_state.segments = event.get("segments", [])
        st.session_state.open_turn = None
        st.session_state.status = "Final result received"
    elif event_type == "finished":
        st.session_state.status = "Diarization complete"
    elif event_type == "error":
        st.session_state.status = event.get("message", "Kombucha-1 returned an error.")


def _speaker_number(speaker_id: str) -> int:
    """Show people one-based labels even when the API IDs start at zero."""
    try:
        return int(speaker_id) + 1
    except (TypeError, ValueError):
        return 1


def _speaker_name(speaker_id: str) -> str:
    labels = st.session_state.setdefault("speaker_labels", {})
    return labels.get(speaker_id, f"Speaker {_speaker_number(speaker_id)}")


def _speaker_colour(speaker_id: str) -> str:
    return SPEAKER_COLOURS[int(speaker_id) % len(SPEAKER_COLOURS)]


def _turn_duration(segment: dict[str, Any]) -> float:
    return max(0, segment["end_ms"] - segment["start_ms"]) / 1_000


def _load_transcript() -> list[dict[str, Any]]:
    if not SAMPLE_TRANSCRIPT.exists():
        return []
    return json.loads(SAMPLE_TRANSCRIPT.read_text(encoding="utf-8"))


def _load_uploaded_transcript(uploaded_file: Any) -> list[dict[str, Any]]:
    """Load the small timestamped-transcript JSON format used by this demo."""
    if uploaded_file is None:
        return []

    payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The transcript JSON must contain a list of transcript lines.")
    for line in payload:
        if not {"start_ms", "end_ms", "text"}.issubset(line):
            raise ValueError(
                "Every transcript line needs start_ms, end_ms, and text fields."
            )
    return payload


def _write_uploaded_wav(uploaded_file: Any) -> Path:
    """Give the streaming client a temporary file without retaining user audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
        temporary_file.write(uploaded_file.getvalue())
        return Path(temporary_file.name)


def _speaker_for_line(line: dict[str, Any]) -> str | None:
    candidates = list(st.session_state.segments)
    if st.session_state.open_turn:
        candidates.append(st.session_state.open_turn)

    best_speaker = None
    best_overlap = 0
    for segment in candidates:
        overlap = max(
            0,
            min(line["end_ms"], segment["end_ms"])
            - max(line["start_ms"], segment["start_ms"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment["speaker"]
    return best_speaker


def _labelled_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **line,
            "speaker": speaker,
            "label": _speaker_name(speaker) if speaker is not None else None,
        }
        for line in transcript
        for speaker in [_speaker_for_line(line)]
    ]


def _markdown_export(transcript: list[dict[str, Any]]) -> str:
    lines = ["# Speaker-labelled discussion", ""]
    for line in _labelled_transcript(transcript):
        start = line["start_ms"] / 1_000
        end = line["end_ms"] / 1_000
        label = line["label"] or "Speaker not assigned yet"
        lines.append(f"- **{label}** ({start:.1f}s–{end:.1f}s): {line['text']}")
    return "\n".join(lines)


def _render_status(container: Any) -> None:
    container.metric("Status", st.session_state.status)


def _render_timeline(container: Any) -> None:
    with container.container():
        if not st.session_state.segments:
            st.info("Waiting for the first completed speaker turn.")
            return

        for segment in st.session_state.segments:
            speaker = segment["speaker"]
            start = segment["start_ms"] / 1_000
            end = segment["end_ms"] / 1_000
            st.markdown(
                (
                    "<div style='border-left: 5px solid "
                    f"{_speaker_colour(speaker)}; padding: 0.35rem 0.7rem; margin: 0.3rem 0; "
                    "border-radius: 0.25rem; background: #f8fafc;'>"
                    f"<strong>{escape(_speaker_name(speaker))}</strong> · {start:.1f}s–{end:.1f}s"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        if st.session_state.open_turn:
            turn = st.session_state.open_turn
            st.caption(f"{_speaker_name(turn['speaker'])} is speaking…")


def _render_transcript(container: Any, transcript: list[dict[str, Any]]) -> None:
    with container.container():
        for line in _labelled_transcript(transcript):
            start = line["start_ms"] / 1_000
            end = line["end_ms"] / 1_000
            if line["speaker"] is None:
                heading = "Transcript"
                colour = "#64748B"
            else:
                heading = line["label"]
                colour = _speaker_colour(line["speaker"])
            st.markdown(
                (
                    "<div style='border-left: 4px solid "
                    f"{colour}; padding: 0.45rem 0.75rem; margin: 0.35rem 0; "
                    "border-radius: 0.25rem; background: #f8fafc;'>"
                    f"<strong>{escape(heading)}</strong> · <span style='color:#64748B'>{start:.1f}s–{end:.1f}s</span><br>"
                    f"{escape(line['text'])}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


if "status" not in st.session_state:
    _reset_run()

st.title("Live Speaker Diarization")
st.caption(
    "Turn a timestamped transcript into a speaker-labelled conversation record with Kombucha-1."
)

token = os.getenv("KOMBUCHA_API_KEY", "")
if not token:
    st.error("Add KOMBUCHA_API_KEY to .env before running the demo.")
elif not SAMPLE_AUDIO.exists():
    st.warning("The bundled demo audio is missing. Add a 16 kHz mono PCM WAV before testing it.")
else:
    source = st.radio(
        "Choose audio to diarize",
        ["Test the built-in sample", "Upload your own WAV"],
        horizontal=True,
    )
    uploaded_audio = None
    uploaded_transcript = None
    transcript_error = None

    if source == "Test the built-in sample":
        transcript = _load_transcript()
        audio_ready = SAMPLE_AUDIO.exists()
        source_description = "A multi-speaker conversation with a plain timestamped transcript."
    else:
        uploaded_audio = st.file_uploader(
            "Upload a 16-bit PCM WAV (16 kHz, mono)", type=["wav"]
        )
        uploaded_transcript = st.file_uploader(
            "Add a timestamped transcript (optional JSON)", type=["json"]
        )
        try:
            transcript = _load_uploaded_transcript(uploaded_transcript)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            transcript = []
            transcript_error = str(error)
        audio_ready = uploaded_audio is not None
        source_description = (
            "Upload audio for speaker turns and talk time. Add a transcript to also produce "
            "a speaker-attributed export."
        )

    if transcript_error:
        st.error(transcript_error)

    left, right = st.columns([1, 2])
    with right:
        st.subheader("Live speaker timeline")
        st.caption("Kombucha-1 speaker turns appear here as the discussion is streamed.")
        timeline_area = st.empty()
        _render_timeline(timeline_area)
        with st.expander("View transcript before and after diarization", expanded=False):
            plain, labelled = st.tabs(["Plain transcript", "Speaker-labelled transcript"])
            with plain:
                st.caption("This is the existing timestamped transcript. It has no speaker identity.")
                for line in transcript:
                    st.write(f"{line['start_ms'] / 1_000:.1f}s–{line['end_ms'] / 1_000:.1f}s  ·  {line['text']}")
            with labelled:
                st.caption("Speaker labels are added as Kombucha-1 returns live turns.")
                transcript_area = st.empty()
                _render_transcript(transcript_area, transcript)

    with left:
        st.subheader("Audio input")
        st.write(source_description)
        if source == "Test the built-in sample":
            st.info(
                "This built-in sample is a real, nine-speaker public debate. Upload a call or "
                "meeting to test your own material."
            )
            with st.expander("Play sample audio", expanded=False):
                st.audio(SAMPLE_AUDIO, format="audio/wav")
        elif uploaded_audio:
            with st.expander("Play uploaded audio", expanded=False):
                st.audio(uploaded_audio, format="audio/wav")

        st.markdown(
            "**What this adds beyond a raw API response**\n\n"
            "- maps speaker turns onto an existing transcript\n"
            "- lets a reviewer rename anonymous speakers\n"
            "- exports a labelled record for QA or follow-up systems"
        )
        status_area = st.empty()
        button_label = (
            "Test built-in sample"
            if source == "Test the built-in sample"
            else "Run diarization on uploaded audio"
        )
        if st.button(
            button_label,
            type="primary",
            use_container_width=True,
            disabled=not audio_ready,
        ):
            _reset_run()
            _render_status(status_area)

            temporary_audio = None
            audio_path = SAMPLE_AUDIO
            if uploaded_audio is not None:
                temporary_audio = _write_uploaded_wav(uploaded_audio)
                audio_path = temporary_audio

            def render_event(event: dict[str, Any]) -> None:
                _handle_event(event)
                _render_status(status_area)
                _render_timeline(timeline_area)
                _render_transcript(transcript_area, transcript)

            try:
                stream_wav(audio_path, token, render_event)
            except KombuchaError as error:
                st.error(str(error))
            except Exception as error:  # Keep API failures readable in the demo.
                st.error(f"Streaming failed: {error}")
            finally:
                if temporary_audio:
                    temporary_audio.unlink(missing_ok=True)
        _render_status(status_area)
        if st.session_state.session_id:
            st.caption(f"Session: {st.session_state.session_id}")

if st.session_state.segments:
    st.divider()
    st.subheader("Speaker roster")
    totals: dict[str, float] = defaultdict(float)
    for segment in st.session_state.segments:
        totals[segment["speaker"]] += _turn_duration(segment)

    columns = st.columns(min(4, len(totals)))
    for index, (speaker, total) in enumerate(sorted(totals.items())):
        with columns[index % len(columns)]:
            st.session_state.setdefault("speaker_labels", {})
            st.session_state.speaker_labels[speaker] = st.text_input(
                f"Speaker {_speaker_number(speaker)}",
                value=_speaker_name(speaker),
                key=f"label_{speaker}",
            )
            st.caption(f"{total:.1f}s of talk time")

    st.download_button(
        "Export speaker turns as JSON",
        json.dumps(st.session_state.segments, ensure_ascii=False, indent=2),
        file_name="speaker_turns.json",
        mime="application/json",
    )
    if transcript:
        st.download_button(
            "Export labelled transcript as Markdown",
            _markdown_export(transcript),
            file_name="speaker_labelled_discussion.md",
            mime="text/markdown",
        )
        st.download_button(
            "Export labelled transcript as JSON",
            json.dumps(_labelled_transcript(transcript), ensure_ascii=False, indent=2),
            file_name="speaker_labelled_discussion.json",
            mime="application/json",
        )
