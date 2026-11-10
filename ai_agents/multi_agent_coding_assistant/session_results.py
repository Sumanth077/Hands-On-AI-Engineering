"""Session-state helpers for generated coding pipeline results."""

from collections.abc import MutableMapping


PIPELINE_RESULT_KEYS = ("plan", "draft_code", "review", "final_code")


def clear_pipeline_results(session_state: MutableMapping[str, object]) -> None:
    """Remove only generated pipeline outputs from a Streamlit session."""
    for key in PIPELINE_RESULT_KEYS:
        session_state.pop(key, None)
