import os
import streamlit as st
from dotenv import load_dotenv
from agents import create_llm, run_planner, run_coder, run_reviewer, run_final_coder

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Coding Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .badge-planner  { background:#dbeafe; color:#1e40af; }
    .badge-coder    { background:#dcfce7; color:#166534; }
    .badge-reviewer { background:#fef9c3; color:#854d0e; }
    .badge-final    { background:#f3e8ff; color:#6b21a8; }
    .stExpander > div:first-child { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _strip_code_fences(text: str) -> str:
    """Remove markdown ```python ... ``` fences that the model may wrap around code."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 Multi-Agent Coding Assistant")
st.caption(
    "Describe a coding task and watch three specialized AI agents collaborate to produce "
    "high-quality, reviewed Python code."
)
st.divider()

# ── API key handling ──────────────────────────────────────────────────────────
api_key = os.getenv("MISTRAL_API_KEY", "")
if not api_key:
    st.warning(
        "No `MISTRAL_API_KEY` found in environment. "
        "Add it to a `.env` file or paste it below.",
        icon="⚠️",
    )
    api_key = st.text_input(
        "Mistral API Key",
        type="password",
        placeholder="Enter your Mistral API key…",
    )

# ── Task input ────────────────────────────────────────────────────────────────
task = st.text_area(
    "Describe your coding task",
    placeholder=(
        "e.g. Build a Python function that reads a CSV file and returns "
        "a dictionary of summary statistics (mean, median, std dev, min, max) "
        "for each numeric column."
    ),
    height=110,
)

run_button = st.button("🚀 Run Pipeline", type="primary", disabled=not (api_key and task))

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_button:
    if not api_key:
        st.error("Please provide your Mistral API key.")
        st.stop()
    if not task.strip():
        st.error("Please describe a coding task.")
        st.stop()

    llm = create_llm(api_key)

    # Step 1 — Planner
    st.markdown(
        '<span class="agent-badge badge-planner">🗺️ Planner Agent</span>',
        unsafe_allow_html=True,
    )
    with st.spinner("Planner is breaking down the task…"):
        plan = run_planner(llm, task)
    st.session_state["plan"] = plan

    # Step 2 — Coder (first draft)
    st.markdown(
        '<span class="agent-badge badge-coder">💻 Coder Agent — First Draft</span>',
        unsafe_allow_html=True,
    )
    with st.spinner("Coder is writing the first draft…"):
        draft_code = run_coder(llm, plan)
    st.session_state["draft_code"] = draft_code

    # Step 3 — Reviewer
    st.markdown(
        '<span class="agent-badge badge-reviewer">🔍 Reviewer Agent</span>',
        unsafe_allow_html=True,
    )
    with st.spinner("Reviewer is auditing the code…"):
        review = run_reviewer(llm, draft_code)
    st.session_state["review"] = review

    # Step 4 — Coder (final version)
    st.markdown(
        '<span class="agent-badge badge-final">✨ Coder Agent — Final Version</span>',
        unsafe_allow_html=True,
    )
    with st.spinner("Coder is applying feedback and producing the final version…"):
        final_code = run_final_coder(llm, plan, draft_code, review)
    st.session_state["final_code"] = final_code

    st.success("Pipeline complete!", icon="✅")
    st.divider()

# ── Results ───────────────────────────────────────────────────────────────────
if all(k in st.session_state for k in ("plan", "draft_code", "review", "final_code")):

    with st.expander("🗺️ Plan", expanded=True):
        st.markdown(
            '<span class="agent-badge badge-planner">Planner Agent</span>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state["plan"])

    with st.expander("💻 First Draft Code", expanded=False):
        st.markdown(
            '<span class="agent-badge badge-coder">Coder Agent</span>',
            unsafe_allow_html=True,
        )
        clean_draft = _strip_code_fences(st.session_state["draft_code"])
        st.code(clean_draft, language="python")

    with st.expander("🔍 Review Feedback", expanded=False):
        st.markdown(
            '<span class="agent-badge badge-reviewer">Reviewer Agent</span>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state["review"])

    with st.expander("✨ Final Code", expanded=True):
        st.markdown(
            '<span class="agent-badge badge-final">Coder Agent — Improved</span>',
            unsafe_allow_html=True,
        )
        clean_final = _strip_code_fences(st.session_state["final_code"])
        st.code(clean_final, language="python")

        # Streamlit's st.code already renders a copy icon in the top-right corner of
        # the code block. The button below provides an explicit fallback with feedback.
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("📋 Copy Final Code"):
                st.session_state["copy_triggered"] = True

        if st.session_state.get("copy_triggered"):
            st.toast("Use the copy icon on the code block above — browser security "
                     "prevents clipboard access from a button outside the code widget.",
                     icon="📋")
            # Reset flag so toast only fires once per click
            st.session_state["copy_triggered"] = False
