from __future__ import annotations

import os

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.client import LinerAPIError, LinerClient
from agent.harness import InvestigationHarness

load_dotenv()

st.set_page_config(page_title="Self-Driving Data Analyst", layout="wide")

DEMO_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ecommerce_demo")

# ---------- session state ----------

if "con" not in st.session_state:
    st.session_state.con = duckdb.connect(database=":memory:")
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "loaded_tables" not in st.session_state:
    st.session_state.loaded_tables = []
if "log" not in st.session_state:
    st.session_state.log = []
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "charts" not in st.session_state:
    st.session_state.charts = []
if "running" not in st.session_state:
    st.session_state.running = False


def read_csv_robust(file_or_path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_or_path)
    except UnicodeDecodeError:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        try:
            return pd.read_csv(file_or_path, encoding="cp1252")
        except UnicodeDecodeError as exc:
            name = getattr(file_or_path, "name", file_or_path)
            raise ValueError(
                f"Could not read '{name}' as UTF-8 or Windows-1252 (cp1252) CSV. "
                "Please check the file's encoding."
            ) from exc


def escape_markdown_dollars(text: str) -> str:
    """Streamlit's st.markdown/st.write treat single $...$ as inline LaTeX
    math, which corrupts model-generated numbers like "$131,364.30". This
    app has no legitimate use for LaTeX math, so escape literal dollar
    signs in any model-generated text before displaying it."""
    if not text:
        return text
    return text.replace("$", "\\$")


def load_dataframe(name: str, df: pd.DataFrame) -> None:
    st.session_state.con.register(name, df)
    st.session_state.dataframes[name] = df
    if name not in st.session_state.loaded_tables:
        st.session_state.loaded_tables.append(name)


def load_demo_dataset() -> None:
    if not os.path.isdir(DEMO_DATA_DIR):
        st.error(
            f"Demo dataset not found at {DEMO_DATA_DIR}. Run "
            "`python scripts/generate_sample_data.py` first."
        )
        return
    for fname in os.listdir(DEMO_DATA_DIR):
        if fname.endswith(".csv"):
            table_name = fname.removesuffix(".csv")
            df = read_csv_robust(os.path.join(DEMO_DATA_DIR, fname))
            load_dataframe(table_name, df)


# ---------- sidebar ----------

with st.sidebar:
    st.header("Dataset")

    uploaded_files = st.file_uploader("Upload CSV file(s)", type="csv", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            table_name = f.name.removesuffix(".csv")
            df = read_csv_robust(f)
            load_dataframe(table_name, df)

    if st.button("Load demo dataset (e-commerce)"):
        load_demo_dataset()

    if st.session_state.loaded_tables:
        st.success(f"Loaded: {', '.join(st.session_state.loaded_tables)}")
    else:
        st.info("No dataset loaded yet.")

    st.divider()
    st.header("Investigation")

    objective_mode = st.radio("Mode", ["Specific objective", "Autopilot: find what matters"])
    if objective_mode == "Specific objective":
        objective = st.text_area(
            "What do you want investigated?",
            placeholder="Why have sales been declining over the last three months?",
        )
    else:
        objective = "Explore this dataset broadly and find the most important patterns, changes, or anomalies happening in the business."

    depth = st.select_slider("Analysis depth", options=["Quick", "Standard", "Deep"], value="Standard")

    max_steps = st.number_input("Max steps", min_value=5, max_value=60, value=25)
    max_tool_calls = st.number_input("Max tool calls", min_value=5, max_value=100, value=40)

    start_disabled = not st.session_state.loaded_tables or st.session_state.running
    start = st.button("Start Investigation", type="primary", disabled=start_disabled)

# ---------- main area ----------

st.title("Self-Driving Data Analyst")
st.caption("Give it a dataset and an objective. It decides how to investigate it.")

if start and not objective:
    st.warning("Please enter an objective before starting.")
elif start:
    st.session_state.log = []
    st.session_state.final_state = None
    st.session_state.charts = []
    st.session_state.running = True

    try:
        client = LinerClient()
    except LinerAPIError as exc:
        st.error(str(exc))
        st.session_state.running = False
        st.stop()

    harness = InvestigationHarness(
        client=client,
        con=st.session_state.con,
        dataframes=st.session_state.dataframes,
        max_steps=int(max_steps),
        max_tool_calls=int(max_tool_calls),
    )

    progress_box = st.status("Investigating...", expanded=True)
    log_lines: list[str] = []
    last_round: int | None = None

    try:
        for event in harness.investigate(objective=objective, depth=depth):
            if event.type == "tool_call":
                record = event.payload["record"]
                if record.model_call_index != last_round:
                    separator = f"-- round {record.model_call_index} --"
                    log_lines.append(separator)
                    progress_box.write(separator)
                    last_round = record.model_call_index
                icon = "x" if record.is_error else "-"
                line = f"[{record.call_index}] {icon} {record.tool_name}({record.arguments})"
                log_lines.append(line)
                progress_box.write(line)
            elif event.type == "loop_warning":
                log_lines.append("!! Loop detected, prompting the agent to reassess.")
                progress_box.write("Loop detected, prompting the agent to reassess...")
            elif event.type in ("finished", "budget_exhausted"):
                st.session_state.final_state = event.payload["state"]

        st.session_state.charts = harness.charts
        st.session_state.log = log_lines
        progress_box.update(label="Investigation complete", state="complete", expanded=False)
    except LinerAPIError as exc:
        st.error(f"Liner Model API error: {exc}")
    finally:
        st.session_state.running = False

# ---------- results ----------

if st.session_state.final_state:
    state = st.session_state.final_state

    st.subheader("Final Report")
    if state.stop_reason == "budget_exhausted":
        st.warning(escape_markdown_dollars(state.final_report))
    if state.root_cause:
        conf = f" ({state.confidence:.0%} confidence)" if state.confidence is not None else ""
        st.markdown(f"**Root cause:** {escape_markdown_dollars(state.root_cause)}{conf}")
    st.write(escape_markdown_dollars(state.final_report))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Findings")
        if state.findings:
            for f in sorted(state.findings, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.importance]):
                st.markdown(f"**[{f.importance.upper()}]** {escape_markdown_dollars(f.finding)}")
                for e in f.evidence:
                    st.caption(f"- {escape_markdown_dollars(e)}")
        else:
            st.caption("No findings were explicitly saved during this run.")

    with col2:
        st.subheader("Hypotheses Explored")
        if state.hypotheses:
            for h in state.hypotheses:
                st.markdown(f"**[{h.status}]** {escape_markdown_dollars(h.hypothesis)}")
        else:
            st.caption("No hypotheses were explicitly tracked during this run.")

    if st.session_state.charts:
        st.subheader("Charts")
        for i, fig in enumerate(st.session_state.charts):
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")

    st.subheader("Investigation Log")
    with st.expander(f"{state.step_count} model calls · {len(state.tool_calls)} tool calls", expanded=False):
        for line in st.session_state.log:
            st.text(line)

    st.subheader("Usage")
    u = state.usage
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Agent steps", state.step_count)
    m2.metric("Model calls", u.model_calls)
    m3.metric("Tokens used", u.total_tokens)
    m4.metric("Estimated cost", f"${u.estimated_cost_usd():.4f}")
    st.caption(
        "Cost estimate uses Liner Model API's published liner-mark-1.0 pricing "
        "($1 / $6 / $0.10 per 1M input / output / cached tokens), which is what "
        "you're billed regardless of which underlying model the orchestrator "
        "routed any given request to."
    )
