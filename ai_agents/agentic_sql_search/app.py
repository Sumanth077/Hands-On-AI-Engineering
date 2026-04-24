"""
Agentic SQL Search — Streamlit UI
Run: streamlit run app.py
"""

import sqlite3
import tempfile
from pathlib import Path

import streamlit as st

from database import DB_PATH, init_db
from agent import run_agent, set_db_path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic SQL Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure the bundled DB exists on every startup
init_db()


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_schema_map(db_path) -> dict[str, list[tuple]]:
    """Return {table_name: [column_info, ...]} for the given database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            schema[table] = cursor.fetchall()
        conn.close()
        return schema
    except Exception:
        return {}


def get_active_db_path() -> Path:
    """Return the currently active database path from session state."""
    return st.session_state.get("active_db_path", DB_PATH)


def render_steps(steps: list):
    """Render agent intermediate steps inside a Streamlit expander."""
    with st.expander("🔎 Agent Reasoning", expanded=False):
        for i, step in enumerate(steps, 1):
            tool_name = step["tool"]
            tool_input = step["input"]
            tool_output = step["output"]

            st.markdown(f"**Step {i} — `{tool_name}`**")

            if tool_name == "execute_sql":
                query = tool_input.get("query", str(tool_input))
                st.code(query, language="sql")
            elif tool_name == "load_skill":
                skill = tool_input.get("skill_name", str(tool_input))
                st.markdown(f"Loading skill: `{skill}`")
            else:
                st.code(str(tool_input))

            preview = tool_output[:600] + (" …" if len(tool_output) > 600 else "")
            st.markdown("**Result:**")
            st.text(preview)

            if i < len(steps):
                st.divider()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Agentic SQL Search")
    st.divider()

    # ── Database selector ──
    st.markdown("### 🗄️ Database")
    db_mode = st.radio(
        "Choose a database",
        options=["Bundled E-Commerce Dataset", "Upload Your Own"],
        index=0,
        label_visibility="collapsed",
    )

    if db_mode == "Upload Your Own":
        uploaded_file = st.file_uploader(
            "Upload a SQLite database",
            type=["db", "sqlite", "sqlite3"],
            help="Upload any SQLite .db file. The agent will inspect its schema automatically.",
        )
        if uploaded_file is not None:
            # Save to a temp file that persists for this session
            if "uploaded_db_path" not in st.session_state or \
               st.session_state.get("uploaded_db_name") != uploaded_file.name:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".db", delete=False, dir=Path(__file__).parent / "data"
                )
                tmp.write(uploaded_file.getbuffer())
                tmp.close()
                st.session_state["uploaded_db_path"] = tmp.name
                st.session_state["uploaded_db_name"] = uploaded_file.name
                st.session_state["messages"] = []  # clear chat for new DB

            active_path = Path(st.session_state["uploaded_db_path"])
            st.session_state["active_db_path"] = active_path
            set_db_path(active_path)
            st.success(f"Using: {uploaded_file.name}")
        else:
            # No file uploaded yet — fall back to bundled
            st.session_state["active_db_path"] = DB_PATH
            set_db_path(DB_PATH)
            st.caption("No file uploaded yet. Showing bundled dataset schema.")
    else:
        st.session_state["active_db_path"] = DB_PATH
        set_db_path(DB_PATH)
        st.caption("20 customers · 25 products · 40 orders · 5 categories")

    st.divider()

    # ── Schema viewer (reflects active DB) ──
    active_db = get_active_db_path()
    st.markdown("### 📊 Schema")
    schema_map = get_schema_map(active_db)
    if schema_map:
        for table_name, columns in schema_map.items():
            with st.expander(f"`{table_name}`"):
                for col in columns:
                    pk_badge = " 🔑" if col[5] else ""
                    st.markdown(f"`{col[1]}` — *{col[2]}*{pk_badge}")
    else:
        st.caption("No tables found in this database.")

    st.divider()

    # ── Sample questions (only for bundled dataset) ──
    if db_mode == "Bundled E-Commerce Dataset":
        st.markdown("### 💡 Try These")
        SAMPLE_QUESTIONS = [
            "Who are the top 5 customers by total spend?",
            "Which product category generates the most revenue?",
            "What are the 3 best-selling products by units sold?",
            "How many orders are in each status?",
            "What is the average order value per country?",
            "List all products with less than 100 units in stock.",
            "Which customers placed more than 2 orders?",
            "What's the total revenue from Electronics?",
        ]
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"btn_{q[:20]}"):
                st.session_state["pending_q"] = q
        st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🔍 Agentic SQL Search")

active_db = get_active_db_path()
if active_db == DB_PATH:
    st.markdown(
        "Ask any question about the e-commerce data in plain English. "
        "The agent writes the SQL, runs it, and explains the results."
    )
else:
    st.markdown(
        f"Querying **{st.session_state.get('uploaded_db_name', 'uploaded database')}**. "
        "The agent will inspect the schema and write the SQL for you."
    )

st.divider()

# Init chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            render_steps(msg["steps"])

# Resolve input — typed or clicked from sidebar
question: str | None = st.chat_input("Ask a question about the data…")
if "pending_q" in st.session_state:
    question = st.session_state.pop("pending_q")

if question:
    # Ensure agent is pointed at the active DB before running
    set_db_path(get_active_db_path())

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Querying the database…"):
            try:
                result = run_agent(question)
                answer = result.get("answer", "No answer generated.")
                steps = result.get("steps", [])
            except Exception as e:
                answer = f"❌ Error: {e}"
                steps = []

        st.markdown(answer)
        if steps:
            render_steps(steps)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
    })
