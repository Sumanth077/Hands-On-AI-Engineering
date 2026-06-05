"""Streamlit UI for the Personal Finance Agent."""

from __future__ import annotations

import tempfile

import streamlit as st
from dotenv import load_dotenv

from agent import (
    SAMPLE_CSV,
    ask_finance_agent,
    ingest_csv,
    load_transactions_df,
)

load_dotenv()

st.set_page_config(
    page_title="Personal Finance Agent",
    page_icon="💰",
    layout="wide",
)

SAMPLE_QUESTIONS = [
    "How much did I spend on food last month?",
    "What's my biggest unnecessary expense?",
    "Break down my spending by category.",
    "Suggest a weekly budget plan based on my habits.",
    "How much did I spend on transport in the last 7 days?",
]


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader(
            "Upload bank statement or transaction CSV",
            type=["csv"],
            help="CSV should include date, description, and amount (or debit/credit columns).",
        )

        if st.button("Load sample transactions.csv", use_container_width=True):
            summary = ingest_csv(SAMPLE_CSV, replace=True)
            st.session_state.data_loaded = True
            st.session_state.import_summary = summary
            st.success(f"Loaded {summary['rows_imported']} sample transactions.")

        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                temp_path = tmp.name
            try:
                summary = ingest_csv(temp_path, replace=True)
                st.session_state.data_loaded = True
                st.session_state.import_summary = summary
                st.success(
                    f"Imported {summary['rows_imported']} transactions "
                    f"({summary['date_range']})."
                )
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

        if st.session_state.get("data_loaded"):
            summary = st.session_state.get("import_summary", {})
            if summary:
                st.metric("Total spending", f"${summary.get('total_spending', 0):,.2f}")
                breakdown = summary.get("category_breakdown", {})
                if breakdown:
                    st.caption("Category totals")
                    for cat, val in breakdown.items():
                        st.write(f"**{cat}:** ${val:,.2f}")


def _render_transactions_preview() -> None:
    df = load_transactions_df()
    if df.empty:
        return
    with st.expander("Transaction preview", expanded=False):
        display = df.copy()
        display["amount"] = display["amount"].map(lambda x: f"${x:,.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    _init_session()
    _render_sidebar()

    st.title("Personal Finance Agent")
    st.markdown(
        "Upload a transaction CSV, then ask natural language questions about your spending. "
        "The agent categorizes transactions and answers using your SQLite-backed history."
    )

    _render_transactions_preview()

    if not st.session_state.data_loaded and load_transactions_df().empty:
        st.info(
            "No transactions loaded yet. Upload a CSV in the sidebar or click "
            "**Load sample transactions.csv** to try the demo."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}"):
            st.session_state.pending_question = q

    prompt = st.chat_input("Ask about your finances...")
    if st.session_state.get("pending_question"):
        prompt = st.session_state.pop("pending_question")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your transactions..."):
                history: list[tuple[str, str]] = []
                prior = st.session_state.messages[:-1]
                idx = 0
                while idx < len(prior) - 1:
                    if prior[idx]["role"] == "user" and prior[idx + 1]["role"] == "assistant":
                        history.append((prior[idx]["content"], prior[idx + 1]["content"]))
                        idx += 2
                    else:
                        idx += 1
                try:
                    answer = ask_finance_agent(prompt, chat_history=history)
                except ValueError as exc:
                    answer = f"Configuration error: {exc}"
                except Exception as exc:
                    answer = (
                        f"I ran into an error: {exc}\n\n"
                        "Check that `ORQ_API_KEY` is set and your Orq.ai router has "
                        "`deepseek-v4-flash` available."
                    )
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
