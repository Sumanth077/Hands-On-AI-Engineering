"""Streamlit chat UI for the LangChain Data Agent."""

from __future__ import annotations

import ast
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import run_query

st.set_page_config(
    page_title="LangChain Data Agent",
    page_icon="🎵",
    layout="wide",
)

st.title("LangChain Data Agent")
st.caption(
    "Ask questions about the Chinook music store database in plain English. "
    "Queries are translated to read-only SQL and answered with charts when relevant."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Database:** Chinook SQLite (artists, albums, tracks, sales)

        **Model:** kimi-k2.6 via [Orq.ai](https://orq.ai)
        """
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def _column_names_from_sql(sql: str | None, num_cols: int) -> list[str]:
    """Derive display column names from the SELECT list when results are bare tuples."""
    if not sql or num_cols < 1:
        return [f"col_{i + 1}" for i in range(num_cols)]

    match = re.search(r"\bSELECT\b(.+?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not match:
        return [f"col_{i + 1}" for i in range(num_cols)]

    select_part = match.group(1).strip()
    if select_part == "*":
        return [f"col_{i + 1}" for i in range(num_cols)]

    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for char in select_part + ",":
        if char == "(":
            depth += 1
            buf.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            buf.append(char)
        elif char == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(char)

    names: list[str] = []
    for part in parts:
        alias = re.search(
            r"\bAS\s+(?:[`\"']?)([A-Za-z_][\w$]*)(?:[`\"']?)\s*$",
            part,
            re.IGNORECASE,
        )
        if alias:
            names.append(alias.group(1))
            continue
        token = part.strip().rstrip(",").split()[-1]
        token = token.strip("`\"'[]")
        names.append(token.split(".")[-1] if token else f"col_{len(names) + 1}")

    if len(names) >= num_cols:
        return names[:num_cols]
    return names + [f"col_{i + 1}" for i in range(len(names), num_cols)]


def _parse_results(raw: str | None, sql: str | None = None) -> pd.DataFrame | None:
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            rows = ast.literal_eval(text)
            if not rows:
                return None
            if isinstance(rows[0], dict):
                return pd.DataFrame(rows)
            if isinstance(rows[0], tuple):
                cols = _column_names_from_sql(sql, len(rows[0]))
                return pd.DataFrame(rows, columns=cols)
        except (SyntaxError, ValueError):
            pass

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    header = re.split(r"\s*\|\s*", lines[0].strip("| "))
    data_rows = []
    for line in lines[2:]:
        if set(line.replace("|", "").strip()) <= {"-", ":"}:
            continue
        data_rows.append(re.split(r"\s*\|\s*", line.strip("| ")))

    if header and data_rows:
        width = len(header)
        normalized = [row[:width] + [""] * (width - len(row)) for row in data_rows]
        df = pd.DataFrame(normalized, columns=header)
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
        return df

    return None


def _build_chart(df: pd.DataFrame):
    if df.empty or len(df) < 2:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        if len(df.columns) >= 2:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Query results")
            return fig
        return None

    value_col = numeric_cols[-1]
    label_col = non_numeric[0] if non_numeric else df.columns[0]

    if label_col == value_col:
        return None

    plot_df = df[[label_col, value_col]].head(20).copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return None

    if len(plot_df) > 12:
        fig = px.bar(
            plot_df,
            x=label_col,
            y=value_col,
            title="Query results",
            labels={label_col: label_col, value_col: value_col},
        )
    else:
        fig = px.pie(
            plot_df,
            names=label_col,
            values=value_col,
            title="Query results",
        )
    fig.update_layout(xaxis_tickangle=-35, margin=dict(l=20, r=20, t=40, b=80))
    return fig


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            with st.expander("SQL query"):
                st.code(message["sql"], language="sql")
        if message.get("chart_df") is not None:
            chart = _build_chart(message["chart_df"])
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
        elif message.get("raw_results"):
            df = _parse_results(message["raw_results"], message.get("sql"))
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)

if prompt := st.chat_input("Ask a question about the Chinook database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = run_query(prompt, history=history)
                answer = result["answer"]
                sql = result.get("sql")
                raw = result.get("raw_results")
                df = _parse_results(raw, sql)

                st.markdown(answer)
                if sql:
                    with st.expander("SQL query"):
                        st.code(sql, language="sql")
                chart = _build_chart(df) if df is not None else None
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                elif df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sql": sql,
                        "raw_results": raw,
                        "chart_df": df,
                    }
                )
            except Exception as exc:
                error_text = f"Something went wrong: {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text}
                )
