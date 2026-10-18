"""Streamlit interface for the self-evolving code review agent."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from self_evolving_agent.config import load_settings
from self_evolving_agent.embedder import Embedder
from self_evolving_agent.graph import build_graph, resume_review, start_review
from self_evolving_agent.llm import LocalLLM
from self_evolving_agent.memory import ReviewMemory, connect_client


SAMPLE_DIFF = """diff --git a/orders.py b/orders.py
index 4c720ab..89bd4cd 100644
--- a/orders.py
+++ b/orders.py
@@ -18,8 +18,13 @@ def get_order(order_id: str, user_id: str):
-    query = f"SELECT * FROM orders WHERE id = '{order_id}'"
-    return db.execute(query).fetchone()
+    order = db.execute(
+        "SELECT * FROM orders WHERE id = ?", (order_id,)
+    ).fetchone()
+    if order is None:
+        return None
+    return order

 def delete_order(order_id: str, user_id: str):
-    return db.execute("DELETE FROM orders WHERE id = ?", (order_id,))
+    return db.execute("DELETE FROM orders WHERE id = ?", (order_id,))
"""


st.set_page_config(
    page_title="Self-Evolving Code Review Agent",
    page_icon=":material/model_training:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading local embedding model...")
def load_resources() -> tuple[Any, ReviewMemory, LocalLLM, Any]:
    settings = load_settings()
    embedder = Embedder(settings.embedding_model)
    client = connect_client(settings.actian_url, settings.actian_access_token)
    memory = ReviewMemory(
        client=client,
        embedder=embedder,
        insight_top_k=settings.insight_top_k,
        trajectory_top_k=settings.trajectory_top_k,
        min_score=settings.min_relevance_score,
    )
    memory.setup()
    llm = LocalLLM(settings.ollama_host, settings.ollama_model)
    graph = build_graph(memory, llm)
    return settings, memory, llm, graph


def initialize_state() -> None:
    st.session_state.setdefault("pending_review", None)
    st.session_state.setdefault("completed_review", None)
    st.session_state.setdefault("diff_input", "")


def load_sample() -> None:
    st.session_state.diff_input = SAMPLE_DIFF
    st.session_state.pending_review = None
    st.session_state.completed_review = None


def clear_workspace() -> None:
    st.session_state.diff_input = ""
    st.session_state.pending_review = None
    st.session_state.completed_review = None


def uploaded_text(uploaded_file: Any) -> str:
    return uploaded_file.getvalue().decode("utf-8", errors="replace")


def severity_color(severity: str) -> str:
    return {
        "critical": "red",
        "high": "red",
        "medium": "orange",
        "low": "blue",
    }.get(severity, "gray")


initialize_state()

st.title("Self-Evolving Code Review Agent")
st.caption(
    "Reviews code with remembered team conventions, then learns from every accept, reject, and edit."
)

try:
    settings, memory, llm, graph = load_resources()
except Exception as exc:
    st.error(f"Startup failed: {exc}", icon=":material/error:")
    st.info(
        "Copy `.env.example` to `.env`, start VectorAI DB, and make sure Ollama is running."
    )
    st.stop()

with st.sidebar:
    st.subheader("Local stack")
    ok, model_message = llm.check()
    (st.success if ok else st.warning)(model_message)
    st.caption(f"Actian: {settings.actian_url}")
    st.caption(f"Embeddings: {settings.embedding_model.split('/')[-1]}")

    counts = memory.counts()
    st.subheader("Learning memory")
    st.metric("Learned insights", counts["insights"], border=True)
    st.metric("Review trajectories", counts["trajectories"], border=True)

    if st.button("Clear current review", icon=":material/refresh:", width="stretch"):
        clear_workspace()
        st.rerun()

review_tab, insights_tab, progress_tab, trajectories_tab = st.tabs(
    [
        ":material/rate_review: Review",
        ":material/psychology: Insights",
        ":material/trending_down: Progress",
        ":material/history: Trajectories",
    ]
)

with review_tab:
    st.subheader("Review a change")
    st.caption("Paste code or a unified diff. You can also upload a local source or patch file.")

    with st.container(horizontal=True, vertical_alignment="bottom"):
        uploaded = st.file_uploader(
            "Upload code or diff",
            type=["diff", "patch", "txt", "py", "js", "jsx", "ts", "tsx", "java", "go", "rs"],
            label_visibility="collapsed",
        )
        st.button("Load sample diff", icon=":material/science:", on_click=load_sample)

    if uploaded is not None and st.session_state.diff_input != uploaded_text(uploaded):
        st.session_state.diff_input = uploaded_text(uploaded)

    with st.form("review_form"):
        language = st.selectbox(
            "Language",
            ["Auto-detect", "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "Other"],
        )
        team_context = st.text_input(
            "Team context",
            placeholder="Example: FastAPI service; correctness and security matter more than style",
        )
        diff_text = st.text_area(
            "Diff or code",
            key="diff_input",
            height=320,
            placeholder="Paste a unified diff or code snippet here...",
        )
        run_review = st.form_submit_button(
            "Run review",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

    if run_review:
        if not diff_text.strip():
            st.warning("Paste or upload code before running a review.")
        elif not ok:
            st.error(model_message)
        else:
            with st.status("Reviewing with learned team guidance...", expanded=True) as status:
                st.write("Retrieving relevant insights and past reviews")
                try:
                    review_id, result = start_review(
                        graph,
                        diff_text=diff_text,
                        language=language,
                        team_context=team_context,
                    )
                except Exception as exc:
                    status.update(label="Review failed", state="error")
                    st.error(str(exc))
                else:
                    status.update(label="Review ready for engineer feedback", state="complete")
                    st.session_state.pending_review = {
                        "review_id": review_id,
                        "result": result,
                    }
                    st.session_state.completed_review = None
                    st.rerun()

    pending = st.session_state.pending_review
    if pending:
        result = pending["result"]
        comments = result.get("comments", [])
        st.subheader("Agent review")
        st.write(result.get("summary", ""))

        recalled_insights = result.get("insights", [])
        recalled_trajectories = result.get("trajectories", [])
        st.caption(
            f"Used {len(recalled_insights)} relevant insights and "
            f"{len(recalled_trajectories)} similar trajectories."
        )

        with st.form("feedback_form"):
            feedback_rows = []
            if not comments:
                st.success("No actionable issues found.", icon=":material/check_circle:")

            for index, comment in enumerate(comments, start=1):
                with st.container(border=True):
                    color = severity_color(comment["severity"])
                    st.markdown(
                        f"**Comment {index}** :{color}-badge[{comment['severity']}] "
                        f":gray-badge[{comment['category']}]"
                    )
                    st.caption(f"Location: {comment['line']}")
                    st.write(comment["comment"])
                    st.markdown(f"**Suggestion:** {comment['suggestion']}")

                    action = st.segmented_control(
                        f"Decision for comment {index}",
                        ["Accept", "Reject", "Edit"],
                        key=f"decision_{pending['review_id']}_{comment['comment_id']}",
                    )
                    edited = st.text_area(
                        f"Edited comment {index}",
                        placeholder="Required only when Edit is selected",
                        key=f"edit_{pending['review_id']}_{comment['comment_id']}",
                    )
                    note = st.text_input(
                        f"Reason or team convention {index}",
                        placeholder="Optional, but useful for rejected or edited comments",
                        key=f"note_{pending['review_id']}_{comment['comment_id']}",
                    )
                    feedback_rows.append((comment, action, edited, note))

            teach_agent = st.form_submit_button(
                "Save feedback and teach agent",
                type="primary",
                icon=":material/model_training:",
                width="stretch",
            )

        if teach_agent:
            errors = []
            feedback = []
            for index, (comment, action, edited, note) in enumerate(feedback_rows, start=1):
                if action is None:
                    errors.append(f"Choose a decision for comment {index}.")
                    continue
                if action == "Edit" and not edited.strip():
                    errors.append(f"Add the edited text for comment {index}.")
                    continue
                feedback.append(
                    {
                        "comment_id": comment["comment_id"],
                        "action": action.lower(),
                        "edited_comment": edited.strip(),
                        "note": note.strip(),
                    }
                )

            if errors:
                for error in errors:
                    st.warning(error)
            else:
                with st.status("Reflecting on engineer feedback...", expanded=True) as status:
                    st.write("Distilling reusable review insights")
                    try:
                        completed = resume_review(graph, pending["review_id"], feedback)
                    except Exception as exc:
                        status.update(label="Learning failed", state="error")
                        st.error(str(exc))
                    else:
                        status.update(label="Learning stored in Actian", state="complete")
                        st.session_state.completed_review = completed
                        st.session_state.pending_review = None
                        st.rerun()

    completed = st.session_state.completed_review
    if completed:
        metrics = completed.get("metrics", {})
        st.success(
            f"Stored this trajectory and added {metrics.get('insights_added', 0)} learned insights.",
            icon=":material/check_circle:",
        )
        st.write(completed.get("reflection_summary", ""))
        new_insights = completed.get("new_insights", [])
        for insight in new_insights:
            with st.container(border=True):
                st.badge(insight["polarity"], color="blue")
                st.write(insight["rule"])
                st.caption(insight["rationale"])

with insights_tab:
    st.subheader("Learned review rules")
    insights = memory.list_insights()
    if not insights:
        st.info("No insights yet. Complete a review and submit feedback to create the first rules.")
    else:
        insight_df = pd.DataFrame(insights)
        visible = [
            column
            for column in ["rule", "polarity", "scope", "confidence", "rationale", "created_at"]
            if column in insight_df.columns
        ]
        st.dataframe(insight_df[visible], hide_index=True, width="stretch")

with progress_tab:
    st.subheader("Feedback trend")
    trajectories = memory.list_trajectories()
    if not trajectories:
        st.info("Progress appears after the first feedback cycle is stored.")
    else:
        trajectory_df = pd.DataFrame(trajectories)
        total_comments = int(trajectory_df["comment_count"].sum())
        total_rejected = int(trajectory_df["rejected_count"].sum())
        latest_rate = float(trajectory_df.iloc[-1]["rejection_rate"])
        overall_rate = total_rejected / total_comments if total_comments else 0.0

        with st.container(horizontal=True):
            st.metric("Completed reviews", len(trajectory_df), border=True)
            st.metric("Comments reviewed", total_comments, border=True)
            st.metric("Overall rejection rate", f"{overall_rate:.0%}", border=True)
            st.metric("Latest rejection rate", f"{latest_rate:.0%}", border=True)

        chart_df = trajectory_df[["created_at", "rejection_rate"]].copy()
        chart_df["Review"] = range(1, len(chart_df) + 1)
        chart_df["Rejected comments"] = chart_df["rejection_rate"] * 100
        st.line_chart(
            chart_df,
            x="Review",
            y="Rejected comments",
            y_label="Rejected comments (%)",
        )
        st.caption(
            "A downward trend is evidence that generated comments need less correction. "
            "It is not proof of correctness on its own."
        )

with trajectories_tab:
    st.subheader("Past review trajectories")
    trajectories = memory.list_trajectories()
    if not trajectories:
        st.info("No review trajectories stored yet.")
    else:
        trajectory_df = pd.DataFrame(trajectories)
        visible = [
            column
            for column in [
                "created_at",
                "language",
                "review_summary",
                "accepted_count",
                "rejected_count",
                "edited_count",
                "rejection_rate",
            ]
            if column in trajectory_df.columns
        ]
        st.dataframe(
            trajectory_df[visible].sort_values("created_at", ascending=False),
            hide_index=True,
            width="stretch",
            column_config={
                "rejection_rate": st.column_config.NumberColumn(
                    "Rejection rate", format="percent"
                )
            },
        )
