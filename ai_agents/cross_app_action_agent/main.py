"""Streamlit interface for the Cross-App Action Agent."""

from __future__ import annotations

import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from agent import build_graph, user_message
from mcp_client import LinerActionsClient

load_dotenv()
st.set_page_config(page_title="Cross-App Action Agent", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner=False)
def bridge() -> LinerActionsClient:
    """Keep the Liner OAuth client alive across Streamlit reruns."""
    return LinerActionsClient()


@st.cache_resource(show_spinner=False)
def runtime(api_key: str) -> Any:
    """Keep the graph and its API client alive for the active browser session."""
    graph = build_graph(
        bridge(),
        os.environ.get("AI_MODEL", "zai/glm-5.3"),
        os.environ.get("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1"),
        api_key,
    )
    return graph


def thread_config() -> dict[str, dict[str, str]]:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def show_messages(messages: list[dict[str, str]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def save_result(result: dict[str, Any], graph: Any) -> None:
    interrupts = result.get("__interrupt__", [])
    st.session_state.pending = interrupts[0].value if interrupts else None
    state = result if not interrupts else graph.get_state(thread_config()).values
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "type", "") == "ai":
        text = str(messages[-1].content)
        if not st.session_state.chat or st.session_state.chat[-1]["content"] != text:
            st.session_state.chat.append({"role": "assistant", "content": text})


def active_api_key() -> str:
    return st.session_state.get("gateway_api_key", "").strip()


def run_request(prompt: str, status: Any) -> None:
    api_key = active_api_key()
    if not api_key:
        raise RuntimeError("Enter a model API key before starting a task.")
    graph = runtime(api_key)
    status.write("The model is selecting the next Liner tool.")
    result: dict[str, Any] | None = None
    for update in graph.stream(user_message(prompt, st.session_state.tools), thread_config()):
        if "agent" in update:
            status.write("The model selected a tool. Running the next safe step.")
        if "execute" in update:
            status.write("Liner is processing the requested action.")
        if "__interrupt__" in update:
            result = update
            status.write("The action is ready for your approval.")
    if result is None:
        result = graph.get_state(thread_config()).values
    save_result(result, graph)


def resume(decision: str) -> None:
    api_key = active_api_key()
    if not api_key:
        raise RuntimeError("Enter the same model API key to continue this task.")
    graph = runtime(api_key)
    result = graph.invoke(Command(resume={"decision": decision}), thread_config())
    save_result(result, graph)


def reset_chat() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.chat = []
    st.session_state.pending = None


if "chat" not in st.session_state:
    st.session_state.chat = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "gateway_api_key" not in st.session_state:
    st.session_state.gateway_api_key = os.environ.get("AI_GATEWAY_API_KEY", "")

st.title("Cross-App Action Agent")
st.caption("Ask once. Review every external action before it runs.")

left, right = st.columns([1.7, 1])
with right:
    st.subheader("Model")
    st.text_input(
        "Model API key",
        type="password",
        key="gateway_api_key",
        help="Stored only in this browser session. You can also set AI_GATEWAY_API_KEY in .env.",
    )
    st.caption("Model: `zai/glm-5.3`")

    st.subheader("Liner connection")
    st.write("Connect Liner once. Connect Slack, GitHub, or Notion only when an action needs it.")
    if st.button("Connect Liner Actions", width="stretch"):
        try:
            with st.spinner("Your browser will open for Liner sign-in."):
                tools = bridge().connect_sync()
            st.session_state.tools = [tool.as_openai_tool() for tool in tools]
            st.success(f"Connected. Liner exposed {len(tools)} tools.")
        except Exception as error:
            st.error(f"Could not connect to Liner Actions: {error}")

    if st.button("Clear conversation", width="stretch"):
        reset_chat()
        st.rerun()

    st.subheader("Suggested demo")
    st.code(
        "Summarise the latest messages in your Slack channel and create a GitHub issue "
        "for a bug mentioned there.",
        language=None,
    )

    if st.session_state.pending:
        st.subheader("Approval required")
        pending = st.session_state.pending
        st.write(pending["message"])
        st.caption(pending["tool_name"])
        st.json(pending["arguments"])
        approve, reject = st.columns(2)
        if approve.button("Approve", type="primary", width="stretch"):
            try:
                resume("approve")
                st.rerun()
            except RuntimeError as error:
                st.error(str(error))
        if reject.button("Reject", width="stretch"):
            try:
                resume("reject")
                st.rerun()
            except RuntimeError as error:
                st.error(str(error))

with left:
    if not st.session_state.chat:
        st.info("Connect Liner Actions, then ask for work across your connected apps.")
    show_messages(st.session_state.chat)
    prompt = st.chat_input("For example: summarise the latest customer feedback and open a GitHub issue")
    if prompt:
        if "tools" not in st.session_state:
            st.warning("Connect Liner Actions before starting a task.")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            completed = False
            try:
                with st.status("Starting the agent...", expanded=True) as status:
                    run_request(prompt, status)
                    status.update(label="Agent turn complete", state="complete")
                completed = True
            except RuntimeError as error:
                st.error(str(error))
            if completed:
                st.rerun()

