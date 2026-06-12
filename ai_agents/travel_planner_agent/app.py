"""Travel Planner Agent - Streamlit chat interface."""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent import create_travel_agent, extract_agent_reply

load_dotenv()

st.set_page_config(
    page_title="Travel Planner Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _check_env() -> list[str]:
    required = ["ORQ_API_KEY", "OPENWEATHER_API_KEY", "EXCHANGERATE_API_KEY"]
    return [key for key in required if not os.getenv(key)]


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        missing = _check_env()
        if missing:
            st.session_state.agent = None
            st.session_state.env_error = (
                f"Missing API keys: {', '.join(missing)}. "
                "Copy `.env.example` to `.env` and add your keys."
            )
        else:
            try:
                st.session_state.agent = create_travel_agent()
                st.session_state.env_error = None
            except Exception as exc:
                st.session_state.agent = None
                st.session_state.env_error = str(exc)


def _build_messages() -> list:
    messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def main() -> None:
    _init_session()

    with st.sidebar:
        st.title("✈️ Travel Planner")
        st.markdown(
            "Plan your trip through conversation - weather, budget, "
            "highlights, and packing in one place."
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("Travel Planner Agent")
    st.caption("Describe your trip in natural language - I'll research and build your plan.")

    if st.session_state.get("env_error"):
        st.error(st.session_state.env_error)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Where would you like to go?"):
        if not st.session_state.agent:
            st.error(
                st.session_state.get("env_error")
                or "Agent not initialized. Check your environment variables."
            )
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Researching your trip..."):
                try:
                    result = st.session_state.agent.invoke(
                        {"messages": _build_messages()}
                    )
                    answer = extract_agent_reply(result)
                except Exception as exc:
                    answer = f"Something went wrong: {exc}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
