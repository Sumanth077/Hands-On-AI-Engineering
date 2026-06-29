"""Streamlit entry point for the LLM Agri Bot chat UI."""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from agent import run_agent, stream_agent_response

load_dotenv()

st.set_page_config(
    page_title="LLM Agri Bot",
    page_icon="🌾",
    layout="wide",
)

WELCOME_MESSAGE = (
    "Welcome to LLM Agri Bot! I can help with crop health, weather, pest control, "
    "and planting seasons. Ask a question like: "
    "'When should I plant tomatoes in spring?' or "
    "'What is the weather in Des Moines?'"
)

EXAMPLE_QUESTIONS = [
    "When should I plant corn in Iowa?",
    "What is the weather in Nairobi today?",
    "How do I control aphids on tomato plants?",
    "When is the best time to harvest wheat?",
    "What planting season does rice follow in tropical regions?",
]


def init_session_state() -> None:
    """Initialize chat message history in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]


def render_sidebar() -> None:
    """Render the sidebar with app info and example questions."""
    with st.sidebar:
        st.title("LLM Agri Bot")
        st.markdown(
            """
            **Your multi-tool farming assistant**

            Ask questions in plain English about:
            - Crop health and diseases
            - Local weather conditions
            - Pest control strategies
            - Planting and harvest timing
            """
        )
        st.divider()
        st.subheader("Example questions")
        for question in EXAMPLE_QUESTIONS:
            st.markdown(f"- {question}")


def render_chat_history() -> None:
    """Display stored messages in the chat interface."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_assistant_response(user_prompt: str) -> str:
    """Run the agent and return the assistant reply for a user prompt."""
    if not os.getenv("MISTRAL_API_KEY"):
        return (
            "MISTRAL_API_KEY is missing. Copy `.env.example` to `.env` and add your API keys."
        )

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Thinking and checking tools..."):
                callback_container = st.container()
                st_callback = StreamlitCallbackHandler(callback_container)

                streamed_parts: list[str] = []
                try:
                    for token in stream_agent_response(
                        user_prompt,
                        chat_history=st.session_state.messages,
                    ):
                        streamed_parts.append(token)
                        full_response = "".join(streamed_parts)
                        response_placeholder.markdown(full_response + "▌")
                except Exception:
                    full_response = run_agent(
                        user_prompt,
                        chat_history=st.session_state.messages,
                        callbacks=[st_callback],
                    )
                    response_placeholder.markdown(full_response)

            if full_response.endswith("▌"):
                full_response = full_response[:-1]
            response_placeholder.markdown(full_response)
            return full_response
        except Exception as exc:
            error_message = f"Sorry, something went wrong: {exc}"
            response_placeholder.markdown(error_message)
            return error_message


def main() -> None:
    """Run the Streamlit app entry point."""
    init_session_state()
    render_sidebar()

    st.title("LLM Agri Bot")
    st.caption("Multi-tool farming assistant powered by Mistral and LangChain")

    render_chat_history()

    if user_prompt := st.chat_input("Ask about crops, weather, pests, or planting seasons..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        assistant_response = get_assistant_response(user_prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )


if __name__ == "__main__":
    main()
