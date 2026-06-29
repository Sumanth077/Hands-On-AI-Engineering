"""Documentation QnA Agent - Streamlit chat UI."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(message)s")

from agent_service import answer_question, fetch_documentation

load_dotenv()

st.set_page_config(
    page_title="Documentation QnA Agent",
    page_icon="📚",
    layout="wide",
)


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "doc_url": None,
        "doc_loaded": False,
        "doc_content": None,
        "load_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def run_async(coro):
    return asyncio.run(coro)


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.doc_loaded = False
    st.session_state.doc_content = None
    st.session_state.load_summary = None


def main() -> None:
    init_session_state()

    st.title("Documentation QnA Agent")
    st.caption(
        "Chat with any documentation by URL. Powered by Fetch MCP and DeepSeek V4 Flash."
    )

    with st.container():
        col_input, col_button = st.columns([5, 1])
        with col_input:
            doc_url = st.text_input(
                "Documentation URL",
                placeholder="https://docs.example.com/getting-started",
                label_visibility="collapsed",
            )
        with col_button:
            load_clicked = st.button("Load", type="primary", use_container_width=True)

    if load_clicked:
        url = doc_url.strip()
        if not is_valid_url(url):
            st.error(
                "Invalid URL. Enter a valid HTTP or HTTPS link, "
                "for example https://docs.python.org/3/"
            )
        else:
            if st.session_state.doc_url != url:
                reset_conversation()
            st.session_state.doc_url = url
            with st.spinner("Fetching documentation..."):
                try:
                    content, summary = run_async(fetch_documentation(url))
                    st.session_state.doc_loaded = True
                    st.session_state.doc_content = content
                    st.session_state.load_summary = summary
                except Exception as exc:
                    st.session_state.doc_loaded = False
                    st.session_state.doc_content = None
                    st.session_state.load_summary = None
                    message = str(exc)
                    if "404 page not found" in message and "status code 404" not in message:
                        st.error(
                            "The NVIDIA NIM model endpoint returned 404. "
                            f"Check NVIDIA_MODEL in .env (current default: moonshotai/kimi-k2.5). "
                            f"Details: {exc}"
                        )
                    else:
                        st.error(f"Failed to fetch documentation: {exc}")

    if st.session_state.doc_url and st.session_state.doc_loaded:
        st.success(f"Loaded: {st.session_state.doc_url}")
        if st.session_state.load_summary:
            with st.expander("Documentation summary", expanded=False):
                st.markdown(st.session_state.load_summary)
    elif st.session_state.doc_url and not st.session_state.doc_loaded:
        st.warning("Documentation is not loaded yet. Click Load to fetch it.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the documentation..."):
        if not st.session_state.doc_url:
            st.warning("Enter and load a documentation URL before asking questions.")
            return

        if not st.session_state.doc_loaded:
            st.warning("Load the documentation URL first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history = st.session_state.messages[:-1]
                    response = run_async(
                        answer_question(
                            st.session_state.doc_url,
                            prompt,
                            history,
                            st.session_state.doc_content or "",
                        )
                    )
                except Exception as exc:
                    response = f"Sorry, I could not answer that question: {exc}"
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
