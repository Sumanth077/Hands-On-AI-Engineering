import os
import asyncio
import uuid

# Load .env before any ADK/Google imports so API keys are available
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()  # Allow asyncio.run() inside Streamlit's existing event loop

import streamlit as st
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents import root_agent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Travel Planning Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("✈️ AI Travel Planner")
    st.markdown("---")
    st.markdown(
        "### How to use\n"
        "Type a natural language travel request in the chat below.\n\n"
        "**Example prompts:**\n"
        "- *Plan a 5-day trip to Paris in June, budget $2000, I love art and food*\n"
        "- *Weekend getaway to Tokyo, budget $1500, interested in anime and street food*\n"
        "- *10 days in Italy visiting Rome and Florence, $3000 budget, history lover*\n\n"
        "After the plan is generated, you can ask follow-up questions!"
    )
    st.markdown("---")
    st.markdown(
        "### Powered by\n"
        "- 🤖 Google ADK + Gemini\n"
        "- 🔍 Tavily Search API\n"
        "- 🗺️ OpenStreetMap / Nominatim\n"
        "- 🌐 Streamlit"
    )
    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # API key status indicators
    st.markdown("### API Status")
    google_ok = bool(os.getenv("GOOGLE_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    st.markdown(
        f"{'✅' if google_ok else '❌'} Google API Key\n\n"
        f"{'✅' if tavily_ok else '❌'} Tavily API Key"
    )
    if not google_ok or not tavily_ok:
        st.warning("Add missing API keys to your `.env` file.")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"

if "adk_runner" not in st.session_state:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="ai-travel-planning-agent",
        session_service=session_service,
    )
    st.session_state.adk_runner = runner
    st.session_state.adk_session_service = session_service

# ---------------------------------------------------------------------------
# Agent runner helper
# ---------------------------------------------------------------------------

async def _stream_response(runner: Runner, session_id: str, user_id: str, content: types.Content) -> str:
    """Collect the final response text from a single runner.run_async() call."""
    final_text = ""
    async for event in runner.run_async(
        session_id=session_id,
        user_id=user_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = "\n".join(
                    part.text for part in event.content.parts if hasattr(part, "text")
                )
            break
    return final_text


async def _run_agent_async(runner: Runner, session_id: str, user_id: str, message: str) -> str:
    """Run the ADK agent and collect the final response text."""
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)],
    )

    session_service = st.session_state.adk_session_service
    app_name = "ai-travel-planning-agent"

    # Ensure the session exists in the session service
    try:
        await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

    try:
        final_text = await _stream_response(runner, session_id, user_id, content)
    except Exception as e:
        if "Session not found" in str(e):
            new_session_id = str(uuid.uuid4())
            await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=new_session_id,
            )
            st.session_state.session_id = new_session_id
            final_text = await _stream_response(runner, new_session_id, user_id, content)
        else:
            raise

    return final_text or "I couldn't generate a response. Please try again."


def get_travel_plan(message: str) -> str:
    """Synchronous wrapper for the async ADK runner."""
    runner: Runner = st.session_state.adk_runner
    session_id: str = st.session_state.session_id
    user_id: str = st.session_state.user_id

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            _run_agent_async(runner, session_id, user_id, message)
        )
    except RuntimeError:
        return asyncio.run(
            _run_agent_async(runner, session_id, user_id, message)
        )

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------

st.title("🌍 AI Travel Planning Agent")
st.caption(
    "Describe your ideal trip and I'll coordinate flight, hotel, and itinerary specialists to build your complete travel plan."
)

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Welcome message on first load
if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome = (
            "👋 Welcome! I'm your AI Travel Planning Agent.\n\n"
            "Tell me about your dream trip and I'll coordinate three specialist agents "
            "(Flight Search, Hotel Finder, and Itinerary Planner) to build you a complete travel plan.\n\n"
            "**Try something like:**\n"
            "> *Plan a 5-day trip to Paris in June, budget $2000, I love art and food*"
        )
        st.markdown(welcome)

# Chat input
if prompt := st.chat_input("Describe your travel plans... (e.g. '5 days in Bali, June, $1500, beaches and culture')"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agents and show response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Coordinating Flight, Hotel & Itinerary agents — this may take 30-60 seconds..."):
            response = get_travel_plan(prompt)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
