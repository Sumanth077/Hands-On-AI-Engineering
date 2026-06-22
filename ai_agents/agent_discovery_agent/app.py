"""
Agent Discovery Agent: Streamlit UI for searching and comparing AI agents across multiple registries.
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent_discovery_agent.tools import (
    get_agent_details,
    get_search_facets,
    get_similar_agents,
    search_agents,
)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = "gemini-3-flash-preview"

SYSTEM_INSTRUCTION = """\
You are an AI Agent Discovery Assistant with access to a universal registry \
that indexes AI agents across five protocols: NANDA, MCP, Virtuals Protocol, A2A, and ERC-8004.

You help users discover, explore, and compare agents through natural language.

When presenting results:
- Always state which registry each agent comes from
- Include the agent name, a short description, and its UAID
- Highlight key capabilities or use cases
- Format results clearly — use markdown lists or tables where helpful
- When a UAID appears, call it out so the user can reference it for follow-up queries
- For comparisons, present agents side by side

Tool usage:
- search_agents() — for keyword or capability-based searches
- get_agent_details() — for full info on a specific agent (requires UAID)
- get_similar_agents() — to find alternatives or compare options (requires UAID)
- get_search_facets() — when users ask what categories, registries, or filters exist
"""

TOOL_FUNCTIONS = {
    "search_agents": search_agents,
    "get_agent_details": get_agent_details,
    "get_similar_agents": get_similar_agents,
    "get_search_facets": get_search_facets,
}

EXAMPLE_QUERIES = [
    "🔍 Find code review agents",
    "📈 Show trading bots on Virtuals Protocol",
    "🗄️ What MCP servers handle databases?",
    "💬 Find customer support agents",
    "📂 What categories of agents are available?",
    "⛓️ Find DeFi agents on ERC-8004",
    "🤖 Show me A2A compatible agents",
    "🔒 Find security and audit agents",
]

REGISTRIES = [
    ("NANDA", "MIT's network for AI agents", "#6366f1"),
    ("MCP", "Model Context Protocol servers", "#8b5cf6"),
    ("Virtuals", "On-chain agents on Base", "#a855f7"),
    ("A2A", "Google's Agent-to-Agent protocol", "#ec4899"),
    ("ERC-8004", "Ethereum on-chain agents", "#f97316"),
]

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agent Discovery Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp { background-color: #0f172a; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background-color: #1e293b;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span {
    color: #e2e8f0 !important;
}

[data-testid="stChatMessage"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessage"] p { color: #e2e8f0 !important; }
[data-testid="stChatMessage"] code {
    background-color: #0f172a !important;
    color: #a5f3fc !important;
    border-radius: 4px !important;
}
[data-testid="stChatMessage"] pre {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }

.stButton button {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    text-align: left !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background-color: #334155 !important;
    border-color: #6366f1 !important;
    color: #f1f5f9 !important;
}

hr { border-color: #334155 !important; }
.stSpinner > div { border-top-color: #6366f1 !important; }

.registry-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
    width: 100%;
}
.hero-title {
    font-size: 30px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 6px;
}
.hero-sub {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 20px;
}
.empty-state {
    text-align: center;
    padding: 48px 20px;
    color: #64748b;
}
.empty-state h3 { color: #94a3b8; font-size: 20px; margin-bottom: 8px; }
.empty-state p  { font-size: 14px; }
.example-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    cursor: pointer;
    font-size: 13px;
    color: #cbd5e1;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.example-card:hover { border-color: #6366f1; }
.status-ok {
    background: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #4ade80;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 8px;
}
.status-missing {
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ── Agent logic ────────────────────────────────────────────────────────────────

def get_client(api_key: str) -> genai.Client:
    """Create and return a Google GenAI client authenticated with the provided API key."""
    return genai.Client(api_key=api_key)


def run_agent(client: genai.Client, history: list, user_message: str) -> tuple[str, list]:
    """Run one full agent turn, handling tool calls in a loop."""
    history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[search_agents, get_agent_details, get_similar_agents, get_search_facets],
    )

    while True:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=history,
            config=config,
        )

        candidate = response.candidates[0]
        history.append(candidate.content)

        function_calls = [
            p for p in candidate.content.parts
            if hasattr(p, "function_call") and p.function_call
        ]

        if not function_calls:
            text_parts = [
                p.text for p in candidate.content.parts
                if hasattr(p, "text") and p.text
            ]
            return "\n".join(text_parts), history

        # Execute tool calls
        tool_results = []
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fn_name = part.function_call.name
                fn_args = dict(part.function_call.args)
                try:
                    result = TOOL_FUNCTIONS[fn_name](**fn_args)
                except Exception as e:
                    result = {"error": str(e)}

                tool_results.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_name,
                            response={"result": json.dumps(result, default=str)},
                        )
                    )
                )

        history.append(types.Content(role="tool", parts=tool_results))


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Agent Discovery")
    st.markdown(
        '<p style="color:#94a3b8;font-size:13px;margin-top:-8px;">'
        "Search 5 AI agent registries in one place</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # API key
    st.markdown("**Configuration**")
    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        placeholder="AIza...",
        help="Get one free at aistudio.google.com",
    )

    if api_key:
        st.markdown('<div class="status-ok">● Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-missing">○ API key required</div>', unsafe_allow_html=True)

    st.divider()

    # Session stats
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Registries", 5)

    st.divider()

    # Example queries
    st.markdown("**Quick queries**")
    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True, key=f"ex_{example}"):
            st.session_state.pending_query = example

    st.divider()

    # Supported registries
    st.markdown("**Supported Registries**")
    for name, desc, color in REGISTRIES:
        st.markdown(
            f'<div class="registry-badge" style="background:rgba(99,102,241,0.1);'
            f'border:1px solid {color};color:{color};">'
            f"● {name} <span style='color:#94a3b8;font-weight:400;font-size:11px;'>"
            f"— {desc}</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.total_queries = 0
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("""
<div>
    <div class="hero-title">🔍 Agent Discovery Agent</div>
    <div class="hero-sub">
        Search and compare AI agents across <strong style="color:#c7d2fe;">NANDA</strong>,
        <strong style="color:#c7d2fe;">MCP</strong>,
        <strong style="color:#c7d2fe;">Virtuals Protocol</strong>,
        <strong style="color:#c7d2fe;">A2A</strong>, and
        <strong style="color:#c7d2fe;">ERC-8004</strong>
        through a single natural language interface.
        Powered by <strong style="color:#c7d2fe;">Gemini 3 Flash</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>What agents are you looking for?</h3>
        <p>Try a quick query from the sidebar, or type anything below.<br>
        You can search by use case, registry, capability, or ask for a comparison.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Input handling ─────────────────────────────────────────────────────────────

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = st.chat_input("Ask anything about AI agents...")

# ── Run agent ──────────────────────────────────────────────────────────────────

if prompt:
    if not api_key:
        st.error("Enter your Google API key in the sidebar to get started.")
        st.stop()

    # Strip emoji prefix from quick query buttons
    clean_prompt = prompt.split(" ", 1)[-1] if prompt and prompt[0] in "🔍📈🗄️💬📂⛓️🤖🔒" else prompt

    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching registries..."):
            try:
                client = get_client(api_key)
                response_text, updated_history = run_agent(
                    client,
                    st.session_state.history,
                    clean_prompt,
                )
                st.session_state.history = updated_history
                st.session_state.total_queries += 1
                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
            except Exception as e:
                error_msg = f"**Error:** {e}"
                st.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
