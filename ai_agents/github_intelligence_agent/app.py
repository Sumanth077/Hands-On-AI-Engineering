"""
GitHub Intelligence Agent — Streamlit UI
Research anything on GitHub through natural language.
Powered by Gemini 3 Flash + GitHub MCP Server via Haystack.

Usage:
    uv run streamlit run app.py
"""

from dotenv import load_dotenv

load_dotenv()

import os
import time
import streamlit as st
from haystack.dataclasses import ChatMessage

from agent import build_agent

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GitHub Intelligence Agent",
    page_icon="🐙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GitHub Dark Theme CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
.stApp { background-color: #0d1117; color: #c9d1d9; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9d1d9 !important;
}

/* ── Inputs ── */
.stTextInput input {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
}
.stTextInput input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}
.stTextInput input::placeholder { color: #484f58 !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #484f58 !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessage"] p { color: #c9d1d9 !important; }
[data-testid="stChatMessage"] code {
    background-color: #0d1117 !important;
    color: #79c0ff !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
}
[data-testid="stChatMessage"] pre {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* ── Buttons ── */
.stButton button {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
    text-align: left !important;
}
.stButton button:hover {
    background-color: #30363d !important;
    border-color: #58a6ff !important;
    color: #f0f6fc !important;
}

/* ── Divider ── */
hr { border-color: #30363d !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #238636 !important; }

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    width: 100%;
    margin-bottom: 4px;
}
.status-connected {
    background-color: rgba(35,134,54,0.15);
    border: 1px solid #238636;
    color: #3fb950;
}
.status-missing {
    background-color: rgba(218,54,51,0.15);
    border: 1px solid #da3633;
    color: #f85149;
}
.status-partial {
    background-color: rgba(187,128,9,0.15);
    border: 1px solid #bb8009;
    color: #d29922;
}

/* ── Stat cards ── */
.stats-row {
    display: flex;
    gap: 8px;
    margin: 4px 0 12px 0;
}
.stat-card {
    flex: 1;
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
}
.stat-number {
    font-size: 22px;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1.2;
}
.stat-label {
    font-size: 10px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Tool badges ── */
.tool-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #30363d;
}
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.25);
    color: #58a6ff;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, monospace;
}

/* ── Response meta ── */
.response-meta {
    font-size: 11px;
    color: #484f58;
    margin-top: 8px;
    padding-top: 6px;
}

/* ── Hero header ── */
.hero {
    border-bottom: 1px solid #30363d;
    padding-bottom: 20px;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 28px;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 6px;
}
.hero p {
    color: #8b949e;
    font-size: 15px;
    margin: 0;
}
.hero strong { color: #c9d1d9; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #484f58;
}
.empty-state h3 { color: #8b949e; font-size: 18px; margin-bottom: 8px; }
.empty-state p  { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_tools_used" not in st.session_state:
    st.session_state.total_tools_used = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🐙 GitHub Intelligence Agent")
    st.markdown('<p style="color:#8b949e;font-size:13px;margin-top:-8px;">Powered by Gemini 3 Flash + GitHub MCP</p>', unsafe_allow_html=True)

    st.divider()

    # Credentials
    st.markdown("**Configuration**")

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        placeholder="AIza...",
        help="Get yours free at https://aistudio.google.com",
    )

    github_pat = st.text_input(
        "GitHub Personal Access Token",
        type="password",
        value=os.getenv("GITHUB_PAT", ""),
        placeholder="ghp_...",
        help="GitHub → Settings → Developer Settings → Personal Access Tokens. Read-only is enough.",
    )

    # Connection status badge
    has_gemini = bool(gemini_api_key)
    has_pat = bool(github_pat)

    if has_gemini and has_pat:
        st.markdown('<div class="status-badge status-connected">● Ready to research</div>', unsafe_allow_html=True)
    elif has_gemini or has_pat:
        missing = "GitHub PAT" if not has_pat else "Gemini API Key"
        st.markdown(f'<div class="status-badge status-partial">◐ Missing {missing}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-missing">○ Credentials required</div>', unsafe_allow_html=True)

    st.divider()

    # Session stats
    st.markdown("**Session Stats**")
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-number">{st.session_state.total_queries}</div>
            <div class="stat-label">Queries</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{st.session_state.total_tools_used}</div>
            <div class="stat-label">Tools Used</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Example queries
    st.markdown("**Quick queries**")
    examples = [
        "🔥 Top 5 repos by karpathy",
        "🐛 Open issues in huggingface/transformers",
        "🚀 What has Microsoft shipped on GitHub lately?",
        "📈 Trending Python AI repos by stars",
        "👥 Top contributors to pytorch/pytorch",
        "🔍 Search repos about RAG pipelines",
    ]
    for example in examples:
        if st.button(example, use_container_width=True, key=f"ex_{example}"):
            st.session_state.pending_query = example

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_tools_used = 0
        st.rerun()

# ── Main Area ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🐙 GitHub Intelligence Agent</h1>
    <p>Research anything on GitHub through natural language. Powered by <strong>Gemini 3 Flash</strong>
    and GitHub's official MCP server. Uses Haystack's <strong>SearchableToolset</strong>
    to discover tools on demand — keeping context lean across 40+ GitHub API endpoints.</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools"):
            badges = "".join([f'<span class="tool-badge">⚙ {t}</span>' for t in message["tools"]])
            st.markdown(f'<div class="tool-badges">{badges}</div>', unsafe_allow_html=True)
        if message.get("response_time"):
            tool_count = len(message.get("tools", []))
            st.markdown(
                f'<div class="response-meta">⏱ {message["response_time"]:.1f}s &nbsp;·&nbsp; '
                f'⚙ {tool_count} tool{"s" if tool_count != 1 else ""} called</div>',
                unsafe_allow_html=True,
            )

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>What do you want to know about GitHub?</h3>
        <p>Try one of the quick queries in the sidebar, or type anything below.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Handle Input ───────────────────────────────────────────────────────────────

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = st.chat_input("Ask anything about GitHub...")

# ── Run Agent ──────────────────────────────────────────────────────────────────

if prompt:
    if not gemini_api_key or not github_pat:
        st.error("Enter your Gemini API Key and GitHub PAT in the sidebar to get started.")
        st.stop()

    # Strip emoji prefix from quick query buttons
    clean_prompt = prompt.split(" ", 1)[-1] if prompt[0] in "🔥🐛🚀📈👥🔍" else prompt

    # Display user message
    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    # Run agent
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = {"text": ""}
        start_time = time.time()

        def streaming_callback(chunk):
            if chunk.content:
                full_response["text"] += chunk.content
                placeholder.markdown(full_response["text"] + "▌")

        try:
            with st.spinner("Researching GitHub..."):
                agent = build_agent(github_pat, gemini_api_key, streaming_callback)
                result = agent.run([ChatMessage.from_user(clean_prompt)])

            elapsed = time.time() - start_time

            # Finalise response text
            if not full_response["text"]:
                for msg in reversed(result.get("messages", [])):
                    if hasattr(msg, "text") and msg.text:
                        full_response["text"] = msg.text
                        break
                    if hasattr(msg, "content") and msg.content:
                        full_response["text"] = msg.content
                        break

            final_text = full_response["text"] or "_No response received. Try rephrasing your query._"
            placeholder.markdown(final_text)

            # Extract tools used from result messages
            tools_used = []
            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = getattr(tc, "tool_name", None)
                        if name and name not in tools_used:
                            tools_used.append(name)

            # Tool badges
            if tools_used:
                badges = "".join([f'<span class="tool-badge">⚙ {t}</span>' for t in tools_used])
                st.markdown(f'<div class="tool-badges">{badges}</div>', unsafe_allow_html=True)

            # Response metadata
            tool_count = len(tools_used)
            st.markdown(
                f'<div class="response-meta">⏱ {elapsed:.1f}s &nbsp;·&nbsp; '
                f'⚙ {tool_count} tool{"s" if tool_count != 1 else ""} called</div>',
                unsafe_allow_html=True,
            )

            # Update session stats
            st.session_state.total_queries += 1
            st.session_state.total_tools_used += tool_count

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "tools": tools_used,
                "response_time": elapsed,
            })

        except Exception as exc:
            error_msg = f"**Error:** {exc}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
