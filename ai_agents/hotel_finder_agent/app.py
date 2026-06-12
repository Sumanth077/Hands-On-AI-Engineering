"""
Streamlit UI for the Hotel Finder Agent.

Uses Orq.ai (qwen3.6-flash via OpenAI-compatible SDK) as the LLM
and the Trivago MCP server for hotel search.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

TRIVAGO_MCP_URL = "https://mcp.trivago.com/mcp"
ORQ_BASE_URL = "https://my.orq.ai/v3/router"
MODEL_ID = "alibaba/qwen3.6-flash"

SYSTEM_INSTRUCTION = f"""You are a helpful hotel search assistant powered by Trivago.

Today's date is {date.today().strftime("%B %d, %Y")}.

Help users find hotels based on natural language queries. Extract:
- Location (city, region, or country)
- Check-in and check-out dates
- Number of adults, children, and rooms
- Price range, star rating, and amenities if mentioned

Tool usage:
1. Use trivago-search-suggestions first to resolve the location name and get the right IDs
2. Use trivago-accommodation-search for the main hotel search
3. Use trivago-accommodation-radius-search when you have GPS coordinates

When presenting results:
- Show hotel name, star rating, price per night, and review score
- Include a booking link where available
- Highlight standout amenities (pool, breakfast, free cancellation, parking)
- Present the top 3-5 options clearly using markdown tables or lists

If no dates are provided, ask the user before searching.
If only a check-in date is given, default to a one-night stay.
"""

EXAMPLE_QUERIES = [
    "🏨 Hotels in Barcelona for next weekend",
    "💰 Budget hotels under $100 in Amsterdam",
    "⭐ 5-star hotels in Dubai for 2 adults",
    "🏖️ Beach hotels in Bali, check-in August 10",
    "🌆 Business hotels near Times Square NYC",
    "🏕️ Boutique hotels in Kyoto with breakfast included",
]

# ── Async agent loop ───────────────────────────────────────────────────────────

async def run_hotel_agent(api_key: str, messages: list[ChatCompletionMessageParam]) -> str:
    """
    Run one full agent turn:
    1. Open a Trivago MCP session via Streamable HTTP
    2. Fetch tool schemas and convert to OpenAI format
    3. Run the OpenAI tool-calling loop until a final answer is produced
    """
    client = OpenAI(base_url=ORQ_BASE_URL, api_key=api_key)

    async with streamablehttp_client(
        TRIVAGO_MCP_URL,
        timeout=30,
        sse_read_timeout=120,
        terminate_on_close=False,  # Trivago returns 501 on DELETE — skip it
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch Trivago tool schemas
            tools_result = await session.list_tools()
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools_result.tools
            ]

            # Prepend system prompt
            full_messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                *messages,
            ]

            # Tool-calling loop
            while True:
                response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=full_messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                msg = response.choices[0].message

                if msg.tool_calls:
                    # Append assistant message with tool calls
                    full_messages.append(msg)  # type: ignore[arg-type]

                    # Execute each tool call via the MCP session
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        result = await session.call_tool(tc.function.name, args)

                        # Collect text content from MCP result parts
                        content_parts = [
                            part.text
                            for part in result.content
                            if hasattr(part, "text")
                        ]
                        content = "\n".join(content_parts) if content_parts else "No results."

                        full_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": content,
                        })
                else:
                    return msg.content or "I couldn't find any results. Try rephrasing your query."


def _run_in_thread(api_key: str, messages: list[ChatCompletionMessageParam]) -> str:
    """Run the async agent in a dedicated thread with its own event loop."""
    import anyio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_hotel_agent(api_key, messages))
    except (anyio.ClosedResourceError, Exception) as exc:
        # Re-raise anything that isn't just MCP session teardown noise
        msg = str(exc)
        if "ClosedResourceError" in msg or "Session termination" in msg:
            return "Search completed but encountered a cleanup error. Please try again."
        raise
    finally:
        loop.close()


def run_agent_sync(api_key: str, messages: list[ChatCompletionMessageParam]) -> str:
    """Run in a separate thread to avoid Streamlit's event loop conflicts."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread, api_key, messages)
        return future.result(timeout=120)


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hotel Finder Agent",
    page_icon="🏨",
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
[data-testid="stChatMessage"] table {
    color: #e2e8f0 !important;
    border-collapse: collapse;
    width: 100%;
}
[data-testid="stChatMessage"] th {
    background-color: #1a2744 !important;
    color: #93c5fd !important;
    padding: 8px 12px;
    border: 1px solid #334155;
}
[data-testid="stChatMessage"] td {
    padding: 8px 12px;
    border: 1px solid #334155;
}
[data-testid="stChatMessage"] tr:nth-child(even) td {
    background-color: #172033 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
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
    border-color: #3b82f6 !important;
    color: #f1f5f9 !important;
}

hr { border-color: #334155 !important; }
.stSpinner > div { border-top-color: #3b82f6 !important; }

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
    st.session_state.history: list[ChatCompletionMessageParam] = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_searches" not in st.session_state:
    st.session_state.total_searches = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏨 Hotel Finder")
    st.markdown(
        '<p style="color:#94a3b8;font-size:13px;margin-top:-8px;">'
        "Natural language hotel search powered by Trivago</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**Configuration**")
    api_key = st.text_input(
        "Orq.ai API Key",
        type="password",
        value=os.getenv("ORQ_API_KEY", ""),
        placeholder="orq-...",
        help="Get one at orq.ai",
    )

    if api_key:
        st.markdown('<div class="status-ok">● Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-missing">○ API key required</div>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    col1.metric("Searches", st.session_state.total_searches)
    col2.metric("Data Source", "Trivago")

    st.divider()

    st.markdown("**Quick searches**")
    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True, key=f"ex_{example}"):
            st.session_state.pending_query = example

    st.divider()

    st.markdown("**What you can search by**")
    criteria = [
        ("📍", "Location (city, country, region)"),
        ("📅", "Check-in and check-out dates"),
        ("👥", "Adults, children, rooms"),
        ("⭐", "Star rating (1–5 stars)"),
        ("💰", "Price range and budget"),
        ("🏊", "Amenities (pool, parking, breakfast)"),
    ]
    for icon, label in criteria:
        st.markdown(
            f'<p style="color:#94a3b8;font-size:13px;margin:4px 0;">'
            f'{icon} {label}</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.total_searches = 0
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("""
<div>
    <div class="hero-title">🏨 Hotel Finder Agent</div>
    <div class="hero-sub">
        Search hotels worldwide using natural language.
        Powered by <strong style="color:#93c5fd;">Trivago</strong> and
        <strong style="color:#93c5fd;">qwen3.6-flash</strong> via Orq.ai.
    </div>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>Where do you want to stay?</h3>
        <p>Tell me the city, your dates, and any preferences.<br>
        I'll search Trivago and show you the best options.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Input handling ─────────────────────────────────────────────────────────────

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = st.chat_input("Where are you headed?")

# ── Run agent ──────────────────────────────────────────────────────────────────

if prompt:
    if not api_key:
        st.error("Enter your Orq.ai API key in the sidebar to get started.")
        st.stop()

    # Strip leading emoji from quick query buttons
    clean_prompt = prompt
    if prompt and prompt[0] in "🏨💰⭐🏖️🌆🏕️":
        parts = prompt.split(" ", 1)
        clean_prompt = parts[1] if len(parts) > 1 else prompt

    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    st.session_state.history.append({"role": "user", "content": clean_prompt})

    with st.chat_message("user"):
        st.markdown(clean_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Trivago..."):
            try:
                response_text = run_agent_sync(api_key, list(st.session_state.history))
                st.session_state.total_searches += 1
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.history.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"**Error:** {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
