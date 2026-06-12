"""
Cal Scheduling Agent: Streamlit UI for managing Cal.com appointments through natural language.
"""

import os
from datetime import date, datetime

import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.calcom import CalComTools
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID = "gemini-3-flash-preview"

EXAMPLE_QUERIES = [
    "📅 What slots are free this week?",
    "📝 Book a meeting for tomorrow at 2pm",
    "🗓️ Show my upcoming bookings",
    "🔄 Reschedule my next appointment",
    "❌ Cancel my booking",
    "🌍 Find slots in New York timezone",
    "📧 Check bookings for john@example.com",
    "⏰ What's available next Monday?",
]


def build_instructions(timezone: str = "UTC") -> str:
    """Return the system instruction string for the agent, injecting today's date and the user's timezone."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    return f"""\
You are a friendly and efficient calendar scheduling assistant with access to \
the user's Cal.com calendar. Today is {today}.

You can help users by:
- Finding available time slots using get_available_slots(start_date, end_date)
- Creating new bookings using create_booking(start_time, name, email)
- Viewing existing bookings using get_upcoming_bookings(email)
- Rescheduling bookings using reschedule_booking(booking_uid, new_start_time, reason)
- Cancelling bookings using cancel_booking(booking_uid, reason)

IMPORTANT STEPS for booking:
1. First check available slots using get_available_slots
2. Confirm all details (time, name, email) with the user before booking
3. Create the booking using create_booking
4. Verify by calling get_upcoming_bookings with the user's email

Guidelines:
- Dates should be in YYYY-MM-DD format
- Times should be in YYYY-MM-DDTHH:MM:SS+TZ format (e.g. 2024-06-03T14:00:00+01:00)
- The user's timezone is {timezone}. Use it for all datetime conversions
- When showing available slots, group by date and show up to 8 per day
- Before creating a booking, show a summary and ask "Shall I confirm this booking?"
- Be concise and warm. Don't ask for information you already have
- Format results clearly using markdown tables or bullet lists
"""


def build_agent(google_api_key: str, calcom_api_key: str, event_type_id: str, timezone: str) -> Agent:
    """Create a fresh Agno agent with the provided credentials."""
    os.environ["GOOGLE_API_KEY"] = google_api_key
    calcom_tools = CalComTools(
        api_key=calcom_api_key,
        event_type_id=event_type_id if event_type_id else None,
        user_timezone=timezone,
    )
    return Agent(
        name="Cal Scheduling Agent",
        instructions=[build_instructions(timezone)],
        model=Gemini(id=MODEL_ID),
        tools=[calcom_tools],
        markdown=True,
    )


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cal Scheduling Agent",
    page_icon="📅",
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
[data-testid="stChatMessage"] p  { color: #e2e8f0 !important; }
[data-testid="stChatMessage"] li { color: #e2e8f0 !important; }
[data-testid="stChatMessage"] td { color: #e2e8f0 !important; }
[data-testid="stChatMessage"] th {
    color: #c7d2fe !important;
    background-color: #1e3a5f !important;
}
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
    border-collapse: collapse !important;
    width: 100% !important;
}
[data-testid="stChatMessage"] tr {
    border-bottom: 1px solid #334155 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.2) !important;
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
    border-color: #10b981 !important;
    color: #f1f5f9 !important;
}

hr { border-color: #334155 !important; }
.stSpinner > div { border-top-color: #10b981 !important; }

.status-ok {
    background: rgba(16,185,129,0.15);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 4px;
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
    margin-bottom: 4px;
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
.capability-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 6px;
    width: 100%;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6ee7b7;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "last_creds" not in st.session_state:
    st.session_state.last_creds = None

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📅 Cal Scheduling Agent")
    st.markdown(
        '<p style="color:#94a3b8;font-size:13px;margin-top:-8px;">'
        "Manage your Cal.com calendar with natural language</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**Configuration**")

    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        placeholder="AIza...",
        help="Get one free at aistudio.google.com",
    )
    calcom_api_key = st.text_input(
        "Cal.com API Key",
        type="password",
        value=os.getenv("CALCOM_API_KEY", ""),
        placeholder="cal_live_...",
        help="From cal.com/settings/developer/api-keys",
    )
    event_type_id = st.text_input(
        "Event Type ID (optional)",
        value=os.getenv("CALCOM_EVENT_TYPE_ID", ""),
        placeholder="123456",
        help="Optional. Scopes availability checks to a specific event type.",
    )
    timezone = st.selectbox(
        "Your timezone",
        options=[
            "UTC", "Europe/London", "America/New_York", "America/Los_Angeles",
            "America/Chicago", "Asia/Kolkata", "Asia/Tokyo", "Australia/Sydney",
            "Europe/Paris", "Europe/Berlin", "Asia/Dubai", "Africa/Lagos",
        ],
        index=0,
    )

    all_set = bool(google_api_key and calcom_api_key)

    if all_set:
        st.markdown('<div class="status-ok">● Ready</div>', unsafe_allow_html=True)
    else:
        missing = []
        if not google_api_key: missing.append("Google API key")
        if not calcom_api_key: missing.append("Cal.com API key")
        st.markdown(
            f'<div class="status-missing">○ Missing: {", ".join(missing)}</div>',
            unsafe_allow_html=True,
        )

    # Rebuild agent when credentials or timezone change
    current_creds = (google_api_key, calcom_api_key, event_type_id, timezone)
    if all_set and current_creds != st.session_state.last_creds:
        st.session_state.agent = build_agent(google_api_key, calcom_api_key, event_type_id, timezone)
        st.session_state.last_creds = current_creds

    st.divider()

    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Messages", len(st.session_state.messages))

    st.divider()

    st.markdown("**Quick queries**")
    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True, key=f"ex_{example}"):
            st.session_state.pending_query = example

    st.divider()

    st.markdown("**Capabilities**")
    for icon, label in [
        ("📅", "Check availability"),
        ("📝", "Create bookings"),
        ("🗓️", "View appointments"),
        ("🔄", "Reschedule"),
        ("❌", "Cancel bookings"),
        ("🌍", "Timezone conversion"),
    ]:
        st.markdown(
            f'<div class="capability-badge">{icon} {label}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        # Reset agent to clear its internal history too
        if all_set:
            st.session_state.agent = build_agent(google_api_key, calcom_api_key, event_type_id, timezone)
        st.session_state.total_queries = 0
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("""
<div>
    <div class="hero-title">📅 Cal Scheduling Agent</div>
    <div class="hero-sub">
        Manage your <strong style="color:#6ee7b7;">Cal.com</strong> calendar
        through natural conversation. Check availability, book appointments,
        reschedule or cancel — with automatic timezone handling.
        Powered by <strong style="color:#6ee7b7;">Gemini 3 Flash</strong> and
        <strong style="color:#6ee7b7;">Agno</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>What would you like to schedule?</h3>
        <p>Try a quick query from the sidebar, or just type naturally.<br>
        You can check availability, book a meeting, reschedule, or cancel.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = st.chat_input("Ask anything about your schedule...")

# ── Run agent ──────────────────────────────────────────────────────────────────

if prompt:
    if not all_set:
        st.error("Please enter your Google API key, Cal.com API key, and Event Type ID in the sidebar.")
        st.stop()

    clean_prompt = (
        prompt.split(" ", 1)[-1]
        if prompt and prompt[0] in "📅📝🗓️🔄❌🌍📧⏰"
        else prompt
    )

    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking your calendar..."):
            try:
                agent: Agent = st.session_state.agent
                response = agent.run(clean_prompt)
                response_text = response.content or ""
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
