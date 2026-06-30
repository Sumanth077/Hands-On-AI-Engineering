"""Streamlit dashboard for email triage and draft reply generation."""

import streamlit as st
from dotenv import load_dotenv

from flow import run_email_flow

load_dotenv()

st.set_page_config(
    page_title="Email Auto Responder",
    page_icon="📧",
    layout="wide",
)

CATEGORY_STYLES = {
    "inquiry": {"label": "Inquiry", "color": "#1E88E5", "background": "#E3F2FD"},
    "complaint": {"label": "Complaint", "color": "#E53935", "background": "#FFEBEE"},
    "follow_up": {"label": "Follow Up", "color": "#FB8C00", "background": "#FFF3E0"},
    "marketing": {"label": "Marketing", "color": "#8E24AA", "background": "#F3E5F5"},
    "notification": {"label": "Notification", "color": "#00897B", "background": "#E0F2F1"},
    "newsletter": {"label": "Newsletter", "color": "#43A047", "background": "#E8F5E9"},
    "spam": {"label": "Spam", "color": "#757575", "background": "#F5F5F5"},
}

NO_RESPONSE_MESSAGES = {
    "marketing": "No response generated. This is a promotional email.",
    "notification": "No response generated. This is an automated notification.",
    "newsletter": "No response generated. This is a newsletter or content digest.",
    "spam": "No response generated. This message was classified as spam.",
}


def render_category_badge(category: str) -> None:
    """Render a color-coded HTML badge for the email category."""
    style = CATEGORY_STYLES.get(category, CATEGORY_STYLES["inquiry"])
    st.markdown(
        f"""
        <span style="
            background-color: {style['background']};
            color: {style['color']};
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid {style['color']};
        ">
            {style['label']}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_needs_response_indicator(needs_response: bool) -> None:
    """Render a badge indicating whether the email needs a reply."""
    if needs_response:
        label = "Response needed"
        color = "#1565C0"
        background = "#E3F2FD"
    else:
        label = "No response needed"
        color = "#616161"
        background = "#EEEEEE"

    st.markdown(
        f"""
        <span style="
            background-color: {background};
            color: {color};
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
            border: 1px solid {color};
        ">
            {label}
        </span>
        """,
        unsafe_allow_html=True,
    )


def get_no_response_message(category: str) -> str:
    """Return the UI message shown when no draft reply is generated."""
    return NO_RESPONSE_MESSAGES.get(
        category,
        "No response generated. This message does not require a reply.",
    )


def render_email_card(email_result: dict) -> None:
    """Render a bordered card with email details and draft response."""
    with st.container(border=True):
        header_col, badge_col = st.columns([4, 1])
        with header_col:
            st.markdown(f"**From:** {email_result['sender']}")
            st.markdown(f"**Subject:** {email_result['subject']}")
        with badge_col:
            render_category_badge(email_result.get("category", "inquiry"))
            render_needs_response_indicator(
                email_result.get("needs_response", True)
            )

        st.markdown("##### Original Email")
        st.text_area(
            "Original message",
            value=email_result.get("body", ""),
            height=180,
            disabled=True,
            label_visibility="collapsed",
        )

        st.markdown("##### Drafted Response")
        draft_response = email_result.get("draft_response")
        needs_response = email_result.get("needs_response", True)
        category = email_result.get("category", "inquiry")

        if not needs_response or not draft_response:
            st.info(get_no_response_message(category))
        else:
            st.text_area(
                "Draft response",
                value=draft_response,
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )


def main() -> None:
    """Run the Streamlit email auto responder dashboard."""
    with st.sidebar:
        st.title("Email Auto Responder")
        st.markdown(
            """
            An automated email triage and response agent that reads unread Gmail
            messages, classifies their intent across seven categories, flags whether
            each message needs a reply, and drafts professional responses when needed.
            """
        )
        st.markdown("---")
        st.markdown("**Example use cases**")
        st.markdown(
            """
            - Triage customer support inbox by intent
            - Draft first-pass replies for inquiries and follow-ups
            - Flag complaints for priority follow-up
            - Skip replies for marketing, notifications, and newsletters
            - Filter spam before manual review
            """
        )
        st.markdown("---")
        st.caption("Configure NVIDIA_API_KEY, EMAIL_ADDRESS, and APP_PASSWORD in .env")

    st.title("Email Auto Responder")
    st.markdown(
        "Click **Check Emails** to fetch unread Gmail messages and generate draft replies."
    )

    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.error = None
        st.session_state.checked = False

    if st.button("Check Emails", type="primary", use_container_width=False):
        with st.spinner("Fetching and processing emails..."):
            flow_output = run_email_flow()
            st.session_state.results = flow_output.get("results", [])
            st.session_state.error = flow_output.get("error")
            st.session_state.checked = True

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.checked and not st.session_state.error:
        if not st.session_state.results:
            st.info(
                "No unread emails found. Your inbox is clear, or all messages have already been read."
            )
        else:
            st.success(f"Processed {len(st.session_state.results)} unread email(s).")
            for email_result in st.session_state.results:
                render_email_card(email_result)


if __name__ == "__main__":
    main()
