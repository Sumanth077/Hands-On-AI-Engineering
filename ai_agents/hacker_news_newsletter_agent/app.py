import os
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HN Newsletter Agent",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stForm"] { border: none; padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📰 Hacker News Newsletter Agent")
st.markdown(
    "Fetch today's top HN stories, summarise them with AI, and deliver a "
    "polished newsletter straight to your inbox — powered by **Gemma 4 via Google AI Studio**."
)
st.divider()

# ── Env-var health check ──────────────────────────────────────────────────────
missing = [v for v in ("GOOGLE_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD") if not os.getenv(v)]
if missing:
    st.warning(
        f"⚠️ Missing environment variables: `{'`, `'.join(missing)}`. "
        "Copy `.env.example` to `.env` and fill in the values before running."
    )

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "newsletter_html": None,
    "recipient_email": "",
    "send_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


def extract_html(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    m = re.search(r"<!DOCTYPE html>.*?</html>", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip()
    m = re.search(r"<html.*?</html>", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return cleaned.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Generate
# ══════════════════════════════════════════════════════════════════════════════
with st.form("newsletter_form"):
    email_input = st.text_input(
        "📧 Your Email Address",
        placeholder="you@example.com",
        help="The newsletter will be delivered here.",
    )

    submitted = st.form_submit_button(
        "🚀 Generate Newsletter",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not email_input or "@" not in email_input:
        st.error("❌ Please enter a valid email address.")
        st.stop()
    if missing:
        st.error(f"❌ Configure missing env vars first: `{'`, `'.join(missing)}`")
        st.stop()

    from tools import fetch_top_stories, extract_content, generate_newsletter

    # Clear any previous generation
    st.session_state.newsletter_html = None
    st.session_state.send_result = None
    st.session_state.topic_note = None

    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("🔍 Fetching top Hacker News stories…")
        progress.progress(10)
        stories = fetch_top_stories(count=10)

        status.info(f"📄 Scraping content from {len(stories)} articles…")
        for i, story in enumerate(stories):
            story["content"] = extract_content(story["url"])
            progress.progress(10 + int(60 * (i + 1) / len(stories)))

        status.info("✍️ Generating newsletter with Gemma 4…")
        progress.progress(75)
        raw_output = generate_newsletter(stories)

        progress.progress(100)
        status.empty()
        progress.empty()

        newsletter_html = extract_html(raw_output)
        st.session_state.newsletter_html = newsletter_html
        st.session_state.recipient_email = email_input
        st.rerun()

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Pipeline error: {exc}")
        with st.expander("Full traceback"):
            st.exception(exc)
        st.info("Check your API keys and Gmail credentials in `.env`.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Edit & Send (shown only after a newsletter has been generated)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.newsletter_html is not None:
    st.divider()
    st.subheader("📄 Newsletter HTML")
    st.caption(
        f"Recipient: **{st.session_state.recipient_email}**"
        + "  |  Edit the HTML below before sending."
    )

    import streamlit.components.v1 as components
    components.html(st.session_state.newsletter_html, height=620, scrolling=True)

    already_sent = bool(
        st.session_state.send_result
        and "successfully" in st.session_state.send_result.lower()
    )

    col_send, col_back = st.columns([2, 1])

    with col_send:
        if st.button(
            "📬 Send Newsletter",
            type="primary",
            use_container_width=True,
            disabled=already_sent,
        ):
            from tools import send_email

            with st.spinner("Sending via Gmail SMTP…"):
                result_msg = send_email(
                    newsletter=st.session_state.newsletter_html,
                    recipient=st.session_state.recipient_email,
                )
            st.session_state.send_result = result_msg
            st.rerun()

    with col_back:
        if st.button("↩ Start Over", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if st.session_state.send_result:
        if already_sent:
            st.success(f"🎉 {st.session_state.send_result}")
        else:
            st.error(f"Delivery failed: {st.session_state.send_result}")
