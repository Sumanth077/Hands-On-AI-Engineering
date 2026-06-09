"""
Brand Monitor - Streamlit UI
Workflow: Configure keys in sidebar -> Enter brand name -> Run -> View reports
"""

import os

import streamlit as st
from dotenv import load_dotenv

from brand_monitor_agent.main import run_brand_monitor

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Brand Monitor",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  footer { display: none !important; }

  .header-bar {
    background: #111827;
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 20px;
  }
  .header-bar h1 { color: #ffffff; margin: 0 0 6px; font-size: 22px; font-weight: 700; }
  .header-bar p  { color: #9ca3af; margin: 0; font-size: 13px; line-height: 1.5; }

  /* Platform section header */
  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 0 10px;
    border-bottom: 2px solid #e5e7eb;
    margin-bottom: 16px;
  }
  .section-icon {
    font-size: 18px;
    line-height: 1;
  }
  .section-title {
    font-size: 15px;
    font-weight: 700;
    color: #111827;
  }

  /* Sentiment pill */
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .3px;
    margin-left: 6px;
    vertical-align: middle;
  }
  .pill-positive  { background: #d1fae5; color: #065f46; }
  .pill-neutral   { background: #fef9c3; color: #713f12; }
  .pill-negative  { background: #fee2e2; color: #991b1b; }
  .pill-error     { background: #fee2e2; color: #991b1b; }

  /* Scoped markdown body */
  .report-body { font-size: 14px; line-height: 1.8; color: #1f2937; }
  .report-body h3 {
    font-size: 13px; font-weight: 700; color: #111827;
    margin: 18px 0 4px; text-transform: uppercase;
    letter-spacing: .4px;
  }
  .report-body p  { margin: 4px 0 10px; }
  .report-body ul { padding-left: 18px; margin: 4px 0 10px; }
  .report-body li { margin-bottom: 4px; }
  .report-body strong { color: #111827; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="header-bar">
  <h1>🔎 Brand Monitor</h1>
  <p>Get a structured intelligence report on any brand across Web, YouTube,
     Twitter/X, and LinkedIn in one run.<br>
     Powered by <a href="https://www.scrapingdog.com" target="_blank"
     style="color:#60a5fa;">Scrapingdog</a> for data collection
     and DeepSeek V4 Flash for analysis.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: API keys
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## API Keys")

    st.markdown("#### Scrapingdog")
    scrapingdog_key = st.text_input(
        "Scrapingdog API Key",
        value=os.getenv("SCRAPINGDOG_API_KEY", ""),
        type="password",
        placeholder="Paste your key",
    )
    st.markdown("[Get your key](https://scrapingdog.com)")

    st.divider()

    st.markdown("#### DeepSeek V4 Flash")
    orq_key = st.text_input(
        "API Key",
        value=os.getenv("ORQ_API_KEY", ""),
        type="password",
        placeholder="Paste your key",
    )

    st.divider()
    st.caption("Keys are used only for this session and are never stored.")

# ---------------------------------------------------------------------------
# Brand input row
# ---------------------------------------------------------------------------

col_input, col_run = st.columns([5, 1])
with col_input:
    brand = st.text_input(
        "Brand or Company Name",
        placeholder="e.g. Notion, OpenAI, Stripe...",
        label_visibility="collapsed",
    )
with col_run:
    run_btn = st.button("Run", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLATFORM_META = {
    "web":      ("Web",       "🌐"),
    "youtube":  ("YouTube",   "▶"),
    "twitter":  ("Twitter/X", "𝕏"),
    "linkedin": ("LinkedIn",  "💼"),
}


def _detect_sentiment_pill(content: str) -> str:
    """Return an HTML pill based on keywords in the report."""
    lower = content.lower()
    if lower.startswith("error"):
        return '<span class="pill pill-error">Error</span>'
    pos = lower.count("positive") + lower.count("strong") + lower.count("well-regarded")
    neg = lower.count("negative") + lower.count("criticism") + lower.count("complaint") + lower.count("concern")
    if pos > neg + 1:
        return '<span class="pill pill-positive">Positive</span>'
    if neg > pos + 1:
        return '<span class="pill pill-negative">Negative</span>'
    return '<span class="pill pill-neutral">Mixed</span>'


def render_platform_report(platform: str, content: str):
    label, icon = PLATFORM_META.get(platform, (platform.title(), ""))
    pill = _detect_sentiment_pill(content)

    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-icon">{icon}</span>'
        f'<span class="section-title">{label}</span>'
        f'{pill}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if content.lower().startswith("error"):
        st.error(content)
    else:
        st.markdown(content)


def render_results(results: dict):
    tab_web, tab_yt, tab_tw, tab_li = st.tabs(
        ["🌐  Web", "▶  YouTube", "𝕏  Twitter/X", "💼  LinkedIn"]
    )
    with tab_web:
        render_platform_report("web", results.get("web", "No data."))
    with tab_yt:
        render_platform_report("youtube", results.get("youtube", "No data."))
    with tab_tw:
        render_platform_report("twitter", results.get("twitter", "No data."))
    with tab_li:
        render_platform_report("linkedin", results.get("linkedin", "No data."))


def clear_results():
    st.session_state.pop("last_brand", None)
    st.session_state.pop("last_results", None)


# ---------------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------------

if run_btn:
    if not brand.strip():
        st.warning("Please enter a brand or company name.")
        st.stop()
    if not scrapingdog_key.strip() or not orq_key.strip():
        st.error("Please fill in both API keys in the sidebar.")
        st.stop()

    with st.spinner(f"Scanning {brand} across 4 platforms — takes about a minute..."):
        try:
            results = run_brand_monitor(
                brand=brand.strip(),
                scrapingdog_key=scrapingdog_key.strip(),
                orq_key=orq_key.strip(),
            )
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.stop()

    st.session_state["last_brand"] = brand.strip()
    st.session_state["last_results"] = results

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if "last_results" in st.session_state:
    results = st.session_state["last_results"]
    brand_label = st.session_state.get("last_brand", "")

    # Results header row: title + clear button
    res_col, clear_col = st.columns([5, 1])
    with res_col:
        st.markdown(f"### Intelligence Report — {brand_label}")
    with clear_col:
        if st.button("Clear", use_container_width=True, on_click=clear_results):
            st.rerun()

    st.divider()
    render_results(results)
