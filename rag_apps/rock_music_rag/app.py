"""
Rock Music RAG — Streamlit UI
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag import DEFAULT_BANDS, RockRAG

load_dotenv()

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rock Music RAG",
    page_icon="🎸",
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
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li { color: #e2e8f0 !important; }

[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #e11d48 !important;
    box-shadow: 0 0 0 3px rgba(225,29,72,0.2) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }

[data-testid="stChatMessage"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
}
[data-testid="stChatMessage"] p  { color: #e2e8f0 !important; }
[data-testid="stChatMessage"] li { color: #e2e8f0 !important; }
[data-testid="stChatMessage"] code {
    background-color: #0f172a !important;
    color: #a5f3fc !important;
}

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
    border-color: #e11d48 !important;
    color: #f1f5f9 !important;
}

hr { border-color: #334155 !important; }
.stSpinner > div { border-top-color: #e11d48 !important; }

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
.band-pill {
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(225,29,72,0.1);
    border: 1px solid rgba(225,29,72,0.3);
    color: #fda4af;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin-bottom: 5px;
    width: 100%;
}
.chunk-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 3px solid #e11d48;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #cbd5e1;
}
.chunk-source {
    color: #64748b;
    font-size: 11px;
    margin-top: 6px;
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
.gemma-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(225,29,72,0.08);
    border: 1px solid rgba(225,29,72,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    width: 100%;
    font-size: 12px;
    color: #fda4af;
}
.empty-state {
    text-align: center;
    padding: 48px 20px;
    color: #64748b;
}
.empty-state h3 { color: #94a3b8; font-size: 20px; margin-bottom: 8px; }
.empty-state p  { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Example questions ──────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "🎤 Who was the lead singer of Audioslave?",
    "🎵 What was Nirvana's breakthrough album in 1991?",
    "🎸 Who guests on Dire Straits' Money for Nothing?",
    "🥁 Who formed Green Day and when?",
    "📀 What is Evanescence's debut album called?",
    "🏷️ What was Sum 41's original name?",
    "🎧 Were The Smiths influential?",
    "🎼 What story does American Idiot tell?",
]

# ── Session state ──────────────────────────────────────────────────────────────

if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "loading_band" not in st.session_state:
    st.session_state.loading_band = None

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_rag(api_key: str) -> RockRAG:
    """Return existing RAG instance or create a new one."""
    if st.session_state.rag is None:
        st.session_state.rag = RockRAG(api_key=api_key)
    return st.session_state.rag


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎸 Rock Music RAG")
    st.markdown(
        '<p style="color:#94a3b8;font-size:13px;margin-top:-8px;">'
        "Build your own rock knowledge base and ask anything</p>",
        unsafe_allow_html=True,
    )

    st.divider()

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

    # Band management
    st.markdown("**Add a Band**")
    new_band = st.text_input(
        "Band name",
        placeholder="e.g. Radiohead",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    add_clicked = col1.button("➕ Add", use_container_width=True, disabled=not api_key)
    defaults_clicked = col2.button("🎸 Load defaults", use_container_width=True, disabled=not api_key)

    st.divider()

    # Show indexed bands
    rag_instance = st.session_state.rag
    if rag_instance and rag_instance.band_titles():
        bands = rag_instance.band_titles()
        st.markdown(f"**Knowledge Base** ({len(bands)} bands · {rag_instance.index_size()} chunks)")
        for band in bands:
            col_b, col_r = st.columns([4, 1])
            col_b.markdown(
                f'<div class="band-pill">🎵 {band}</div>',
                unsafe_allow_html=True,
            )
            if col_r.button("✕", key=f"rm_{band}", help=f"Remove {band}"):
                rag_instance.remove_band(band)
                st.rerun()
    else:
        st.markdown(
            '<p style="color:#64748b;font-size:13px;">No bands indexed yet.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Gemma 4 capabilities
    st.markdown("**Powered by Gemma 4**")
    for icon, label in [
        ("🌍", "140 languages supported"),
        ("🧠", "26B parameters"),
        ("⚡", "Fast via Google AI API"),
        ("🔍", "Strong Q&A (85.2% MMLU)"),
    ]:
        st.markdown(
            f'<div class="gemma-badge">{icon} {label}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Handle sidebar actions ─────────────────────────────────────────────────────

if add_clicked and new_band.strip() and api_key:
    st.session_state.loading_band = new_band.strip()

if defaults_clicked and api_key:
    st.session_state.loading_band = "__defaults__"

# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("""
<div>
    <div class="hero-title">🎸 Rock Music RAG</div>
    <div class="hero-sub">
        Build a custom rock knowledge base from Wikipedia and ask anything about your favourite bands.
        Powered by <strong style="color:#fda4af;">Gemma 4</strong> and
        <strong style="color:#fda4af;">BM25 retrieval</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# Band loading (runs before chat so progress bar shows in the right place)
if st.session_state.loading_band:
    target = st.session_state.loading_band
    rag = get_rag(api_key)

    if target == "__defaults__":
        to_load = [b for b in DEFAULT_BANDS if b not in rag.bands]
        if to_load:
            progress = st.progress(0, text="Loading default bands from Wikipedia...")
            for i, band in enumerate(to_load):
                progress.progress((i) / len(to_load), text=f"Fetching {band}...")
                try:
                    rag.add_band(band)
                except Exception as e:
                    st.warning(f"Could not load '{band}': {e}")
            progress.progress(1.0, text="Done!")
            progress.empty()
        else:
            st.info("Default bands are already loaded.")
    else:
        with st.spinner(f"Fetching '{target}' from Wikipedia..."):
            try:
                title = rag.add_band(target)
                st.success(f"Added **{title}** ({rag.index_size()} total chunks)")
            except Exception as e:
                st.error(f"Could not find '{target}' on Wikipedia: {e}")

    st.session_state.loading_band = None
    st.rerun()

# Example question buttons
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>Ask anything about rock music</h3>
        <p>Add bands in the sidebar, then ask questions below.<br>
        Try a quick question to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True, key=f"eq_{i}"):
            st.session_state.pending_question = q

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chunks"):
            with st.expander(f"📄 {len(message['chunks'])} retrieved chunks"):
                for chunk in message["chunks"]:
                    st.markdown(
                        f'<div class="chunk-card">{chunk["content"]}'
                        f'<div class="chunk-source">🎵 {chunk["band"]} · '
                        f'<a href="{chunk["url"]}" target="_blank" '
                        f'style="color:#64748b;">{chunk["url"]}</a></div></div>',
                        unsafe_allow_html=True,
                    )

# ── Input handling ─────────────────────────────────────────────────────────────

if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    prompt = st.chat_input("Ask anything about your rock bands...")

# ── Run RAG ────────────────────────────────────────────────────────────────────

if prompt:
    if not api_key:
        st.error("Enter your Google API key in the sidebar to get started.")
        st.stop()

    rag = get_rag(api_key)

    if not rag.band_titles():
        st.warning("Add at least one band in the sidebar before asking questions.")
        st.stop()

    # Strip emoji prefix from example buttons
    clean_prompt = prompt.split(" ", 1)[-1] if prompt and prompt[0] in "🎤🎵🎸🥁📀🏷️🎧🎼" else prompt

    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                result = rag.query(clean_prompt)
                st.markdown(result.answer)

                chunk_data = [
                    {"content": c.content, "band": c.band, "url": c.url}
                    for c in result.retrieved_chunks
                ]
                if chunk_data:
                    with st.expander(f"📄 {len(chunk_data)} retrieved chunks"):
                        for chunk in chunk_data:
                            st.markdown(
                                f'<div class="chunk-card">{chunk["content"]}'
                                f'<div class="chunk-source">🎵 {chunk["band"]} · '
                                f'<a href="{chunk["url"]}" target="_blank" '
                                f'style="color:#64748b;">{chunk["url"]}</a></div></div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "chunks": chunk_data,
                })
            except Exception as e:
                error_msg = f"**Error:** {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
