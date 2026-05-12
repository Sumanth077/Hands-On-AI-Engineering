"""
Multimodal RAG — Streamlit UI
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag import SOURCE_ICONS, MultimodalRAG

load_dotenv()

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
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

[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.2) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }

.stButton button {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background-color: #334155 !important;
    border-color: #8b5cf6 !important;
    color: #f1f5f9 !important;
}
hr { border-color: #334155 !important; }
.stSpinner > div { border-top-color: #8b5cf6 !important; }

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
.source-row {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 6px;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 12px;
    color: #c4b5fd;
    word-break: break-all;
}
.chunk-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 3px solid #8b5cf6;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.5;
}
.chunk-source {
    color: #64748b;
    font-size: 11px;
    margin-top: 6px;
}
.hero-title { font-size: 30px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
.hero-sub   { color: #94a3b8; font-size: 15px; margin-bottom: 20px; }
.empty-state { text-align: center; padding: 48px 20px; color: #64748b; }
.empty-state h3 { color: #94a3b8; font-size: 20px; margin-bottom: 8px; }
.empty-state p  { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None

EXAMPLE_QUESTIONS = [
    "🔍 What are the main topics covered in the sources?",
    "📊 Summarize everything in the knowledge base",
    "🖼️ What does the image show?",
    "🎵 What is discussed in the audio?",
    "🎬 Summarize the video content",
    "📄 What does the PDF say about...",
]

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 Multimodal RAG")
    st.markdown(
        '<p style="color:#94a3b8;font-size:13px;margin-top:-8px;">'
        "Index text, URLs, PDFs, images, audio, and video — ask anything</p>",
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
        if st.session_state.rag is None:
            st.session_state.rag = MultimodalRAG(api_key=api_key)
    else:
        st.markdown('<div class="status-missing">○ API key required</div>', unsafe_allow_html=True)

    st.divider()

    # ── Source ingestion ───────────────────────────────────────────────────────
    st.markdown("**Add to Knowledge Base**")

    source_type = st.selectbox(
        "Source type",
        options=["📝 Text", "🌐 URL", "📄 PDF", "🖼️ Image", "🎵 Audio", "🎬 Video"],
        label_visibility="collapsed",
    )
    stype = source_type.split(" ", 1)[1].lower()  # "text", "url", "pdf", etc.

    ingested = False

    if stype == "text":
        text_input = st.text_area(
            "Paste text",
            placeholder="Paste any text content here...",
            height=120,
            label_visibility="collapsed",
        )
        text_label = st.text_input("Label (optional)", placeholder="e.g. Meeting notes", key="text_label")
        if st.button("➕ Add text", use_container_width=True, disabled=not api_key):
            if text_input.strip():
                with st.spinner("Indexing text..."):
                    rag: MultimodalRAG = st.session_state.rag
                    n = rag.add_text(text_input.strip(), label=text_label or "Pasted text")
                st.success(f"Added {n} chunks")
                ingested = True

    elif stype == "url":
        url_input = st.text_input("URL", placeholder="https://example.com/article", label_visibility="collapsed")
        if st.button("➕ Add URL", use_container_width=True, disabled=not api_key):
            if url_input.strip():
                with st.spinner(f"Fetching {url_input}..."):
                    try:
                        rag: MultimodalRAG = st.session_state.rag
                        n = rag.add_url(url_input.strip())
                        st.success(f"Added {n} chunks")
                        ingested = True
                    except Exception as e:
                        st.error(f"Failed to fetch URL: {e}")

    elif stype == "pdf":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if pdf_file:
            size_mb = pdf_file.size / (1024 * 1024)
            if size_mb > 10:
                st.warning(f"Large PDF ({size_mb:.1f} MB). Only the first 100 pages will be indexed.")
        if st.button("➕ Add PDF", use_container_width=True, disabled=not (api_key and pdf_file)):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name
            with st.spinner("Loading PDF..."):
                try:
                    rag: MultimodalRAG = st.session_state.rag
                    n = rag.add_pdf(tmp_path)
                    st.success(f"Added {n} chunks from {pdf_file.name}")
                    ingested = True
                except Exception as e:
                    st.error(f"Failed to load PDF: {e}")

    elif stype == "image":
        img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "gif"], label_visibility="collapsed")
        if st.button("➕ Add image", use_container_width=True, disabled=not (api_key and img_file)):
            suffix = "." + img_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(img_file.read())
                tmp_path = tmp.name
            with st.spinner("Uploading to Gemini File API and describing..."):
                try:
                    rag: MultimodalRAG = st.session_state.rag
                    n = rag.add_image(tmp_path)
                    st.success(f"Described and indexed {img_file.name}")
                    ingested = True
                except Exception as e:
                    st.error(f"Failed to process image: {e}")

    elif stype == "audio":
        audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "ogg", "m4a", "flac"], label_visibility="collapsed")
        if st.button("➕ Add audio", use_container_width=True, disabled=not (api_key and audio_file)):
            suffix = "." + audio_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            with st.spinner("Uploading and transcribing audio... (this may take a moment)"):
                try:
                    rag: MultimodalRAG = st.session_state.rag
                    n = rag.add_audio(tmp_path)
                    st.success(f"Transcribed and indexed {audio_file.name} ({n} chunks)")
                    ingested = True
                except Exception as e:
                    st.error(f"Failed to process audio: {e}")

    elif stype == "video":
        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "webm", "avi"], label_visibility="collapsed")
        if st.button("➕ Add video", use_container_width=True, disabled=not (api_key and video_file)):
            suffix = "." + video_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
            with st.spinner("Uploading and analysing video... (this may take a moment)"):
                try:
                    rag: MultimodalRAG = st.session_state.rag
                    n = rag.add_video(tmp_path)
                    st.success(f"Analysed and indexed {video_file.name} ({n} chunks)")
                    ingested = True
                except Exception as e:
                    st.error(f"Failed to process video: {e}")

    if ingested:
        st.rerun()

    st.divider()

    # Indexed sources list
    rag_inst: MultimodalRAG | None = st.session_state.rag
    if rag_inst and rag_inst.sources:
        st.markdown(f"**Knowledge Base** ({rag_inst.source_count()} sources · {rag_inst.chunk_count()} chunks)")
        for src in rag_inst.sources:
            icon = SOURCE_ICONS.get(src.source_type, "📁")
            label = src.label if len(src.label) <= 35 else src.label[:32] + "..."
            st.markdown(
                f'<div class="source-row">'
                f'{icon} <span style="flex:1;">{label}</span>'
                f'<span style="color:#475569;margin-left:auto;">{src.chunks} chunks</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="color:#64748b;font-size:13px;">No sources indexed yet.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Reset knowledge base", use_container_width=True):
        if api_key:
            st.session_state.rag = MultimodalRAG(api_key=api_key)
        st.session_state.messages = []
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("""
<div>
    <div class="hero-title">🧠 Multimodal RAG</div>
    <div class="hero-sub">
        Index <strong style="color:#c4b5fd;">text</strong>,
        <strong style="color:#c4b5fd;">URLs</strong>,
        <strong style="color:#c4b5fd;">PDFs</strong>,
        <strong style="color:#c4b5fd;">images</strong>,
        <strong style="color:#c4b5fd;">audio</strong>, and
        <strong style="color:#c4b5fd;">video</strong> into a shared knowledge base.
        Ask questions and get grounded answers with citations.
        Powered by <strong style="color:#c4b5fd;">Gemini 3 Flash</strong> and
        <strong style="color:#c4b5fd;">Gemini Embedding 2</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander(f"📎 {len(msg['chunks'])} retrieved sources"):
                for chunk in msg["chunks"]:
                    icon = SOURCE_ICONS.get(chunk["source_type"], "📁")
                    st.markdown(
                        f'<div class="chunk-card">{chunk["content"]}'
                        f'<div class="chunk-source">{icon} {chunk["source_label"]}</div></div>',
                        unsafe_allow_html=True,
                    )

# Empty state with example questions
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>Ask anything across your sources</h3>
        <p>Add sources in the sidebar, then ask questions below.<br>
        Gemini 3 Flash sees the actual files — not just descriptions.</p>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True, key=f"eq_{i}"):
            st.session_state.pending_q = q

# ── Input ──────────────────────────────────────────────────────────────────────

if st.session_state.pending_q:
    prompt = st.session_state.pending_q
    st.session_state.pending_q = None
else:
    prompt = st.chat_input("Ask anything about your sources...")

# ── Run RAG ────────────────────────────────────────────────────────────────────

if prompt:
    if not api_key:
        st.error("Enter your Google API key in the sidebar.")
        st.stop()

    clean = prompt.split(" ", 1)[-1] if prompt and prompt[0] in "🔍📊🖼️🎵🎬📄" else prompt

    st.session_state.messages.append({"role": "user", "content": clean})
    with st.chat_message("user"):
        st.markdown(clean)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                rag: MultimodalRAG = st.session_state.rag
                result = rag.query(clean)
                st.markdown(result.answer)

                chunk_data = [
                    {
                        "content": d.page_content,
                        "source_type": d.metadata.get("source_type", "unknown"),
                        "source_label": d.metadata.get("source_label", "unknown"),
                    }
                    for d in result.retrieved_docs
                ]

                if chunk_data:
                    with st.expander(f"📎 {len(chunk_data)} retrieved sources"):
                        for chunk in chunk_data:
                            icon = SOURCE_ICONS.get(chunk["source_type"], "📁")
                            st.markdown(
                                f'<div class="chunk-card">{chunk["content"]}'
                                f'<div class="chunk-source">{icon} {chunk["source_label"]}</div></div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "chunks": chunk_data,
                })
            except Exception as e:
                msg = f"**Error:** {e}"
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
