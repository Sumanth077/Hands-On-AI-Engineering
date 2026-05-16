"""
RAG Agent with Database Routing - Streamlit UI.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from rag_agent import PipelineResult, build_pipeline, run_pipeline
from rag_agent.databases import add_documents, doc_count

load_dotenv()

# -- Constants -----------------------------------------------------------------

DB_LABELS = {
    "products": ("🛍️", "Products DB", "#0ea5e9"),
    "support": ("🎧", "Support DB", "#10b981"),
    "financial": ("💰", "Financial DB", "#f59e0b"),
}

EXAMPLE_QUERIES = [
    "What are the specs and price of the TechPro X1 laptop?",
    "How do I reset my password?",
    "What pricing plans are available?",
    "How do I set up two-factor authentication?",
    "What were the Q1 2025 revenue figures?",
    "How do I invite team members to my workspace?",
    "What payment methods do you accept?",
    "What is the return policy for physical products?",
    "Tell me about the DataFlow Analytics Suite features.",
    "How much does the Enterprise plan cost?",
]

# -- Page setup ----------------------------------------------------------------

st.set_page_config(
    page_title="RAG Agent with Database Routing",
    page_icon="🔀",
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
[data-testid="stSidebar"] span { color: #e2e8f0 !important; }

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

[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.2) !important;
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
    border-color: #0ea5e9 !important;
    color: #f1f5f9 !important;
}

hr { border-color: #334155 !important; }
.stSpinner > div { border-top-color: #0ea5e9 !important; }
.stExpander { border: 1px solid #334155 !important; border-radius: 8px !important; }

.route-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 8px;
}
.fallback-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(168,85,247,0.15);
    border: 1px solid #a855f7;
    color: #c084fc;
    margin-bottom: 8px;
}
.status-ok {
    background: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #4ade80;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
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
}
.hero-title { font-size: 28px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
.hero-sub { color: #94a3b8; font-size: 15px; margin-bottom: 20px; }
.empty-state { text-align: center; padding: 48px 20px; color: #64748b; }
.empty-state h3 { color: #94a3b8; font-size: 20px; margin-bottom: 8px; }
.empty-state p  { font-size: 14px; }
.db-pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 500; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# -- Session state -------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata" not in st.session_state:
    st.session_state.metadata: dict[int, PipelineResult] = {}
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "fallback_count" not in st.session_state:
    st.session_state.fallback_count = 0
if "doc_counts" not in st.session_state:
    st.session_state.doc_counts: dict[str, int] = {"products": 0, "support": 0, "financial": 0}


def _split_text(raw: str) -> list[str]:
    """Split raw text into chunks separated by blank lines."""
    return [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]


def _extract_pdf(uploaded_file) -> list[str]:
    """Extract text from a PDF and split into page-level chunks."""
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.extend(_split_text(text))
    return chunks


# -- Sidebar -------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔀 Database Routing RAG")
    st.markdown(
        '<p style="color:#94a3b8;font-size:13px;margin-top:-8px;">'
        "Queries routed across 3 specialized databases</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Configuration**")
    orq_api_key = st.text_input(
        "Orq.ai API Key",
        type="password",
        value=os.getenv("ORQ_API_KEY", ""),
        placeholder="orq-...",
        help="Get one at orq.ai",
    )

    api_key = orq_api_key  # used by downstream checks
    if orq_api_key:
        st.markdown('<div class="status-ok">● Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-missing">○ API key required</div>', unsafe_allow_html=True)

    st.divider()

    # -- Upload documents ------------------------------------------------------

    st.markdown("**Upload Documents**")
    st.markdown(
        '<p style="color:#94a3b8;font-size:12px;margin-top:-6px;">'
        "Upload a .txt file or paste text into each database. "
        "Separate chunks with a blank line.</p>",
        unsafe_allow_html=True,
    )

    if api_key:
        for db_key, (icon, label, color) in DB_LABELS.items():
            count = st.session_state.doc_counts[db_key]
            with st.expander(f"{icon} {label}  ({count} docs)", expanded=False):
                uploaded = st.file_uploader(
                    "Upload file",
                    type=["txt", "pdf"],
                    key=f"upload_{db_key}",
                    label_visibility="collapsed",
                )
                pasted = st.text_area(
                    "Or paste text",
                    key=f"paste_{db_key}",
                    height=100,
                    placeholder="Paste content here. Separate chunks with a blank line.",
                    label_visibility="collapsed",
                )
                if st.button(f"Add to {label}", key=f"add_{db_key}", use_container_width=True):
                    if st.session_state.pipeline is None:
                        with st.spinner("Initializing databases..."):
                            st.session_state.pipeline = build_pipeline(orq_api_key)

                    client, embeddings, _ = st.session_state.pipeline

                    chunks: list[str] = []
                    if uploaded is not None:
                        if uploaded.name.lower().endswith(".pdf"):
                            chunks.extend(_extract_pdf(uploaded))
                        else:
                            content = uploaded.read().decode("utf-8", errors="ignore")
                            chunks.extend(_split_text(content))
                    if pasted.strip():
                        chunks.extend(_split_text(pasted))

                    if chunks:
                        with st.spinner(f"Embedding {len(chunks)} chunk(s)..."):
                            added = add_documents(client, embeddings, db_key, chunks)
                        st.session_state.doc_counts[db_key] += added
                        st.success(f"Added {added} chunk(s).")
                        st.rerun()
                    else:
                        st.warning("No content found. Upload a file or paste some text.")
    else:
        st.markdown(
            '<p style="color:#64748b;font-size:12px;">Enter your API key above to upload documents.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Fallbacks", st.session_state.fallback_count)

    st.divider()

    st.markdown("**Example queries**")
    for q in EXAMPLE_QUERIES:
        if st.button(q[:55] + ("..." if len(q) > 55 else ""), use_container_width=True, key=f"ex_{q}"):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.metadata = {}
        st.session_state.total_queries = 0
        st.session_state.fallback_count = 0
        st.rerun()

# -- Main area -----------------------------------------------------------------

st.markdown("""
<div>
    <div class="hero-title">🔀 RAG Agent with Database Routing</div>
    <div class="hero-sub">
        Queries are intelligently routed across
        <strong style="color:#0ea5e9;">Products</strong>,
        <strong style="color:#10b981;">Support</strong>, and
        <strong style="color:#f59e0b;">Financial</strong> databases.
        When no relevant documents are found, a
        <strong style="color:#c084fc;">web fallback agent</strong> takes over.
    </div>
</div>
""", unsafe_allow_html=True)

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and i in st.session_state.metadata:
            result = st.session_state.metadata[i]
            icon, label, color = DB_LABELS.get(result.routing.database, ("🔀", "Unknown", "#64748b"))

            if result.used_fallback:
                st.markdown(
                    f'<div class="fallback-badge">🌐 Web Fallback (no docs in {label})</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="route-badge" style="background:rgba(14,165,233,0.1);'
                    f'border:1px solid {color};color:{color};">'
                    f'{icon} Routed to {label}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(message["content"])

        if message["role"] == "assistant" and i in st.session_state.metadata:
            result = st.session_state.metadata[i]
            if result.docs:
                with st.expander(f"Sources ({len(result.docs)} documents)", expanded=False):
                    for j, doc in enumerate(result.docs):
                        st.markdown(
                            f"**[{j+1}]** *(score: {doc.score:.2f})*\n\n{doc.text}",
                        )
            with st.expander("Routing reasoning", expanded=False):
                st.markdown(f"*{result.routing.reasoning}*")

total_docs = sum(st.session_state.doc_counts.values())

if not st.session_state.messages:
    if total_docs == 0:
        st.markdown("""
        <div class="empty-state">
            <h3>Upload your documents first</h3>
            <p>Open a database section in the sidebar, upload a .txt file or paste text,<br>
            then ask questions and watch the router decide which database to use.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <h3>Ask anything</h3>
            <p>Your question will be routed to the right database automatically.<br>
            Try the example queries in the sidebar or type your own below.</p>
        </div>
        """, unsafe_allow_html=True)

# -- Input handling ------------------------------------------------------------

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = st.chat_input(
        "Ask about your documents..." if total_docs > 0 else "Upload documents in the sidebar to get started...",
        disabled=(not api_key),
    )

# -- Run pipeline --------------------------------------------------------------

if prompt:
    if not orq_api_key:
        st.error("Enter your Orq.ai API key in the sidebar to get started.")
        st.stop()

    if st.session_state.pipeline is None:
        with st.spinner("Setting up databases and agents..."):
            st.session_state.pipeline = build_pipeline(orq_api_key)

    client, embeddings, orq_client = st.session_state.pipeline

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing and retrieving..."):
            try:
                result = run_pipeline(
                    query=prompt,
                    client=client,
                    embeddings=embeddings,
                    orq_client=orq_client,
                )

                icon, label, color = DB_LABELS.get(result.routing.database, ("🔀", "Unknown", "#64748b"))

                if result.used_fallback:
                    st.markdown(
                        '<div class="fallback-badge">🌐 Web Fallback (no docs found)</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.fallback_count += 1
                else:
                    st.markdown(
                        f'<div class="route-badge" style="background:rgba(14,165,233,0.1);'
                        f'border:1px solid {color};color:{color};">'
                        f'{icon} Routed to {label}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(result.answer)

                if result.docs:
                    with st.expander(f"Sources ({len(result.docs)} documents)", expanded=False):
                        for j, doc in enumerate(result.docs):
                            st.markdown(f"**[{j+1}]** *(score: {doc.score:.2f})*\n\n{doc.text}")

                with st.expander("Routing reasoning", expanded=False):
                    st.markdown(f"*{result.routing.reasoning}*")

                msg_index = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
                st.session_state.metadata[msg_index] = result
                st.session_state.total_queries += 1

            except Exception as e:
                error_msg = f"**Error:** {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
