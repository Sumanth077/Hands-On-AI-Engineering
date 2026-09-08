"""
Citation-Aware Deep Research Agent -- Streamlit app.

Upload a long PDF, get a quick overview, then ask questions and receive answers
that cite the exact pages they came from. Parsing is done by LlamaParse; the
answers are generated locally by Ollama.

The UI is built around the point of the project: every answer sits next to the
source text it came from, with the matching words highlighted, the page stamped
on it, and a relevance bar -- so you can see the answer is grounded, not guessed.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import re
import tempfile
from pathlib import Path

import streamlit as st

from citation_agent.config import SETTINGS, configure_settings
from citation_agent.indexer import build_or_load_index
from citation_agent.overview import generate_overview
from citation_agent.parser import parse_pdf
from citation_agent.query import Citation, ask, make_query_engine

st.set_page_config(page_title="Citation-Aware Research Agent", page_icon="📄", layout="wide")

configure_settings()

# Doc-agnostic questions that demo well on almost any long PDF.
SUGGESTED_QUESTIONS = [
    "Summarize this document in a few sentences.",
    "What are the key findings or main claims?",
    "What are the most important numbers or results?",
    "What limitations or caveats are mentioned?",
]

_STOPWORDS = {
    "what", "which", "does", "this", "that", "with", "from", "have", "were",
    "are", "the", "and", "for", "how", "many", "much", "into", "over", "about",
    "document", "mentioned", "reported", "important", "most",
}


# ---------------------------------------------------------------------------
# Small HTML helpers for the source cards
# ---------------------------------------------------------------------------

def _highlight(text: str, question: str, limit: int = 700) -> str:
    """Escape source text and <mark> the meaningful words from the question."""
    snippet = text.strip()
    if len(snippet) > limit:
        snippet = snippet[:limit].rsplit(" ", 1)[0] + " …"
    escaped = html.escape(snippet)

    terms = {
        w for w in re.findall(r"[A-Za-z0-9]+", question.lower())
        if len(w) > 3 and w not in _STOPWORDS
    }
    if not terms:
        return escaped
    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)),
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def _score_bar(score: float | None) -> str:
    """A thin relevance bar; retrieval scores are roughly 0..1."""
    if score is None:
        return ""
    pct = int(max(0.0, min(1.0, score)) * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0 2px">'
        f'<div style="flex:1;background:#E5E7EB;border-radius:4px;height:6px">'
        f'<div style="width:{pct}%;background:#4F46E5;height:6px;border-radius:4px"></div>'
        f'</div>'
        f'<span style="font-size:11px;color:#64748B">relevance {score:.2f}</span>'
        f'</div>'
    )


def _render_card(c: Citation, question: str) -> None:
    page = f"p. {c.page}" if c.page is not None else "page n/a"
    st.markdown(
        f'<div style="border:1px solid #E5E7EB;border-radius:10px;padding:12px 14px;'
        f'margin-bottom:12px;background:#FFFFFF">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-weight:700;color:#4F46E5">[{c.number}] {page}</span>'
        f'<span style="font-size:12px;color:#64748B">{html.escape(c.file_name)}</span>'
        f'</div>'
        f'{_score_bar(c.score)}'
        f'<div style="font-size:13px;color:#1E293B;line-height:1.55;margin-top:8px;'
        f'max-height:240px;overflow:auto">{_highlight(c.text, question)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_turn(turn: dict, rich: bool) -> None:
    """Render one Q&A turn. The latest turn gets the side-by-side source view."""
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if rich and turn["citations"]:
            answer_col, source_col = st.columns([3, 2], gap="large")
            with answer_col:
                st.markdown(turn["answer"])
            with source_col:
                st.markdown("**Sources**")
                for c in turn["citations"]:
                    _render_card(c, turn["question"])
        else:
            st.markdown(turn["answer"])
            for c in turn["citations"]:
                page = f"p. {c.page}" if c.page is not None else "page n/a"
                score = f" · relevance {c.score:.2f}" if c.score is not None else ""
                with st.expander(f"[{c.number}] {c.file_name} · {page}{score}"):
                    st.markdown(c.text)


def _prepare_document(file_bytes: bytes, file_name: str):
    """Parse + index a PDF (both cached), returning a citation-aware query engine."""
    doc_key = hashlib.md5(file_bytes).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    with st.status("Parsing with LlamaParse and indexing locally...", expanded=False):
        documents = parse_pdf(tmp_path, file_name=file_name)
        index = build_or_load_index(documents, doc_key)
        engine = make_query_engine(index)

    return documents, engine


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Setup")
    st.text(f"Chat model   : {SETTINGS.ollama_model}")
    st.text(f"Embeddings   : {SETTINGS.ollama_embed_model}")
    st.text(f"Ollama       : {SETTINGS.ollama_base_url}")
    if not SETTINGS.llama_cloud_api_key:
        st.error("LLAMA_CLOUD_API_KEY is not set. Add it to .env.")
    st.caption(
        "LlamaParse (cloud) parses the PDF. Everything else -- embeddings, "
        "retrieval, and the answers -- runs locally on Ollama."
    )
    if st.button("New document", width="stretch"):
        st.session_state.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Citation-Aware Deep Research Agent")
st.caption("Upload a long PDF, ask questions, and get answers that cite their source pages.")

# The main architectural point, made visible: one cloud call to parse, everything
# else local.
st.markdown(
    '<div style="display:flex;gap:8px;margin:2px 0 14px">'
    '<span style="background:#DCFCE7;color:#166534;padding:4px 12px;border-radius:999px;'
    'font-size:12px;font-weight:600">● Local · Ollama (embeddings + answers)</span>'
    '<span style="background:#EEF2FF;color:#3730A3;padding:4px 12px;border-radius:999px;'
    'font-size:12px;font-weight:600">☁ Cloud · LlamaParse (parse only)</span>'
    '</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a PDF (paper, report, spec, contract)", type=["pdf"])

if uploaded is not None and st.session_state.get("file_name") != uploaded.name:
    documents, engine = _prepare_document(uploaded.getvalue(), uploaded.name)
    st.session_state.file_name = uploaded.name
    st.session_state.engine = engine
    st.session_state.overview = generate_overview(documents)
    st.session_state.chat = []

if "engine" in st.session_state:
    with st.expander("Document overview", expanded=True):
        st.markdown(st.session_state.overview)

    chat = st.session_state.get("chat", [])

    # Empty state: offer clickable starter questions so a cold demo lands instantly.
    pending: str | None = None
    if not chat:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"suggest_{i}", width="stretch"):
                pending = q

    # Render the conversation; only the last turn gets the rich side-by-side view.
    for idx, turn in enumerate(chat):
        _render_turn(turn, rich=(idx == len(chat) - 1))

    typed = st.chat_input("Ask a question about the document")
    question = typed or pending

    if question:
        with st.spinner("Reading the relevant pages and answering..."):
            answer = ask(st.session_state.engine, question)
        st.session_state.chat.append(
            {"question": question, "answer": answer.text, "citations": answer.citations}
        )
        st.rerun()
else:
    st.info(
        "Upload a PDF to begin. The first parse of a document takes a moment; "
        "after that it is cached."
    )
