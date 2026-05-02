import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from config import DEFAULT_TOP_K, DEFAULT_MAX_ENTITIES
from src.document_processor import (
    load_document, chunk_text, compute_hash,
    is_cached, add_to_cache, list_cached_docs,
)
from src.graph_indexer import (
    build_doc_graph, merge_into_combined,
    save_doc_graph, load_combined_graph,
)
from src.vector_indexer import (
    check_ollama, get_chroma_collection,
    index_chunks, collection_has_doc,
)
from src.graph_retriever import retrieve_from_graph
from src.vector_retriever import retrieve_from_vector
from src.context_fusion import fuse_context, format_graph_context, format_vector_context
from src.mistral_client import generate_answer

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
.graph-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #4a9eff;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
}
.vector-box {
    background: linear-gradient(135deg, #1a2e1a 0%, #162e16 100%);
    border: 1px solid #4aff6e;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
}
.answer-box {
    background: linear-gradient(135deg, #2e1a1a 0%, #2e1616 100%);
    border: 1px solid #ff6e4a;
    border-radius: 10px;
    padding: 20px;
    margin: 12px 0;
}
.metric-card {
    background: #1e1e2e;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}
.status-ok { color: #4aff6e; font-weight: bold; }
.status-err { color: #ff4a4a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Cached resources ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_mistral_client():
    from mistralai.client import Mistral
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return None
    return Mistral(api_key=api_key)


@st.cache_resource(show_spinner=False)
def _get_collection():
    try:
        return get_chroma_collection()
    except Exception as e:
        return None


# ─── Session state init ──────────────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = load_combined_graph()

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_graph_results" not in st.session_state:
    st.session_state.last_graph_results = None

if "last_vector_results" not in st.session_state:
    st.session_state.last_vector_results = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔮 Hybrid RAG")
    st.caption("Knowledge Graph + Vector Retrieval")

    # Status checks
    st.subheader("System Status")
    col1, col2 = st.columns(2)

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    mistral_ok = bool(api_key)
    with col1:
        if mistral_ok:
            st.markdown('<span class="status-ok">✓ Mistral</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✗ Mistral</span>', unsafe_allow_html=True)

    ollama_ok, ollama_msg = check_ollama()
    with col2:
        if ollama_ok:
            st.markdown('<span class="status-ok">✓ Ollama</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✗ Ollama</span>', unsafe_allow_html=True)

    if not mistral_ok:
        st.warning("Set MISTRAL_API_KEY in your .env file.")
    if not ollama_ok:
        st.warning(ollama_msg)

    st.divider()

    # Retrieval settings
    st.subheader("Retrieval Settings")
    top_k = st.slider(
        "Top-K vector chunks",
        min_value=1, max_value=20,
        value=DEFAULT_TOP_K,
        help="Number of document chunks retrieved via semantic search",
    )
    max_entities = st.slider(
        "Max graph entities",
        min_value=5, max_value=50,
        value=DEFAULT_MAX_ENTITIES,
        help="Maximum number of entities retrieved from the knowledge graph",
    )

    st.divider()

    # Document upload
    st.subheader("Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        help="Documents are dual-indexed: entities → knowledge graph, chunks → ChromaDB",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        try:
            text = load_document(file_bytes, uploaded_file.name)
            doc_hash = compute_hash(text)
            cached = is_cached(doc_hash)

            if cached:
                st.success(f"Already indexed: {uploaded_file.name}")
                st.caption(f"Chunks: {cached['num_chunks']} | Entities: {cached['num_entities']}")
            else:
                st.info(f"Ready to index: {uploaded_file.name}")
                st.caption(f"Size: {len(text):,} characters")

                if st.button("Process Document", type="primary", use_container_width=True):
                    if not mistral_ok:
                        st.error("Mistral API key required for entity extraction.")
                    elif not ollama_ok:
                        st.error("Ollama required for embeddings.")
                    else:
                        chunks = chunk_text(text)
                        collection = _get_collection()
                        mistral_client = _get_mistral_client()

                        st.markdown("**Ingestion Progress**")
                        phase_label = st.empty()
                        progress_bar = st.progress(0)
                        stat_label = st.empty()

                        # --- Phase 1: Graph indexing ---
                        phase_label.markdown("Phase 1/2: Building knowledge graph...")
                        graph_progress = [0]

                        def graph_cb(done, total):
                            graph_progress[0] = done
                            pct = int(done / total * 50)
                            progress_bar.progress(pct)
                            stat_label.caption(f"Extracting entities: chunk {done}/{total}")

                        G_doc = build_doc_graph(chunks, mistral_client, progress_cb=graph_cb)
                        save_doc_graph(G_doc, doc_hash)
                        G_combined = merge_into_combined(G_doc)
                        st.session_state.graph = G_combined
                        num_entities = G_doc.number_of_nodes()

                        # --- Phase 2: Vector indexing ---
                        phase_label.markdown("Phase 2/2: Embedding chunks into ChromaDB...")
                        vec_progress = [0]

                        def vec_cb(done, total):
                            vec_progress[0] = done
                            pct = 50 + int(done / total * 50)
                            progress_bar.progress(pct)
                            stat_label.caption(f"Embedding chunks: {done}/{total}")

                        index_chunks(chunks, collection, doc_hash, progress_cb=vec_cb)

                        progress_bar.progress(100)
                        phase_label.empty()
                        stat_label.empty()

                        add_to_cache(doc_hash, uploaded_file.name, len(chunks), num_entities)
                        st.success(f"Indexed {len(chunks)} chunks, {num_entities} entities")
                        st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.divider()

    # Indexed documents list
    st.subheader("Indexed Documents")
    cached_docs = list_cached_docs()
    G = st.session_state.graph
    st.caption(f"Graph: {G.number_of_nodes()} entities, {G.number_of_edges()} relationships")

    if cached_docs:
        for doc in cached_docs:
            with st.expander(f"📄 {doc['filename']}", expanded=False):
                st.caption(f"Chunks: {doc['num_chunks']}")
                st.caption(f"Entities: {doc['num_entities']}")
                st.caption(f"Hash: {doc['hash'][:12]}...")
    else:
        st.caption("No documents indexed yet.")

# ─── Main area ────────────────────────────────────────────────────────────────

st.title("🔮 Hybrid RAG System")
st.markdown(
    "Combines **Knowledge Graph** retrieval and **Vector** retrieval "
    "into a unified pipeline for answering complex questions."
)

# Architecture diagram (text-based)
with st.expander("How it works", expanded=False):
    st.markdown("""
    ```
    Document Upload
         │
         ├──► GraphRAG (Mistral) ──► NetworkX Graph ──► JSON
         │         Entity + Relationship Extraction
         │
         └──► Text Chunker ──► nomic-embed-text ──► ChromaDB
                   Semantic Embeddings

    User Question
         │
         ├──► Graph Retrieval (parallel) ──► Entities & Relationships
         │
         └──► Vector Retrieval (parallel) ──► Top-K Chunks
                          │
                          ▼
                   Context Fusion
                          │
                          ▼
                  Mistral Small 4
                          │
                          ▼
              Answer + Sources Display
    ```
    """)

st.divider()

# Q&A section
docs_ready = bool(list_cached_docs())
G = st.session_state.graph

if not docs_ready:
    st.info("Upload and process at least one document in the sidebar to begin.")
else:
    st.subheader("Ask a Question")

    with st.form("qa_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder="What are the main concepts discussed? Who are the key people involved? How do X and Y relate?",
            height=80,
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted and question.strip():
        if not mistral_ok:
            st.error("Mistral API key required. Set MISTRAL_API_KEY in .env")
        elif not ollama_ok:
            st.error(f"Ollama required for embeddings: {ollama_msg}")
        else:
            collection = _get_collection()
            mistral_client = _get_mistral_client()
            G = st.session_state.graph

            with st.spinner("Running parallel retrieval..."):
                retrieval_times = {}

                def run_graph():
                    t0 = time.time()
                    result = retrieve_from_graph(G, question, max_entities)
                    retrieval_times["graph"] = round(time.time() - t0, 2)
                    return result

                def run_vector():
                    t0 = time.time()
                    result = retrieve_from_vector(collection, question, top_k)
                    retrieval_times["vector"] = round(time.time() - t0, 2)
                    return result

                with ThreadPoolExecutor(max_workers=2) as executor:
                    graph_future = executor.submit(run_graph)
                    vector_future = executor.submit(run_vector)
                    graph_results = graph_future.result()
                    vector_results = vector_future.result()

            with st.spinner("Generating answer with Mistral Small 4..."):
                t0 = time.time()
                fused = fuse_context(graph_results, vector_results)
                answer = generate_answer(mistral_client, question, fused)
                gen_time = round(time.time() - t0, 2)

            st.session_state.last_answer = answer
            st.session_state.last_graph_results = graph_results
            st.session_state.last_vector_results = vector_results
            st.session_state.last_question = question
            st.session_state.retrieval_times = retrieval_times
            st.session_state.gen_time = gen_time

    # Display results
    if st.session_state.last_answer:
        st.subheader("Answer")

        # Timing metrics
        rt = getattr(st.session_state, "retrieval_times", {})
        gt = getattr(st.session_state, "gen_time", 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Graph retrieval", f"{rt.get('graph', 0):.2f}s")
        m2.metric("Vector retrieval", f"{rt.get('vector', 0):.2f}s")
        m3.metric("Generation", f"{gt:.2f}s")
        m4.metric(
            "Sources",
            f"{len(st.session_state.last_graph_results.get('entities', []))}E + "
            f"{len(st.session_state.last_vector_results)}C"
        )

        st.markdown(
            f'<div class="answer-box">{st.session_state.last_answer}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Sources")
        col_graph, col_vector = st.columns(2)

        with col_graph:
            gr = st.session_state.last_graph_results
            n_ent = len(gr.get("entities", []))
            n_rel = len(gr.get("relationships", []))
            st.markdown(
                f'<div style="color:#4a9eff;font-weight:bold;font-size:1.05rem;">'
                f'Graph Sources ({n_ent} entities, {n_rel} relationships)</div>',
                unsafe_allow_html=True,
            )

            if gr.get("entities"):
                with st.expander("Entities", expanded=True):
                    for e in gr["entities"]:
                        badge_color = {
                            "PERSON": "#ff9999", "ORGANIZATION": "#99ccff",
                            "LOCATION": "#99ff99", "TECHNOLOGY": "#ffcc99",
                            "CONCEPT": "#cc99ff", "EVENT": "#ffff99",
                        }.get(e["type"], "#cccccc")
                        st.markdown(
                            f'<span style="background:{badge_color};color:#000;'
                            f'border-radius:4px;padding:1px 6px;font-size:0.75rem;">'
                            f'{e["type"]}</span> **{e["name"]}**',
                            unsafe_allow_html=True,
                        )
                        if e["description"]:
                            st.caption(e["description"])

            if gr.get("relationships"):
                with st.expander("Relationships", expanded=True):
                    for r in gr["relationships"]:
                        rels = " | ".join(r["relations"])
                        st.markdown(f"`{r['source']}` → **{rels}** → `{r['target']}`")
                        if r["description"]:
                            st.caption(r["description"])

            if not gr.get("entities") and not gr.get("relationships"):
                st.caption("No graph context found for this query.")

        with col_vector:
            vr = st.session_state.last_vector_results
            st.markdown(
                f'<div style="color:#4aff6e;font-weight:bold;font-size:1.05rem;">'
                f'Vector Sources ({len(vr)} chunks)</div>',
                unsafe_allow_html=True,
            )

            if vr:
                for i, chunk in enumerate(vr, 1):
                    sim = chunk.get("similarity", 0.0)
                    sim_pct = int(sim * 100)
                    with st.expander(f"Chunk {i} — similarity {sim_pct}%", expanded=(i == 1)):
                        st.markdown(chunk["text"])
                        meta = chunk.get("metadata", {})
                        if meta.get("doc_hash"):
                            st.caption(f"Doc: {meta['doc_hash'][:12]}... | Chunk #{meta.get('chunk_index', '?')}")
            else:
                st.caption("No vector context found for this query.")

    elif submitted and question.strip():
        pass  # errors already shown above

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Hybrid RAG System — GraphRAG (NetworkX) + Vector (ChromaDB) | "
    "Embeddings: nomic-embed-text via Ollama | LLM: Mistral Small 4"
)
