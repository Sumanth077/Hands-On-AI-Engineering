import uuid
import json
import os

import networkx as nx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from src.cache_manager import CacheManager
from src.document_processor import DocumentProcessor
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.llm_client import LLMClient
from src.search_engine import SearchEngine
from src.vector_store import VectorStore

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Knowledge System",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1f77b4;}
    .sub-header {font-size: 1.1rem; color: #555; margin-bottom: 1.5rem;}
    .entity-card {
        background: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 4px;
    }
    .rel-card {
        background: #fff8f0;
        border-left: 4px solid #ff7f0e;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 4px;
    }
    .answer-box {
        background: #f8fff8;
        border: 1px solid #2ca02c;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached component initialisation ──────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def get_components():
    graph_store = GraphStore()
    vector_store = VectorStore()
    entity_extractor = EntityExtractor()
    search_engine = SearchEngine(graph_store, vector_store)
    cache_manager = CacheManager()
    doc_processor = DocumentProcessor()
    llm_client = LLMClient()
    return (
        graph_store,
        vector_store,
        entity_extractor,
        search_engine,
        cache_manager,
        doc_processor,
        llm_client,
    )


# ── Community detection + summarisation ──────────────────────────────────────

COMMUNITY_SUMMARY_PROMPT = """You are summarizing a group of related entities from a knowledge graph.

Entity group ({count} entities):
{entities}

Write a 2–4 sentence summary describing:
1. What this group of entities represents as a theme or domain
2. Key characteristics and relationships within this group
3. Why these entities are connected

Summary:"""


def run_community_detection(
    graph_store: GraphStore, llm_client: LLMClient, status_fn
) -> int:
    full_graph = graph_store.get_full_graph()
    entities = full_graph["entities"]
    relationships = full_graph["relationships"]

    if len(entities) < 2:
        return 0

    G = nx.Graph()
    G.add_nodes_from(e["name"] for e in entities)
    for r in relationships:
        G.add_edge(r["source"], r["target"], weight=r.get("weight", 0.5))

    # Remove isolated nodes for cleaner communities
    G_connected = G.copy()
    isolated = [n for n in G_connected.nodes if G_connected.degree(n) == 0]
    G_connected.remove_nodes_from(isolated)

    if G_connected.number_of_nodes() < 2:
        return 0

    communities_raw = list(
        nx.algorithms.community.greedy_modularity_communities(G_connected)
    )

    community_count = 0
    for i, comm in enumerate(communities_raw):
        comm_list = list(comm)
        if len(comm_list) < 2:
            continue

        # Assign community id to entities
        cid = f"community_{i}"
        for name in comm_list:
            graph_store.set_entity_community(name, cid)

        # Build summary prompt using entity descriptions
        descs = graph_store.get_entity_descriptions(comm_list[:15])
        entity_lines = "\n".join(
            f"- {d['name']} ({d.get('type','OTHER')}): {d.get('description','')}"
            for d in descs
        )

        status_fn(f"Summarising community {i + 1}/{len(communities_raw)}…")

        try:
            summary = llm_client.complete(
                COMMUNITY_SUMMARY_PROMPT.format(
                    count=len(comm_list), entities=entity_lines
                )
            )
        except Exception:
            summary = f"Community of {len(comm_list)} entities: " + ", ".join(
                comm_list[:5]
            )

        graph_store.create_community(cid, comm_list, summary, level=0)
        community_count += 1

    return community_count


# ── Document ingestion pipeline ───────────────────────────────────────────────


def ingest_document(
    file_content: bytes,
    filename: str,
    graph_store: GraphStore,
    vector_store: VectorStore,
    entity_extractor: EntityExtractor,
    doc_processor: DocumentProcessor,
    cache_manager: CacheManager,
    llm_client: LLMClient,
) -> dict:
    file_hash = cache_manager.file_hash(file_content)

    if cache_manager.is_processed(file_hash):
        return {
            "status": "cached",
            "message": f"'{filename}' was already processed — skipping.",
        }

    # ── Step 1: Load and chunk ────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Loading and chunking document…")

    try:
        chunks = doc_processor.process(file_content, filename)
    except Exception as e:
        progress_bar.empty()
        return {"status": "error", "message": f"Document loading failed: {e}"}

    if not chunks:
        progress_bar.empty()
        return {"status": "error", "message": "No text could be extracted from the document."}

    doc_id = str(uuid.uuid4())
    total_chunks = len(chunks)

    # ── Step 2: Create document node ─────────────────────────────────────────
    graph_store.create_document(doc_id, filename, file_hash)
    progress_bar.progress(0.05, text=f"Indexed document node. Processing {total_chunks} chunks…")

    # ── Step 3: Extract entities per chunk ───────────────────────────────────
    all_entity_names_per_chunk: dict[str, list] = {}

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx}"
        chunk_text = chunk.page_content

        # Store chunk in Neo4j
        graph_store.create_chunk(chunk_id, chunk_text, doc_id)

        # Extract entities and relationships
        extraction = entity_extractor.extract(chunk_text, chunk_id)
        chunk_entity_names = []

        for entity in extraction["entities"]:
            graph_store.upsert_entity(
                entity["name"],
                entity["type"],
                entity["description"],
                chunk_id,
            )
            chunk_entity_names.append(entity["name"])

        for rel in extraction["relationships"]:
            graph_store.upsert_relationship(
                rel["source"], rel["target"], rel["description"], rel["weight"]
            )

        all_entity_names_per_chunk[chunk_id] = chunk_entity_names

        # Store chunk in ChromaDB with entity metadata
        vector_store.add_chunk(
            chunk_id,
            chunk_text,
            {
                "document_id": doc_id,
                "filename": filename,
                "chunk_index": str(idx),
                "entity_names": "|".join(chunk_entity_names),
            },
        )

        pct = 0.05 + 0.75 * ((idx + 1) / total_chunks)
        progress_bar.progress(
            pct,
            text=f"Extracting entities: chunk {idx + 1}/{total_chunks} "
            f"({len(chunk_entity_names)} entities found)…",
        )

    # ── Step 4: Community detection ───────────────────────────────────────────
    progress_bar.progress(0.82, text="Running community detection on knowledge graph…")

    n_communities = run_community_detection(
        graph_store,
        llm_client,
        lambda msg: progress_bar.progress(0.85, text=msg),
    )

    # ── Step 5: Finalise ──────────────────────────────────────────────────────
    progress_bar.progress(1.0, text="Done!")
    cache_manager.mark_processed(file_hash, filename, doc_id, total_chunks)
    progress_bar.empty()

    return {
        "status": "success",
        "message": (
            f"Processed **{filename}**: {total_chunks} chunks, "
            f"{n_communities} communities detected."
        ),
        "doc_id": doc_id,
        "chunks": total_chunks,
        "communities": n_communities,
    }


# ── UI helpers ────────────────────────────────────────────────────────────────


TYPE_COLORS = {
    "PERSON": "#4c72b0",
    "ORGANIZATION": "#dd8452",
    "LOCATION": "#55a868",
    "TECHNOLOGY": "#c44e52",
    "CONCEPT": "#8172b2",
    "EVENT": "#937860",
}


def _type_badge(entity_type: str) -> str:
    color = TYPE_COLORS.get(entity_type, "#888")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:10px;font-size:0.72rem;font-weight:600;">'
        f"{entity_type}</span>"
    )


def render_entity_list(entities: list):
    if not entities:
        st.caption("No entities retrieved.")
        return
    for e in entities[:20]:
        name = e.get("name", "")
        etype = e.get("type", "OTHER")
        desc = (e.get("description") or "")[:200]
        col_name, col_tag = st.columns([4, 1])
        with col_name:
            st.markdown(f"**{name}**")
            if desc:
                st.caption(desc)
        with col_tag:
            st.markdown(_type_badge(etype), unsafe_allow_html=True)
        st.divider()


def render_relationship_list(relationships: list):
    if not relationships:
        st.caption("No relationships retrieved.")
        return
    for r in relationships[:15]:
        source = r.get("source", "")
        target = r.get("target", "")
        desc = (r.get("description") or "")[:200]
        weight = r.get("weight", 0.5)
        st.markdown(f"**{source}** → **{target}** &nbsp; `weight {weight:.2f}`", unsafe_allow_html=True)
        if desc:
            st.caption(desc)
        st.divider()


# ── Main layout ───────────────────────────────────────────────────────────────


def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-header">🕸️ GraphRAG Knowledge System</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Upload documents → Build a knowledge graph → Ask complex questions</div>',
        unsafe_allow_html=True,
    )

    # ── Component init ────────────────────────────────────────────────────────
    try:
        (
            graph_store,
            vector_store,
            entity_extractor,
            search_engine,
            cache_manager,
            doc_processor,
            llm_client,
        ) = get_components()
    except Exception as e:
        st.error(f"Failed to initialise components: {e}")
        st.info(
            "Check your `.env` file — make sure NEO4J_URI, NEO4J_PASSWORD, "
            "MISTRAL_API_KEY are set and Ollama is running on http://localhost:11434"
        )
        get_components.clear()
        return

    # Auto-clear stale cache when the GraphStore class has been updated since
    # the resource was last cached (e.g. after a hot-reload added new methods).
    if not hasattr(graph_store, "ping"):
        get_components.clear()
        st.rerun()

    # ── Graph store health check ──────────────────────────────────────────────
    db_ok, db_msg = graph_store.ping()
    if not db_ok:
        st.error(f"Graph store unavailable: {db_msg}")
        get_components.clear()
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📄 Document Ingestion")

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Documents are chunked, entity-extracted, and stored in Neo4j + ChromaDB.",
        )

        if uploaded_files and st.button("⚙️ Process Documents", type="primary", use_container_width=True):
            for uf in uploaded_files:
                st.subheader(f"Processing: {uf.name}")
                content = uf.read()
                result = ingest_document(
                    content,
                    uf.name,
                    graph_store,
                    vector_store,
                    entity_extractor,
                    doc_processor,
                    cache_manager,
                    llm_client,
                )
                if result["status"] == "success":
                    st.success(result["message"])
                elif result["status"] == "cached":
                    st.info(result["message"])
                else:
                    st.error(result["message"])

        st.divider()
        st.subheader("📋 Processed Documents")
        processed = cache_manager.get_all()
        if processed:
            for doc in processed:
                with st.expander(doc["filename"]):
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    st.write(f"**Processed:** {doc['processed_at'][:19]}")
        else:
            st.caption("No documents processed yet.")

        st.divider()
        st.subheader("📊 Graph Stats")
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.metric("Chunks", vector_store.count())
            except Exception:
                st.metric("Chunks", "—")
        with col2:
            try:
                communities = graph_store.get_all_communities()
                st.metric("Communities", len(communities))
            except Exception:
                st.metric("Communities", "—")

        if st.button("🔄 Refresh Stats", use_container_width=True):
            st.rerun()

        if st.button("♻️ Reset App Cache", use_container_width=True, help="Force-reinitialise all components (use after code changes or connection errors)"):
            get_components.clear()
            st.rerun()

    # ── Main panel ────────────────────────────────────────────────────────────
    col_query, col_mode = st.columns([4, 1])

    with col_mode:
        search_mode = st.radio(
            "Search mode",
            ["🔍 Local", "🌐 Global"],
            help=(
                "**Local**: pinpoint entities in matching chunks. "
                "**Global**: synthesise across whole-graph communities."
            ),
        )

    with col_query:
        question = st.text_area(
            "Ask a question about your documents",
            placeholder="e.g. 'What are the key relationships between X and Y?' or 'What themes emerge across the documents?'",
            height=100,
        )

    ask_col, _ = st.columns([1, 4])
    with ask_col:
        ask = st.button("🚀 Ask", type="primary", use_container_width=True)

    if ask and question.strip():
        with st.spinner("Retrieving graph context and generating answer…"):
            try:
                if "Local" in search_mode:
                    result = search_engine.local_search(question.strip())
                else:
                    result = search_engine.global_search(question.strip())
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

        # ── Answer ────────────────────────────────────────────────────────────
        mode_label = "Local Search" if "Local" in search_mode else "Global Search"
        st.markdown(f"### Answer — {mode_label}")
        st.markdown(result["answer"])

        # ── Supporting context ────────────────────────────────────────────────
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        source_chunks = result.get("source_chunks", [])

        if entities or relationships:
            st.markdown("---")
            st.markdown("#### Supporting Knowledge Graph Context")
            col_ent, col_rel = st.columns(2)
            with col_ent:
                st.markdown(f"**Entities ({len(entities)})**")
                render_entity_list(entities)
            with col_rel:
                st.markdown(f"**Relationships ({len(relationships)})**")
                render_relationship_list(relationships)

        if source_chunks:
            with st.expander(f"📄 Source Excerpts ({len(source_chunks)})"):
                for i, chunk in enumerate(source_chunks):
                    st.markdown(f"**Excerpt {i + 1}**")
                    st.write(chunk)
                    if i < len(source_chunks) - 1:
                        st.divider()
        elif "Global" in search_mode and result.get("communities_used", 0):
            st.caption(
                f"Global search synthesised across {result['communities_used']} communities."
            )

    elif ask and not question.strip():
        st.warning("Please enter a question before clicking Ask.")

    # ── Welcome state ─────────────────────────────────────────────────────────
    if not ask:
        st.markdown("---")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.info(
                "**1. Upload Documents**\n\n"
                "Use the sidebar to upload PDF or TXT files. "
                "Each document is chunked and entities are extracted."
            )
        with info_col2:
            st.info(
                "**2. Knowledge Graph Built**\n\n"
                "Entities and relationships are stored in Neo4j. "
                "Text embeddings go into ChromaDB via Ollama."
            )
        with info_col3:
            st.info(
                "**3. Ask Questions**\n\n"
                "Use **Local** search for specific entity lookups, "
                "or **Global** search for broad thematic questions."
            )


if __name__ == "__main__":
    main()
