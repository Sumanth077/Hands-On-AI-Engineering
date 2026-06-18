"""
Customer Query Routing and Resolution Agent — Streamlit UI

User-facing chat interface. The orchestrator, routing details, and scores are
hidden from the chat view and available only in a per-message "Details" expander
for developers and reviewers.

Pipeline (internal):
  1. Embed        -- all-MiniLM-L6-v2, 384-dim
  2. Retrieve     -- VectorAI DB, 4 collections in parallel
  3. Orchestrate  -- LLM agent decides resolve or escalate
  4. Route        -- department from best semantic match
  5a. Resolve     -- Ministral 3B, RAG-grounded answer
  5b. Escalate    -- Ministral 3B, empathetic handoff + template
  6. Memory       -- resolved queries optionally saved to VectorAI DB
"""

from __future__ import annotations

import streamlit as st

from customer_query_routing_agent.config import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    ESCALATION_SENTIMENT_THRESHOLD,
    ESCALATION_URGENCY_THRESHOLD,
)
from customer_query_routing_agent.embedder import Embedder
from customer_query_routing_agent.orchestrator import triage, OrchestratorDecision
from customer_query_routing_agent.resolver import Resolver
from customer_query_routing_agent.router import route
from customer_query_routing_agent.vectorstore import (
    get_client,
    get_collection_counts,
    seed_all,
    setup_collections,
    store_resolved_query,
    search_memory,
)

# ---------------------------------------------------------------------------
# Example messages shown when the chat is empty
# ---------------------------------------------------------------------------

EXAMPLE_MESSAGES = [
    "I haven't received my order and it's been over two weeks. Where is it?",
    "I was charged twice for the same order. I need this fixed immediately.",
    "I want to return an item I bought last week. What's the process?",
    "I am absolutely furious. This is the third time I've contacted you and nothing has been done. I'm considering legal action.",
    "My account shows a charge I don't recognise. This could be fraud.",
    "How long does a refund usually take?",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Support",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar: static content (no client dependency)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## How to use")
    st.markdown(
        "**1. Enter your message or complaint** in the box below.\n\n"
        "**2. Submit it.** The agent will automatically decide whether to "
        "resolve it or pass it to a human agent.\n\n"
        "**3. Review the response.** If your issue was resolved, the answer "
        "is saved so similar queries are handled faster in future.\n\n"
        "If your message needs a human, you'll receive a reference number "
        "and a time window for when someone will get back to you."
    )
    st.divider()

# ---------------------------------------------------------------------------
# Init (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Getting things ready...")
def init():
    client = get_client()
    client.connect()
    embedder = Embedder()
    setup_collections(client)
    seed_all(client, embedder)
    resolver = Resolver()
    return client, embedder, resolver


client, embedder, resolver = init()
counts = get_collection_counts(client)

# ---------------------------------------------------------------------------
# Sidebar: dynamic content (needs client)
# ---------------------------------------------------------------------------

with st.sidebar:
    with st.expander("Pipeline"):
        st.markdown(
            "<div style='display:flex;flex-direction:column;gap:6px;padding:4px 0'>"
            "<div style='display:flex;gap:6px;align-items:center;flex-wrap:wrap'>"
            "<span style='background:#1e3a5f;color:#e8f4fd;padding:3px 9px;border-radius:10px;font-size:12px'>① Embed</span>"
            "<span style='color:#888;font-size:12px'>→</span>"
            "<span style='background:#1e3a5f;color:#e8f4fd;padding:3px 9px;border-radius:10px;font-size:12px'>② Retrieve</span>"
            "<span style='color:#888;font-size:12px'>→</span>"
            "<span style='background:#1e3a5f;color:#e8f4fd;padding:3px 9px;border-radius:10px;font-size:12px'>③ Orchestrate</span>"
            "<span style='color:#888;font-size:12px'>→</span>"
            "<span style='background:#1e3a5f;color:#e8f4fd;padding:3px 9px;border-radius:10px;font-size:12px'>④ Route</span>"
            "</div>"
            "<div style='display:flex;gap:6px;align-items:center;padding-left:8px'>"
            "<span style='color:#888;font-size:12px'>↳</span>"
            "<span style='background:#1a4731;color:#d4edda;padding:3px 9px;border-radius:10px;font-size:12px'>⑤a Resolve</span>"
            "<span style='color:#888;font-size:12px'>or</span>"
            "<span style='background:#5c1a1a;color:#f8d7da;padding:3px 9px;border-radius:10px;font-size:12px'>⑤b Escalate</span>"
            "<span style='color:#888;font-size:12px'>→</span>"
            "<span style='background:#1e3a5f;color:#e8f4fd;padding:3px 9px;border-radius:10px;font-size:12px'>⑥ Memory</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Knowledge base"):
        st.metric("FAQ entries", counts.get("product_faq", 0))
        st.metric("Policy docs", counts.get("product_docs", 0))
        st.metric("Resolved tickets", counts.get("resolved_tickets", 0))
        st.metric("Agent memory", counts.get("resolved_queries", 0))

    with st.expander("Technical details"):
        st.markdown("**Models**")
        st.caption(
            "Embeddings: all-MiniLM-L6-v2 (384-dim)  \n"
            "LLM: Ministral 3B Q4_K_M  \n"
            "Fully offline after first run"
        )
        st.markdown("**Escalation thresholds**")
        st.caption(
            f"Sentiment: {ESCALATION_SENTIMENT_THRESHOLD}  \n"
            f"Urgency: {ESCALATION_URGENCY_THRESHOLD}  \n"
            f"High confidence: {HIGH_CONFIDENCE_THRESHOLD}  \n"
            f"Low confidence: {LOW_CONFIDENCE_THRESHOLD}"
        )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Each message is a dict:
    # {"role": "user"|"assistant", "content": str,
    #  "meta": None | {
    #    "path": str, "decision": OrchestratorDecision,
    #    "routing": RoutingResult, "saved": bool, "query": str
    #  }}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Support Assistant")
st.caption("Ask a question or describe your issue and we'll take it from there.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_memory = st.tabs(["Chat", "Past resolutions"])

# ===========================================================================
# TAB 1: CHAT
# ===========================================================================

with tab_chat:

    # -- Clear button (only shown when there are messages) -------------------
    if st.session_state.messages:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        st.markdown("")

    # -- Example messages (only shown when chat is empty) --------------------
    if not st.session_state.messages:
        st.markdown("**Try one of these or type your own below:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_MESSAGES):
            with cols[i % 2]:
                if st.button(example, key=f"ex_{i}", use_container_width=True):
                    st.session_state["pending_query"] = example
                    st.rerun()

    # -- Render existing messages --------------------------------------------
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                meta = msg.get("meta") or {}

                # Save to memory (resolve path, not yet saved)
                if meta.get("path") == "resolve":
                    if not meta.get("saved"):
                        if st.button("Save to memory", key=f"save_{i}", type="secondary"):
                            store_resolved_query(
                                client,
                                query=meta["query"],
                                query_vector=meta["routing"].query_vector,
                                resolution=msg["content"],
                                department=meta["routing"].department,
                            )
                            st.session_state.messages[i]["meta"]["saved"] = True
                            st.rerun()
                    else:
                        st.caption("Saved to memory")

                # Details expander (hidden by default — for devs/reviewers)
                decision: OrchestratorDecision = meta.get("decision")
                routing = meta.get("routing")
                if decision and routing:
                    with st.expander("Details"):
                        st.caption(f"**Path:** {meta['path']}")
                        if decision.reasoning:
                            st.caption(f"**Orchestrator reasoning:** {decision.reasoning}")
                        st.caption(
                            f"**Sentiment:** {decision.sentiment_score:.2f}  |  "
                            f"**Urgency:** {decision.urgency_score:.2f}  |  "
                            f"**Legal flag:** {decision.legal_flag}  |  "
                            f"**Repeat contact:** {decision.repeat_contact}"
                        )
                        if decision.escalation_reasons:
                            st.caption("**Escalation reasons:** " + "; ".join(decision.escalation_reasons))
                        st.caption(
                            f"**Department:** {routing.department}  |  "
                            f"**Confidence:** {routing.confidence:.0%} ({routing.confidence_label})"
                        )
                        sc = routing.source_counts
                        st.caption(
                            f"**Sources:** {sc.get('faq',0)} FAQ  |  "
                            f"{sc.get('docs',0)} docs  |  "
                            f"{sc.get('tickets',0)} tickets  |  "
                            f"{sc.get('memory',0)} memory"
                        )

    # -- Handle pending query from example button click ----------------------
    pending = st.session_state.pop("pending_query", None)

    # -- Chat input ----------------------------------------------------------
    user_input = st.chat_input("Enter your message or complaint...")
    if user_input:
        pending = user_input

    # -- Run pipeline --------------------------------------------------------
    if pending:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": pending, "meta": None})

        with st.chat_message("user"):
            st.markdown(pending)

        with st.chat_message("assistant"):
            with st.spinner("Looking into this..."):
                routing = route(pending, client, embedder)
                decision = triage(pending, routing.confidence, resolver.llm)

            with st.spinner("Preparing a response..."):
                if decision.path == "escalate":
                    response = resolver.escalate(pending, routing, decision)
                else:
                    response = resolver.resolve(pending, routing)

            st.markdown(response)

            meta = {
                "path": decision.path,
                "decision": decision,
                "routing": routing,
                "saved": False,
                "query": pending,
            }

            if decision.path == "resolve":
                if st.button("Save to memory", key="save_new", type="secondary"):
                    store_resolved_query(
                        client,
                        query=pending,
                        query_vector=routing.query_vector,
                        resolution=response,
                        department=routing.department,
                    )
                    meta["saved"] = True

            with st.expander("Details"):
                st.caption(f"**Path:** {decision.path}")
                if decision.reasoning:
                    st.caption(f"**Orchestrator reasoning:** {decision.reasoning}")
                st.caption(
                    f"**Sentiment:** {decision.sentiment_score:.2f}  |  "
                    f"**Urgency:** {decision.urgency_score:.2f}  |  "
                    f"**Legal flag:** {decision.legal_flag}  |  "
                    f"**Repeat contact:** {decision.repeat_contact}"
                )
                if decision.escalation_reasons:
                    st.caption("**Escalation reasons:** " + "; ".join(decision.escalation_reasons))
                st.caption(
                    f"**Department:** {routing.department}  |  "
                    f"**Confidence:** {routing.confidence:.0%} ({routing.confidence_label})"
                )
                sc = routing.source_counts
                st.caption(
                    f"**Sources:** {sc.get('faq',0)} FAQ  |  "
                    f"{sc.get('docs',0)} docs  |  "
                    f"{sc.get('tickets',0)} tickets  |  "
                    f"{sc.get('memory',0)} memory"
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "meta": meta}
        )


# ===========================================================================
# TAB 2: PAST RESOLUTIONS
# ===========================================================================

with tab_memory:
    st.markdown("### Past resolutions")
    st.caption(
        "Every message you save from the chat is stored here. "
        "The agent draws on these when handling similar queries."
    )

    memory_count = counts.get("resolved_queries", 0)

    if memory_count == 0:
        st.info(
            "Nothing saved yet. After the agent resolves a query in the chat, "
            "click Save to memory to store it here.",
            icon="💡",
        )
    else:
        st.metric("Saved resolutions", memory_count)
        st.markdown("")

        mem_query = st.text_input(
            "Search",
            placeholder="e.g. charged twice, missing order, refund delay",
        )

        if mem_query.strip():
            vec = embedder.embed(mem_query.strip())
            hits = search_memory(client, vec, top_k=5)
            if hits:
                for hit in hits:
                    dept = hit.payload.get("department", "")
                    q_text = hit.payload.get("query", "")
                    with st.expander(f"{q_text[:80]}...  —  {dept}"):
                        st.markdown(f"**Query:** {q_text}")
                        st.divider()
                        st.markdown(hit.payload.get("resolution", ""))
                        st.caption(f"Similarity: {hit.score:.3f}  |  Department: {dept}")
            else:
                st.info("No matching entries found.")
