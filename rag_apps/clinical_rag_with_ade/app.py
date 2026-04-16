import streamlit as st
import os
from processor import ClinicalRAGProcessor

st.set_page_config(page_title="ADE Clinical RAG", layout="wide", page_icon="💊")

# Custom CSS for a medical theme
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 10px; }
    .source-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #007bff; }
    .sidebar-text { font-size: 0.9rem; color: #555; }
    </style>
    """, unsafe_allow_html=True) 

# Initialize Processor
if "processor" not in st.session_state:
    st.session_state.processor = ClinicalRAGProcessor()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # 1. ADE Description Section
    with st.expander("ℹ️ About LandingAI ADE", expanded=True):
        st.markdown("""
        **Agentic Document Extraction (ADE)** is a visual-first parsing technology. 
        Unlike traditional OCR, it treats documents as visual artifacts.
        
        **Why it matters for Clinical RAG:**
        - **Preserves Tables:** Complex medical dosage or lab tables remain structured.
        - **Hierarchy:** Understands headers and nested sections.
        - **Grounding:** Provides exact (x, y) coordinates for every piece of data.
        """)
        st.markdown("[Learn more in the ADE Docs](https://docs.landing.ai/ade/ade-overview)")

    st.divider()

    # 2. File Upload Section
    st.header("📂 Document Ingestion")
    # We use a key for the file uploader so we can reset it manually
    uploaded_file = st.file_uploader(
        "Upload Clinical PDF", 
        type="pdf", 
        key=st.session_state.get("file_uploader_key", "file_uploader")
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        process_btn = st.button("🚀 Process", use_container_width=True)
    with col2:
        # 3. Clear/Reset Button
        clear_btn = st.button("🗑️ Clear All", use_container_width=True, type="secondary")

    if process_btn and uploaded_file:
        with st.spinner("ADE is extracting visual structure..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                count = st.session_state.processor.ingest_document(temp_path)
                st.success(f"Indexed {count} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Logic for Clear Button
    if clear_btn:
        st.session_state.processor.clear_database()
        st.session_state.messages = []
        # Update the key to force the file_uploader to reset
        st.session_state["file_uploader_key"] = st.session_state.get("file_uploader_key", "file_uploader") + "1"
        st.rerun()

# --- MAIN UI ---
st.title("🏥 Clinical RAG with ADE")
st.markdown("""
    This system uses **LandingAI ADE** to parse dense medical PDFs and **Mistral Large** to provide 
    grounded reasoning over clinical data. Every answer can be traced back to its visual source.
""")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about dosage, adverse effects, or clinical findings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence..."):
            answer, sources = st.session_state.processor.ask(prompt)
            st.markdown(answer)
            
            # Visual Grounding Panel
            with st.expander("🔍 Source Evidence & Visual Grounding (BBox)"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Evidence {i+1}**")
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.info(f"📄 Page: {doc.metadata.get('page')}")
                        st.json({
                            "bounding_box": {
                                "left": round(doc.metadata.get('bbox_left', 0), 3),
                                "top": round(doc.metadata.get('bbox_top', 0), 3),
                                "right": round(doc.metadata.get('bbox_right', 0), 3),
                                "bottom": round(doc.metadata.get('bbox_bottom', 0), 3)
                            }
                        })
                    with c2:
                        st.markdown(f"```markdown\n{doc.page_content}\n```")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})