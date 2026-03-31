import streamlit as st
from agno.models.ollama import Ollama
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.milvus import Milvus
from agno.tools.firecrawl import FirecrawlTools
import os
from dotenv import load_dotenv
import base64
from sentence_transformers import SentenceTransformer
import concurrent.futures
import time
from datetime import datetime
from queue import Queue

import torch
torch.classes.__path__ = []    



# --- Caching for performance ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")
# @st.cache_resource(show_spinner=False)
# def get_embedding_model():
#     return SentenceTransformer("BAAI/bge-small-en-v1.5")

# Custom Embedder using cached model
class CustomEmbedder:
    def __init__(self):
        self.model = get_embedding_model()
        self.dimensions = 384  # Required by agno Milvus wrapper

    def get_embedding(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def get_embedding_and_usage(self, texts: list[str]):
        embeddings = self.embed(texts)
        usage = {
            "input_tokens": sum(len(t.split()) for t in texts),
            "output_vectors": len(embeddings),
        }
        return embeddings, usage

load_dotenv()

firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

# Milvus Vector DB
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "rag_documents_local"

# Configure Milvus with the custom embedder
vector_db = Milvus(
    collection=COLLECTION_NAME,
    uri=MILVUS_URI,
    embedder=CustomEmbedder()
)

def get_rag_agent(knowledge_base: PDFKnowledgeBase, model_id="qwen3:1.7b", debug_mode=True):
    model = Ollama(id=model_id)
    instructions = [
        "You are concise, quick and to the point. Don't display your thinking process.",
        "Do not show your reasoning, thoughts, or internal process to the user.",
        "Only output the final answer and relevant citations.",
        "Never display step-by-step thinking or tool call details.",
        "1. Knowledge Base Search:",
        "   - ALWAYS start by searching the knowledge base using search_knowledge_base tool",
        "   - Analyze ALL returned documents thoroughly before responding",
        "   - If multiple documents are returned, synthesize the information coherently",
        "2. External Search:",
        "   - If knowledge base search yields insufficient results, use FireCrawl",
        "   - Use the search_web tool to search the web for the most recent information",
        "   - Focus on reputable sources and recent information",
        "   - Cross-reference information from multiple sources when possible",
        "3. Citation Precision:",
        "   - Reference page numbers and section headers",
        "   - Distinguish between main content and appendices",
        "4. Response Quality:",
        "   - Provide specific citations and sources for claims",
        "   - Structure responses with clear sections and bullet points when appropriate",
        "   - Include relevant quotes from source materials",
        "   - Avoid hedging phrases like 'based on my knowledge' or 'depending on the information'",
        "5. Response Structure:",
        "   - Use markdown for formatting technical content",
        "   - Create bullet points for lists found in documents",
        "   - Preserve important formatting from original PDF",
        "6. User Interaction:",
        "   - Ask for clarification if the query is ambiguous",
        "   - Break down complex questions into manageable parts",
        "   - Proactively suggest related topics or follow-up questions",
        "7. Error Handling:",
        "   - If no relevant information is found, clearly state this",
        "   - Suggest alternative approaches or questions",
        "   - Be transparent about limitations in available information",
    ]

    return Agent(
        model=model,
        knowledge=knowledge_base,
        description="You are a helpful Agent called 'Agentic RAG' assisting with questions about a PDF document.",
        instructions=instructions,
        search_knowledge=True,
        markdown=True,
        tools=[FirecrawlTools(api_key=firecrawl_api_key, scrape=False, crawl=True)],
        show_tool_calls=False,  #make True to debug and see what tools are being called
        add_datetime_to_instructions=False,
        debug_mode=False,       #make True if required
        stream=True,
        enable_agentic_memory=True,
        read_chat_history=True,
        read_tool_call_history=True
    )

def process_pdf(pdf_path, progress_queue):
    try:
        for i in range(1, 6):
            time.sleep(0.2)  # Simulate progress
            progress_queue.put(i * 20)
        kb = PDFKnowledgeBase(path=pdf_path, vector_db=vector_db)
        kb.load(recreate=True)
        agent = get_rag_agent(kb)
        return kb, agent
    except Exception as e:
        progress_queue.put(("error", str(e)))
        raise e

# Set Streamlit page config
st.set_page_config(
    page_title="PDF RAG Agent (Local R1)",
    page_icon="📚",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
.stApp { color: #FAFAFA; }
.stSidebar > div:first-child {
    background-image: linear-gradient(to bottom, #262936, #1e202a);
}
[data-testid="stChatMessage"] {
    background-color: rgba(74, 74, 106, 0.4);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
[data-testid="stChatMessage"] p { color: inherit; }
.pdf-preview-container {
    border: 1px solid #4A4A6A;
    border-radius: 0.5rem;
    padding: 0.5rem;
    background-color: #262936;
    margin-bottom: 1rem;
}
.user-bubble {
    background-color: #2d2d4d;
    color: #fff;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    position: relative;
    min-height: 2.5rem;
}
.assistant-bubble {
    background-color: #3a3a5a;
    color: #fff;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    position: relative;
    min-height: 2.5rem;
}
.timestamp {
    font-size: 0.8em;
    color: #aaa;
    position: absolute;
    bottom: 0.25rem;
    right: 0.75rem;
}
.message-content {
    margin-bottom: 1.5rem;
}
.message-content p {
    margin: 0;
    padding: 0;
}
.message-content ul, .message-content ol {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}
.message-content pre {
    margin: 0.5rem 0;
    padding: 0.5rem;
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 0.25rem;
}
.message-content code {
    background-color: rgba(0, 0, 0, 0.2);
    padding: 0.1rem 0.3rem;
    border-radius: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# PDF Viewer
def display_pdf_preview(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300px"></iframe>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Session State Init
for key in ["messages", "document_loaded", "agent", "knowledge_base", "processing", "progress"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key == "progress":
            st.session_state[key] = 0
        else:
            st.session_state[key] = None

# --- Sidebar: Upload PDF ---
with st.sidebar:
    with st.expander("Upload and Process PDF", expanded=True):
        if st.session_state.processing:
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_container = st.empty()
            status_container.info("Processing document... Please wait.")
        elif st.session_state.document_loaded:
            st.markdown("✅ Document indexed!")
        else:
            uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
            if uploaded_file:
                pdf_path = "temp_uploaded.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                display_pdf_preview(pdf_path)
                if st.button("Process Document"):
                    st.session_state.processing = True
                    st.session_state.progress = 0
                    progress_container = st.empty()
                    progress_bar = progress_container.progress(0)
                    status_container = st.empty()
                    status_container.info("Processing document... Please wait.")
                    progress_queue = Queue()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(process_pdf, pdf_path, progress_queue)
                        while not future.done():
                            try:
                                progress = progress_queue.get_nowait()
                                if isinstance(progress, tuple) and progress[0] == "error":
                                    st.error(f"Error during processing: {progress[1]}")
                                    break
                                progress_bar.progress(progress)
                            except:
                                pass
                            time.sleep(0.1)
                        try:
                            kb, agent = future.result()
                            progress_bar.progress(100)
                            st.session_state.knowledge_base = kb
                            st.session_state.agent = agent
                            st.session_state.document_loaded = True
                            st.session_state.messages = []
                            st.session_state.processing = False
                            st.session_state.progress = 100
                            progress_container.empty()
                            status_container.empty()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during processing: {str(e)}")
                            st.session_state.processing = False
                            progress_container.empty()
                            status_container.empty()
                            st.rerun()

# --- Main Chat Area ---
st.markdown("# Chat with your PDF")
if st.session_state.processing:
    st.info("Processing document... Please wait.")
elif not st.session_state.document_loaded:
    st.info("👈 Upload and process a PDF to begin.")
else:
    with st.container():
        for msg in st.session_state.messages:
            timestamp = msg.get("timestamp")
            if not timestamp:
                timestamp = datetime.now().strftime("%H:%M:%S")
                msg["timestamp"] = timestamp
            bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
            with st.chat_message(msg["role"]):
                st.markdown(f'<div class="{bubble_class}"><div class="message-content">{msg["content"]}</div><span class="timestamp">{timestamp}</span></div>', unsafe_allow_html=True)
    prompt = st.chat_input("Ask something about your document...")
    if prompt:
        now = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": now})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble"><div class="message-content">{prompt}</div><span class="timestamp">{now}</span></div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            msg_box = st.empty()
            with st.spinner("Lemme think about it..."):
                try:
                    full_response = ""
                    for chunk in st.session_state.agent.run(prompt):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                            msg_box.markdown(f'<div class="assistant-bubble"><div class="message-content">{full_response}</div></div>', unsafe_allow_html=True)
                        else:
                            full_response += str(chunk)
                            msg_box.markdown(f'<div class="assistant-bubble"><div class="message-content">{full_response}</div></div>', unsafe_allow_html=True)
                    content = full_response
                    now = datetime.now().strftime("%H:%M:%S")
                    msg_box.markdown(f'<div class="assistant-bubble"><div class="message-content">{full_response}</div><span class="timestamp">{now}</span></div>', unsafe_allow_html=True)
                except Exception as e:
                    content = f"Error: {e}"
                    msg_box.markdown(f'<div class="assistant-bubble"><div class="message-content">{content}</div></div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": content, "timestamp": now})
        # Only rerun if document is processed, not after every message