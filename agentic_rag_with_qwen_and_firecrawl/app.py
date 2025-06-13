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

# Load embedder
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

class CustomEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
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
</style>
""", unsafe_allow_html=True)

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

# Create Agent
def get_rag_agent(knowledge_base: PDFKnowledgeBase, model_id="qwen3:8b", debug_mode=True):
    model = Ollama(id=model_id)
    instructions = [
        "1. Knowledge Base Search:",
        "   - ALWAYS start by searching the knowledge base using search_knowledge_base tool",
        "   - Analyze ALL returned documents thoroughly before responding",
        "   - If multiple documents are returned, synthesize the information coherently",
        "2. External Search:",
        "   - If knowledge base search yields insufficient results, use duckduckgo_search",
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
        description="You are a helpful Agent called 'Agentic RAG' assisting with questions about a PDF document.", # Updated description
        instructions=instructions,
        search_knowledge=True,
        markdown=True,
        tools=[FirecrawlTools(api_key=firecrawl_api_key, scrape=False, crawl=True)],
        show_tool_calls=True,
        add_datetime_to_instructions=False,
        debug_mode=debug_mode,
    )

# PDF Viewer
def display_pdf_preview(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300px"></iframe>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Session State Init
for key in ["messages", "document_loaded", "agent", "knowledge_base"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None

# Sidebar: Upload PDF
with st.sidebar:
    st.markdown("## Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        pdf_path = "temp_uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        display_pdf_preview(pdf_path)

        if not st.session_state.document_loaded:
            if st.button("Process Document"):
                with st.spinner("Indexing document..."):
                    kb = PDFKnowledgeBase(path=pdf_path, vector_db=vector_db)

                    kb.load(recreate=True)
                    agent = get_rag_agent(kb)

                    st.session_state.knowledge_base = kb
                    st.session_state.agent = agent
                    st.session_state.document_loaded = True
                    st.session_state.messages = []
                    st.rerun()

        else:
            st.markdown("✅ Document indexed!")

# Main Chat Area
st.markdown("# Chat with your PDF")
if not st.session_state.document_loaded:
    st.info("👈 Upload and process a PDF to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            msg_box = st.empty()
            try:
                response = st.session_state.agent.run(prompt)
                content = response.content
                msg_box.markdown(content)
            except Exception as e:
                content = f"Error: {e}"
                st.error(content)

        st.session_state.messages.append({"role": "assistant", "content": content})
        st.rerun()