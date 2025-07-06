import streamlit as st
# Import OpenAI model and embedder classes from Agno
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
# Removed Ollama imports
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.milvus import Milvus
from agno.tools.duckduckgo import DuckDuckGoTools
import os
from dotenv import load_dotenv
from typing import Optional 
import time
import base64

# Custom CSS for styling
st.set_page_config(
    page_title="PDF RAG Agent (OpenAI)", 
    page_icon="📚",
    layout="wide"
)

# Apply custom CSS styling (Reverted to previous style + minor tweaks)
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        /* Remove specific app background to allow default Streamlit theme or user settings */
        /* background-color: #1E1E2E; */ 
        color: #FAFAFA; /* Keep light text for dark theme */
    }
    .stSidebar > div:first-child {
        background-image: linear-gradient(to bottom, #262936, #1e202a); /* Dark sidebar */
    }
    /* Ensure chat messages stand out slightly */
    [data-testid="stChatMessage"] {
        background-color: rgba(74, 74, 106, 0.4); 
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    [data-testid="stChatMessage"] p {
        color: inherit; /* Inherit text color */
    }
    /* PDF Preview container styling */
    .pdf-preview-container {
        border: 1px solid #4A4A6A; 
        border-radius: 0.5rem;
        padding: 0.5rem;
        background-color: #262936; /* Match sidebar bg */
        margin-bottom: 1rem; /* Add margin below preview */
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Load environment variables
load_dotenv()

# Initialize session state 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

# Define the agent creation function using OpenAI with updated instructions
def get_rag_agent(
    knowledge_base: PDFKnowledgeBase, 
    model_id: str = "gpt-4o-mini", 
    debug_mode: bool = True,
) -> Agent:
    """Creates and configures an Agentic RAG Agent for PDF interaction using OpenAI."""
    
    model = OpenAIChat(id=model_id) 
    # Updated detailed instructions
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
    pdf_rag_agent: Agent = Agent(
        model=model,
        knowledge=knowledge_base,
        description="You are a helpful Agent called 'Agentic RAG' assisting with questions about a PDF document.", # Updated description
        instructions=instructions,
        search_knowledge=True,
        markdown=True,
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        add_datetime_to_instructions=False,
        debug_mode=debug_mode,
    )
    return pdf_rag_agent

# Configure OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("OpenAI API Key not found. Please set the OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
        st.stop()

# Configure Milvus with OpenAI Embedder
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "rag_documents_openai" 
vector_db = Milvus(
    collection=COLLECTION_NAME, 
    uri=MILVUS_URI,
    embedder=OpenAIEmbedder() 
)

# Function to display a PDF preview (Using st.markdown)
def display_pdf_preview(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Main layout
st.markdown("# PDF RAG Agent with Milvus and Agno 📄") 

# Sidebar for document upload
with st.sidebar:
    st.markdown("## Add your documents!")
    uploaded_file = st.file_uploader("Choose your .pdf file", type=['pdf'], key="pdf_uploader")
    
    if uploaded_file is not None:
        temp_pdf_path = "temp_uploaded.pdf" 
        # Check if the file content is different from the one potentially already processed
        # This requires storing the name/hash of the last processed file if needed for robust check
        # Simple check based on file uploader state for now
        if not st.session_state.document_loaded or st.session_state.get("processed_file_name") != uploaded_file.name:
            st.session_state.document_loaded = False # Reset if new file uploaded
            st.session_state.messages = []
            st.session_state.agent = None

        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        file_details = f"{uploaded_file.name} ({round(len(uploaded_file.getvalue())/1024/1024, 2)} MB)" 
        st.write(file_details)
        
        st.markdown("## PDF Preview")
        st.markdown('<div class="pdf-preview-container">', unsafe_allow_html=True)
        display_pdf_preview(temp_pdf_path)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show process button only if not already processed
        if not st.session_state.document_loaded:
            if st.button("Process Document"):
                with st.spinner("Indexing your document..."): 
                    try:
                        knowledge_base = PDFKnowledgeBase(
                            path=temp_pdf_path,
                            vector_db=vector_db,
                        )
                        knowledge_base.load(recreate=True)
                        agent = get_rag_agent(knowledge_base=knowledge_base)
                        
                        st.session_state.knowledge_base = knowledge_base
                        st.session_state.agent = agent
                        st.session_state.document_loaded = True
                        st.session_state.processed_file_name = uploaded_file.name # Store processed file name
                        st.session_state.messages = [] 
                        st.rerun() # Rerun to update UI immediately
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        st.error("Ensure your OpenAI API Key is valid and has embedding permissions.")
                        import traceback
                        st.error(traceback.format_exc())
        else:
             # Show processed indicator instead of the button
             st.markdown("✅ Document Processed Successfully!")

    else:
        # Clear state if file is removed
        if st.session_state.document_loaded:
             st.session_state.document_loaded = False
             st.session_state.messages = []
             st.session_state.agent = None
             st.session_state.pop("processed_file_name", None) # Remove processed file name
             st.rerun() # Rerun to update main area

# Main chat area
st.markdown("## Chat")

if not st.session_state.document_loaded:
    st.info("👈 Please upload and process a PDF document in the sidebar to start chatting.")
else:
    # Display chat messages from history using st.chat_message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input using st.chat_input
    if prompt := st.chat_input("Ask a question about the document..."): 
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty() 
            full_response_content = ""
            with st.spinner("Thinking..."):
                try:
                    response_obj = st.session_state.agent.run(prompt)
                    full_response_content = response_obj.content 
                    message_placeholder.markdown(full_response_content)
                except Exception as e:
                    error_message = f"Error generating response: {e}"
                    st.error(error_message)
                    full_response_content = error_message 
                    import traceback
                    st.error(traceback.format_exc())
        
        st.session_state.messages.append({"role": "assistant", "content": full_response_content})
        st.rerun() # Rerun after getting response to update display immediately 