import os
import shutil
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# LandingAI ADE
from landingai_ade import LandingAIADE

# LangChain & Mistral AI
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

class ClinicalRAGProcessor:
    def __init__(self, db_path="./chroma_db"):
        """
        Initializes the Clinical RAG Processor.
        - LandingAI ADE for visual-first document parsing.
        - Mistral AI for embeddings and high-reasoning LLM.
        - ChromaDB as the vector store for structured Markdown chunks.
        """
        self.ade_client = LandingAIADE()
        self.embeddings = MistralAIEmbeddings(model="mistral-embed")
        self.llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
        self.db_path = db_path
        self.vector_store = None

    def ingest_document(self, file_path: str) -> int:
        """
        Parses a PDF using ADE and indexes structured chunks into ChromaDB.
        Returns the number of chunks indexed.
        """
        print(f"--- ADE Parsing Started: {file_path} ---")
        
        # ADE expects a pathlib.Path object for local file parsing
        response = self.ade_client.parse(
            document=Path(file_path), 
            model="dpt-2-latest"
        )

        documents = []
        for chunk in response.chunks:
            # Capture visual grounding metadata (BBoxes and Page numbers)
            # This allows the UI to show exactly where the answer came from.
            metadata = {
                "chunk_id": chunk.id,
                "chunk_type": chunk.type,
                "page": chunk.grounding.page + 1, # Convert to 1-indexed for human readability
                "bbox_left": chunk.grounding.box.left if chunk.grounding.box else 0,
                "bbox_top": chunk.grounding.box.top if chunk.grounding.box else 0,
                "bbox_right": chunk.grounding.box.right if chunk.grounding.box else 0,
                "bbox_bottom": chunk.grounding.box.bottom if chunk.grounding.box else 0,
            }
            
            # Use chunk.markdown to preserve table structures and document hierarchy
            documents.append(Document(page_content=chunk.markdown, metadata=metadata))

        # Initialize/Update the vector store with the new clinical chunks
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_name="clinical_docs"
        )
        return len(documents)

    def ask(self, query: str) -> Tuple[str, List[Document]]:
        """
        Retrieves relevant structured context and generates a grounded response.
        """
        if not self.vector_store:
            # Attempt to load existing vector store if it's not in memory
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name="clinical_docs"
            )

        # Retrieval: Fetch top 5 relevant Markdown chunks
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        context_docs = retriever.invoke(query)
        
        # Build the context string for Mistral Large
        context_text = "\n\n".join([
            f"[Source: Page {d.metadata['page']}]\n{d.page_content}" 
            for d in context_docs
        ])
        
        system_prompt = f"""
        You are a highly precise Clinical Data Analyst. 
        Your task is to answer the user's question based ONLY on the provided document context.
        The context is provided in Markdown format, preserving tables and medical hierarchies.
        
        Rules:
        1. If tables are involved, verify data across rows/columns.
        2. Provide direct, evidence-based answers.
        3. Reference specific page numbers from the context.
        4. If the info is missing, state that clearly.

        CONTEXT:
        {context_text}
        """
        
        # Structured Reasoning using Mistral Large
        response = self.llm.invoke([
            ("system", system_prompt),
            ("human", query)
        ])
        
        return response.content, context_docs

    def clear_database(self):
        """
        Deletes the physical ChromaDB folder and resets the vector store object.
        Essential for clearing the UI to upload new documents.
        """
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.vector_store = None
        print(f"Database at {self.db_path} has been wiped.")