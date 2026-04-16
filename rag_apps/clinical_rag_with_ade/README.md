# 🏥 Clinical RAG with ADE
![Clinical RAG Architecture](assets/clinical_rag.png)

## Overview

Clinical documents like trial results, lab reports, and prescribing information are difficult for traditional RAG systems to process. Standard PDF parsers "flatten" the text, destroying the structure of multi-column layouts and dense medical tables. 

This project solves the **Ingestion Bottleneck** by using **LandingAI’s Agentic Document Extraction (ADE)**. ADE treats documents as visual artifacts, preserving hierarchical headers and complex tables as structured Markdown. By combining this with **Mistral Large’s** superior reasoning and **ChromaDB**, medical professionals can query dense PDFs and receive answers that are strictly grounded in the document's visual layout.

## Features

- **Visual-First Parsing:** Utilizes LandingAI ADE to extract semantic chunks while preserving the integrity of medical tables.
- **Markdown-Based Retrieval:** Ensures the LLM "sees" tables in their original structure rather than unordered text strings.
- **Visual Grounding:** Every retrieved chunk includes normalized bounding box coordinates (`bbox`) and page numbers.
- **Mistral Reasoning:** Uses `mistral-large-latest` to perform complex reasoning over clinical data.
- **Audit Trails:** A "Source Evidence" panel in the UI allows users to trace every answer back to its exact location in the source PDF.
- **Session Management:** Built-in functionality to clear clinical data and switch between documents securely.

## Tech Stack

**Frameworks & Libraries:**
- [LangChain](https://python.langchain.com/): Orchestration of the RAG pipeline.
- [LandingAI ADE](https://docs.landing.ai/ade/ade-overview): Agentic Document Extraction for structured parsing.
- [Mistral AI](https://docs.mistral.ai/): `mistral-embed` for vectorization and `mistral-large-latest` for reasoning.
- **Vector Database:** [ChromaDB](https://www.trychroma.com/).
- **Web Framework:** [Streamlit](https://streamlit.io/).
- **Package Manager:** [uv](https://docs.astral.sh/uv/).

## Prerequisites

Before you begin, ensure you have:

- Python 3.12 or higher.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your machine.
- API keys for:
  - [LandingAI](https://va.landing.ai/my/settings/api-key) (`VISION_AGENT_API_KEY`)
  - [Mistral AI](https://console.mistral.ai/) (`MISTRAL_API_KEY`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/rag_apps/clinical_rag_with_ade
```

### 2. Install Dependencies
Using `uv`, simply run the sync command. This will create a virtual environment and install all required packages from the lockfile:
```bash
uv sync
```

### 3. Set Up Environment Variables
Copy the example environment file and fill in your keys:
```bash
cp .env.example .env
```
Edit `.env`:
```text
VISION_AGENT_API_KEY=your_landingai_key_here
MISTRAL_API_KEY=your_mistral_key_here
```

## Usage

### Running the Application

To start the clinical dashboard:

```bash
uv run streamlit run app.py
```

### Example Workflow

1. **Upload:** Drop a clinical trial PDF (e.g., NEJM results) into the sidebar.
2. **Ingest:** Click "🚀 Process" to trigger ADE visual parsing and ChromaDB indexing.
3. **Query:** Ask: *"What was the incidence of Grade 3 adverse events in the 50mg cohort?"*
4. **Verify:** Expand the "🔍 Source Evidence" to see the Markdown table and the bounding box coordinates pointing to the exact page.
5. **Reset:** Use the "🗑️ Clear All" button to wipe the local database and upload a new study.

## Project Structure

```
clinical_rag_with_ade/
├── app.py                 # Streamlit UI with Visual Grounding panel
├── processor.py           # Core RAG Logic (ADE Parsing, Indexing, Mistral Query)
├── schemas.py             # Pydantic models for Document Chunks
├── .env                   # API Keys (Git ignored)
├── pyproject.toml         # uv project configuration
└── chroma_db/             # Local persistent vector database (Auto-generated)
```

## How It Works

![Clinical RAG workflow](assets/workflow.png)

**Technical Details:**
- **Visual-First Ingestion:** `processor.py` sends the PDF to LandingAI's `dpt-2-latest` model. It returns Markdown + Grounding metadata.
- **Preserved Hierarchy:** Instead of splitting by character count, the system indexes semantic units (chunks). A full table is kept as a single unit to maintain context.
- **Reflective Retrieval:** LangChain's `VectorStoreRetriever` fetches the top-K chunks. These are enriched with metadata (page, bbox) before being sent to the LLM.
- **Grounded Reasoning:** Mistral Large receives a system prompt instructing it to answer *only* using the provided context, referencing page numbers for high-stakes medical accuracy.

[⬆ Back to Top](#clinical-rag-with-ade)