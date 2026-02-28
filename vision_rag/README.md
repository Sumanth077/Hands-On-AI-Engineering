# VisionRAG (Multimodal Search & VQA)

A visual-first search engine that allows you to "talk" to your images and PDF documents. It combines visual similarity search with advanced Visual Question Answering (VQA).

## Principal Functionalities
- **Visual Retrieval Augmented Generation (VisionRAG)**: Unlike text-only RAG, this system breaks down PDFs and images into visual segments and uses **Cohere Embed-v3** to find the most relevant visual context for your question.
- **Multimodal Question Answering**: Powered by **OpenAI Vision**, the system "looks" at the retrieved image or PDF page to answer specific questions about charts, diagrams, or photos.
- **PDF-to-Visual Indexing**: High-fidelity conversion of PDF pages into searchable visual chunks using **PyMuPDF**.
- **Interactive Visual Chat**: A dashboard that shows you exactly which part of the document the AI is looking at when it gives its answer.

## Technical Context
- **Intelligence**: OpenAI `gpt-4o-mini` (Vision-enabled) for multimodal understanding.
- **Embeddings**: Cohere `embed-english-v3.0` for high-accuracy visual search.
- **Processing**: PyMuPDF and Pillow for document-to-image pipeline.

## Setup & Execution

### 1. Requirements
This project runs within the shared repository environment. Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Environment
The application pulls your `OPENAI_API_KEY` and `COHERE_API_KEY` from the root `.env` file.

### 3. Launch
```bash
streamlit run app.py
```

## How to Use
1. **Upload**: Load any image (PNG/JPG) or a multi-page PDF document.
2. **Interact**: Ask a question like "What does the growth chart on page 3 indicate?" or "Describe the logo in this image."
3. **Verify**: The system will display the most relevant visual segment side-by-side with its AI-generated answer.
