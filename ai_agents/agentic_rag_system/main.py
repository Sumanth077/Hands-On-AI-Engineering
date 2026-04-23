import gradio as gr
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rag_graph import app, embeddings, CHROMA_DB_PATH


# ── PDF Processing ─────────────────────────────────────────────────────────────

def process_pdf(file):
    if file is None:
        return (
            gr.update(value="⚠️ No file selected. Please upload a PDF first."),
            gr.update(interactive=False),
        )
    try:
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)

        # Gradio 4+ returns a string path; older versions return a file-like object
        file_path = file if isinstance(file, str) else file.name
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)

        return (
            gr.update(value=f"✅ Ready — {len(chunks)} chunks indexed from {len(docs)} pages."),
            gr.update(interactive=True),
        )
    except Exception as e:
        return (
            gr.update(value=f"❌ Processing failed: {str(e)}"),
            gr.update(interactive=False),
        )


def clear_pdf():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    return (
        None,  # reset file input
        gr.update(value="🗑️ Document cleared. Upload a new PDF to continue."),
        gr.update(interactive=False),
    )


# ── Question Answering ─────────────────────────────────────────────────────────

def ask_question(question):
    if not question or not question.strip():
        return gr.update(value=""), gr.update(value="*Ask a question to get started.*")

    if not os.path.exists(CHROMA_DB_PATH):
        return (
            gr.update(value="⚠️ No document loaded. Upload and process a PDF first."),
            gr.update(value=""),
        )

    try:
        result = app.invoke({
            "question": question.strip(),
            "refined_query": "",
            "context": "",
            "reflection": "",
            "answer": "",
            "iterations": 0,
            "reflection_log": [],
        })

        reflection_log = result.get("reflection_log", [])
        iterations = result.get("iterations", 0)
        answer = result.get("answer", "No answer was generated.")

        # Format reflection log for display
        separator = "\n" + "─" * 60 + "\n"
        log_text = f"Total retrieval iterations: {iterations}\n" + separator
        log_text += separator.join(reflection_log) if reflection_log else "No reflection data."

        return gr.update(value=log_text), gr.update(value=answer)

    except Exception as e:
        return (
            gr.update(value=f"❌ Error during inference: {str(e)}"),
            gr.update(value=""),
        )


def clear_question():
    return "", gr.update(value=""), gr.update(value="*Ask a question to get started.*")


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Self-Reflective Agentic RAG",
    css="""
        .header-box { text-align: center; padding: 8px 0 4px 0; }
        .section-label { font-weight: 600; margin-bottom: 4px; }
    """,
) as demo:

    # ── Header ──
    gr.Markdown(
        """
        <div class="header-box">

        # 🔄 Self-Reflective Agentic RAG
        **Powered by Gemma 4 · gemini-embedding-001 · LangGraph**

        </div>

        > This system doesn't answer right after retrieval. It **grades** the retrieved context for relevance and sufficiency,
        > **rewrites the query** if context is lacking, and only generates an answer once the context passes validation —
        > reducing hallucinations through an iterative retrieval loop.
        """,
        elem_classes="header-box",
    )

    gr.Markdown("---")

    # ── Main layout ──
    with gr.Row(equal_height=False):

        # Left column — Document Management
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📄 Document")
            file_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                file_count="single",
            )
            with gr.Row():
                proc_btn = gr.Button("⚙️ Process", variant="primary", scale=2)
                clear_pdf_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
            status_box = gr.Textbox(
                label="Status",
                value="No document loaded.",
                interactive=False,
                lines=2,
            )

        # Right column — Q&A
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask a Question")
            q_input = gr.Textbox(
                label="Question",
                placeholder="What does the document say about…?",
                lines=3,
            )
            with gr.Row():
                ask_btn = gr.Button(
                    "🔍 Ask",
                    variant="primary",
                    interactive=False,  # locked until PDF is processed
                    scale=3,
                )
                clear_q_btn = gr.Button("✖ Clear", variant="secondary", scale=1)

    gr.Markdown("---")

    # ── Reflection Loop ──
    with gr.Accordion("🔄 Self-Reflection Loop  (retrieval grading & query refinement)", open=False):
        gr.Markdown(
            "_Each iteration shows the LLM's verdict on the retrieved context. "
            "A `NO` verdict triggers query rewriting and a new retrieval attempt._"
        )
        reflection_log_box = gr.Textbox(
            label="Grading Log",
            lines=12,
            interactive=False,
            placeholder="Reflection logs will appear here after you ask a question.",
        )

    # ── Final Answer ──
    gr.Markdown("### 📝 Answer")
    answer_box = gr.Markdown(value="*Ask a question to get started.*")

    # ── Event wiring ──
    proc_btn.click(
        fn=process_pdf,
        inputs=[file_input],
        outputs=[status_box, ask_btn],
    )
    clear_pdf_btn.click(
        fn=clear_pdf,
        outputs=[file_input, status_box, ask_btn],
    )
    ask_btn.click(
        fn=ask_question,
        inputs=[q_input],
        outputs=[reflection_log_box, answer_box],
    )
    clear_q_btn.click(
        fn=clear_question,
        outputs=[q_input, reflection_log_box, answer_box],
    )


if __name__ == "__main__":
    demo.launch()
