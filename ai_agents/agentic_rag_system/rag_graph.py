import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file!")

# Gemma 4 via Gemini API — temperature=0 for deterministic grading
llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=api_key,
    temperature=0,
)

# gemini-embedding-001 with retrieval_document task type
# langchain-google-genai automatically switches to retrieval_query for embed_query calls
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
    task_type="retrieval_document",
)

MAX_ITERATIONS = 3
CHROMA_DB_PATH = "./chroma_db"


def extract_text(response) -> str:
    """
    Safely extract a plain string from an LLM response.
    langchain-google-genai can return response.content as either a str
    or a list of content blocks (e.g. [{"type": "text", "text": "..."}]).
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return str(content).strip()


class GraphState(TypedDict):
    question: str          # Original user question — never mutated
    refined_query: str     # Current search query (rewritten if needed)
    context: str           # Retrieved chunks joined as string
    reflection: str        # Latest grading output from the LLM
    answer: str            # Final generated answer
    iterations: int        # How many retrieval loops have run
    reflection_log: List[str]  # Full history of every grading step


def get_db() -> Chroma:
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


# ── Node 1: Retrieve ──────────────────────────────────────────────────────────
def retrieve(state: GraphState) -> dict:
    """Retrieve top-k chunks using the current query (original or rewritten)."""
    query = state.get("refined_query") or state["question"]
    db = get_db()
    docs = db.similarity_search(query, k=4)
    context = "\n\n".join(
        [f"[Chunk {i + 1}]:\n{d.page_content}" for i, d in enumerate(docs)]
    )
    return {
        "context": context,
        "refined_query": query,
        "iterations": state.get("iterations", 0) + 1,
    }


# ── Node 2: Grade Retrieval ───────────────────────────────────────────────────
def grade_retrieval(state: GraphState) -> dict:
    """
    LLM judges whether the retrieved context is relevant and sufficient.
    Outputs a structured verdict and, if needed, a refined search query.
    """
    prompt = f"""You are a strict retrieval quality judge for a RAG system.

Your job: decide if the retrieved context is good enough to answer the question accurately.

Question: {state['question']}

Retrieved Context:
{state['context']}

Evaluate the context on three criteria:
1. RELEVANCE — Does it directly address the question?
2. SUFFICIENCY — Does it contain enough detail for a complete answer?
3. CONSISTENCY — Are there contradictions between chunks?

Reply in this EXACT format (no extra lines):
VERDICT: YES
REASON: <one sentence>
REFINED_QUERY: NONE

OR if context is not good enough:
VERDICT: NO
REASON: <one sentence explaining what is missing or wrong>
REFINED_QUERY: <a better, more specific search query to find the missing information>"""

    response = llm.invoke(prompt)
    content = extract_text(response)

    log_entry = f"Iteration {state.get('iterations', 1)}\n{content}"
    reflection_log = list(state.get("reflection_log", [])) + [log_entry]

    return {
        "reflection": content,
        "reflection_log": reflection_log,
    }


# ── Node 3: Rewrite Query ─────────────────────────────────────────────────────
def rewrite_query(state: GraphState) -> dict:
    """
    Extract the REFINED_QUERY from the grader's output.
    Falls back to the original question if parsing fails.
    """
    reflection = state.get("reflection", "")
    refined = state["question"]  # safe fallback

    for line in reflection.splitlines():
        line = line.strip()
        if line.upper().startswith("REFINED_QUERY:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate and candidate.upper() != "NONE":
                refined = candidate
                break

    return {"refined_query": refined}


# ── Node 4: Generate Answer ───────────────────────────────────────────────────
def generate(state: GraphState) -> dict:
    """Generate the final answer grounded strictly in the validated context."""
    prompt = f"""You are a precise, helpful assistant. Answer the question using ONLY the provided context.
If the context is insufficient for a complete answer, clearly state what is missing — do not hallucinate.

Question: {state['question']}

Validated Context:
{state['context']}

Write a clear, structured answer grounded in the context above."""

    response = llm.invoke(prompt)
    return {"answer": extract_text(response)}


# ── Router ────────────────────────────────────────────────────────────────────
def should_continue(state: GraphState) -> str:
    """
    Route to 'generate' if context passed grading or max iterations reached.
    Route to 'rewrite' otherwise to refine the query and re-retrieve.
    """
    iterations = state.get("iterations", 0)
    if iterations >= MAX_ITERATIONS:
        return "generate"  # forced exit — generate with best context so far

    reflection = state.get("reflection", "")
    for line in reflection.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            if "YES" in line.upper():
                return "generate"
            break

    return "rewrite"


# ── Build LangGraph ───────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_retrieval", grade_retrieval)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_retrieval")
workflow.add_conditional_edges(
    "grade_retrieval",
    should_continue,
    {
        "generate": "generate",
        "rewrite": "rewrite_query",
    },
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()
