"""
Central configuration. All tuneable constants live here.
Values are overridden by environment variables or a .env file.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# VectorAI DB
# ---------------------------------------------------------------------------

VECTORAI_URL = os.getenv("ACTIAN_VECTORAI_URL", "localhost:6574")
VECTORAI_ACCESS_TOKEN = os.getenv("ACTIAN_VECTORAI_ACCESS_TOKEN")

# Collection names
FAQ_COLLECTION = "product_faq"
DOCS_COLLECTION = "product_docs"
TICKETS_COLLECTION = "resolved_tickets"
MEMORY_COLLECTION = "resolved_queries"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2

# ---------------------------------------------------------------------------
# Language model (Ministral 3B via llama-cpp-python)
# ---------------------------------------------------------------------------

LLM_REPO_ID = "lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF"
LLM_FILENAME = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"

# ---------------------------------------------------------------------------
# Routing thresholds
# ---------------------------------------------------------------------------

# Similarity score above this triggers an auto-resolved response from the knowledge base
HIGH_CONFIDENCE_THRESHOLD = 0.80

# Similarity score below this means no relevant context was found
LOW_CONFIDENCE_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Escalation thresholds (orchestrator)
# ---------------------------------------------------------------------------

# Aggregate negative-sentiment score (0-1) above which the query is escalated
ESCALATION_SENTIMENT_THRESHOLD = 0.50

# Aggregate urgency score (0-1) above which the query is escalated
ESCALATION_URGENCY_THRESHOLD = 0.50

# Number of context documents to pass to the LLM
TOP_K = 4

# ---------------------------------------------------------------------------
# Departments
# ---------------------------------------------------------------------------

DEPARTMENTS = [
    "Returns & Refunds",
    "Billing & Payments",
    "Technical Support",
    "Order Tracking",
    "General Inquiry",
]
