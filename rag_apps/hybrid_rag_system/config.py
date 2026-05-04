from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "processed_docs.json"
COMBINED_GRAPH_FILE = GRAPHS_DIR / "combined_graph.json"

for _d in [DATA_DIR, GRAPHS_DIR, CHROMA_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral-small-latest"
CHROMA_COLLECTION = "hybrid-rag"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

DEFAULT_TOP_K = 5
DEFAULT_MAX_ENTITIES = 15
