import os
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "chroma_db")
UPLOAD_DIR = DATA_DIR

# ─── Model Settings ────────────────────────────────────
# Using Gemini model for LangGraph Agent (e.g., gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Chunking ──────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "4000"))

# ─── Retrieval ─────────────────────────────────────────
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))
COLLECTION_NAME = "multimodal_rag"

# ─── Limits ────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "200"))
