# src/config.py

import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM ──────────────────────────────────────────────────
GROQ_API_KEY: str    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str      = "llama-3.3-70b-versatile"   # hardcoded — env was corrupted

# ── Embeddings ────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE: int      = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int   = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Reflection ────────────────────────────────────────────────
MAX_REFLECTION_RETRIES: int = int(os.getenv("MAX_REFLECTION_RETRIES", "2"))

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR           = pathlib.Path(__file__).parent.parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR   = DATA_DIR / "vector_store"

def validate_config():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")
    print(f"Config OK — Model: {GROQ_MODEL}")

if __name__ == "__main__":
    validate_config()