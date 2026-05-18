# src/config.py

import os
import pathlib
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM ──────────────────────────────────────────────────
GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str     = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# ── Embeddings (local sentence-transformers) ──────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE: int     = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int  = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Reflection ────────────────────────────────────────────────
MAX_REFLECTION_RETRIES: int = int(os.getenv("MAX_REFLECTION_RETRIES", "2"))

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR           = pathlib.Path(__file__).parent.parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR   = DATA_DIR / "vector_store"

def validate_config():
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY missing in .env")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Fix the above config errors first.")
    print("Config OK")

if __name__ == "__main__":
    validate_config()
    print(f"  LLM Model   : {GROQ_MODEL}")
    print(f"  Embeddings  : {EMBEDDING_MODEL}")
    print(f"  Chunk size  : {CHUNK_SIZE}")
    print(f"  Base dir    : {BASE_DIR}")