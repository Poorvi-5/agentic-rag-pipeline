# src/tools/vector_search.py
"""
Tool 1: Vector Semantic Search
Searches the FAISS index built in Step 2.
Returns the most relevant document chunks for a query.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.tools import Tool

from src.config import VECTOR_STORE_DIR, EMBEDDING_MODEL


# ── Load FAISS index + metadata once at import time ───────────
_embedder  = None
_index     = None
_metadata  = None


def _load_resources():
    """Lazy-load the embedder, FAISS index and metadata."""
    global _embedder, _index, _metadata

    if _embedder is None:
        print(" Loading embedding model for search...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    if _index is None:
        faiss_path = VECTOR_STORE_DIR / "index.faiss"
        meta_path  = VECTOR_STORE_DIR / "metadata.pkl"

        if not faiss_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run python ingest.py first."
            )

        _index = faiss.read_index(str(faiss_path))

        with open(meta_path, "rb") as f:
            _metadata = pickle.load(f)

        print(f" FAISS index loaded — {_index.ntotal} vectors")


def search_documents(query: str, top_k: int = 3) -> str:
    """
    Search the vector store for chunks most similar to the query.

    Args:
        query: The user's question as a string
        top_k: Number of top results to return

    Returns:
        Formatted string of top matching chunks with sources
    """
    _load_resources()

    # Embed the query using the same model used during ingestion
    query_vector = _embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Search FAISS — returns distances and indices of top_k results
    distances, indices = _index.search(query_vector, top_k)

    results = []
    for rank, (dist, idx) in enumerate(
        zip(distances[0], indices[0]), start=1
    ):
        if idx == -1:           # FAISS returns -1 for empty slots
            continue

        chunk    = _metadata[idx]
        score    = float(dist)  # cosine similarity (0-1, higher=better)

        results.append(
            f"[Result {rank}] Source: {chunk['source']} "
            f"| Score: {score:.3f}\n"
            f"{chunk['content']}\n"
        )

    if not results:
        return "No relevant documents found in the knowledge base."

    return "\n---\n".join(results)


# ── Wrap as a LangChain Tool ───────────────────────────────────
vector_search_tool = Tool(
    name="vector_search",
    func=search_documents,
    description=(
        "Search the local knowledge base for information from "
        "uploaded documents (PDFs, TXT files). Use this tool when "
        "the user asks about topics that might be in the documents, "
        "such as AI concepts, LangChain, RAG, or any domain-specific "
        "knowledge. Input should be a clear search query string."
    )
)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    query = "What is RAG and how does it work?"
    print(f"\n Query: {query}\n")
    result = search_documents(query)
    print(result)