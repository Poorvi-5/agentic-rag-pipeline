# src/ingestion/embedder.py
"""
Converts text chunks into vector embeddings and stores them
in FAISS (fast similarity search) and ChromaDB (persistent).

Uses sentence-transformers locally — FREE, no API key needed.
Model: all-MiniLM-L6-v2 — small, fast, 384-dimensional vectors
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

from src.config import (
    EMBEDDING_MODEL,
    VECTOR_STORE_DIR,
    PROCESSED_DATA_DIR
)


# ── Load the embedding model once (reused across calls) ───────
print(f" Loading embedding model: {EMBEDDING_MODEL}")
_embedder = SentenceTransformer(EMBEDDING_MODEL)
print(f" Embedding model ready — vector size: "
      f"{_embedder.get_sentence_embedding_dimension()}")


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of strings to a numpy array of vectors.
    Shape: (num_texts, embedding_dim)  e.g. (25, 384)
    """
    vectors = _embedder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # L2 normalize for cosine similarity
    )
    return vectors


def save_to_faiss(chunks: List[Dict], vectors: np.ndarray) -> None:
    """
    Build a FAISS index from vectors and save it to disk.
    Also saves chunk metadata separately as a pickle file.
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    dim = vectors.shape[1]  # embedding dimension (384)

    # IndexFlatIP = Inner Product search (cosine sim with normalized vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save FAISS index
    faiss_path = VECTOR_STORE_DIR / "index.faiss"
    faiss.write_index(index, str(faiss_path))

    # Save chunk metadata (text + source info) alongside the index
    meta_path = VECTOR_STORE_DIR / "metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n FAISS index saved → {faiss_path}")
    print(f" Metadata saved   → {meta_path}")
    print(f" Vectors in index : {index.ntotal}")


def save_to_chromadb(chunks: List[Dict], vectors: np.ndarray) -> None:
    """
    Store chunks in ChromaDB for persistent retrieval.
    ChromaDB stores: documents (text), embeddings, metadata, ids
    """
    chroma_path = str(VECTOR_STORE_DIR / "chroma_db")
    
    # PersistentClient saves data to disk automatically
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection to avoid duplicates on re-run
    try:
        client.delete_collection("rag_documents")
    except Exception:
        pass

    collection = client.create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "cosine"}
    )

    # ChromaDB needs lists, not numpy arrays
    ids        = [f"chunk_{i}" for i in range(len(chunks))]
    documents  = [c["content"] for c in chunks]
    embeddings = vectors.tolist()
    metadatas  = [
        {
            "source":       c["source"],
            "file_type":    c["file_type"],
            "chunk_id":     str(c["chunk_id"]),
            "total_chunks": str(c["total_chunks"])
        }
        for c in chunks
    ]

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(ids), batch_size),
                  desc="Saving to ChromaDB"):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"\n ChromaDB saved   → {chroma_path}")
    print(f" Documents stored : {collection.count()}")


def save_chunks_json(chunks: List[Dict]) -> None:
    """Save chunks as JSON for inspection/debugging."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PROCESSED_DATA_DIR / "chunks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f" Chunks JSON      → {json_path}")


def run_embedding_pipeline(chunks: List[Dict]) -> np.ndarray:
    """
    Master function — embed all chunks and save to both stores.
    
    Args:
        chunks: List of chunk dicts from chunker.py
    Returns:
        vectors: numpy array of embeddings
    """
    print(f"\n Embedding {len(chunks)} chunks...")

    # Extract just the text content for embedding
    texts = [c["content"] for c in chunks]

    # Generate vectors
    vectors = embed_texts(texts)
    print(f" Vector shape: {vectors.shape}")  # e.g. (25, 384)

    # Save to both vector stores
    save_to_faiss(chunks, vectors)
    save_to_chromadb(chunks, vectors)

    # Save human-readable JSON
    save_chunks_json(chunks)

    return vectors


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import RAW_DATA_DIR
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents

    docs   = load_documents(str(RAW_DATA_DIR))
    chunks = chunk_documents(docs)
    vecs   = run_embedding_pipeline(chunks)
    print(f"\n Embedding pipeline complete!")
    print(f" Sample vector (first 5 dims): {vecs[0][:5]}")