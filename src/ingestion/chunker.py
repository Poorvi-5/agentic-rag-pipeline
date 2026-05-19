# src/ingestion/chunker.py
"""
Splits loaded documents into smaller overlapping chunks.
Why chunks? LLMs have context limits. Smaller chunks = 
more precise retrieval. Overlap = no context lost at boundaries.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split each document into overlapping chunks.

    Args:
        documents: List of dicts from loader.py
                   Each has: content, source, file_type

    Returns:
        List of chunk dicts with:
        content, source, file_type, chunk_id, total_chunks
    """

    # RecursiveCharacterTextSplitter tries to split on:
    # paragraphs → sentences → words → characters
    # It's the smartest splitter for natural text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # max chars per chunk
        chunk_overlap=CHUNK_OVERLAP,    # overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        # Split this document's content
        raw_chunks = splitter.split_text(doc["content"])

        print(f"\n {doc['source']} → {len(raw_chunks)} chunks")

        for i, chunk_text in enumerate(raw_chunks):
            all_chunks.append({
                "content":      chunk_text,
                "source":       doc["source"],
                "file_type":    doc["file_type"],
                "chunk_id":     i,
                "total_chunks": len(raw_chunks)
            })

            # Show a preview of first chunk
            if i == 0:
                print(f"   First chunk preview: "
                      f"{chunk_text[:80]}...")

    print(f"\n Total chunks created: {len(all_chunks)}")
    return all_chunks


def print_chunk_stats(chunks: List[Dict]) -> None:
    """Print statistics about the chunks."""
    lengths = [len(c["content"]) for c in chunks]
    print(f"\n Chunk Statistics:")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Avg length   : {sum(lengths)//len(lengths)} chars")
    print(f"   Min length   : {min(lengths)} chars")
    print(f"   Max length   : {max(lengths)} chars")


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import RAW_DATA_DIR
    from src.ingestion.loader import load_documents

    docs = load_documents(str(RAW_DATA_DIR))
    chunks = chunk_documents(docs)
    print_chunk_stats(chunks)