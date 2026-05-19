# src/ingestion/loader.py
"""
Loads documents from data/raw/ folder.
Supports: PDF, TXT, DOCX files.
Returns a list of dicts: {content, source, file_type}
"""

import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_txt(file_path: Path) -> str:
    """Load a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: Path) -> str:
    """Load a PDF file and extract all text."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"  Warning: Could not read PDF {file_path.name}: {e}")
        return ""


def load_docx(file_path: Path) -> str:
    """Load a Word document."""
    try:
        from docx import Document
        doc = Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"  Warning: Could not read DOCX {file_path.name}: {e}")
        return ""


def load_documents(raw_data_dir: str) -> List[Dict]:
    """
    Scan the raw data directory and load all supported files.
    
    Returns:
        List of dicts with keys: content, source, file_type
    """
    raw_path = Path(raw_data_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    # Supported file extensions
    supported = {".txt", ".pdf", ".docx"}
    
    # Find all supported files
    files = [
        f for f in raw_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    ]

    if not files:
        raise ValueError(f"No supported files found in {raw_data_dir}")

    print(f"\n Found {len(files)} file(s) in {raw_data_dir}")
    
    documents = []

    for file_path in tqdm(files, desc="Loading files"):
        ext = file_path.suffix.lower()
        
        if ext == ".txt":
            content = load_txt(file_path)
            file_type = "txt"
        elif ext == ".pdf":
            content = load_pdf(file_path)
            file_type = "pdf"
        elif ext == ".docx":
            content = load_docx(file_path)
            file_type = "docx"
        else:
            continue

        # Skip empty files
        if not content.strip():
            print(f"  Skipping empty file: {file_path.name}")
            continue

        documents.append({
            "content": content,
            "source": file_path.name,
            "file_type": file_type
        })

        print(f"  Loaded: {file_path.name} "
              f"({len(content)} chars)")

    print(f"\n Successfully loaded {len(documents)} document(s)")
    return documents


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import RAW_DATA_DIR
    docs = load_documents(str(RAW_DATA_DIR))
    for doc in docs:
        print(f"\n--- {doc['source']} ---")
        print(doc["content"][:200], "...")