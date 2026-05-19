# ingest.py  ← run this to process all your documents
"""
Master ingestion script.
Run this whenever you add new documents to data/raw/

Usage:
    python ingest.py
"""

from rich.console import Console
from rich.panel import Panel

from src.config import validate_config, RAW_DATA_DIR
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents, print_chunk_stats
from src.ingestion.embedder import run_embedding_pipeline

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]Agentic RAG — Document Ingestion Pipeline[/bold cyan]\n"
        "Loads → Chunks → Embeds → Stores",
        border_style="cyan"
    ))

    # Step 1: Validate config
    console.print("\n[bold]Step 1:[/bold] Validating config...")
    validate_config()

    # Step 2: Load documents
    console.print("\n[bold]Step 2:[/bold] Loading documents from data/raw/...")
    documents = load_documents(str(RAW_DATA_DIR))

    # Step 3: Chunk documents
    console.print("\n[bold]Step 3:[/bold] Chunking documents...")
    chunks = chunk_documents(documents)
    print_chunk_stats(chunks)

    # Step 4: Embed and store
    console.print("\n[bold]Step 4:[/bold] Embedding and storing vectors...")
    vectors = run_embedding_pipeline(chunks)

    # Done
    console.print(Panel.fit(
        f"[bold green]Ingestion Complete![/bold green]\n"
        f"Documents : {len(documents)}\n"
        f"Chunks    : {len(chunks)}\n"
        f"Vectors   : {vectors.shape}",
        border_style="green"
    ))
    console.print("\nYour knowledge base is ready. Proceed to Step 3!\n")


if __name__ == "__main__":
    main()