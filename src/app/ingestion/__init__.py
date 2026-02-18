"""Ingestion package — parse, chunk, summarize, and index PDFs."""

from src.app.ingestion.parser import parse_pdf, parse_directory
from src.app.ingestion.chunker import chunk_elements, process_chunks
from src.app.ingestion.indexer import create_vector_store, load_vector_store, add_documents
from src.app.ingestion.pipeline import run_ingestion_pipeline, run_ingestion_directory

__all__ = [
    "parse_pdf",
    "parse_directory",
    "chunk_elements",
    "process_chunks",
    "create_vector_store",
    "load_vector_store",
    "add_documents",
    "run_ingestion_pipeline",
    "run_ingestion_directory",
]
