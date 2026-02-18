"""
Ingestion pipeline orchestrator.

Chains together: parse → chunk → process (AI summarize) → index.
"""
import os

from langchain_chroma import Chroma

from src.app.ingestion.parser import parse_pdf, parse_directory
from src.app.ingestion.chunker import chunk_elements, process_chunks
from src.app.ingestion.indexer import create_vector_store, load_vector_store, add_documents
from src.app.config import CHROMA_PERSIST_DIR
from src.app.logger import get_logger
from src.app.utils import timer

logger = get_logger(__name__)


@timer
def run_ingestion_pipeline(
    pdf_path: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
    extract_images: bool = True,
) -> Chroma:
    """
    Run the complete RAG ingestion pipeline on a single PDF.

    Pipeline: parse PDF → chunk by title → AI-summarize multimodal chunks → index in ChromaDB.

    Args:
        pdf_path: Path to the PDF file.
        persist_dir: Directory for ChromaDB persistence.
        extract_images: Whether to extract images from the PDF.

    Returns:
        Chroma vector store instance with indexed documents.
    """
    source_filename = os.path.basename(pdf_path)

    logger.info("🚀 Starting RAG Ingestion Pipeline")
    logger.info("=" * 50)

    # Step 1: Parse
    elements = parse_pdf(pdf_path, extract_images=extract_images)

    # Step 2: Chunk
    chunks = chunk_elements(elements)

    # Step 3: Process (classify + AI summarize)
    documents = process_chunks(chunks, source_filename)

    # Step 4: Index
    vectorstore = create_vector_store(documents, persist_dir=persist_dir)

    logger.info("=" * 50)
    logger.info("🎉 Ingestion pipeline completed!")
    return vectorstore


@timer
def run_ingestion_directory(
    dir_path: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
    extract_images: bool = True,
) -> Chroma:
    """
    Run the ingestion pipeline on all PDFs in a directory.

    Args:
        dir_path: Path to directory containing PDF files.
        persist_dir: Directory for ChromaDB persistence.
        extract_images: Whether to extract images from PDFs.

    Returns:
        Chroma vector store instance with all indexed documents.
    """
    logger.info("🚀 Starting Directory Ingestion Pipeline")
    logger.info("=" * 50)

    parsed = parse_directory(dir_path, extract_images=extract_images)

    if not parsed:
        raise ValueError(f"No PDFs found in {dir_path}")

    all_documents = []
    for filename, elements in parsed.items():
        chunks = chunk_elements(elements)
        documents = process_chunks(chunks, filename)
        all_documents.extend(documents)

    vectorstore = create_vector_store(all_documents, persist_dir=persist_dir)

    logger.info("=" * 50)
    logger.info(f"🎉 Indexed {len(all_documents)} documents from {len(parsed)} PDF(s)!")
    return vectorstore
