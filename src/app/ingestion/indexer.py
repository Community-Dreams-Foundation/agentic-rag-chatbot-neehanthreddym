"""
Embedding generation and ChromaDB vector store management.

Creates, loads, and manages the ChromaDB vector store for
retrieval-augmented generation.
"""
import os
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from src.app.config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.app.logger import get_logger
from src.app.utils import timer

logger = get_logger(__name__)


def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Return the configured embedding model."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@timer
def create_vector_store(
    documents: list[Document],
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Create a ChromaDB vector store from documents and persist to disk.

    Args:
        documents: List of LangChain Document objects to index.
        persist_dir: Directory path for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Chroma vector store instance.
    """
    logger.info(f"🔮 Creating embeddings and storing in ChromaDB ({persist_dir})...")

    embedding_model = _get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"✅ Vector store created with {len(documents)} documents")
    return vectorstore


def load_vector_store(
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        persist_dir: Directory path for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Chroma vector store instance.

    Raises:
        FileNotFoundError: If the persist directory doesn't exist.
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"ChromaDB directory not found: {persist_dir}. "
            "Run the ingestion pipeline first."
        )

    logger.info(f"📂 Loading vector store from {persist_dir}...")
    embedding_model = _get_embedding_model()

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    logger.info(f"✅ Loaded vector store with {count} documents")
    return vectorstore


def add_documents(
    vectorstore: Chroma,
    documents: list[Document],
) -> None:
    """
    Add new documents to an existing vector store.

    Args:
        vectorstore: Existing Chroma vector store instance.
        documents: List of LangChain Document objects to add.
    """
    logger.info(f"➕ Adding {len(documents)} documents to vector store...")
    vectorstore.add_documents(documents)
    count = vectorstore._collection.count()
    logger.info(f"✅ Vector store now contains {count} documents")
