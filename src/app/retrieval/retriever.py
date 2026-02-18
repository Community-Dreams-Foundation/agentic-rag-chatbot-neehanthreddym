"""
Document retriever with context formatting for citation support.

Wraps ChromaDB similarity search and formats retrieved documents
with source attribution markers for grounded citations.
"""
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.app.config import TOP_K
from src.app.logger import get_logger

logger = get_logger(__name__)


def get_retriever(vectorstore: Chroma, top_k: int = TOP_K):
    """
    Create a retriever from a Chroma vector store.

    Args:
        vectorstore: Chroma vector store instance.
        top_k: Number of documents to retrieve.

    Returns:
        LangChain retriever instance.
    """
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def retrieve(retriever, query: str) -> list[Document]:
    """
    Retrieve relevant documents for a query.

    Args:
        retriever: LangChain retriever instance.
        query: User query string.

    Returns:
        List of retrieved Document objects.
    """
    logger.info(f"🔍 Retrieving top documents for: {query[:80]}...")
    docs = retriever.invoke(query)
    logger.info(f"✅ Retrieved {len(docs)} documents")
    return docs


def format_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into a context string with citation markers.

    Each document block is labeled with its source and chunk ID so
    the LLM can produce grounded citations.

    Args:
        docs: List of retrieved Document objects.

    Returns:
        Formatted context string with citation markers.
    """
    if not docs:
        return "No relevant documents found."

    context_parts = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        has_tables = doc.metadata.get("has_tables", False)
        has_images = doc.metadata.get("has_images", False)

        header = f"[Source: {source}, Chunk {chunk_id}]"
        content = doc.page_content

        # Include original tables if available for richer LLM context
        if has_tables and "original_content" in doc.metadata:
            try:
                original = json.loads(doc.metadata["original_content"])
                tables = original.get("tables_html", [])
                if tables:
                    content += "\n\nTABLES:\n" + "\n".join(
                        f"Table {j+1}: {t}" for j, t in enumerate(tables)
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        context_parts.append(f"--- Document {i} ---\n{header}\n{content}")

    return "\n\n".join(context_parts)
