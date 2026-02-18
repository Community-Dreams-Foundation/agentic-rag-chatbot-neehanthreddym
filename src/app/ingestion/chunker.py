"""
Text chunker with multimodal AI summarization.

Chunks parsed PDF elements by title, classifies content types,
and generates AI summaries for chunks containing tables or images.
"""
import json
import time

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from unstructured.chunking.title import chunk_by_title

from src.app.config import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_NEW_AFTER_N_CHARS,
    CHUNK_OVERLAP,
)
from src.app.logger import get_logger
from src.app.utils import timer, get_llm

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Element classification
# ---------------------------------------------------------------------------

def separate_content_types(chunk) -> dict:
    """
    Classify a chunk's elements into text, tables, and images.

    Args:
        chunk: An unstructured CompositeElement containing sub-elements.

    Returns:
        Dict with keys: 'text', 'tables_html', 'images', 'types'.
    """
    text_parts = []
    tables_html = []
    images_base64 = []
    content_types = set()

    for element in chunk.metadata.orig_elements:
        el_type = type(element).__name__

        if el_type == "Table":
            content_types.add("table")
            html = getattr(element.metadata, "text_as_html", None)
            if html:
                tables_html.append(html)

        elif el_type == "Image":
            content_types.add("image")
            payload = getattr(element.metadata, "image_base64", None)
            if payload:
                images_base64.append(payload)

        else:
            content_types.add("text")
            text_parts.append(str(element))

    return {
        "text": "\n".join(text_parts),
        "tables_html": tables_html,
        "images": images_base64,
        "types": sorted(content_types),
    }


# ---------------------------------------------------------------------------
# AI summary for multimodal content
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """You are a research document analyst. Given the following content from an academic paper, produce a single dense, keyword-rich paragraph that captures all important details. Your summary will be used as the search index for retrieval-augmented generation, so maximize retrieval relevance.

TEXT:
{text}

{tables_section}

Respond ONLY with the summary paragraph — no preamble."""


def create_ai_summary(text: str, tables_html: list, images_base64: list) -> str:
    """
    Generate an AI summary for a chunk that contains tables and/or images.

    Uses Gemini's multimodal capabilities to interpret visual content
    alongside raw text.

    Args:
        text: The raw text content of the chunk.
        tables_html: List of HTML table strings.
        images_base64: List of base64-encoded image strings.

    Returns:
        A dense summary string optimized for retrieval.
    """
    llm = get_llm()

    tables_section = ""
    if tables_html:
        tables_section = "TABLES:\n" + "\n".join(
            f"Table {i+1}:\n{t}" for i, t in enumerate(tables_html)
        )

    prompt_text = _SUMMARY_PROMPT.format(
        text=text or "(no text)",
        tables_section=tables_section,
    )

    # Build multimodal message content
    message_content = [{"type": "text", "text": prompt_text}]

    for img_b64 in images_base64:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    message = HumanMessage(content=message_content)
    response = llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Chunk processing pipeline
# ---------------------------------------------------------------------------

def chunk_elements(elements: list) -> list:
    """
    Chunk parsed elements by title boundaries.

    Args:
        elements: List of unstructured Element objects from parser.

    Returns:
        List of CompositeElement chunks.
    """
    logger.info("📐 Chunking elements by title...")
    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_MAX_CHARACTERS,
        new_after_n_chars=CHUNK_NEW_AFTER_N_CHARS,
        overlap=CHUNK_OVERLAP,
    )
    logger.info(f"✅ Created {len(chunks)} chunks")
    return chunks


@timer
def process_chunks(chunks: list, source_filename: str) -> list:
    """
    Process raw chunks into LangChain Documents with metadata.

    For chunks containing tables or images, generates an AI summary
    to maximize retrieval relevance. Pure-text chunks keep their
    original content.

    Args:
        chunks: List of CompositeElement chunks from chunk_elements().
        source_filename: Name of the source PDF file.

    Returns:
        List of LangChain Document objects with rich metadata.
    """
    logger.info(f"🔬 Processing {len(chunks)} chunks with AI summaries...")
    documents = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_id = i + 1
        logger.info(f"\tProcessing chunk {chunk_id}/{total}")

        content_data = separate_content_types(chunk)
        has_tables = len(content_data["tables_html"]) > 0
        has_images = len(content_data["images"]) > 0

        logger.debug(
            f"Chunk {chunk_id}/{total} — Types: {content_data['types']} | "
            f"Tables: {len(content_data['tables_html'])} | "
            f"Images: {len(content_data['images'])}"
        )

        # AI-summarize multimodal chunks
        # use raw text for text-only
        if has_tables or has_images:
            logger.info(f"  Chunk {chunk_id}/{total}: summarizing multimodal content...")
            try:
                page_content = create_ai_summary(
                    content_data["text"],
                    content_data["tables_html"],
                    content_data["images"],
                )
                logger.info(f"✅ AI summary ({len(page_content)} chars)")
            except Exception as e:
                logger.error(f"  Chunk {chunk_id}/{total}: AI summary failed: {e}")
                page_content = content_data["text"]
            time.sleep(1)  # rate-limit guard
        else:
            logger.debug(f"  Chunk {chunk_id}/{total}: text-only, using raw content")
            page_content = content_data["text"]

        page_content = page_content or content_data["text"] or ""

        doc = Document(
            page_content=page_content,
            metadata={
                "source": source_filename,
                "chunk_id": chunk_id,
                "has_tables": has_tables,
                "has_images": has_images,
                "content_types": content_data["types"],
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables_html"],
                    "images_base64": content_data["images"],
                }),
            },
        )
        documents.append(doc)

    logger.info(f"✅ Processed {len(documents)} chunks into Documents")
    return documents
