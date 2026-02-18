"""
Grounded answer generator with citation support.

Uses Gemini or Groq LLM to generate answers grounded in retrieved
context, with structured citations.
"""
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.app.generation.prompts import RAG_SYSTEM_PROMPT
from src.app.retrieval.retriever import format_context
from src.app.utils import timer, get_llm
from src.app.logger import get_logger

logger = get_logger(__name__)


def _extract_citations(answer_text: str, docs: list[Document]) -> list[dict]:
    """
    Extract citation references from the answer text.

    Looks for [Source: filename, Chunk N] patterns and maps them
    to the actual retrieved documents.

    Args:
        answer_text: The LLM's generated answer.
        docs: The retrieved documents used as context.

    Returns:
        List of citation dicts with source, chunk_id, and snippet.
    """
    pattern = r"\[Source:\s*(.+?),\s*Chunk\s*(\d+)\]"
    matches = re.findall(pattern, answer_text)

    citations = []
    seen = set()

    for source, chunk_id_str in matches:
        key = (source.strip(), int(chunk_id_str))
        if key in seen:
            continue
        seen.add(key)

        # Find the matching document
        snippet = ""
        for doc in docs:
            if (doc.metadata.get("source") == key[0] and
                    doc.metadata.get("chunk_id") == key[1]):
                snippet = doc.page_content[:200]
                break

        citations.append({
            "source": key[0],
            "chunk_id": key[1],
            "snippet": snippet,
        })

    return citations

@timer
def generate_answer(query: str, context_docs: list[Document]) -> dict:
    """
    Generate a grounded answer with citations.

    Args:
        query: The user's question.
        context_docs: Retrieved documents providing context.

    Returns:
        Dict with keys:
          - answer: The generated answer text.
          - citations: List of citation dicts.
          - sources_used: List of unique source filenames.
    """
    llm = get_llm()

    # Format the context with citation markers
    context = format_context(context_docs)

    # Build the system prompt with context injected
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    logger.info(f"🤖 Generating answer for: {query}")
    response = llm.invoke(messages)
    answer_text = response.content

    # Extract structured citations
    citations = _extract_citations(answer_text, context_docs)
    sources_used = list({c["source"] for c in citations})

    logger.info(
        f"✅ Answer generated with {len(citations)} citation(s) "
        f"from {len(sources_used)} source(s)"
    )

    return {
        "answer": answer_text,
        "citations": citations,
        "sources_used": sources_used,
    }
