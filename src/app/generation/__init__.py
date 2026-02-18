"""Generation package — grounded answer generation with citations."""

from src.app.generation.generator import get_llm, generate_answer
from src.app.generation.prompts import RAG_SYSTEM_PROMPT, MEMORY_EXTRACTION_PROMPT

__all__ = [
    "get_llm",
    "generate_answer",
    "RAG_SYSTEM_PROMPT",
    "MEMORY_EXTRACTION_PROMPT",
]
