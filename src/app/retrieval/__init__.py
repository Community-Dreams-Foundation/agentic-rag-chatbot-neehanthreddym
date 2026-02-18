"""Retrieval package — document retrieval with citation-aware formatting."""

from src.app.retrieval.retriever import get_retriever, retrieve, format_context

__all__ = [
    "get_retriever",
    "retrieve",
    "format_context",
]
