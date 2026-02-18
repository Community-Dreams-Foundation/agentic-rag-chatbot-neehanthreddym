"""
Centralized configuration for the Agentic RAG Chatbot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Google Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# LLM_MODEL = "gemini-2.5-pro"
LLM_TEMPERATURE = 0

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Embeddings
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "arxiv_papers"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 5

# --- Memory ---
USER_MEMORY_PATH = "USER_MEMORY.md"
COMPANY_MEMORY_PATH = "COMPANY_MEMORY.md"
MEMORY_CONFIDENCE_THRESHOLD = 0.7

# --- Paths ---
SAMPLE_DOCS_DIR = "sample_docs"
ARTIFACTS_DIR = "artifacts"
