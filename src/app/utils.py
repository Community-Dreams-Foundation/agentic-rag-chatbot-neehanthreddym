import time
from functools import wraps

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.app.config import LLM_MODEL, LLM_TEMPERATURE, LLM_PROVIDER, GROQ_MODEL
from src.app.logger import get_logger

logger = get_logger(__name__)


def timer(base_function):
    """Decorator to measure and log function execution time."""
    @wraps(base_function)
    def enhanced_function(*args, **kwargs):
        start_time = time.time()
        result = base_function(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{base_function.__name__} completed in {elapsed:.2f}s")
        return result
    return enhanced_function


def get_llm():
    """
    Initialize and return the configured LLM.

    Uses LLM_PROVIDER config to select between Gemini and Groq.

    Returns:
        LangChain chat model instance.
    """
    if LLM_PROVIDER == "groq":
        logger.info(f"Using Groq LLM: {GROQ_MODEL}")
        return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)
    else:
        logger.info(f"Using Gemini LLM: {LLM_MODEL}")
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE
        )