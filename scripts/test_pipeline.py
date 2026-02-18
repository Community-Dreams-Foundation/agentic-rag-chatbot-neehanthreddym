"""End-to-end pipeline test — run from project root."""
import shutil, os
from src.app.ingestion.pipeline import run_ingestion_pipeline
from src.app.retrieval.retriever import get_retriever, retrieve
from src.app.generation.generator import generate_answer

DB = "test_chroma_db"

# 1. Ingest
vs = run_ingestion_pipeline("sample_docs/TinyLoRA_2602.04118v1.pdf", persist_dir=DB)

# 2. Retrieve
retriever = get_retriever(vs, top_k=3)
docs = retrieve(retriever, "What is TinyLoRA?")

# 3. Generate
result = generate_answer("What is TinyLoRA?", docs)
print(result["answer"])
print(f"Citations: {result['citations']}")

# Cleanup
shutil.rmtree(DB, ignore_errors=True)