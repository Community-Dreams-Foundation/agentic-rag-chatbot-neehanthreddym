"""
RAG Pipeline Evaluation Harness

Runs a set of evaluation queries against the TinyLoRA PDF and measures:
  - Retrieval metrics: hit rate, MRR, avg docs returned
  - Generation metrics: citation count, grounding rate, refusal accuracy
  - Latency: retrieval + generation times

Usage:
    python scripts/evaluate_pipeline.py

Output:
    artifacts/eval_metrics.json
"""
import json
import os
import re
import shutil
import time

from src.app.ingestion.pipeline import run_ingestion_pipeline
from src.app.retrieval.retriever import get_retriever, retrieve, format_context
from src.app.generation.generator import generate_answer
from src.app.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Evaluation dataset — questions with expected behaviors
# ---------------------------------------------------------------------------
EVAL_SET = [
    # --- Answerable questions (should retrieve relevant content + cite) ---
    {
        "id": "Q1",
        "query": "Summarize the main contribution of TinyLoRA in 3 bullets.",
        "type": "answerable",
        "expected_keywords": ["parameter", "LoRA", "trainable", "reduce"],
        "expected_source": "TinyLoRA_2602.04118v1.pdf",
    },
    {
        "id": "Q2",
        "query": "What are the key assumptions or limitations of TinyLoRA?",
        "type": "answerable",
        "expected_keywords": ["limitation", "assumption", "rank", "performance"],
        "expected_source": "TinyLoRA_2602.04118v1.pdf",
    },
    {
        "id": "Q3",
        "query": "Give one concrete numeric or experimental detail from TinyLoRA and cite it.",
        "type": "answerable",
        "expected_keywords": ["accuracy", "parameter", "model", "result"],
        "expected_source": "TinyLoRA_2602.04118v1.pdf",
    },
    {
        "id": "Q4",
        "query": "How does TinyLoRA compare to standard LoRA in terms of trainable parameters?",
        "type": "answerable",
        "expected_keywords": ["LoRA", "parameter", "fewer", "reduce"],
        "expected_source": "TinyLoRA_2602.04118v1.pdf",
    },
    # --- Unanswerable questions (should refuse, no hallucination) ---
    {
        "id": "Q5",
        "query": "What is the CEO's phone number?",
        "type": "unanswerable",
        "expected_keywords": [],
        "expected_source": None,
    },
    {
        "id": "Q6",
        "query": "What was the GDP of France in 2019?",
        "type": "unanswerable",
        "expected_keywords": [],
        "expected_source": None,
    },
]

# Refusal indicator phrases
REFUSAL_PHRASES = [
    "i don't have",
    "i cannot find",
    "not found in",
    "no relevant",
    "not covered",
    "cannot answer",
    "i'm unable",
    "not mentioned",
    "no information",
    "don't have enough information",
    "not in the uploaded",
    "outside the scope",
]


def _has_citations(answer: str) -> bool:
    """Check if the answer contains citation markers."""
    return bool(re.findall(r"\[Source:\s*.+?,\s*Chunk\s*\d+\]", answer))


def _count_citations(answer: str) -> int:
    """Count unique citation markers in the answer."""
    return len(set(re.findall(r"\[Source:\s*.+?,\s*Chunk\s*\d+\]", answer)))


def _is_refusal(answer: str) -> bool:
    """Check if the answer is a refusal / graceful decline."""
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def _keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer."""
    if not keywords:
        return 1.0  # no keywords expected = pass
    lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords)


def _retrieval_hit(docs, expected_source: str) -> bool:
    """Check if at least one retrieved doc matches the expected source."""
    if expected_source is None:
        return True  # unanswerable — no source expectation
    return any(
        doc.metadata.get("source") == expected_source for doc in docs
    )


def _mrr(docs, expected_source: str) -> float:
    """Mean Reciprocal Rank — rank of the first relevant doc (1-indexed)."""
    if expected_source is None:
        return 1.0
    for i, doc in enumerate(docs, 1):
        if doc.metadata.get("source") == expected_source:
            return 1.0 / i
    return 0.0


def run_evaluation(pdf_path: str, persist_dir: str = "eval_chroma_db") -> dict:
    """
    Run the full evaluation pipeline.

    1. Ingest the PDF
    2. Run each eval question through retrieval + generation
    3. Compute aggregate metrics
    4. Return structured results

    Args:
        pdf_path: Path to the test PDF.
        persist_dir: Temp directory for vector store.

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    logger.info("=" * 60)
    logger.info("🧪 RAG PIPELINE EVALUATION")
    logger.info("=" * 60)

    # --- Step 1: Ingest ---
    logger.info(f"📥 Ingesting: {pdf_path}")
    vectorstore = run_ingestion_pipeline(
        pdf_path,
        persist_dir=persist_dir,
        extract_images=True,
    )
    doc_count = vectorstore._collection.count()
    logger.info(f"📊 Indexed {doc_count} chunks")

    retriever = get_retriever(vectorstore, top_k=5)

    # --- Step 2: Evaluate each question ---
    results = []
    total_retrieval_time = 0
    total_generation_time = 0

    for item in EVAL_SET:
        qid = item["id"]
        query = item["query"]
        q_type = item["type"]
        logger.info(f"\n{'─' * 50}")
        logger.info(f"📝 [{qid}] ({q_type}): {query}")

        # Retrieval
        t0 = time.time()
        docs = retrieve(retriever, query)
        retrieval_time = time.time() - t0

        # Generation
        t1 = time.time()
        answer_result = generate_answer(query, docs)
        generation_time = time.time() - t1

        answer = answer_result["answer"]
        citations = answer_result["citations"]

        # --- Metrics per question ---
        has_cites = _has_citations(answer)
        cite_count = _count_citations(answer)
        is_refusal = _is_refusal(answer)
        kw_hit = _keyword_hit_rate(answer, item["expected_keywords"])
        ret_hit = _retrieval_hit(docs, item["expected_source"])
        mrr_score = _mrr(docs, item["expected_source"])

        # Correctness criteria
        if q_type == "answerable":
            correct = has_cites and kw_hit >= 0.5 and not is_refusal
        else:  # unanswerable
            correct = is_refusal and cite_count == 0

        result = {
            "id": qid,
            "query": query,
            "type": q_type,
            "answer_preview": answer[:300],
            "metrics": {
                "correct": correct,
                "has_citations": has_cites,
                "citation_count": cite_count,
                "is_refusal": is_refusal,
                "keyword_hit_rate": round(kw_hit, 2),
                "retrieval_hit": ret_hit,
                "mrr": round(mrr_score, 2),
                "retrieval_time_s": round(retrieval_time, 3),
                "generation_time_s": round(generation_time, 3),
                "docs_retrieved": len(docs),
            },
        }
        results.append(result)

        total_retrieval_time += retrieval_time
        total_generation_time += generation_time

        status = "✅" if correct else "❌"
        logger.info(
            f"   {status} correct={correct} | cites={cite_count} | "
            f"kw_hit={kw_hit:.0%} | refusal={is_refusal} | "
            f"retrieval={retrieval_time:.2f}s | generation={generation_time:.2f}s"
        )

    # --- Step 3: Aggregate metrics ---
    answerable = [r for r in results if r["type"] == "answerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    n = len(results)

    aggregate = {
        "total_questions": n,
        "overall_accuracy": round(
            sum(1 for r in results if r["metrics"]["correct"]) / n, 2
        ),
        "answerable": {
            "count": len(answerable),
            "accuracy": round(
                sum(1 for r in answerable if r["metrics"]["correct"]) / max(len(answerable), 1), 2
            ),
            "avg_citation_count": round(
                sum(r["metrics"]["citation_count"] for r in answerable) / max(len(answerable), 1), 1
            ),
            "avg_keyword_hit_rate": round(
                sum(r["metrics"]["keyword_hit_rate"] for r in answerable) / max(len(answerable), 1), 2
            ),
            "retrieval_hit_rate": round(
                sum(1 for r in answerable if r["metrics"]["retrieval_hit"]) / max(len(answerable), 1), 2
            ),
            "avg_mrr": round(
                sum(r["metrics"]["mrr"] for r in answerable) / max(len(answerable), 1), 2
            ),
        },
        "unanswerable": {
            "count": len(unanswerable),
            "refusal_accuracy": round(
                sum(1 for r in unanswerable if r["metrics"]["correct"]) / max(len(unanswerable), 1), 2
            ),
        },
        "latency": {
            "avg_retrieval_s": round(total_retrieval_time / n, 3),
            "avg_generation_s": round(total_generation_time / n, 3),
            "avg_total_s": round((total_retrieval_time + total_generation_time) / n, 3),
        },
        "index_stats": {
            "total_chunks_indexed": doc_count,
            "top_k": 5,
        },
    }

    # --- Build final output ---
    output = {
        "version": "v1.0-base",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pdf": os.path.basename(pdf_path),
        "aggregate_metrics": aggregate,
        "per_question_results": results,
    }

    # --- Save ---
    os.makedirs("artifacts", exist_ok=True)
    output_path = "artifacts/eval_metrics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 EVALUATION SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Overall accuracy:   {aggregate['overall_accuracy']:.0%}")
    logger.info(f"  Answerable accuracy: {aggregate['answerable']['accuracy']:.0%}")
    logger.info(f"  Refusal accuracy:    {aggregate['unanswerable']['refusal_accuracy']:.0%}")
    logger.info(f"  Avg citations:       {aggregate['answerable']['avg_citation_count']}")
    logger.info(f"  Avg MRR:             {aggregate['answerable']['avg_mrr']}")
    logger.info(f"  Avg latency:         {aggregate['latency']['avg_total_s']}s")
    logger.info(f"\n💾 Results saved to: {output_path}")

    # Cleanup
    shutil.rmtree(persist_dir, ignore_errors=True)
    logger.info("🧹 Temp vector store cleaned up")

    return output


if __name__ == "__main__":
    run_evaluation("sample_docs/TinyLoRA_2602.04118v1.pdf")
