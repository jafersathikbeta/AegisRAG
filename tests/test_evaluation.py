"""Evaluation suite: dynamic retrieval + extractive QA keyword checks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import get_qa_pipeline, run_dynamic_evaluation  # noqa: E402
from ingestion import ingest_text  # noqa: E402
from vector_store import VectorStore  # noqa: E402


def test_dynamic_evaluation_keyword_match():
    """User-supplied cases against an ingested corpus; keywords must hit answers."""
    corpus = (
        "AegisRAG is an offline retrieval-augmented generation system. "
        "It uses sentence-transformers for embeddings and FAISS for vector search. "
        "The extractive QA model is deepset/roberta-base-squad2. "
        "NetworkGuard blocks outbound network connections to keep data local."
    )
    cases = [
        {
            "question": "What embedding library does AegisRAG use?",
            "expected_keywords": ["sentence-transformers", "embeddings"],
        },
        {
            "question": "Which model is used for extractive QA?",
            "expected_keywords": ["roberta", "squad2"],
        },
        {
            "question": "What does NetworkGuard do?",
            "expected_keywords": ["network", "block"],
        },
    ]

    vs = VectorStore()
    vs.add_documents(ingest_text(corpus, source="test_corpus"))
    qa = get_qa_pipeline()
    report = run_dynamic_evaluation(cases, vs, qa)

    assert report["total"] == 3
    assert report["passed"] == 3
    assert report["accuracy"] == 1.0
    for case_result in report["results"]:
        assert case_result["passed"] is True
        assert len(case_result["matched_keywords"]) >= 1


def test_dynamic_evaluation_empty_cases():
    vs = VectorStore()
    qa = get_qa_pipeline()
    report = run_dynamic_evaluation([], vs, qa)
    assert report["total"] == 0
    assert report["accuracy"] == 0.0
    assert "error" in report
