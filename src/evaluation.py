"""Retrieval + extractive QA evaluation suite."""

from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from config import settings
from pii import redact_pii
from vector_store import VectorStore, store as global_store

_QA_PIPELINE = None


class ExtractiveQAPipeline:
    """
    Extractive QA compatible with transformers.pipeline('question-answering').

    Transformers v5 removed the built-in question-answering task; this wrapper
    uses AutoModelForQuestionAnswering on deepset/roberta-base-squad2.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, question: str, context: str) -> dict[str, Any]:
        if not context or not context.strip():
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))
        if end_idx < start_idx:
            end_idx = start_idx

        start_score = float(torch.softmax(start_logits, dim=-1)[start_idx])
        end_score = float(torch.softmax(end_logits, dim=-1)[end_idx])
        score = (start_score + end_score) / 2.0

        input_ids = inputs["input_ids"][0]
        answer_ids = input_ids[start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "score": score,
            "start": start_idx,
            "end": end_idx,
        }


def get_qa_pipeline():
    """
    Lazy-load extractive QA.

    Prefers transformers.pipeline('question-answering') when available (v4);
    falls back to ExtractiveQAPipeline on transformers v5+.
    """
    global _QA_PIPELINE
    if _QA_PIPELINE is not None:
        return _QA_PIPELINE

    try:
        from transformers import pipeline

        _QA_PIPELINE = pipeline(
            "question-answering",
            model=settings.QA_MODEL,
            tokenizer=settings.QA_MODEL,
        )
    except (KeyError, ValueError):
        _QA_PIPELINE = ExtractiveQAPipeline(settings.QA_MODEL)
    return _QA_PIPELINE


def answer_question(
    question: str,
    vector_store: VectorStore | None = None,
    top_k: int | None = None,
    qa_pipeline: Callable | None = None,
) -> dict[str, Any]:
    """
    Redact PII from the query, retrieve top-k chunks, run extractive QA.

    Returns answer text, score, context, and cited sources.
    """
    vs = vector_store if vector_store is not None else global_store
    k = top_k or settings.TOP_K
    qa = qa_pipeline if qa_pipeline is not None else get_qa_pipeline()

    clean_q = redact_pii(question).text
    hits = vs.search(clean_q, top_k=k)
    if not hits:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "score": 0.0,
            "sources": [],
            "context": "",
            "query": clean_q,
        }

    context = "\n\n".join(h["text"] for h in hits)
    result = qa(question=clean_q, context=context)
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    score = float(result.get("score", 0.0)) if isinstance(result, dict) else 0.0

    sources = [
        {
            "source": h.get("source", "unknown"),
            "chunk_id": h.get("chunk_id"),
            "score": h.get("score"),
            "excerpt": h.get("text", "")[:200],
        }
        for h in hits
    ]

    return {
        "answer": answer,
        "score": score,
        "sources": sources,
        "context": context,
        "query": clean_q,
    }


def _normalize_keywords(raw: Any) -> list[str]:
    """Accept a list or comma-separated string of keywords."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [k.strip() for k in raw.split(",") if k.strip()]
    return [str(k).strip() for k in raw if str(k).strip()]


def run_dynamic_evaluation(
    test_cases: list[dict[str, Any]],
    vector_store: VectorStore,
    qa_pipeline: Callable,
) -> dict[str, Any]:
    """
    Run user-supplied evaluation cases against an existing index.

    Each case: ``{"question": "...", "expected_keywords": ["..."] | "a, b"}``.
    A case passes when at least one expected keyword appears in the QA answer.
    """
    if not test_cases:
        return {
            "results": [],
            "passed": 0,
            "total": 0,
            "accuracy": 0.0,
            "error": "No test cases provided.",
        }

    results: list[dict[str, Any]] = []
    passed = 0

    for case in test_cases:
        question = (case.get("question") or "").strip()
        keywords = _normalize_keywords(case.get("expected_keywords"))
        if not question:
            continue

        out = answer_question(
            question,
            vector_store=vector_store,
            qa_pipeline=qa_pipeline,
        )
        answer = out.get("answer") or ""
        answer_l = answer.lower()
        matched = [kw for kw in keywords if kw.lower() in answer_l]
        ok = len(matched) > 0 if keywords else False
        if ok:
            passed += 1

        results.append(
            {
                "question": question,
                "answer": answer,
                "score": out.get("score", 0.0),
                "expected_keywords": keywords,
                "matched_keywords": matched,
                "passed": ok,
                "sources": out.get("sources") or [],
            }
        )

    total = len(results)
    return {
        "results": results,
        "passed": passed,
        "total": total,
        "accuracy": (passed / total) if total else 0.0,
    }
