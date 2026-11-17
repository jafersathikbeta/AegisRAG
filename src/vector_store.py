"""FAISS-backed vector store with local sentence-transformers embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import faiss_compat as faiss

from config import settings

_MODEL = None


def get_embed_model():
    """Lazy-load the sentence-transformers embedding model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(settings.EMBED_MODEL)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into L2-normalized float32 vectors."""
    model = get_embed_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


class VectorStore:
    """In-memory FAISS index with metadata for retrieved chunks."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict[str, Any]] = []
        self.dim: int | None = None

    def add_documents(self, docs: list[dict[str, Any]]) -> int:
        """Embed and add document chunks. Returns number added."""
        if not docs:
            return 0
        texts = [d["text"] for d in docs]
        vectors = embed_texts(texts)
        if self.index is None:
            self.dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.documents.extend(docs)
        return len(docs)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return top-k most similar chunks for the query."""
        k = top_k or settings.TOP_K
        if self.index is None or self.index.ntotal == 0:
            return []
        q = embed_texts([query])
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = dict(self.documents[int(idx)])
            doc["score"] = float(score)
            results.append(doc)
        return results

    def clear(self) -> None:
        """Reset the index."""
        self.index = None
        self.documents = []
        self.dim = None

    def save(self, path: str | Path) -> None:
        """Persist FAISS index and document metadata."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        with path.with_suffix(".json").open("w", encoding="utf-8") as fh:
            json.dump(self.documents, fh)

    def load(self, path: str | Path) -> None:
        """Load FAISS index and document metadata."""
        import json

        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".json")
        if faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
            self.dim = self.index.d
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                self.documents = json.load(fh)


# Shared singleton used by the app and evaluation suite
store = VectorStore()
