"""Pre-cache Hugging Face models used by AegisRAG (run during install)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing project `src` modules when run from installer
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    print("Downloading / caching sentence-transformers embeddings…", flush=True)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("all-MiniLM-L6-v2")
    print("Embeddings cached.", flush=True)

    print("Downloading / caching deepset/roberta-base-squad2…", flush=True)
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    name = "deepset/roberta-base-squad2"
    AutoTokenizer.from_pretrained(name)
    AutoModelForQuestionAnswering.from_pretrained(name)
    print("QA model cached.", flush=True)
    print("Model pre-download complete.", flush=True)


if __name__ == "__main__":
    main()
