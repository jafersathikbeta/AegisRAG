"""Multimodal ingestion guards and PII-before-chunk path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion import (  # noqa: E402
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    chunk_text,
    ingest_text,
    validate_file_size,
)
from pii import redact_pii  # noqa: E402


def test_supported_extensions_include_media():
    for ext in (".txt", ".pdf", ".png", ".jpg", ".jpeg", ".mp4", ".avi"):
        assert ext in SUPPORTED_EXTENSIONS


def test_validate_file_size_limit(tmp_path: Path):
    small = tmp_path / "ok.txt"
    small.write_text("hello", encoding="utf-8")
    validate_file_size(small)

    big = tmp_path / "big.bin"
    # Sparse-ish write without allocating full 50MB+ in RAM twice
    with big.open("wb") as fh:
        fh.seek(MAX_FILE_BYTES)
        fh.write(b"x")
    with pytest.raises(ValueError, match="exceeds"):
        validate_file_size(big)


def test_pii_applied_before_chunk_tokens():
    raw = "Contact secret@example.com about case."
    docs = ingest_text(raw, source="unit")
    joined = " ".join(d["text"] for d in docs)
    assert "secret@example.com" not in joined
    assert "[EMAIL_REDACTED]" in joined


def test_redact_then_chunk_order():
    """Simulate multimodal extract → redact → chunk pipeline order."""
    extracted = "SSN 123-45-6789 appears in the slide deck."
    redacted = redact_pii(extracted, log=False).text
    assert "123-45-6789" not in redacted
    chunks = chunk_text(redacted, chunk_size=40)
    assert chunks
    assert all("123-45-6789" not in c for c in chunks)
