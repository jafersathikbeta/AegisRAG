"""Multimodal document ingestion: load, scan, PII-boundary chunk, redact."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from config import settings
from pii import find_pii_spans, redact_pii, scan_for_injection
from security import audit_log

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".mp4",
    ".avi",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".avi"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

ProgressCallback = Callable[[float, str], None]

_WHISPER_MODEL = None
_TESSERACT_CONFIGURED = False


class InjectionBlockedError(ValueError):
    """Raised when OCR/transcript content matches a prompt-injection pattern."""


def configure_tesseract(tesseract_cmd: str | Path | None = None) -> None:
    """Point pytesseract at the local Tesseract binary (Windows default)."""
    global _TESSERACT_CONFIGURED
    import pytesseract

    cmd = Path(tesseract_cmd) if tesseract_cmd else TESSERACT_CMD
    if cmd.exists():
        pytesseract.pytesseract.tesseract_cmd = str(cmd)
    _TESSERACT_CONFIGURED = True


def get_whisper_model():
    """Lazy-load faster-whisper 'base' on CPU (int8). Call before NetworkGuard."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def warm_multimodal_models() -> None:
    """Configure Tesseract + load Whisper before outbound network is blocked."""
    configure_tesseract()
    get_whisper_model()


def _noop_progress(frac: float, msg: str) -> None:
    return None


def validate_file_size(path: str | Path, max_bytes: int = MAX_FILE_BYTES) -> None:
    """Raise ValueError if file exceeds the configured size limit."""
    p = Path(path)
    size = p.stat().st_size
    if size > max_bytes:
        mb = size / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(
            f"File '{p.name}' is {mb:.1f} MB — exceeds the {limit_mb:.0f} MB limit."
        )


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _ocr_image(image) -> str:
    import pytesseract

    if not _TESSERACT_CONFIGURED:
        configure_tesseract()
    return (pytesseract.image_to_string(image) or "").strip()


def load_image_file(
    path: str | Path,
    progress: ProgressCallback | None = None,
) -> str:
    from PIL import Image

    prog = progress or _noop_progress
    prog(0.2, "Running OCR on image (Tesseract)…")
    with Image.open(path) as img:
        text = _ocr_image(img.convert("RGB"))
    prog(0.9, "OCR complete")
    return text


def _extract_video_audio(path: Path, wav_path: Path) -> bool:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip  # type: ignore
        except ImportError:
            return False

    clip = None
    try:
        clip = VideoFileClip(str(path))
        if clip.audio is None:
            return False
        clip.audio.write_audiofile(
            str(wav_path),
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            logger=None,
        )
        return wav_path.exists() and wav_path.stat().st_size > 0
    except Exception:
        return False
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def load_video_file(
    path: str | Path,
    progress: ProgressCallback | None = None,
    frame_interval_sec: float = 5.0,
) -> str:
    import cv2
    from PIL import Image

    prog = progress or _noop_progress
    p = Path(path)
    sections: list[str] = []

    prog(0.05, "Opening video for frame OCR…")
    if not _TESSERACT_CONFIGURED:
        configure_tesseract()

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {p.name}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    interval_frames = max(1, int(fps * frame_interval_sec))

    ocr_parts: list[str] = []
    idx = 0
    sampled = 0
    est_samples = max(1, int(duration / frame_interval_sec) if duration else 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            block = _ocr_image(Image.fromarray(rgb)).strip()
            if block:
                ts = idx / fps
                ocr_parts.append(f"[frame @ {ts:.1f}s] {block}")
            sampled += 1
            frac = 0.05 + 0.45 * min(1.0, sampled / est_samples)
            prog(frac, f"OCR frame {sampled}/{est_samples}…")
        idx += 1

    cap.release()
    if ocr_parts:
        sections.append("### Video Frame OCR\n" + "\n".join(ocr_parts))

    prog(0.55, "Extracting audio for Whisper…")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        if _extract_video_audio(p, wav_path):
            prog(0.65, "Transcribing audio with faster-whisper (CPU, base/int8)…")
            model = get_whisper_model()
            segments, _info = model.transcribe(str(wav_path))
            transcript = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
            if transcript:
                sections.append("### Video Audio Transcript\n" + transcript)
            prog(0.95, "Transcription complete")
        else:
            prog(0.9, "No audio track found — OCR only")

    if not sections:
        return ""
    return "\n\n".join(sections)


def load_document(
    path: str | Path,
    progress: ProgressCallback | None = None,
) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if suffix == ".txt":
        return load_text_file(p)
    if suffix == ".pdf":
        return load_pdf_file(p)
    if suffix in IMAGE_EXTENSIONS:
        return load_image_file(p, progress=progress)
    if suffix in VIDEO_EXTENSIONS:
        return load_video_file(p, progress=progress)
    raise ValueError(f"Unsupported file type: {suffix}")


def _span_covering(spans: list, offset: int):
    """Return the PII span that contains ``offset`` (exclusive end), else None."""
    for span in spans:
        if span.start < offset < span.end:
            return span
    return None


def chunk_text(text: str, chunk_size: int | None = None) -> list[str]:
    """
    Split text into overlapping chunks with PII-boundary awareness.

    If a prospective split would cut through a PII entity, the boundary is
    extended to fully encompass that entity before the chunk is emitted and
    redacted (callers should still redact; this helper preserves whole spans).
    """
    size = chunk_size or settings.CHUNK_SIZE
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    spans = find_pii_spans(text)
    chunks: list[str] = []
    overlap = max(20, size // 10)
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        covering = _span_covering(spans, end)
        if covering is not None and end < n:
            # Extend past the full PII entity so it is never split
            end = min(covering.end, n)
        # Also pull in any PII that starts before end but ends after end
        for span in spans:
            if start < span.end <= end:
                continue
            if span.start < end < span.end:
                end = min(span.end, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        step = max(end - overlap, start + 1)
        covering_start = _span_covering(spans, step)
        if covering_start is not None and covering_start.start >= start:
            # Prefer not to start mid-entity; jump to entity start only if progress
            step = covering_start.start if covering_start.start > start else end
        start = step if step > start else end
    return chunks


def _guard_media_text(text: str, source: str, media: str) -> None:
    """Scan OCR/transcript for prompt injection; raise + ledger on hit."""
    if scan_for_injection(text):
        audit_log(
            "malicious_injection_blocked",
            f"Blocked prompt injection in {media} extract from {source}",
            source=source,
            media=media,
        )
        raise InjectionBlockedError(
            f"Malicious prompt injection detected in {media} content from '{source}'. "
            "Ingestion rejected."
        )


def _chunk_and_redact(raw: str, chunk_size: int | None = None) -> list[str]:
    """
    PII-boundary chunk on raw text, then redact each chunk.

    Extending boundaries around PII before redaction guarantees entities are
    never split across chunks (100% redaction coverage).
    """
    raw_chunks = chunk_text(raw, chunk_size=chunk_size)
    return [redact_pii(c, log=False).text for c in raw_chunks if c.strip()]


def ingest_document(
    path: str | Path,
    chunk_size: int | None = None,
    progress: ProgressCallback | None = None,
    enforce_size_limit: bool = True,
) -> list[dict]:
    """
    Load a multimodal document, scan media extracts for injection,
    PII-boundary-chunk, redact, and return chunk records.
    """
    prog = progress or _noop_progress
    p = Path(path)
    if enforce_size_limit:
        validate_file_size(p)

    suffix = p.suffix.lower()
    prog(0.02, f"Loading {p.name}…")
    raw = load_document(p, progress=prog)

    if suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
        prog(0.93, "Scanning for prompt injection…")
        media = "image" if suffix in IMAGE_EXTENSIONS else "video"
        _guard_media_text(raw, p.name, media)

    prog(0.96, "PII-boundary chunking + redaction…")
    chunks = _chunk_and_redact(raw, chunk_size=chunk_size)
    prog(1.0, f"Ready — {len(chunks)} chunk(s)")
    return [
        {"text": c, "source": p.name, "chunk_id": i}
        for i, c in enumerate(chunks)
    ]


def ingest_text(
    text: str,
    source: str = "inline",
    chunk_size: int | None = None,
) -> list[dict]:
    """Ingest raw text with PII-boundary chunking then redaction."""
    chunks = _chunk_and_redact(text, chunk_size=chunk_size)
    return [
        {"text": c, "source": source, "chunk_id": i}
        for i, c in enumerate(chunks)
    ]
