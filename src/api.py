"""AegisRAG FastAPI server — chat, ingest, metrics + static UI."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure `src/` is importable the same way as the existing modules
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from evaluation import answer_question, get_qa_pipeline
from ingestion import (
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    InjectionBlockedError,
    configure_tesseract,
    ingest_document,
    warm_multimodal_models,
)
from security import NetworkGuard, audit_log, audit_metrics, verify_ledger
from vector_store import get_embed_model, store

_STATIC = _ROOT / "static"
_UPLOAD_DIR = _ROOT / "data" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm local models before any NetworkGuard-protected requests."""
    print(f"AegisRAG: Tesseract → {_TESSERACT_EXE} (exists={_TESSERACT_EXE.exists()})", flush=True)
    configure_tesseract(_TESSERACT_EXE)
    print("AegisRAG: warming embedding model…", flush=True)
    get_embed_model()
    print("AegisRAG: warming QA pipeline…", flush=True)
    get_qa_pipeline()
    print("AegisRAG: warming faster-whisper (CPU/int8)…", flush=True)
    warm_multimodal_models()
    print("AegisRAG: ready", flush=True)
    yield


app = FastAPI(title="AegisRAG", version="2.0.0", lifespan=lifespan)

if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: str
    pii_redacted: int
    network_blocked: bool


@app.get("/")
async def index():
    index_path = _STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html missing")
    return FileResponse(index_path)


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    pii_before = int(audit_metrics().get("total_pii_redacted", 0) or 0)

    if not store.documents:
        return ChatResponse(
            answer=(
                "I don't have any documents yet. Upload a file from the sidebar, "
                "then ask again."
            ),
            sources="",
            pii_redacted=0,
            network_blocked=True,
        )

    with NetworkGuard():
        result = answer_question(message)
        audit_log("chat_query", "Processed secure chat query", query_len=len(message))

    pii_after = int(audit_metrics().get("total_pii_redacted", 0) or 0)
    pii_delta = max(0, pii_after - pii_before)
    sources = result.get("sources") or []
    names = sorted({str(s.get("source") or "unknown") for s in sources})
    sources_str = ", ".join(names)

    return ChatResponse(
        answer=result.get("answer") or "",
        sources=sources_str,
        pii_redacted=pii_delta,
        network_blocked=True,
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        mb = len(data) / (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File is {mb:.1f} MB — max 50 MB")

    dest = _UPLOAD_DIR / Path(file.filename).name
    dest.write_bytes(data)

    try:
        with NetworkGuard():
            docs = ingest_document(dest)
            if not docs:
                raise HTTPException(
                    status_code=422,
                    detail=f"No extractable text in '{dest.name}'",
                )
            n = store.add_documents(docs)
            audit_log(
                "ingestion",
                f"Ingested {dest.name}",
                chunks=n,
                source=dest.name,
                media=suffix,
            )
    except InjectionBlockedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {
        "ok": True,
        "filename": dest.name,
        "chunks": n,
        "index_size": len(store.documents),
        "message": f"Ingested {dest.name} · {n} chunk(s)",
    }


@app.get("/metrics")
async def metrics():
    m = audit_metrics()
    return {
        "total_pii_redacted": m.get("total_pii_redacted", 0),
        "total_network_blocks": m.get("total_network_blocks", 0),
        "total_injection_blocks": m.get("total_injection_blocks", 0),
        "indexed_chunks": len(store.documents),
        "ledger_valid": verify_ledger(),
    }


@app.post("/clear")
async def clear_db():
    store.clear()
    audit_log("clear_database", "Knowledge base cleared")
    return {"ok": True, "indexed_chunks": 0}
