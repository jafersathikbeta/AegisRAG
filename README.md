# AegisRAG

A production-grade, fully offline Retrieval-Augmented Generation (RAG) desktop application. AegisRAG prioritizes data privacy and security by enforcing a strict zero-network policy, automated PII redaction, and local-only AI models.

## Core Architecture

- **Desktop App:** Wrapped in a native desktop window using `pywebview`.
- **Backend:** High-performance `FastAPI` server running locally.
- **Vector Search:** `FAISS` (with a custom NumPy fallback for Python 3.15 compatibility).
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Extractive QA:** `deepset/roberta-base-squad2` (answers are generated strictly from retrieved local document chunks).

## Security & Privacy Features

- **NetworkGuard:** A custom runtime interceptor that blocks outbound sockets and `urlopen` calls, preventing data exfiltration.
- **PII Redaction:** Automatically detects and masks Emails, Phone Numbers (US/IN), SSNs, Credit Cards, IPv4 addresses, Aadhaar, and PAN.
- **Audit Ledger:** Generates a JSON-lines audit trail (`.aegis_audit.log`) for all queries, ingestions, and security blocks. Entries are SHA-256 hash-chained, so any edit or deletion is detectable on verification.
- **Injection Protection:** Blocks malicious document injections during the ingestion pipeline.

## Multimodal Ingestion

Unlike standard text-only RAG systems, AegisRAG extracts text from multiple media types locally:
- **Documents:** PDF (`pypdf`)
- **Images:** OCR via `pytesseract` and `opencv-python`
- **Audio:** Transcription via `faster-whisper`
- **Video:** Frame extraction via `moviepy`

## Setup & Installation (Windows)

1. Clone the repository and create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Note on Python 3.15:** Native `faiss-cpu` wheels are not yet available. AegisRAG includes a custom NumPy `IndexFlatIP` shim (`src/faiss_compat.py`) that mimics the FAISS API perfectly.

3. Run the application:
   ```powershell
   python run.py
   ```
   *This starts the FastAPI server and opens the native desktop window.*

## Configuration

Environment variables (see `.env.example`):

| Variable | Default |
|----------|---------|
| `CHUNK_SIZE` | `500` |
| `TOP_K` | `3` |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` |
| `QA_MODEL` | `deepset/roberta-base-squad2` |

## Testing

```powershell
$env:PYTHONPATH = "src"
pytest -q
```
