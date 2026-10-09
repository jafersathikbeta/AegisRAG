# AegisRAG

Production-grade, fully offline RAG system with NetworkGuard, PII redaction, FAISS retrieval, and extractive QA (`deepset/roberta-base-squad2`).

## Features

- **Offline embeddings** via `sentence-transformers` (`all-MiniLM-L6-v2`)
- **FAISS** vector search (`faiss-cpu`)
- **Extractive QA** — answers only from retrieved chunks
- **NetworkGuard** — blocks outbound sockets / `urlopen` at runtime
- **PII redaction** — Email, Phone (US/IN), SSN, Credit Card, IPv4, Aadhaar, PAN
- **Gradio** dark enterprise dashboard (Chat · Ingestion · Audit · Evaluation)

## Setup (Windows)

```powershell
cd aegis-rag
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.15 note:** `faiss-cpu` has no cp315 wheel yet. AegisRAG falls back to a NumPy `IndexFlatIP` shim (`src/faiss_compat.py`) with the same API. On Python &lt; 3.15, native `faiss-cpu` is used.

First run downloads Hugging Face models (local cache thereafter). No API keys required.

## Run

```powershell
python src\app.py
```

Open **http://127.0.0.1:7860**

## Tests

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

## Configuration

See `.env.example`:

| Variable     | Default                         |
|--------------|---------------------------------|
| `CHUNK_SIZE` | `500`                           |
| `TOP_K`      | `3`                             |
| `EMBED_MODEL`| `all-MiniLM-L6-v2`              |
| `QA_MODEL`   | `deepset/roberta-base-squad2`   |

## Security

- Audit trail: `.aegis_audit.log` (JSON lines)
- Wrap sensitive operations in `NetworkGuard` to raise `PermissionError("Outbound network blocked by AegisRAG")`
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
