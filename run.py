"""Launch AegisRAG as a native desktop app (FastAPI + pywebview)."""

from __future__ import annotations

import sys
import threading
import time

import uvicorn
import webview

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"

_server: uvicorn.Server | None = None


def _run_server() -> None:
    """Run uvicorn in this thread (no reload — incompatible with embedding)."""
    global _server
    config = uvicorn.Config(
        "src.api:app",
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
    )
    _server = uvicorn.Server(config)
    _server.run()


def _wait_until_ready(timeout: float = 180.0) -> None:
    """Block until the FastAPI app accepts connections (models may warm first)."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/metrics", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 — retry until ready
            last_err = exc
            time.sleep(0.4)
    raise RuntimeError(f"AegisRAG server did not become ready: {last_err}")


def _shutdown() -> None:
    """Stop uvicorn and exit the process cleanly after the window closes."""
    global _server
    if _server is not None:
        _server.should_exit = True
    # Hard-exit so daemon threads / model workers cannot keep the process alive
    time.sleep(0.2)
    sys.exit(0)


if __name__ == "__main__":
    server_thread = threading.Thread(target=_run_server, name="uvicorn", daemon=True)
    server_thread.start()

    print(f"AegisRAG: waiting for {URL} …", flush=True)
    _wait_until_ready()
    print("AegisRAG: opening desktop window", flush=True)

    webview.create_window(
        "AegisRAG - Enterprise RAG",
        URL,
        width=1280,
        height=800,
    )
    # Blocks until the native window is closed
    webview.start()
    _shutdown()
