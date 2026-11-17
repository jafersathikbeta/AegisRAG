"""NetworkGuard and cryptographic audit ledger for AegisRAG."""

from __future__ import annotations

import hashlib
import json
import socket
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

AUDIT_LOG_PATH = Path(".aegis_audit.log")
_GENESIS_HASH = "0" * 64
_BLOCK_MSG = "Outbound network blocked by AegisRAG"


def _canonical_payload(record: dict[str, Any]) -> str:
    """Stable JSON for hashing (excludes current_hash)."""
    payload = {k: v for k, v in record.items() if k != "current_hash"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_hash() -> str:
    """Return the current_hash of the last ledger entry, or genesis."""
    if not AUDIT_LOG_PATH.exists():
        return _GENESIS_HASH
    last = _GENESIS_HASH
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            last = ev.get("current_hash") or last
    return last


def audit_log(event_type: str, detail: str, **extra: Any) -> dict[str, Any]:
    """
    Append a chained JSON audit event to .aegis_audit.log.

    Each entry stores ``prev_hash`` (SHA-256 of the prior entry) and
    ``current_hash`` = SHA-256(canonical content including prev_hash).
    """
    prev = _last_hash()
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "detail": detail,
        **extra,
        "prev_hash": prev,
    }
    record["current_hash"] = _sha256(_canonical_payload(record))
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def verify_ledger(path: Path | None = None) -> bool:
    """
    Walk the audit log and verify the SHA-256 hash chain.

    Returns False if any entry is missing hashes, has a broken prev link,
    or if recomputed current_hash does not match (tamper detection).
    """
    log_path = path or AUDIT_LOG_PATH
    if not log_path.exists():
        return True

    expected_prev = _GENESIS_HASH
    with log_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return False

            prev = record.get("prev_hash")
            stored = record.get("current_hash")
            if not prev or not stored:
                return False
            if prev != expected_prev:
                return False

            recomputed = _sha256(_canonical_payload(record))
            if recomputed != stored:
                return False

            expected_prev = stored
    return True


def read_audit_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Read audit events from the log file (oldest first)."""
    if not AUDIT_LOG_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        return events[-limit:]
    return events


def audit_metrics() -> dict[str, int]:
    """Aggregate PII redaction, network-block, and injection-block counts."""
    events = read_audit_events()
    pii_total = 0
    network_blocks = 0
    injection_blocks = 0
    for ev in events:
        et = ev.get("event_type", "")
        if et == "pii_redacted":
            pii_total += int(ev.get("count", 0) or 0)
        elif et == "network_blocked":
            network_blocks += 1
        elif et == "malicious_injection_blocked":
            injection_blocks += 1
    return {
        "total_pii_redacted": pii_total,
        "total_network_blocks": network_blocks,
        "total_injection_blocks": injection_blocks,
    }


class NetworkGuard:
    """
    Context manager that blocks outbound network connections.

    Monkeypatches socket.socket.connect, socket.create_connection, and
    urllib.request.urlopen. Also intercepts requests.Session.request so
    HTTP clients raise PermissionError directly (urllib3 otherwise wraps
    PermissionError as ConnectionError because it subclasses OSError).
    """

    def __init__(self) -> None:
        self._orig_socket_connect: Callable | None = None
        self._orig_create_connection: Callable | None = None
        self._orig_urlopen: Callable | None = None
        self._orig_session_request: Callable | None = None
        self._requests_sessions = None

    def _block(self, detail: str, **extra: Any) -> None:
        audit_log("network_blocked", detail, **extra)
        raise PermissionError(_BLOCK_MSG)

    def __enter__(self) -> "NetworkGuard":
        self._orig_socket_connect = socket.socket.connect
        self._orig_create_connection = socket.create_connection
        self._orig_urlopen = urllib.request.urlopen

        guard = self

        def _blocked_connect(sock_self, address):  # noqa: ANN001
            guard._block(f"socket.connect blocked to {address}", address=str(address))

        def _blocked_create_connection(address, *args, **kwargs):  # noqa: ANN001
            guard._block(
                f"create_connection blocked to {address}",
                address=str(address),
            )

        def _blocked_urlopen(url, *args, **kwargs):  # noqa: ANN001
            guard._block(f"urlopen blocked for {url}", url=str(url))

        socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
        socket.create_connection = _blocked_create_connection  # type: ignore[assignment]
        urllib.request.urlopen = _blocked_urlopen  # type: ignore[assignment]

        try:
            import requests.sessions as req_sessions

            self._requests_sessions = req_sessions
            self._orig_session_request = req_sessions.Session.request

            def _blocked_session_request(session_self, method, url, *args, **kwargs):  # noqa: ANN001
                guard._block(f"requests blocked for {method} {url}", url=str(url))

            req_sessions.Session.request = _blocked_session_request  # type: ignore[method-assign]
        except ImportError:
            self._requests_sessions = None
            self._orig_session_request = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._orig_socket_connect is not None:
            socket.socket.connect = self._orig_socket_connect  # type: ignore[method-assign]
        if self._orig_create_connection is not None:
            socket.create_connection = self._orig_create_connection  # type: ignore[assignment]
        if self._orig_urlopen is not None:
            urllib.request.urlopen = self._orig_urlopen  # type: ignore[assignment]
        if self._requests_sessions is not None and self._orig_session_request is not None:
            self._requests_sessions.Session.request = self._orig_session_request  # type: ignore[method-assign]
