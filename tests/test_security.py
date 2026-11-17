"""NetworkGuard + cryptographic ledger tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import security as security_mod  # noqa: E402
from security import NetworkGuard, audit_log, verify_ledger  # noqa: E402


def test_network_guard_blocks_requests_get():
    with NetworkGuard():
        with pytest.raises(PermissionError, match="Outbound network blocked by AegisRAG"):
            requests.get("http://google.com", timeout=2)


def test_network_guard_restores_after_exit():
    with NetworkGuard():
        with pytest.raises(PermissionError):
            requests.get("http://example.com", timeout=2)
    assert True


def test_audit_ledger_hash_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / ".aegis_audit.log"
    monkeypatch.setattr(security_mod, "AUDIT_LOG_PATH", log_path)

    r1 = audit_log("test_event", "first")
    r2 = audit_log("test_event", "second")

    assert r1["prev_hash"] == "0" * 64
    assert r1["current_hash"]
    assert r2["prev_hash"] == r1["current_hash"]
    assert verify_ledger(log_path) is True


def test_audit_ledger_detects_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / ".aegis_audit.log"
    monkeypatch.setattr(security_mod, "AUDIT_LOG_PATH", log_path)

    audit_log("test_event", "legit")
    audit_log("test_event", "also legit")
    assert verify_ledger(log_path) is True

    # Manually edit the first entry's detail without updating hashes
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(lines[0])
    first["detail"] = "TAMPERED"
    lines[0] = json.dumps(first)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert verify_ledger(log_path) is False
