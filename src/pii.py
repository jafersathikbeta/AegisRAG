"""PII detection/redaction and adversarial prompt-injection scanning."""

from __future__ import annotations

import re
from dataclasses import dataclass

from security import audit_log

# Order matters: more specific / longer patterns first where they overlap.
_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "EMAIL",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        "[EMAIL_REDACTED]",
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "[CREDIT_CARD_REDACTED]",
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN_REDACTED]",
    ),
    (
        "AADHAAR",
        re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
        "[AADHAAR_REDACTED]",
    ),
    (
        "PAN",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        "[PAN_REDACTED]",
    ),
    (
        "PHONE",
        re.compile(
            r"(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"
            r"|(?:\+91[-.\s]?)?[6-9]\d{9}\b"
        ),
        "[PHONE_REDACTED]",
    ),
    (
        "IPV4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IPV4_REDACTED]",
    ),
]

# Common jailbreak / prompt-injection phrases
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you|your)\s+(know|instructions?)", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"you\s+are\s+now\b", re.I),
    re.compile(r"\bbypass\b", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"do\s+not\s+follow\s+(your|the)\s+(rules|guidelines|policy)", re.I),
    re.compile(r"override\s+(your|the)\s+(safety|system|rules)", re.I),
    re.compile(r"developer\s+mode", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+have\s+no\s+restrictions|DAN)\b", re.I),
]


@dataclass
class RedactionResult:
    text: str
    count: int
    entities: dict[str, int]


@dataclass
class PIISpan:
    start: int
    end: int
    name: str


def find_pii_spans(text: str) -> list[PIISpan]:
    """Return non-overlapping PII spans sorted by start offset."""
    if not text:
        return []
    raw: list[PIISpan] = []
    for name, pattern, _token in _PII_PATTERNS:
        for m in pattern.finditer(text):
            raw.append(PIISpan(start=m.start(), end=m.end(), name=name))
    raw.sort(key=lambda s: (s.start, -(s.end - s.start)))

    # Greedy non-overlap (prefer earlier / longer)
    merged: list[PIISpan] = []
    cursor = -1
    for span in raw:
        if span.start < cursor:
            continue
        merged.append(span)
        cursor = span.end
    return merged


def scan_for_injection(text: str) -> bool:
    """
    Return True if ``text`` matches common prompt-injection / jailbreak patterns.
    """
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def redact_pii(text: str, *, log: bool = True) -> RedactionResult:
    """Detect and mask the seven supported PII entity types."""
    if not text:
        return RedactionResult(text="", count=0, entities={})

    entities: dict[str, int] = {}
    result = text
    total = 0

    for name, pattern, token in _PII_PATTERNS:
        matches = list(pattern.finditer(result))
        if not matches:
            continue
        for m in reversed(matches):
            result = result[: m.start()] + token + result[m.end() :]
        entities[name] = len(matches)
        total += len(matches)

    if log and total > 0:
        audit_log(
            "pii_redacted",
            f"Redacted {total} PII entit{'y' if total == 1 else 'ies'}",
            count=total,
            entities=entities,
        )

    return RedactionResult(text=result, count=total, entities=entities)
