from __future__ import annotations

import re
from dataclasses import dataclass


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_cn": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card_cn": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "api_key": re.compile(r"\b(?:sk|ak)-[A-Za-z0-9_-]{16,}\b"),
}


@dataclass(frozen=True)
class PiiFinding:
    kind: str
    value: str


def detect_pii(text: str) -> list[PiiFinding]:
    findings: list[PiiFinding] = []
    for kind, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append(PiiFinding(kind=kind, value=match.group(0)))
    return findings


def redact_pii(text: str) -> str:
    redacted = text
    for kind, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{kind.upper()}]", redacted)
    return redacted


def apply_pii_policy(text: str, *, policy: str, label: str) -> str:
    findings = detect_pii(text)
    if not findings:
        return text

    summary = ", ".join(sorted({finding.kind for finding in findings}))
    if policy == "fail":
        raise ValueError(f"PII detected in {label}: {summary}")
    if policy == "redact":
        return redact_pii(text)
    if policy == "warn":
        print(f"Warning: PII detected in {label}: {summary}")
        return text
    raise ValueError(f"Unsupported RAG_PII_POLICY={policy!r}; use warn/redact/fail")

