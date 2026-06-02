from __future__ import annotations

import re


CITATION_PATTERN = re.compile(r"\[(\d+)]")
REFUSAL_TEXT = "当前知识库没有足够证据"


def extract_citation_numbers(answer: str) -> list[int]:
    return [int(match.group(1)) for match in CITATION_PATTERN.finditer(answer)]


def citation_accuracy(answer: str, evidence_count: int) -> float:
    citations = extract_citation_numbers(answer)
    if not citations:
        return 0.0 if evidence_count else 1.0
    valid = sum(1 for citation in citations if 1 <= citation <= evidence_count)
    return valid / len(citations)


def is_refusal(answer: str) -> bool:
    return REFUSAL_TEXT in answer

