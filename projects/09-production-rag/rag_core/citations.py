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


def term_coverage(text: str, expected_terms: list[str]) -> float:
    terms = [term for term in expected_terms if term]
    if not terms:
        return 1.0
    normalized = text.lower()
    matched = sum(1 for term in terms if term.lower() in normalized)
    return matched / len(terms)


def unsupported_term_rate(answer: str, evidence_text: str, unsupported_terms: list[str]) -> float:
    terms = [term for term in unsupported_terms if term]
    if not terms:
        return 0.0
    normalized_answer = answer.lower()
    normalized_evidence = evidence_text.lower()
    unsupported = [
        term
        for term in terms
        if term.lower() in normalized_answer and term.lower() not in normalized_evidence
    ]
    return len(unsupported) / len(terms)


def faithfulness_score(answer: str, evidence_text: str, unsupported_terms: list[str]) -> float:
    return 1.0 - unsupported_term_rate(answer, evidence_text, unsupported_terms)
