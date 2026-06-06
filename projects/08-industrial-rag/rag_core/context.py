from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from rag_core.text_utils import tokenize
from rag_core.types import PackingStats, SearchHit


@dataclass(frozen=True)
class PackingDecision:
    doc_id: str
    chunk_index: int
    title: str
    rerank_score: float | None
    text_chars: int
    used_chars_before: int
    decision: str
    reason: str


def default_text_units(text: str) -> int:
    return len(tokenize(text))


def pack_context(
    hits: list[SearchHit],
    *,
    max_selected: int | None = None,
    max_chars: int,
    max_chunks_per_doc: int,
    min_rerank_score: float | None,
    text_unit_counter: Callable[[str], int] | None = None,
) -> tuple[list[SearchHit], PackingStats]:
    selected, stats, _ = explain_context_packing(
        hits,
        max_selected=max_selected,
        max_chars=max_chars,
        max_chunks_per_doc=max_chunks_per_doc,
        min_rerank_score=min_rerank_score,
        text_unit_counter=text_unit_counter,
    )
    return selected, stats


def explain_context_packing(
    hits: list[SearchHit],
    *,
    max_selected: int | None = None,
    max_chars: int,
    max_chunks_per_doc: int,
    min_rerank_score: float | None,
    text_unit_counter: Callable[[str], int] | None = None,
) -> tuple[list[SearchHit], PackingStats, list[PackingDecision]]:
    selected: list[SearchHit] = []
    per_doc: Counter[str] = Counter()
    used_chars = 0
    dropped_by_score = 0
    dropped_by_doc_limit = 0
    dropped_by_budget = 0
    decisions: list[PackingDecision] = []
    count_text_units = text_unit_counter or default_text_units

    for hit in hits:
        text_len = count_text_units(hit.text)
        if (
            min_rerank_score is not None
            and hit.rerank_score is not None
            and hit.rerank_score < min_rerank_score
        ):
            dropped_by_score += 1
            decisions.append(
                packing_decision(
                    hit,
                    text_len=text_len,
                    used_chars_before=used_chars,
                    decision="drop",
                    reason="below_min_rerank_score",
                )
            )
            continue
        if max_selected is not None and len(selected) >= max_selected:
            decisions.append(
                packing_decision(
                    hit,
                    text_len=text_len,
                    used_chars_before=used_chars,
                    decision="drop",
                    reason="context_hit_limit",
                )
            )
            continue
        if per_doc[hit.doc_id] >= max_chunks_per_doc:
            dropped_by_doc_limit += 1
            decisions.append(
                packing_decision(
                    hit,
                    text_len=text_len,
                    used_chars_before=used_chars,
                    decision="drop",
                    reason="max_chunks_per_doc",
                )
            )
            continue

        if selected and used_chars + text_len > max_chars:
            dropped_by_budget += 1
            decisions.append(
                packing_decision(
                    hit,
                    text_len=text_len,
                    used_chars_before=used_chars,
                    decision="drop",
                    reason="context_char_budget",
                )
            )
            continue
        if not selected and text_len > max_chars:
            selected.append(hit)
            per_doc[hit.doc_id] += 1
            used_chars += text_len
            decisions.append(
                packing_decision(
                    hit,
                    text_len=text_len,
                    used_chars_before=0,
                    decision="select",
                    reason="first_chunk_exceeds_budget",
                )
            )
            continue

        selected.append(hit)
        per_doc[hit.doc_id] += 1
        used_chars += text_len
        decisions.append(
            packing_decision(
                hit,
                text_len=text_len,
                used_chars_before=used_chars - text_len,
                decision="select",
                reason="fits_budget",
            )
        )

    return selected, PackingStats(
        selected_count=len(selected),
        dropped_by_score=dropped_by_score,
        dropped_by_doc_limit=dropped_by_doc_limit,
        dropped_by_budget=dropped_by_budget,
    ), decisions


def packing_decision(
    hit: SearchHit,
    *,
    text_len: int,
    used_chars_before: int,
    decision: str,
    reason: str,
) -> PackingDecision:
    return PackingDecision(
        doc_id=hit.doc_id,
        chunk_index=hit.chunk_index,
        title=hit.title,
        rerank_score=hit.rerank_score,
        text_chars=text_len,
        used_chars_before=used_chars_before,
        decision=decision,
        reason=reason,
    )
