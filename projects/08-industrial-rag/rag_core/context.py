from __future__ import annotations

from collections import Counter

from rag_core.types import PackingStats, SearchHit


def pack_context(
    hits: list[SearchHit],
    *,
    max_chars: int,
    max_chunks_per_doc: int,
    min_rerank_score: float | None,
) -> tuple[list[SearchHit], PackingStats]:
    selected: list[SearchHit] = []
    per_doc: Counter[str] = Counter()
    used_chars = 0
    dropped_by_score = 0
    dropped_by_doc_limit = 0
    dropped_by_budget = 0

    for hit in hits:
        if (
            min_rerank_score is not None
            and hit.rerank_score is not None
            and hit.rerank_score < min_rerank_score
        ):
            dropped_by_score += 1
            continue
        if per_doc[hit.doc_id] >= max_chunks_per_doc:
            dropped_by_doc_limit += 1
            continue

        text_len = len(hit.text)
        if selected and used_chars + text_len > max_chars:
            dropped_by_budget += 1
            continue
        if not selected and text_len > max_chars:
            # Keep one truncated-looking source rather than returning no context.
            selected.append(hit)
            per_doc[hit.doc_id] += 1
            used_chars += text_len
            continue

        selected.append(hit)
        per_doc[hit.doc_id] += 1
        used_chars += text_len

    return selected, PackingStats(
        selected_count=len(selected),
        dropped_by_score=dropped_by_score,
        dropped_by_doc_limit=dropped_by_doc_limit,
        dropped_by_budget=dropped_by_budget,
    )

