from __future__ import annotations

import os
from dataclasses import replace

from rag_core.context import root_source_id
from rag_core.types import SearchHit


def group_selected_doc_ids(doc_ids: list[str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for doc_id in doc_ids:
        normalized = str(doc_id or "").strip()
        if not normalized:
            continue
        source_id = root_source_id(normalized)
        group = groups.setdefault(source_id, [])
        if normalized not in group:
            group.append(normalized)
    return list(groups.items())


def annotate_retrieval_source(hits: list[SearchHit], source_id: str) -> list[SearchHit]:
    return [
        replace(
            hit,
            metadata={
                **hit.metadata,
                "retrieval_source_id": source_id,
            },
        )
        for hit in hits
    ]


def round_robin_hit_groups(groups: list[list[SearchHit]]) -> list[SearchHit]:
    output: list[SearchHit] = []
    seen: set[str] = set()
    max_length = max((len(group) for group in groups), default=0)
    for index in range(max_length):
        for group in groups:
            if index >= len(group):
                continue
            hit = group[index]
            if hit.id in seen:
                continue
            seen.add(hit.id)
            output.append(hit)
    return output


def should_fan_out_source_retrieval(groups: list[tuple[str, list[str]]]) -> bool:
    return 1 < len(groups) <= source_fanout_limit()


def per_source_candidate_limit(candidate_limit: int, source_count: int) -> int:
    return max(
        4,
        (max(candidate_limit, 10) + max(1, source_count) - 1)
        // max(1, source_count),
    )


def context_chunks_per_source(
    configured_limit: int,
    context_limit: int,
    selected_source_groups: list[tuple[str, list[str]]],
) -> int:
    if len(selected_source_groups) == 1:
        return max(configured_limit, context_limit)
    return configured_limit


def source_fanout_limit() -> int:
    try:
        return max(1, int(os.environ.get("RAG_RETRIEVAL_SOURCE_FANOUT_LIMIT", "8")))
    except ValueError:
        return 8
