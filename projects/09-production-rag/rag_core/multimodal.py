from __future__ import annotations

from dataclasses import replace

from rag_core.types import SearchHit


def reciprocal_rank_fusion(
    channels: list[tuple[str, list[SearchHit]]],
    *,
    limit: int,
    k: int = 60,
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    best_hits: dict[str, SearchHit] = {}
    channel_ranks: dict[str, dict[str, int]] = {}
    channel_scores: dict[str, dict[str, float]] = {}

    for channel_name, hits in channels:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            if hit.id not in best_hits or hit.score > best_hits[hit.id].score:
                best_hits[hit.id] = hit
            channel_ranks.setdefault(hit.id, {})[channel_name] = rank
            channel_scores.setdefault(hit.id, {})[channel_name] = hit.score

    ranked_ids = sorted(
        scores,
        key=lambda hit_id: (
            scores[hit_id],
            -min(channel_ranks[hit_id].values()),
            best_hits[hit_id].score,
        ),
        reverse=True,
    )
    fused: list[SearchHit] = []
    for hit_id in ranked_ids[:limit]:
        hit = best_hits[hit_id]
        metadata = {
            **hit.metadata,
            "fusion": {
                "rrf_score": scores[hit_id],
                "channels": channel_ranks[hit_id],
                "channel_scores": channel_scores[hit_id],
            },
        }
        fused.append(replace(hit, score=scores[hit_id], metadata=metadata))
    return fused
