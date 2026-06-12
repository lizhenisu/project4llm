from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model
from rag_core.io import write_jsonl
from rag_core.milvus_store import (
    build_filter_expr,
    connect,
    dense_search,
    ensure_collection,
    hybrid_search,
    sparse_search,
)
from rag_core.rerankers import build_reranker
from rag_core.text_utils import lexical_overlap_score
from rag_core.types import SearchHit


@dataclass(frozen=True)
class CandidateDiagnosis:
    doc_id: str
    chunk_index: int
    title: str
    source_type: str
    dense_rank: int | None
    sparse_rank: int | None
    hybrid_rank: int | None
    rerank_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    hybrid_score: float | None
    rerank_score: float | None
    lexical_overlap: float
    text_preview: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain dense/sparse/hybrid/rerank candidate rankings for a query."
    )
    parser.add_argument("query")
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat to allow multiple groups.",
    )
    parser.add_argument("--doc-version", type=int)
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Restrict retrieval to a source type. Repeat for multiple types.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--json-output", type=Path, help="Write diagnosis rows as JSONL.")
    args = parser.parse_args()

    rows = diagnose_retrieval(
        args.query,
        tenant_id=args.tenant_id,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
        candidate_limit=args.candidate_limit,
        limit=args.limit,
    )
    print_diagnosis(rows)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.json_output, [asdict(row) for row in rows])


def diagnose_retrieval(
    query: str,
    *,
    tenant_id: str,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    source_types: list[str] | None = None,
    candidate_limit: int = 20,
    limit: int = 8,
) -> list[CandidateDiagnosis]:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    embedding_model = build_embedding_model(config)
    query_vector = embedding_model.encode([query])[0]
    filter_expr = build_filter_expr(
        tenant_id=tenant_id,
        allowed_acl_groups=acl_groups,
        doc_version=doc_version,
        embedding_model=embedding_model.model_name,
        source_types=source_types,
    )
    dense_hits = dense_search(
        client,
        collection_name=config.collection_name,
        query_vector=query_vector,
        filter_expr=filter_expr,
        limit=candidate_limit,
    )
    sparse_hits = sparse_search(
        client,
        collection_name=config.collection_name,
        query_text=query,
        filter_expr=filter_expr,
        limit=candidate_limit,
    )
    hybrid_hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=query_vector,
        query_text=query,
        filter_expr=filter_expr,
        limit=candidate_limit,
    )
    reranked = build_reranker(config).rerank(
        query,
        hybrid_hits,
        limit=min(limit, len(hybrid_hits)),
    )
    return build_diagnosis_rows(
        query,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        hybrid_hits=hybrid_hits,
        reranked=reranked,
        limit=limit,
    )


def build_diagnosis_rows(
    query: str,
    *,
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    hybrid_hits: list[SearchHit],
    reranked: list[SearchHit],
    limit: int,
) -> list[CandidateDiagnosis]:
    hits_by_id: dict[str, SearchHit] = {}
    for hit in [*dense_hits, *sparse_hits, *hybrid_hits, *reranked]:
        hits_by_id.setdefault(hit.id, hit)

    dense_rank = rank_by_id(dense_hits)
    sparse_rank = rank_by_id(sparse_hits)
    hybrid_rank = rank_by_id(hybrid_hits)
    rerank_rank = rank_by_id(reranked)
    dense_score = score_by_id(dense_hits)
    sparse_score = score_by_id(sparse_hits)
    hybrid_score = score_by_id(hybrid_hits)
    rerank_score = {hit.id: hit.rerank_score for hit in reranked}

    ordered_ids = sorted(
        hits_by_id,
        key=lambda hit_id: (
            rerank_rank.get(hit_id, 10_000),
            hybrid_rank.get(hit_id, 10_000),
            dense_rank.get(hit_id, 10_000),
            sparse_rank.get(hit_id, 10_000),
        ),
    )
    rows: list[CandidateDiagnosis] = []
    for hit_id in ordered_ids[:limit]:
        hit = hits_by_id[hit_id]
        rows.append(
            CandidateDiagnosis(
                doc_id=hit.doc_id,
                chunk_index=hit.chunk_index,
                title=hit.title,
                source_type=hit.source_type,
                dense_rank=dense_rank.get(hit_id),
                sparse_rank=sparse_rank.get(hit_id),
                hybrid_rank=hybrid_rank.get(hit_id),
                rerank_rank=rerank_rank.get(hit_id),
                dense_score=dense_score.get(hit_id),
                sparse_score=sparse_score.get(hit_id),
                hybrid_score=hybrid_score.get(hit_id),
                rerank_score=rerank_score.get(hit_id),
                lexical_overlap=lexical_overlap_score(query, hit.text),
                text_preview=hit.text[:220].replace("\n", " "),
            )
        )
    return rows


def rank_by_id(hits: list[SearchHit]) -> dict[str, int]:
    return {hit.id: index for index, hit in enumerate(hits, start=1)}


def score_by_id(hits: list[SearchHit]) -> dict[str, float]:
    return {hit.id: hit.score for hit in hits}


def print_diagnosis(rows: list[CandidateDiagnosis]) -> None:
    for row in rows:
        print(
            f"doc={row.doc_id} chunk={row.chunk_index} "
            f"dense={format_rank_score(row.dense_rank, row.dense_score)} "
            f"sparse={format_rank_score(row.sparse_rank, row.sparse_score)} "
            f"hybrid={format_rank_score(row.hybrid_rank, row.hybrid_score)} "
            f"rerank={format_rank_score(row.rerank_rank, row.rerank_score)} "
            f"lexical={row.lexical_overlap:.3f}"
        )
        print(row.text_preview)


def format_rank_score(rank: int | None, score: float | None) -> str:
    if rank is None:
        return "-"
    if score is None:
        return f"#{rank}"
    return f"#{rank}/{score:.4f}"


if __name__ == "__main__":
    main()
