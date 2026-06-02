from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from rag_core.config import DATA_DIR, load_config
from rag_core.embeddings import build_embedding_model
from rag_core.io import read_jsonl
from rag_core.milvus_store import (
    build_filter_expr,
    connect,
    dense_search,
    ensure_collection,
    hybrid_search,
)
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import sparse_embedding
from rag_core.types import SearchHit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval recall, MRR, nDCG, leakage, and latency."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "eval_queries.jsonl",
        help="JSONL eval set.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "rerank"],
        default="hybrid",
        help="Retrieval mode to evaluate.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    embedding_model = build_embedding_model(config)

    recall_hits = 0
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    latencies_ms: list[float] = []
    leakage_failures = 0

    for row in rows:
        started = time.perf_counter()
        hits = run_eval_search(
            row["query"],
            tenant_id=row["tenant_id"],
            limit=args.limit,
            mode=args.mode,
            client=client,
            collection_name=config.collection_name,
            embedding_model=embedding_model,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        returned_doc_ids = [hit.doc_id for hit in hits]
        expected = set(row.get("expected_doc_ids", []))

        if expected:
            matched_ranks = [
                index
                for index, doc_id in enumerate(returned_doc_ids, start=1)
                if doc_id in expected
            ]
            if matched_ranks:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / matched_ranks[0])
            else:
                reciprocal_ranks.append(0.0)
            ndcg_scores.append(ndcg_at_k(returned_doc_ids, expected, args.limit))
        else:
            forbidden_team_b = any(hit.tenant_id != row["tenant_id"] for hit in hits)
            if forbidden_team_b:
                leakage_failures += 1

        print(
            f"query={row['query']} expected={sorted(expected)} "
            f"returned={returned_doc_ids[:args.limit]}"
        )

    answerable_count = sum(1 for row in rows if row.get("expected_doc_ids"))
    recall = recall_hits / answerable_count if answerable_count else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    print(f"recall@{args.limit}: {recall:.3f}")
    print(f"mrr@{args.limit}: {mrr:.3f}")
    print(f"ndcg@{args.limit}: {ndcg:.3f}")
    print(f"avg_latency_ms: {avg_latency:.2f}")
    print(f"permission_leakage_failures: {leakage_failures}")


def run_eval_search(
    query: str,
    *,
    tenant_id: str,
    limit: int,
    mode: str,
    client,
    collection_name: str,
    embedding_model,
) -> list[SearchHit]:
    if mode == "rerank":
        return retrieve_and_rerank(
            query,
            tenant_id=tenant_id,
            candidate_limit=max(limit, 20),
            context_limit=limit,
        ).hits

    query_vector = embedding_model.encode([query])[0]
    filter_expr = build_filter_expr(tenant_id=tenant_id)
    if mode == "dense":
        return dense_search(
            client,
            collection_name=collection_name,
            query_vector=query_vector,
            filter_expr=filter_expr,
            limit=limit,
        )
    return hybrid_search(
        client,
        collection_name=collection_name,
        query_vector=query_vector,
        query_sparse=sparse_embedding(query),
        filter_expr=filter_expr,
        limit=limit,
    )


def ndcg_at_k(returned_doc_ids: list[str], expected_doc_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(returned_doc_ids[:k], start=1):
        relevance = 1.0 if doc_id in expected_doc_ids else 0.0
        dcg += relevance / math.log2(rank + 1)

    ideal_relevant = min(len(expected_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / idcg if idcg else 0.0


if __name__ == "__main__":
    main()
