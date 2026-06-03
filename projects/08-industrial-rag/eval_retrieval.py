from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
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
    sparse_search,
)
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import sparse_embedding
from rag_core.types import SearchHit
from search_multimodal import retrieve_multimodal


@dataclass(frozen=True)
class EvalSearchResult:
    hits: list[SearchHit]
    stage_latency_ms: dict[str, float]


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
        choices=["dense", "sparse", "hybrid", "rerank", "multimodal"],
        default="hybrid",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument("--json-output", type=Path, help="Write metrics as JSON.")
    args = parser.parse_args()

    metrics = evaluate_retrieval(
        input_path=args.input,
        limit=args.limit,
        mode=args.mode,
    )
    print(f"recall@{args.limit}: {metrics['recall']:.3f}")
    print(f"mrr@{args.limit}: {metrics['mrr']:.3f}")
    print(f"ndcg@{args.limit}: {metrics['ndcg']:.3f}")
    print(f"avg_latency_ms: {metrics['avg_latency_ms']:.2f}")
    print(f"p95_latency_ms: {metrics['p95_latency_ms']:.2f}")
    print(f"stage_p95_latency_ms: {metrics['stage_p95_latency_ms']}")
    print(f"permission_leakage_failures: {metrics['permission_leakage_failures']}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def evaluate_retrieval(*, input_path: Path, limit: int, mode: str) -> dict[str, float | int | str]:
    rows = read_jsonl(input_path)
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    embedding_model = build_embedding_model(config)

    recall_hits = 0
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    latencies_ms: list[float] = []
    stage_latencies_ms: dict[str, list[float]] = {}
    leakage_failures = 0

    for row in rows:
        started = time.perf_counter()
        search_result = run_eval_search(
            row["query"],
            tenant_id=row["tenant_id"],
            acl_groups=row.get("acl_groups") or None,
            doc_version=row.get("doc_version"),
            source_types=row.get("source_types") or None,
            history=row.get("history") or None,
            limit=limit,
            mode=mode,
            client=client,
            collection_name=config.collection_name,
            embedding_model=embedding_model,
        )
        hits = search_result.hits
        latencies_ms.append((time.perf_counter() - started) * 1000)
        for stage, latency in search_result.stage_latency_ms.items():
            stage_latencies_ms.setdefault(stage, []).append(float(latency))
        expected, returned = eval_targets(row, hits)

        if expected:
            matched_ranks = [
                index
                for index, value in enumerate(returned, start=1)
                if value in expected
            ]
            if matched_ranks:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / matched_ranks[0])
            else:
                reciprocal_ranks.append(0.0)
            ndcg_scores.append(ndcg_at_k(returned, expected, limit))
        else:
            forbidden_team_b = any(hit.tenant_id != row["tenant_id"] for hit in hits)
            if forbidden_team_b:
                leakage_failures += 1

        print(
            f"query={row['query']} expected={sorted(expected)} "
            f"returned={returned[:limit]}"
        )

    answerable_count = sum(
        1 for row in rows if row.get("expected_chunk_ids") or row.get("expected_doc_ids")
    )
    recall = recall_hits / answerable_count if answerable_count else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    return {
        "mode": mode,
        "limit": limit,
        "query_count": len(rows),
        "answerable_count": answerable_count,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "stage_p95_latency_ms": {
            stage: percentile(values, 0.95)
            for stage, values in sorted(stage_latencies_ms.items())
        },
        "permission_leakage_failures": leakage_failures,
    }


def run_eval_search(
    query: str,
    *,
    tenant_id: str,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    source_types: list[str] | None = None,
    history: list[str] | None = None,
    limit: int,
    mode: str,
    client,
    collection_name: str,
    embedding_model,
) -> EvalSearchResult:
    if mode == "rerank":
        result = retrieve_and_rerank(
            query,
            tenant_id=tenant_id,
            candidate_limit=max(limit, 20),
            context_limit=limit,
            acl_groups=acl_groups,
            doc_version=doc_version,
            source_types=source_types,
            history=history,
        )
        return EvalSearchResult(
            hits=result.hits,
            stage_latency_ms=result.trace.stage_latency_ms,
        )
    if mode == "multimodal":
        search_start = time.perf_counter()
        result = retrieve_multimodal(
            query,
            tenant_id=tenant_id,
            candidate_limit=max(limit, 10),
            context_limit=limit,
            acl_groups=acl_groups,
            doc_version=doc_version,
            source_types=source_types or ["image"],
            history=history,
        )
        stage_latency_ms = {
            **result.trace.stage_latency_ms,
            "multimodal_search": elapsed_ms(search_start),
        }
        return EvalSearchResult(
            hits=result.hits,
            stage_latency_ms=stage_latency_ms,
        )

    filter_expr = build_filter_expr(
        tenant_id=tenant_id,
        allowed_acl_groups=acl_groups,
        doc_version=doc_version,
        source_types=source_types,
        embedding_model=embedding_model.model_name,
    )
    sparse_start = time.perf_counter()
    query_sparse = sparse_embedding(query)
    sparse_ms = elapsed_ms(sparse_start)
    if mode == "sparse":
        search_start = time.perf_counter()
        hits = sparse_search(
            client,
            collection_name=collection_name,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            limit=limit,
        )
        return EvalSearchResult(
            hits=hits,
            stage_latency_ms={
                "sparse_query": sparse_ms,
                "milvus_search": elapsed_ms(search_start),
            },
        )

    embedding_start = time.perf_counter()
    query_vector = embedding_model.encode([query])[0]
    embedding_ms = elapsed_ms(embedding_start)
    search_start = time.perf_counter()
    if mode == "dense":
        hits = dense_search(
            client,
            collection_name=collection_name,
            query_vector=query_vector,
            filter_expr=filter_expr,
            limit=limit,
        )
    else:
        hits = hybrid_search(
            client,
            collection_name=collection_name,
            query_vector=query_vector,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            limit=limit,
        )
    return EvalSearchResult(
        hits=hits,
        stage_latency_ms={
            "embedding": embedding_ms,
            "milvus_search": elapsed_ms(search_start),
        },
    )


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def eval_targets(row: dict, hits: list[SearchHit]) -> tuple[set[str], list[str]]:
    expected_chunk_ids = set(row.get("expected_chunk_ids", []))
    if expected_chunk_ids:
        return expected_chunk_ids, [hit_eval_chunk_id(hit, expected_chunk_ids) for hit in hits]
    return set(row.get("expected_doc_ids", [])), [hit.doc_id for hit in hits]


def hit_eval_chunk_id(hit: SearchHit, expected: set[str]) -> str:
    if hit.id in expected:
        return hit.id
    metadata = hit.metadata or {}
    metadata_chunk_id = str(metadata.get("chunk_id", ""))
    if metadata_chunk_id in expected:
        return metadata_chunk_id
    return f"{hit.doc_id}:{hit.chunk_index}"


def ndcg_at_k(returned_doc_ids: list[str], expected_doc_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(returned_doc_ids[:k], start=1):
        relevance = 1.0 if doc_id in expected_doc_ids else 0.0
        dcg += relevance / math.log2(rank + 1)

    ideal_relevant = min(len(expected_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[index]


if __name__ == "__main__":
    main()
