from __future__ import annotations

import argparse
import time

from rag_core.answering import generate_answer
from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model
from rag_core.milvus_store import build_filter_expr, connect, ensure_collection, hybrid_search
from rag_core.rerankers import build_reranker
from rag_core.text_utils import sparse_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG stage latency.")
    parser.add_argument(
        "--query",
        default="RAG 检索变慢时应该排查什么",
        help="Benchmark query.",
    )
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=False)
    embedding_model = build_embedding_model(config)
    reranker = build_reranker(config)
    filter_expr = build_filter_expr(tenant_id=args.tenant_id)

    totals = {
        "embedding_ms": [],
        "search_ms": [],
        "rerank_ms": [],
        "answer_ms": [],
        "total_ms": [],
    }

    for _ in range(args.runs):
        total_start = time.perf_counter()

        start = time.perf_counter()
        query_vector = embedding_model.encode([args.query])[0]
        query_sparse = sparse_embedding(args.query)
        totals["embedding_ms"].append(elapsed_ms(start))

        start = time.perf_counter()
        candidates = hybrid_search(
            client,
            collection_name=config.collection_name,
            query_vector=query_vector,
            query_sparse=query_sparse,
            filter_expr=filter_expr,
            limit=args.candidate_limit,
        )
        totals["search_ms"].append(elapsed_ms(start))

        start = time.perf_counter()
        hits = reranker.rerank(args.query, candidates, limit=args.context_limit)
        totals["rerank_ms"].append(elapsed_ms(start))

        start = time.perf_counter()
        _ = generate_answer(config, args.query, hits)
        totals["answer_ms"].append(elapsed_ms(start))
        totals["total_ms"].append(elapsed_ms(total_start))

    for name, values in totals.items():
        print(
            f"{name}: avg={avg(values):.2f} "
            f"p95={percentile(values, 0.95):.2f} "
            f"min={min(values):.2f} max={max(values):.2f}"
        )


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[index]


if __name__ == "__main__":
    main()

