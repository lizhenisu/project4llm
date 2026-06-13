from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from rag_core.config import FIXTURE_DATA_DIR, load_config
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
        default=FIXTURE_DATA_DIR / "eval_queries.jsonl",
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

    # 下面这些变量都是“先逐 query 收集，再在函数末尾汇总”的评估中间量。
    #
    # recall_hits:
    #   统计有答案 query 中，有多少条在 top-K 结果里至少命中 1 个期望文档/片段。
    # reciprocal_ranks:
    #   每条有答案 query 的 RR = 1 / 第一个命中结果的排名；没命中则记 0。
    # ndcg_scores:
    #   每条有答案 query 的 nDCG@K，衡量相关结果是否排在更靠前的位置。
    # latencies_ms / stage_latencies_ms:
    #   分别记录整条检索链路耗时、以及 embedding / milvus_search / rerank 等阶段耗时。
    # leakage_failures:
    #   对“无 expected 目标”的权限测试 query，统计是否返回了其他 tenant 的结果。
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

        # latency 指从进入当前 query 评估开始，到检索结果返回为止的端到端耗时。
        # stage_latency_ms 则由具体检索函数返回，用于拆解瓶颈，例如向量化慢还是 Milvus 搜索慢。
        latencies_ms.append((time.perf_counter() - started) * 1000)
        for stage, latency in search_result.stage_latency_ms.items():
            stage_latencies_ms.setdefault(stage, []).append(float(latency))

        # eval_targets 会根据评测集字段决定评估粒度：
        # - 如果 row 里有 expected_chunk_ids，就按 chunk 级别评估。
        # - 否则按 expected_doc_ids 做文档级评估。
        # returned 是本次检索返回结果映射后的 id 列表，顺序就是检索排名顺序。
        expected, returned = eval_targets(row, hits)

        if expected:
            # matched_ranks 记录所有命中 expected 的返回位置，排名从 1 开始。
            # 例子：expected={"doc_a"}，returned=["doc_x", "doc_a", "doc_b"]，
            # matched_ranks=[2]，表示第 2 名才首次命中。
            matched_ranks = [
                index
                for index, value in enumerate(returned, start=1)
                if value in expected
            ]
            if matched_ranks:
                # Recall@K 在这里按 query 级别计算：
                # 只要 top-K 里出现任意一个期望目标，这条 query 就算 recall 命中。
                # 最后 recall = recall_hits / answerable_count。
                recall_hits += 1

                # Reciprocal Rank 只关心“第一个相关结果出现得多早”：
                # 第 1 名命中 RR=1.0，第 2 名命中 RR=0.5，第 10 名命中 RR=0.1。
                # MRR 是所有有答案 query 的 RR 平均值。
                reciprocal_ranks.append(1.0 / matched_ranks[0])
            else:
                # top-K 完全没命中时，这条 query 对 MRR 的贡献为 0。
                reciprocal_ranks.append(0.0)

            # nDCG@K 比 Recall@K 和 MRR 更细：
            # - Recall@K 只看有没有命中。
            # - MRR 只看第一个命中排第几。
            # - nDCG@K 会把多个相关结果的位置都计入，并用 log 折扣惩罚靠后的命中。
            ndcg_scores.append(ndcg_at_k(returned, expected, limit))
        else:
            # 没有 expected_doc_ids / expected_chunk_ids 的样本，不参与 recall/mrr/ndcg。
            # 在这个项目里这类样本用于权限泄漏测试：如果返回了非当前 tenant 的 hit，
            # 说明 ACL / tenant filter 没有正确生效。
            forbidden_team_b = any(hit.tenant_id != row["tenant_id"] for hit in hits)
            if forbidden_team_b:
                leakage_failures += 1

        print(
            f"query={row['query']} expected={sorted(expected)} "
            f"returned={returned[:limit]}"
        )

    # answerable_count 是可以计算检索质量指标的 query 数量。
    # 权限泄漏测试样本通常没有 expected，它们只进入 leakage_failures，不进入质量指标分母。
    answerable_count = sum(
        1 for row in rows if row.get("expected_chunk_ids") or row.get("expected_doc_ids")
    )

    # Recall@K = top-K 至少命中一次的有答案 query 数 / 有答案 query 总数。
    # 注意：这是 query-level hit rate，不是“命中的相关文档数 / 所有相关文档数”的细粒度 recall。
    recall = recall_hits / answerable_count if answerable_count else 0.0

    # MRR@K = 每条有答案 query 的 Reciprocal Rank 平均值。
    # 它特别看重第一个正确结果的位置，适合问答/RAG 场景：
    # 第一个证据越靠前，后续组 context 和生成答案越稳定。
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # nDCG@K = 每条 query 的 DCG / 理想 DCG，再对所有有答案 query 求平均。
    # 它适合评估“多个相关文档是否整体排得靠前”。
    ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # 平均延迟容易被极端慢 query 拉高；p95 延迟表示 95% query 不超过这个耗时，
    # 更接近线上体验里“绝大多数请求”的尾延迟表现。
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
    if mode == "sparse":
        search_start = time.perf_counter()
        hits = sparse_search(
            client,
            collection_name=collection_name,
            query_text=query,
            filter_expr=filter_expr,
            limit=limit,
        )
        return EvalSearchResult(
            hits=hits,
            stage_latency_ms={
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
            query_text=query,
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
    # NDCG@K 衡量“前 K 个检索结果里，相关文档是否排得足够靠前”。
    # 这里使用二值相关性：命中 expected_doc_ids 记为 1，不命中记为 0。
    #
    # DCG = Discounted Cumulative Gain，带位置折扣的累计收益：
    # - rank=1 的命中贡献最大：1 / log2(1 + 1) = 1
    # - rank=2 的命中贡献变小：1 / log2(2 + 1)
    # - rank 越靠后，贡献越低，表示“相关结果排得越晚越不理想”。
    dcg = 0.0
    for rank, doc_id in enumerate(returned_doc_ids[:k], start=1):
        relevance = 1.0 if doc_id in expected_doc_ids else 0.0
        dcg += relevance / math.log2(rank + 1)

    # IDCG = Ideal DCG，即理想情况下能拿到的最高 DCG。
    # 如果 expected_doc_ids 有 3 个、k=5，理想排序就是前 3 名全相关；
    # 如果 expected_doc_ids 有 10 个、k=5，最多也只能在前 5 名放 5 个相关结果。
    ideal_relevant = min(len(expected_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))

    # 用 DCG / IDCG 归一化到 0~1：
    # - 1.0 表示前 K 个结果达到了理想排序
    # - 0.0 表示前 K 个结果没有任何相关文档
    # idcg 为 0 说明没有期望文档，无法定义理想排序，这里返回 0。
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[index]


if __name__ == "__main__":
    main()
