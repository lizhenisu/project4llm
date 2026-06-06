from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from answer import answer_query
from answer_multimodal import answer_multimodal_query


STAGE_ORDER = [
    "rewrite",
    "embedding",
    "milvus_search",
    "rerank",
    "text_embedding",
    "text_search",
    "image_embedding",
    "image_search",
    "fusion",
    "context_pack",
    "answer",
    "total",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the real RAG answer pipeline and aggregate stage latency."
    )
    parser.add_argument(
        "--query",
        default="RAG 检索变慢时应该排查什么",
        help="Benchmark query or image path for multimodal mode.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["text", "multimodal"],
        default="text",
        help="Answer pipeline to benchmark.",
    )
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
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="Conversation history item used for query rewrite. Repeat for multiple turns.",
    )
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json-output", type=Path, help="Write benchmark summary as JSON.")
    args = parser.parse_args()

    metrics = benchmark_query(
        query=args.query,
        query_mode=args.query_mode,
        tenant_id=args.tenant_id,
        acl_groups=args.acl_group or None,
        doc_version=args.doc_version,
        source_types=args.source_type or None,
        history=args.history or None,
        candidate_limit=args.candidate_limit,
        context_limit=args.context_limit,
        runs=args.runs,
    )
    print_benchmark(metrics)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def benchmark_query(
    *,
    query: str,
    query_mode: str,
    tenant_id: str,
    acl_groups: list[str] | None = None,
    doc_version: int | None = None,
    source_types: list[str] | None = None,
    history: list[str] | None = None,
    candidate_limit: int,
    context_limit: int,
    runs: int,
) -> dict[str, object]:
    stage_samples: dict[str, list[float]] = {}

    for _ in range(runs):
        total_start = perf_counter()
        result = run_answer_pipeline(
            query=query,
            query_mode=query_mode,
            tenant_id=tenant_id,
            acl_groups=acl_groups,
            doc_version=doc_version,
            source_types=source_types,
            history=history,
            candidate_limit=candidate_limit,
            context_limit=context_limit,
        )
        total_ms = elapsed_ms(total_start)
        for stage, latency in result.trace.stage_latency_ms.items():
            stage_samples.setdefault(stage, []).append(float(latency))
        stage_samples.setdefault("answer", []).append(float(result.generation.latency_ms))
        stage_samples.setdefault("total", []).append(total_ms)

    return {
        "query_mode": query_mode,
        "runs": runs,
        "candidate_limit": candidate_limit,
        "context_limit": context_limit,
        "stage_latency_ms": {
            stage: summarize(values)
            for stage, values in ordered_stage_samples(stage_samples).items()
        },
    }


def run_answer_pipeline(
    *,
    query: str,
    query_mode: str,
    tenant_id: str,
    acl_groups: list[str] | None,
    doc_version: int | None,
    source_types: list[str] | None,
    history: list[str] | None,
    candidate_limit: int,
    context_limit: int,
):
    if query_mode == "multimodal":
        return answer_multimodal_query(
            query,
            tenant_id=tenant_id,
            candidate_limit=candidate_limit,
            context_limit=context_limit,
            acl_groups=acl_groups,
            doc_version=doc_version,
            source_types=source_types,
            history=history,
        )
    return answer_query(
        query,
        tenant_id=tenant_id,
        candidate_limit=candidate_limit,
        context_limit=context_limit,
        acl_groups=acl_groups,
        doc_version=doc_version,
        source_types=source_types,
        history=history,
    )


def ordered_stage_samples(stage_samples: dict[str, list[float]]) -> dict[str, list[float]]:
    ordered: dict[str, list[float]] = {}
    for stage in STAGE_ORDER:
        if stage in stage_samples:
            ordered[stage] = stage_samples[stage]
    for stage in sorted(stage_samples):
        if stage not in ordered:
            ordered[stage] = stage_samples[stage]
    return ordered


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg_ms": round(avg(values), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


def print_benchmark(metrics: dict[str, object]) -> None:
    print(f"query_mode: {metrics['query_mode']}")
    print(f"runs: {metrics['runs']}")
    for stage, summary in metrics["stage_latency_ms"].items():
        print(
            f"{stage}_ms: avg={summary['avg_ms']:.2f} "
            f"p95={summary['p95_ms']:.2f} "
            f"min={summary['min_ms']:.2f} max={summary['max_ms']:.2f}"
        )


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


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
