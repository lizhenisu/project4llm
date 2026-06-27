from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SearchSample:
    index: int
    tenant_id: str
    ok: bool
    latency_ms: float
    status_code: int
    hit_count: int = 0
    candidate_count: int = 0
    reranked_count: int = 0
    stage_latency_ms: dict[str, float] | None = None
    error: str = ""


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    samples = run_load(args)
    summary = build_summary(args, samples, wall_ms=elapsed_ms(started))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if summary["failed"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load-test Production RAG /search without final answer-generation latency."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--token", default="production-rag-fixed-test-login-token")
    parser.add_argument("--tenant-id", default="tenant-fixed-test")
    parser.add_argument(
        "--tenant-count",
        type=int,
        default=1,
        help="Cycle requests across tenant-id or tenant-id-NNNN when greater than one.",
    )
    parser.add_argument("--acl-group", action="append", default=[])
    parser.add_argument("--query", default="总结资料中的核心技术方案")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=5)
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument("--include-all-sources", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_load(args: argparse.Namespace) -> list[SearchSample]:
    total = max(1, args.requests)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(send_search, args, index) for index in range(total)]
        samples = [future.result() for future in concurrent.futures.as_completed(futures)]
    return sorted(samples, key=lambda sample: sample.index)


def send_search(args: argparse.Namespace, index: int) -> SearchSample:
    tenant_id = tenant_for_index(args.tenant_id, max(1, args.tenant_count), index)
    payload = {
        "query": args.query,
        "tenant_id": tenant_id,
        "acl_groups": args.acl_group,
        "doc_ids": args.doc_id,
        "source_types": args.source_type,
        "include_all_sources": args.include_all_sources,
        "candidate_limit": args.candidate_limit,
        "context_limit": args.context_limit,
        "request_id": f"milvus-load-{uuid.uuid4().hex[:12]}",
    }
    headers = {
        "Content-Type": "application/json",
        "X-RAG-Tenant-ID": tenant_id,
    }
    if args.acl_group:
        headers["X-RAG-ACL-Groups"] = ",".join(args.acl_group)
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    request = Request(
        urljoin(args.base_url.rstrip("/") + "/", "search"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            trace = body.get("trace") or {}
            return SearchSample(
                index=index,
                tenant_id=tenant_id,
                ok=True,
                latency_ms=elapsed_ms(started),
                status_code=int(getattr(response, "status", 200)),
                hit_count=len(body.get("hits") or []),
                candidate_count=int(trace.get("candidate_count") or 0),
                reranked_count=int(trace.get("reranked_count") or 0),
                stage_latency_ms={
                    str(stage): float(latency)
                    for stage, latency in (trace.get("stage_latency_ms") or {}).items()
                },
            )
    except HTTPError as exc:
        return failed_sample(index, tenant_id, started, exc.code, safe_error_body(exc))
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        return failed_sample(index, tenant_id, started, 0, str(exc) or exc.__class__.__name__)


def failed_sample(
    index: int,
    tenant_id: str,
    started: float,
    status_code: int,
    error: str,
) -> SearchSample:
    return SearchSample(
        index=index,
        tenant_id=tenant_id,
        ok=False,
        latency_ms=elapsed_ms(started),
        status_code=status_code,
        error=error,
    )


def build_summary(
    args: argparse.Namespace,
    samples: list[SearchSample],
    *,
    wall_ms: float,
) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.ok]
    failed = [sample for sample in samples if not sample.ok]
    stages: dict[str, list[float]] = {}
    for sample in successful:
        for stage, latency in (sample.stage_latency_ms or {}).items():
            stages.setdefault(stage, []).append(latency)
    return {
        "target": args.base_url,
        "endpoint": "/search",
        "requests": len(samples),
        "concurrency": args.concurrency,
        "tenant_count": args.tenant_count,
        "success": len(successful),
        "failed": len(failed),
        "failure_rate": round(len(failed) / max(1, len(samples)), 4),
        "wall_ms": round(wall_ms, 2),
        "throughput_rps": round(len(samples) / max(0.001, wall_ms / 1000.0), 2),
        "latency_ms": summarize([sample.latency_ms for sample in successful]),
        "hit_count": summarize([float(sample.hit_count) for sample in successful]),
        "candidate_count": summarize([float(sample.candidate_count) for sample in successful]),
        "reranked_count": summarize([float(sample.reranked_count) for sample in successful]),
        "stage_latency_ms": {
            stage: summarize(values)
            for stage, values in sorted(stages.items())
        },
        "status_counts": dict(sorted(Counter(str(sample.status_code) for sample in samples).items())),
        "failed_samples": [asdict(sample) for sample in failed[:20]],
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 2),
        "avg": round(statistics.fmean(ordered), 2),
        "p50": round(percentile(ordered, 0.50), 2),
        "p95": round(percentile(ordered, 0.95), 2),
        "max": round(ordered[-1], 2),
    }


def percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def tenant_for_index(base_tenant_id: str, tenant_count: int, index: int) -> str:
    if tenant_count <= 1:
        return base_tenant_id
    return f"{base_tenant_id}-{index % tenant_count:04d}"


def safe_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except OSError:
        return str(exc.reason)


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


if __name__ == "__main__":
    main()
