from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class PollSample:
    tenant_id: str
    endpoint: str
    ok: bool
    latency_ms: float
    status_code: int = 0
    response_items: int = 0
    source_status_counts: dict[str, int] | None = None
    error: str = ""


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    samples = run_poll_load(args)
    wall_ms = elapsed_ms(started)
    payload = build_summary(args, samples, wall_ms=wall_ms)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if any(not sample.ok for sample in samples):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frontend idle polling load test for Production RAG.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--token", default="production-rag-fixed-test-login-token")
    parser.add_argument("--tenant-prefix", default="tenant-frontend-poll")
    parser.add_argument("--acl-groups", default="engineering")
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--think-time-ms", type=float, default=0.0)
    parser.add_argument(
        "--endpoints",
        default="sources,conversations,artifacts",
        help="Comma-separated endpoint groups: sources, conversations, artifacts.",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def run_poll_load(args: argparse.Namespace) -> list[PollSample]:
    users = max(1, args.users)
    endpoints = parse_endpoints(args.endpoints)
    samples: list[PollSample] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(run_user_rounds, args, user_index, endpoints)
            for user_index in range(users)
        ]
        for future in concurrent.futures.as_completed(futures):
            samples.extend(future.result())
    return samples


def run_user_rounds(args: argparse.Namespace, user_index: int, endpoints: list[str]) -> list[PollSample]:
    tenant_id = f"{args.tenant_prefix}-{user_index:04d}"
    samples: list[PollSample] = []
    for _round in range(max(1, args.rounds)):
        for endpoint in endpoints:
            samples.append(poll_endpoint(args, tenant_id=tenant_id, endpoint=endpoint))
        if args.think_time_ms > 0:
            time.sleep(args.think_time_ms / 1000.0)
    return samples


def poll_endpoint(args: argparse.Namespace, *, tenant_id: str, endpoint: str) -> PollSample:
    started = time.perf_counter()
    path = endpoint_path(args, tenant_id=tenant_id, endpoint=endpoint)
    request = Request(
        urljoin(args.base_url.rstrip("/") + "/", path.lstrip("/")),
        headers=headers(args, tenant_id=tenant_id),
        method="GET",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            source_status_counts = count_source_statuses(payload) if endpoint == "sources" else None
            return PollSample(
                tenant_id=tenant_id,
                endpoint=endpoint,
                ok=True,
                latency_ms=elapsed_ms(started),
                status_code=int(getattr(response, "status", 0) or 0),
                response_items=count_items(endpoint, payload),
                source_status_counts=source_status_counts,
            )
    except HTTPError as exc:
        return PollSample(
            tenant_id=tenant_id,
            endpoint=endpoint,
            ok=False,
            latency_ms=elapsed_ms(started),
            status_code=exc.code,
            error=safe_error_body(exc),
        )
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        return PollSample(
            tenant_id=tenant_id,
            endpoint=endpoint,
            ok=False,
            latency_ms=elapsed_ms(started),
            error=str(exc) or exc.__class__.__name__,
        )


def endpoint_path(args: argparse.Namespace, *, tenant_id: str, endpoint: str) -> str:
    if endpoint == "sources":
        return f"/sources?{urlencode({'tenant_id': tenant_id})}"
    if endpoint == "conversations":
        return f"/conversations?{urlencode({'tenant_id': tenant_id})}"
    if endpoint == "artifacts":
        return f"/artifacts?{urlencode({'tenant_id': tenant_id, 'workspace_id': args.workspace_id})}"
    raise ValueError(f"Unsupported endpoint group: {endpoint}")


def headers(args: argparse.Namespace, *, tenant_id: str) -> dict[str, str]:
    output = {
        "X-RAG-Tenant-ID": tenant_id,
        "X-RAG-ACL-Groups": args.acl_groups,
    }
    if args.token:
        output["Authorization"] = f"Bearer {args.token}"
    return output


def parse_endpoints(value: str) -> list[str]:
    allowed = {"sources", "conversations", "artifacts"}
    endpoints = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(endpoints) - allowed)
    if unknown:
        raise ValueError(f"Unsupported endpoint groups: {', '.join(unknown)}")
    return endpoints or ["sources", "conversations", "artifacts"]


def count_items(endpoint: str, payload: dict[str, Any]) -> int:
    if endpoint == "sources":
        return len(payload.get("sources") or [])
    if endpoint == "conversations":
        return len(payload.get("conversations") or [])
    if endpoint == "artifacts":
        return len(payload.get("artifacts") or [])
    return 0


def count_source_statuses(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for source in payload.get("sources") or []:
        if isinstance(source, dict):
            counts[str(source.get("status") or "unknown")] += 1
    return dict(sorted(counts.items()))


def build_summary(args: argparse.Namespace, samples: list[PollSample], *, wall_ms: float) -> dict[str, Any]:
    ok_samples = [sample for sample in samples if sample.ok]
    failed_samples = [sample for sample in samples if not sample.ok]
    by_endpoint: dict[str, dict[str, Any]] = {}
    for endpoint in parse_endpoints(args.endpoints):
        endpoint_samples = [sample for sample in samples if sample.endpoint == endpoint]
        endpoint_ok = [sample for sample in endpoint_samples if sample.ok]
        by_endpoint[endpoint] = {
            "requests": len(endpoint_samples),
            "success": len(endpoint_ok),
            "failed": len(endpoint_samples) - len(endpoint_ok),
            "failure_rate": round((len(endpoint_samples) - len(endpoint_ok)) / max(1, len(endpoint_samples)), 4),
            "latency_ms": summarize_values([sample.latency_ms for sample in endpoint_ok]),
            "response_items": summarize_values([sample.response_items for sample in endpoint_ok]),
            "source_status_counts": summarize_source_status_counts(endpoint_ok) if endpoint == "sources" else {},
        }
    return {
        "target": args.base_url,
        "users": args.users,
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "endpoints": parse_endpoints(args.endpoints),
        "wall_ms": wall_ms,
        "total_requests": len(samples),
        "success": len(ok_samples),
        "failed": len(failed_samples),
        "failure_rate": round(len(failed_samples) / max(1, len(samples)), 4),
        "throughput_rps": round(len(samples) / max(0.001, wall_ms / 1000.0), 2),
        "latency_ms": summarize_values([sample.latency_ms for sample in ok_samples]),
        "by_endpoint": by_endpoint,
        "failed_samples": [sample.__dict__ for sample in failed_samples[:20]],
    }


def summarize_source_status_counts(samples: list[PollSample]) -> dict[str, dict[str, float]]:
    statuses = sorted(
        {
            status
            for sample in samples
            for status in (sample.source_status_counts or {})
        }
    )
    return {
        status: summarize_values([
            (sample.source_status_counts or {}).get(status, 0)
            for sample in samples
        ])
        for status in statuses
    }


def summarize_values(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    data = sorted(float(value) for value in values)
    return {
        "avg": round(statistics.fmean(data), 2),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "min": round(data[0], 2),
        "max": round(data[-1], 2),
    }


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, round((pct / 100) * (len(values) - 1)))
    return round(values[index], 2)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def safe_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception:
        return str(exc)


if __name__ == "__main__":
    main()
