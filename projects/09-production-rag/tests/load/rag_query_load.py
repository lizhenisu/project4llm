from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


EXTERNAL_LLM_STAGES = {"rewrite", "answer"}
EXTERNAL_EMBEDDING_STAGES = {"embedding", "text_embedding", "image_embedding"}
EXTERNAL_RERANK_STAGES = {"rerank"}
DEFAULT_QUESTION = "总结这些资料的核心内容"


@dataclass
class RequestSample:
    index: int
    request_id: str
    ok: bool
    total_ms: float
    status_code: int = 0
    resolved_tenant_id: str = ""
    first_event_ms: float | None = None
    answer_chars: int = 0
    citations: int = 0
    error: str = ""
    error_kind: str = ""
    stages_ms: dict[str, float] = field(default_factory=dict)
    stage_events: int = 0


def main() -> None:
    args = parse_args()
    questions = load_questions(args)
    summary = asyncio.run(run_load(args, questions))
    print_summary(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    strict_failure = args.max_failure_rate is None and bool(summary["failed"])
    if strict_failure or not summary["capacity_gate"]["passed"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load-test a running Production RAG /query/stream endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--endpoint", default="/query/stream")
    parser.add_argument("--token", help="Bearer token, for example production-rag-fixed-test-login-token.")
    parser.add_argument("--tenant-id", default="team_a")
    parser.add_argument(
        "--acl-group",
        action="append",
        default=[],
        help="Allowed ACL group. Repeat for multiple groups.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--questions-file",
        type=Path,
        help="UTF-8 text file with one question per line. Blank lines are ignored.",
    )
    parser.add_argument("--requests", type=positive_int, default=20)
    parser.add_argument("--concurrency", type=positive_int, default=3)
    parser.add_argument("--warmup", type=non_negative_int, default=0)
    parser.add_argument("--candidate-limit", type=positive_int, default=20)
    parser.add_argument("--context-limit", type=positive_int, default=5)
    parser.add_argument("--doc-version", type=int)
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument(
        "--include-all-sources",
        action="store_true",
        help="Exercise all visible ready sources instead of only current documents.",
    )
    parser.add_argument("--history", action="append", default=[])
    parser.add_argument(
        "--query-mode",
        choices=["text", "multimodal"],
        default="text",
        help="Use text for normal RAG query load. Multimodal requires --image-data-url.",
    )
    parser.add_argument("--image-data-url")
    parser.add_argument(
        "--external-mode",
        choices=["real", "mock"],
        default="real",
        help="Report label only. Point backend env vars to mock_external_api.py for mock mode.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-failure-rate", type=non_negative_float)
    parser.add_argument("--max-p95-ms", type=non_negative_float)
    parser.add_argument("--max-first-event-p95-ms", type=non_negative_float)
    parser.add_argument("--min-throughput-rps", type=non_negative_float)
    parser.add_argument("--min-accepted-rate", type=unit_interval)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling new work after the first failed request.",
    )
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> list[str]:
    if not args.questions_file:
        return [args.question]
    lines = [
        line.strip()
        for line in args.questions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise SystemExit(f"No questions found in {args.questions_file}")
    return lines


async def run_load(args: argparse.Namespace, questions: list[str]) -> dict[str, Any]:
    warmup_samples: list[RequestSample] = []
    if args.warmup > 0:
        warmup_samples = await run_batch(args, questions, total=args.warmup, label="warmup")

    started = time.perf_counter()
    samples = await run_batch(args, questions, total=args.requests, label="load")
    wall_ms = elapsed_ms(started)
    return build_summary(args, samples, wall_ms=wall_ms, warmup_samples=warmup_samples)


async def run_batch(
    args: argparse.Namespace,
    questions: list[str],
    *,
    total: int,
    label: str,
) -> list[RequestSample]:
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=max(1, args.concurrency),
        thread_name_prefix=f"rag-query-load-{label}",
    )

    async def run_one(index: int) -> RequestSample:
        async with semaphore:
            if stop.is_set():
                return RequestSample(
                    index=index,
                    request_id="skipped",
                    ok=False,
                    total_ms=0,
                    error="skipped after fail-fast",
                    error_kind="fail_fast_skip",
                )
            question = random.choice(questions)
            sample = await loop.run_in_executor(
                executor,
                send_stream_request,
                args,
                question,
                index,
                label,
            )
            if args.fail_fast and not sample.ok:
                stop.set()
            return sample

    try:
        tasks = [asyncio.create_task(run_one(index)) for index in range(total)]
        samples = []
        for task in asyncio.as_completed(tasks):
            sample = await task
            samples.append(sample)
            status = "ok" if sample.ok else "fail"
            print(
                f"[{label}] #{sample.index + 1} {status} "
                f"total={sample.total_ms:.1f}ms request_id={sample.request_id}",
                file=sys.stderr,
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    samples.sort(key=lambda item: item.index)
    return samples


def send_stream_request(
    args: argparse.Namespace,
    question: str,
    index: int,
    label: str,
) -> RequestSample:
    request_id = f"load-{label}-{uuid.uuid4().hex[:12]}"
    payload = {
        "query": question,
        "query_mode": args.query_mode,
        "image_data_url": args.image_data_url,
        "history": args.history,
        "tenant_id": args.tenant_id,
        "acl_groups": args.acl_group,
        "doc_version": args.doc_version,
        "doc_ids": args.doc_id,
        "source_types": args.source_type,
        "include_all_sources": args.include_all_sources,
        "candidate_limit": args.candidate_limit,
        "context_limit": args.context_limit,
        "request_id": request_id,
    }
    payload = {key: value for key, value in payload.items() if value not in (None, [], "")}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/x-ndjson",
        "X-RAG-Tenant-ID": args.tenant_id,
    }
    if args.acl_group:
        headers["X-RAG-ACL-Groups"] = ",".join(args.acl_group)
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    url = urljoin(args.base_url.rstrip("/") + "/", args.endpoint.lstrip("/"))
    started = time.perf_counter()
    first_event_ms: float | None = None
    stages_ms: dict[str, float] = {}
    stage_events = 0
    answer_chars = 0
    citations = 0
    error = ""
    error_kind = ""
    status_code = 0
    resolved_tenant_id = ""
    ok = False
    try:
        request = Request(url, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=args.timeout) as response:
            status_code = int(getattr(response, "status", 200))
            for raw_line in response:
                if first_event_ms is None:
                    first_event_ms = elapsed_ms(started)
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                event_type = event.get("type")
                if event_type == "stage":
                    stage_events += 1
                    if event.get("status") == "done" and event.get("latency_ms") is not None:
                        stages_ms[str(event.get("stage"))] = float(event["latency_ms"])
                elif event_type == "result":
                    resolved_tenant_id = str((event.get("trace") or {}).get("tenant_id") or "")
                    error = tenant_resolution_error(args.tenant_id, resolved_tenant_id)
                    if error:
                        error_kind = "tenant_resolution_mismatch"
                    else:
                        ok = True
                    answer_chars = len(str(event.get("answer") or ""))
                    citations = len(event.get("citations") or [])
                    trace = event.get("trace") or {}
                    for stage, latency in (trace.get("stage_latency_ms") or {}).items():
                        stages_ms.setdefault(str(stage), float(latency))
                elif event_type == "error":
                    error = str(event.get("detail") or "stream error")
                    error_kind = "stream_error"
    except HTTPError as exc:
        status_code = int(exc.code)
        error = f"HTTP {exc.code}: {safe_error_body(exc)}"
        error_kind = classify_http_error(exc.code, error)
    except json.JSONDecodeError as exc:
        error = str(exc) or exc.__class__.__name__
        error_kind = "invalid_stream_json"
    except (TimeoutError, URLError, OSError) as exc:
        error = str(exc) or exc.__class__.__name__
        error_kind = "transport_or_timeout"

    total_ms = elapsed_ms(started)
    if not ok and not error:
        error = "stream ended without result"
        error_kind = "missing_result"
    return RequestSample(
        index=index,
        request_id=request_id,
        ok=ok,
        total_ms=total_ms,
        status_code=status_code,
        resolved_tenant_id=resolved_tenant_id,
        first_event_ms=first_event_ms,
        answer_chars=answer_chars,
        citations=citations,
        error=error,
        error_kind=error_kind,
        stages_ms=stages_ms,
        stage_events=stage_events,
    )


def safe_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except OSError:
        return exc.reason


def classify_http_error(status_code: int, detail: str) -> str:
    lowered = detail.lower()
    if status_code == 503:
        for kind in ("tenant", "user", "global"):
            if kind in lowered:
                return f"rejected_{kind}"
        return "rejected_capacity"
    return f"http_{status_code}"


def tenant_resolution_error(requested: str, resolved: str) -> str:
    if not resolved or requested == resolved:
        return ""
    return f"tenant_resolution_mismatch: requested={requested} resolved={resolved}"


def build_summary(
    args: argparse.Namespace,
    samples: list[RequestSample],
    *,
    wall_ms: float,
    warmup_samples: list[RequestSample] | None = None,
) -> dict[str, Any]:
    ok_samples = [sample for sample in samples if sample.ok]
    failed_samples = [sample for sample in samples if not sample.ok]
    warmup_samples = warmup_samples or []
    accepted = sum(1 for sample in samples if 200 <= sample.status_code < 300)
    stage_samples: dict[str, list[float]] = {}
    for sample in ok_samples:
        for stage, latency in sample.stages_ms.items():
            stage_samples.setdefault(stage, []).append(latency)

    total_values = [sample.total_ms for sample in ok_samples]
    first_event_values = [
        sample.first_event_ms for sample in ok_samples if sample.first_event_ms is not None
    ]
    stage_summary = {stage: summarize(values) for stage, values in sorted(stage_samples.items())}
    external_summary = summarize_external(stage_samples)
    unattributed_values = [
        max(0.0, sample.total_ms - sum(sample.stages_ms.values()))
        for sample in ok_samples
    ]
    summary = {
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "external_mode": args.external_mode,
        "requests": len(samples),
        "concurrency": args.concurrency,
        "warmup": {
            "requests": len(warmup_samples),
            "success": sum(1 for sample in warmup_samples if sample.ok),
            "failed": sum(1 for sample in warmup_samples if not sample.ok),
        },
        "accepted": accepted,
        "accepted_rate": round(accepted / max(1, len(samples)), 4),
        "success": len(ok_samples),
        "failed": len(failed_samples),
        "failure_rate": round(len(failed_samples) / max(1, len(samples)), 4),
        "wall_ms": round(wall_ms, 2),
        "throughput_rps": round(len(ok_samples) / (wall_ms / 1000), 3) if wall_ms > 0 else 0,
        "latency_ms": summarize(total_values),
        "first_event_ms": summarize(first_event_values),
        "stage_latency_ms": stage_summary,
        "external_latency_ms": external_summary,
        "unattributed_ms": summarize(unattributed_values),
        "answer_chars": summarize([sample.answer_chars for sample in ok_samples]),
        "citations": summarize([sample.citations for sample in ok_samples]),
        "status_counts": dict(
            sorted(Counter(str(sample.status_code) for sample in samples).items())
        ),
        "error_kind_counts": dict(
            sorted(Counter(sample.error_kind or "unknown" for sample in failed_samples).items())
        ),
        "failures": [
            {
                "index": sample.index,
                "request_id": sample.request_id,
                "status_code": sample.status_code,
                "error_kind": sample.error_kind,
                "error": sample.error,
            }
            for sample in failed_samples[:20]
        ],
    }
    summary["capacity_gate"] = build_capacity_gate(args, summary)
    return summary


def build_capacity_gate(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    thresholds = (
        (
            "max_failure_rate",
            getattr(args, "max_failure_rate", None),
            summary["failure_rate"],
            lambda actual, limit: actual <= limit,
        ),
        (
            "max_p95_ms",
            getattr(args, "max_p95_ms", None),
            summary["latency_ms"]["p95"],
            lambda actual, limit: actual <= limit,
        ),
        (
            "max_first_event_p95_ms",
            getattr(args, "max_first_event_p95_ms", None),
            summary["first_event_ms"]["p95"],
            lambda actual, limit: actual <= limit,
        ),
        (
            "min_throughput_rps",
            getattr(args, "min_throughput_rps", None),
            summary["throughput_rps"],
            lambda actual, limit: actual >= limit,
        ),
        (
            "min_accepted_rate",
            getattr(args, "min_accepted_rate", None),
            summary["accepted_rate"],
            lambda actual, limit: actual >= limit,
        ),
    )
    checks = {
        name: {"actual": actual, "threshold": limit, "passed": predicate(actual, limit)}
        for name, limit, actual, predicate in thresholds
        if limit is not None
    }
    return {
        "enabled": bool(checks),
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def summarize_external(stage_samples: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "llm": summarize(sum_parallel_stage_groups(stage_samples, EXTERNAL_LLM_STAGES)),
        "embedding": summarize(sum_parallel_stage_groups(stage_samples, EXTERNAL_EMBEDDING_STAGES)),
        "rerank": summarize(sum_parallel_stage_groups(stage_samples, EXTERNAL_RERANK_STAGES)),
    }


def sum_parallel_stage_groups(
    stage_samples: dict[str, list[float]],
    stages: set[str],
) -> list[float]:
    selected = [values for stage, values in stage_samples.items() if stage in stages]
    if not selected:
        return []
    count = min(len(values) for values in selected)
    return [sum(values[index] for values in selected) for index in range(count)]


def summarize(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    floats = [float(value) for value in values]
    return {
        "avg": round(statistics.fmean(floats), 2),
        "p50": round(percentile(floats, 0.50), 2),
        "p90": round(percentile(floats, 0.90), 2),
        "p95": round(percentile(floats, 0.95), 2),
        "min": round(min(floats), 2),
        "max": round(max(floats), 2),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return ordered[index]


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    main()
