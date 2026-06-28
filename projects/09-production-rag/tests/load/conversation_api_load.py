from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


OPERATIONS = ("create", "update", "list", "read", "delete")


@dataclass(frozen=True)
class ConversationSample:
    index: int
    ok: bool
    total_ms: float
    operation_ms: dict[str, float] = field(default_factory=dict)
    status_code: int = 0
    error: str = ""


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    samples = run_load(args)
    summary = build_summary(args, samples, wall_ms=elapsed_ms(started))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if any(not sample.ok for sample in samples):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent create/update/list/read/delete load for the conversation API."
    )
    parser.add_argument(
        "--base-urls",
        default="http://127.0.0.1:8008",
        help="Comma-separated API origins; each workflow rotates operations across them.",
    )
    parser.add_argument("--token", default="production-rag-fixed-test-login-token")
    parser.add_argument("--tenant-prefix", default="tenant-conversation-load")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def run_load(args: argparse.Namespace) -> list[ConversationSample]:
    users = max(1, args.users)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(run_user_workflow, args, index) for index in range(users)]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def run_user_workflow(args: argparse.Namespace, index: int) -> ConversationSample:
    origins = parse_base_urls(args.base_urls)
    tenant_id = f"{args.tenant_prefix}-{index:06d}"
    conversation_id = f"conversation-load-{uuid.uuid4().hex}"
    started = time.perf_counter()
    operation_ms: dict[str, float] = {}
    created = False
    try:
        timed_request(
            args,
            origins,
            index,
            "create",
            "/conversations",
            tenant_id,
            method="POST",
            payload=conversation_payload(conversation_id, tenant_id, "created", 1),
            operation_ms=operation_ms,
        )
        created = True
        timed_request(
            args,
            origins,
            index,
            "update",
            "/conversations",
            tenant_id,
            method="POST",
            payload=conversation_payload(conversation_id, tenant_id, "updated", 2),
            operation_ms=operation_ms,
        )
        listed = timed_request(
            args,
            origins,
            index,
            "list",
            f"/conversations?{urlencode({'tenant_id': tenant_id})}",
            tenant_id,
            operation_ms=operation_ms,
        )
        if not any(item.get("id") == conversation_id for item in listed.get("conversations") or []):
            raise RuntimeError("saved conversation missing from list")
        loaded = timed_request(
            args,
            origins,
            index,
            "read",
            f"/conversations/{conversation_id}?{urlencode({'tenant_id': tenant_id})}",
            tenant_id,
            operation_ms=operation_ms,
        )
        if loaded.get("title") != "updated" or len(loaded.get("messages") or []) != 2:
            raise RuntimeError("updated conversation was not read consistently")
        timed_request(
            args,
            origins,
            index,
            "delete",
            f"/conversations/{conversation_id}?{urlencode({'tenant_id': tenant_id})}",
            tenant_id,
            method="DELETE",
            operation_ms=operation_ms,
        )
        created = False
        return ConversationSample(index=index, ok=True, total_ms=elapsed_ms(started), operation_ms=operation_ms)
    except HTTPError as exc:
        return ConversationSample(
            index=index,
            ok=False,
            total_ms=elapsed_ms(started),
            operation_ms=operation_ms,
            status_code=exc.code,
            error=safe_error_body(exc),
        )
    except (OSError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
        return ConversationSample(
            index=index,
            ok=False,
            total_ms=elapsed_ms(started),
            operation_ms=operation_ms,
            error=str(exc) or exc.__class__.__name__,
        )
    finally:
        if created:
            try:
                request_json(
                    args,
                    origins[0],
                    f"/conversations/{conversation_id}?{urlencode({'tenant_id': tenant_id})}",
                    tenant_id,
                    method="DELETE",
                )
            except Exception:
                pass


def timed_request(
    args: argparse.Namespace,
    origins: list[str],
    user_index: int,
    operation: str,
    path: str,
    tenant_id: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    operation_ms: dict[str, float],
) -> dict[str, Any]:
    started = time.perf_counter()
    origin = origins[(user_index + OPERATIONS.index(operation)) % len(origins)]
    result = request_json(args, origin, path, tenant_id, method=method, payload=payload)
    operation_ms[operation] = elapsed_ms(started)
    return result


def request_json(
    args: argparse.Namespace,
    origin: str,
    path: str,
    tenant_id: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {args.token}",
        "X-RAG-Tenant-ID": tenant_id,
        "Content-Type": "application/json",
    }
    request = Request(
        urljoin(origin.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def conversation_payload(
    conversation_id: str,
    tenant_id: str,
    title: str,
    message_count: int,
) -> dict[str, Any]:
    return {
        "id": conversation_id,
        "tenant_id": tenant_id,
        "title": title,
        "source_doc_ids": [],
        "messages": [
            {
                "id": f"{conversation_id}-message-{index}",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"synthetic conversation load message {index}",
                "status": "done",
                "created_at": index + 1,
            }
            for index in range(message_count)
        ],
    }


def parse_base_urls(value: str) -> list[str]:
    origins = [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
    if not origins or any(not origin.startswith(("http://", "https://")) for origin in origins):
        raise ValueError("--base-urls must contain at least one HTTP(S) origin")
    return origins


def build_summary(
    args: argparse.Namespace,
    samples: list[ConversationSample],
    *,
    wall_ms: float,
) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.ok]
    failed = [sample for sample in samples if not sample.ok]
    return {
        "targets": parse_base_urls(args.base_urls),
        "users": args.users,
        "concurrency": args.concurrency,
        "wall_ms": wall_ms,
        "success": len(successful),
        "failed": len(failed),
        "failure_rate": round(len(failed) / max(1, len(samples)), 4),
        "workflow_throughput_rps": round(len(samples) / max(0.001, wall_ms / 1000.0), 2),
        "request_throughput_rps": round(
            sum(len(sample.operation_ms) for sample in samples) / max(0.001, wall_ms / 1000.0),
            2,
        ),
        "workflow_latency_ms": summarize([sample.total_ms for sample in successful]),
        "operation_latency_ms": {
            operation: summarize(
                [sample.operation_ms[operation] for sample in successful if operation in sample.operation_ms]
            )
            for operation in OPERATIONS
        },
        "failed_samples": [sample.__dict__ for sample in failed[:20]],
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "avg": round(statistics.fmean(ordered), 2),
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "max": round(ordered[-1], 2),
    }


def percentile(values: list[float], pct: int) -> float:
    index = min(len(values) - 1, round((pct / 100) * (len(values) - 1)))
    return round(values[index], 2)


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def safe_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception:
        return str(exc)


if __name__ == "__main__":
    main()
