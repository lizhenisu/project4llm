from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


ACTIVE_STATUSES = {"uploading", "queued", "processing"}


@dataclass(frozen=True)
class UploadSample:
    index: int
    tenant_id: str
    ok: bool
    status_code: int
    latency_ms: float
    detail: str


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    samples = run_uploads(args)
    upload_elapsed_s = round(time.perf_counter() - start, 3)
    queue_result = wait_for_ingestion_completion(args) if args.wait else {}
    summary = summarize(samples, upload_elapsed_s=upload_elapsed_s, queue_result=queue_result)
    payload = {
        "target": args.base_url,
        "tenant_id": args.tenant_id,
        "tenant_prefix": args.tenant_prefix,
        "users": args.users,
        "docs_per_user": args.docs_per_user,
        "uploads": upload_count(args),
        "concurrency": args.concurrency,
        "file_size_bytes": args.file_size_bytes,
        "summary": summary,
        "sample_limit": args.sample_limit,
        "samples": [sample.__dict__ for sample in samples[: args.sample_limit]],
        "failed_samples": [sample.__dict__ for sample in samples if not sample.ok],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent synthetic upload load test for Production RAG ingestion.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--token", default="")
    parser.add_argument("--tenant-id", default="tenant-fixed-test")
    parser.add_argument("--tenant-prefix", default="tenant-load")
    parser.add_argument("--acl-groups", default="engineering")
    parser.add_argument("--uploads", type=int, default=20)
    parser.add_argument("--users", type=int, default=0)
    parser.add_argument("--docs-per-user", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--file-size-bytes", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--wait", action="store_true", help="Poll /sources until queued/processing tasks finish.")
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def run_uploads(args: argparse.Namespace) -> list[UploadSample]:
    uploads = upload_count(args)
    concurrency = max(1, args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(upload_one, args, index) for index in range(uploads)]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def upload_one(args: argparse.Namespace, index: int) -> UploadSample:
    started = time.perf_counter()
    tenant_id = tenant_for_index(args, index)
    boundary = f"----production-rag-load-{uuid.uuid4().hex}"
    filename = f"synthetic-load-{index:06d}-{uuid.uuid4().hex[:12]}.txt"
    body = multipart_body(
        boundary=boundary,
        fields={
            "tenant_id": tenant_id,
            "acl_groups": args.acl_groups,
            "language": "zh",
        },
        files={
            "file": (
                filename,
                "text/plain",
                synthetic_document(index=index, size_bytes=max(128, args.file_size_bytes)),
            ),
        },
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        "X-RAG-Tenant-ID": tenant_id,
        "X-RAG-ACL-Groups": args.acl_groups,
    }
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    request = Request(
        urljoin(args.base_url.rstrip("/") + "/", "sources/upload"),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            response.read()
            status_code = int(response.status)
        ok = 200 <= status_code < 300
        detail = "accepted" if ok else "http_error"
    except HTTPError as exc:
        status_code = int(exc.code)
        ok = False
        detail = exc.read().decode("utf-8", errors="replace")[:500]
    except URLError as exc:
        status_code = 0
        ok = False
        detail = str(exc.reason)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return UploadSample(index=index, tenant_id=tenant_id, ok=ok, status_code=status_code, latency_ms=latency_ms, detail=detail)


def upload_count(args: argparse.Namespace) -> int:
    if args.users > 0 and args.docs_per_user > 0:
        return args.users * args.docs_per_user
    return max(1, args.uploads)


def tenant_for_index(args: argparse.Namespace, index: int) -> str:
    if args.users > 0 and args.docs_per_user > 0:
        user_index = index // args.docs_per_user
        return f"{args.tenant_prefix}-{user_index:04d}"
    return args.tenant_id


def multipart_body(
    *,
    boundary: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, str, bytes]],
) -> bytes:
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, (filename, content_type, content) in files.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8"),
                content,
                b"\r\n",
            ]
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts)


def synthetic_document(*, index: int, size_bytes: int) -> bytes:
    paragraph = (
        f"Synthetic ingestion load document {index}. "
        "This file is generated for upload throughput testing only. "
        "It contains repeated RAG, embedding, chunking, and source guide terms.\n"
    )
    content = paragraph
    while len(content.encode("utf-8")) < size_bytes:
        content += paragraph
    return content.encode("utf-8")[:size_bytes]


def wait_for_ingestion_completion(args: argparse.Namespace) -> dict[str, object]:
    deadline = time.monotonic() + max(1.0, args.wait_timeout)
    last_inventory: dict[str, int] = {}
    while time.monotonic() < deadline:
        sources = fetch_sources(args)
        counts: dict[str, int] = {}
        for source in sources:
            status = str(source.get("status") or "")
            counts[status] = counts.get(status, 0) + 1
        last_inventory = counts
        active = sum(counts.get(status, 0) for status in ACTIVE_STATUSES)
        if active == 0:
            return {"completed": True, "status_counts": counts}
        time.sleep(max(0.1, args.poll_interval))
    return {"completed": False, "status_counts": last_inventory}


def fetch_sources(args: argparse.Namespace) -> list[dict]:
    query = urlencode({"tenant_id": args.tenant_id})
    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    request = Request(
        urljoin(args.base_url.rstrip("/") + "/", f"sources?{query}"),
        headers=headers,
        method="GET",
    )
    with urlopen(request, timeout=args.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return list(body.get("sources") or [])


def summarize(
    samples: list[UploadSample],
    *,
    upload_elapsed_s: float,
    queue_result: dict[str, object],
) -> dict[str, object]:
    latencies = [sample.latency_ms for sample in samples]
    ok = sum(1 for sample in samples if sample.ok)
    failed = len(samples) - ok
    return {
        "requests": len(samples),
        "accepted": ok,
        "failed": failed,
        "failure_rate": round(failed / max(1, len(samples)), 4),
        "upload_elapsed_s": upload_elapsed_s,
        "accepted_per_second": round(ok / max(0.001, upload_elapsed_s), 3),
        "unique_tenants": len({sample.tenant_id for sample in samples}),
        "latency_ms": {
            "min": min(latencies) if latencies else 0,
            "avg": round(statistics.fmean(latencies), 2) if latencies else 0,
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else 0,
        },
        "ingestion_wait": queue_result,
    }


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return round(ordered[index], 2)


if __name__ == "__main__":
    main()
