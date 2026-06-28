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
from urllib.parse import quote, urlencode, urljoin
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
    source_title: str = ""


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    samples = run_uploads(args)
    upload_elapsed_s = round(time.perf_counter() - start, 3)
    queue_result = wait_for_ingestion_completion(args, samples) if args.wait else {}
    cleanup_result = cleanup_ingested_sources(args, samples) if args.cleanup else {}
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
        "cleanup": cleanup_result,
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
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete only sources created by this synthetic run after measurement.",
    )
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
            payload = json.loads(response.read().decode("utf-8"))
            status_code = int(response.status)
        ok = 200 <= status_code < 300
        detail = "accepted" if ok else "http_error"
        source_title = str((payload.get("sources") or [{}])[0].get("title") or filename)
    except HTTPError as exc:
        status_code = int(exc.code)
        ok = False
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        source_title = filename
    except URLError as exc:
        status_code = 0
        ok = False
        detail = str(exc.reason)
        source_title = filename
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return UploadSample(
        index=index,
        tenant_id=tenant_id,
        ok=ok,
        status_code=status_code,
        latency_ms=latency_ms,
        detail=detail,
        source_title=source_title,
    )


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


def wait_for_ingestion_completion(
    args: argparse.Namespace,
    samples: list[UploadSample],
) -> dict[str, object]:
    expected_by_tenant: dict[str, set[str]] = {}
    for sample in samples:
        if sample.ok and sample.source_title:
            expected_by_tenant.setdefault(sample.tenant_id, set()).add(sample.source_title)
    expected_count = sum(len(titles) for titles in expected_by_tenant.values())
    if expected_count == 0:
        return {
            "completed": True,
            "all_ready": True,
            "expected": 0,
            "matched": 0,
            "missing": 0,
            "status_counts": {},
            "wait_elapsed_s": 0.0,
            "ready_per_second": 0.0,
            "poll_errors": 0,
            "recent_poll_errors": [],
        }

    started = time.perf_counter()
    deadline = time.monotonic() + max(1.0, args.wait_timeout)
    last_inventory: dict[str, int] = {}
    last_matched = 0
    poll_errors = 0
    recent_poll_errors: list[str] = []
    while time.monotonic() < deadline:
        counts: dict[str, int] = {}
        matched = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(1, args.concurrency), len(expected_by_tenant), 50)
        ) as executor:
            inventories = executor.map(
                lambda tenant_id: fetch_sources_safely(args, tenant_id=tenant_id),
                expected_by_tenant,
            )
            for tenant_id, sources, poll_error in inventories:
                if poll_error:
                    poll_errors += 1
                    recent_poll_errors.append(poll_error)
                    recent_poll_errors = recent_poll_errors[-20:]
                    continue
                expected_titles = expected_by_tenant[tenant_id]
                rows_by_title: dict[str, list[dict]] = {}
                for source in sources:
                    title = str(source.get("title") or "")
                    if title in expected_titles:
                        rows_by_title.setdefault(title, []).append(source)
                for title in expected_titles:
                    rows = rows_by_title.get(title) or []
                    if not rows:
                        continue
                    matched += 1
                    status = preferred_source_status(rows)
                    counts[status] = counts.get(status, 0) + 1
        last_inventory = counts
        last_matched = matched
        active = sum(counts.get(status, 0) for status in ACTIVE_STATUSES)
        if active == 0 and matched == expected_count:
            elapsed = round(time.perf_counter() - started, 3)
            ready = counts.get("ready", 0)
            return {
                "completed": True,
                "all_ready": ready == expected_count,
                "expected": expected_count,
                "matched": matched,
                "missing": 0,
                "status_counts": counts,
                "wait_elapsed_s": elapsed,
                "ready_per_second": round(ready / max(0.001, elapsed), 3),
                "poll_errors": poll_errors,
                "recent_poll_errors": recent_poll_errors,
            }
        time.sleep(max(0.1, args.poll_interval))
    elapsed = round(time.perf_counter() - started, 3)
    ready = last_inventory.get("ready", 0)
    return {
        "completed": False,
        "all_ready": False,
        "expected": expected_count,
        "matched": last_matched,
        "missing": expected_count - last_matched,
        "status_counts": last_inventory,
        "wait_elapsed_s": elapsed,
        "ready_per_second": round(ready / max(0.001, elapsed), 3),
        "poll_errors": poll_errors,
        "recent_poll_errors": recent_poll_errors,
    }


def fetch_sources(args: argparse.Namespace, *, tenant_id: str) -> list[dict]:
    query = urlencode({"tenant_id": tenant_id})
    headers = {
        "X-RAG-Tenant-ID": tenant_id,
        "X-RAG-ACL-Groups": args.acl_groups,
    }
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


def fetch_sources_safely(
    args: argparse.Namespace,
    *,
    tenant_id: str,
) -> tuple[str, list[dict], str]:
    try:
        return tenant_id, fetch_sources(args, tenant_id=tenant_id), ""
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return tenant_id, [], f"tenant={tenant_id} status={exc.code} detail={detail}"
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        return tenant_id, [], f"tenant={tenant_id} error={str(exc)[:300]}"


def fetch_sources_with_retries(
    args: argparse.Namespace,
    *,
    tenant_id: str,
    attempts: int = 3,
) -> tuple[list[dict], str]:
    last_error = ""
    for attempt in range(max(1, attempts)):
        _, sources, last_error = fetch_sources_safely(args, tenant_id=tenant_id)
        if not last_error:
            return sources, ""
        if attempt + 1 < attempts:
            time.sleep(min(1.0, max(0.01, args.poll_interval)))
    return [], last_error


def preferred_source_status(rows: list[dict]) -> str:
    priority = {"failed": 4, "processing": 3, "queued": 2, "uploading": 1, "ready": 0}
    return max(
        (str(row.get("status") or "unknown") for row in rows),
        key=lambda status: priority.get(status, -1),
    )


def cleanup_ingested_sources(
    args: argparse.Namespace,
    samples: list[UploadSample],
) -> dict[str, object]:
    expected_by_tenant: dict[str, set[str]] = {}
    for sample in samples:
        if sample.ok and sample.source_title:
            expected_by_tenant.setdefault(sample.tenant_id, set()).add(sample.source_title)
    targets: set[tuple[str, str]] = set()
    failures: list[dict[str, object]] = []
    for tenant_id, expected_titles in expected_by_tenant.items():
        sources, discovery_error = fetch_sources_with_retries(args, tenant_id=tenant_id)
        if discovery_error:
            failures.append(
                {
                    "tenant_id": tenant_id,
                    "doc_id": "",
                    "status": 0,
                    "stage": "discovery",
                    "error": discovery_error,
                }
            )
            continue
        for source in sources:
            if str(source.get("title") or "") in expected_titles:
                doc_id = str(source.get("doc_id") or "")
                if doc_id:
                    targets.add((tenant_id, doc_id))

    deleted = 0
    for tenant_id, doc_id in sorted(targets):
        query = urlencode({"tenant_id": tenant_id})
        headers = {
            "X-RAG-Tenant-ID": tenant_id,
            "X-RAG-ACL-Groups": args.acl_groups,
        }
        if args.token:
            headers["Authorization"] = f"Bearer {args.token}"
        request = Request(
            urljoin(
                args.base_url.rstrip("/") + "/",
                f"sources/{quote(doc_id, safe='')}?{query}",
            ),
            headers=headers,
            method="DELETE",
        )
        try:
            with urlopen(request, timeout=args.timeout) as response:
                response.read()
                if 200 <= int(response.status) < 300:
                    deleted += 1
                else:
                    failures.append({"tenant_id": tenant_id, "doc_id": doc_id, "status": response.status})
        except HTTPError as exc:
            failures.append({"tenant_id": tenant_id, "doc_id": doc_id, "status": exc.code})
        except URLError as exc:
            failures.append({"tenant_id": tenant_id, "doc_id": doc_id, "status": 0, "error": str(exc.reason)})
    return {
        "targets": len(targets),
        "deleted": deleted,
        "failed": len(failures),
        "failures": failures[:20],
    }


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
