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

from ingestion_upload_load import multipart_body, synthetic_document


ACTIVE_SOURCE_STATUSES = {"queued", "processing", "uploading"}


@dataclass(frozen=True)
class WorkflowSample:
    index: int
    tenant_id: str
    ok: bool
    total_ms: float
    upload_ms: float = 0.0
    source_poll_ms: float = 0.0
    pending_save_ms: float = 0.0
    query_ms: float = 0.0
    final_save_ms: float = 0.0
    first_event_ms: float | None = None
    answer_chars: int = 0
    status_code: int = 0
    error: str = ""
    stage_events: int = 0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    samples = run_workflows(args)
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
    parser = argparse.ArgumentParser(description="Frontend-like workflow load test for Production RAG.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--token", default="production-rag-fixed-test-login-token")
    parser.add_argument("--tenant-prefix", default="tenant-frontend-load")
    parser.add_argument("--acl-groups", default="engineering")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--file-size-bytes", type=int, default=512)
    parser.add_argument("--question", default="总结这个文件的核心内容")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--source-poll-timeout", type=float, default=60.0)
    parser.add_argument("--source-poll-interval", type=float, default=1.0)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def run_workflows(args: argparse.Namespace) -> list[WorkflowSample]:
    total = max(1, args.users)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(run_workflow, args, index) for index in range(total)]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def run_workflow(args: argparse.Namespace, index: int) -> WorkflowSample:
    tenant_id = f"{args.tenant_prefix}-{index:04d}"
    started = time.perf_counter()
    doc_ids: list[str] = []
    uploaded_title = ""
    upload_ms = 0.0
    source_poll_ms = 0.0
    pending_save_ms = 0.0
    query_ms = 0.0
    final_save_ms = 0.0
    first_event_ms: float | None = None
    answer_chars = 0
    stage_events = 0
    stage_latency_ms: dict[str, float] = {}
    conversation_id = f"workflow-{uuid.uuid4().hex}"
    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())
    try:
        if not args.skip_upload:
            upload_started = time.perf_counter()
            uploaded, uploaded_title = upload_source(args, tenant_id=tenant_id, index=index)
            upload_ms = elapsed_ms(upload_started)
            doc_ids = [source["doc_id"] for source in uploaded if source.get("doc_id")]
            poll_started = time.perf_counter()
            ready_sources = wait_for_sources(
                args,
                tenant_id=tenant_id,
                uploaded_doc_ids=doc_ids,
                uploaded_title=uploaded_title,
            )
            source_poll_ms = elapsed_ms(poll_started)
            ready_doc_ids = [source["doc_id"] for source in ready_sources if source.get("status") == "ready"]
            doc_ids = ready_doc_ids or doc_ids
        user_message = {
            "id": user_message_id,
            "role": "user",
            "content": args.question,
            "status": "done",
            "request_id": None,
            "citations": [],
            "image_data_url": None,
            "created_at": now_ms(),
            "feedback_rating": None,
            "rag_progress": [],
        }
        pending_message = {
            "id": assistant_message_id,
            "role": "assistant",
            "content": "RAG 调用链启动中...",
            "status": "sending",
            "request_id": None,
            "citations": [],
            "image_data_url": None,
            "created_at": now_ms(),
            "feedback_rating": None,
            "rag_progress": [],
        }
        pending_started = time.perf_counter()
        save_conversation(args, tenant_id=tenant_id, conversation_id=conversation_id, messages=[user_message, pending_message], doc_ids=doc_ids)
        pending_save_ms = elapsed_ms(pending_started)
        response: dict[str, Any] = {"answer": "", "request_id": "", "citations": []}
        if not args.skip_query:
            query_started = time.perf_counter()
            response, first_event_ms, stage_events, stage_latency_ms = query_stream(args, tenant_id=tenant_id, doc_ids=doc_ids)
            query_ms = elapsed_ms(query_started)
        final_message = {
            **pending_message,
            "content": response.get("answer") or "",
            "status": "done",
            "request_id": response.get("request_id") or None,
            "citations": response.get("citations") or [],
            "rag_progress": [],
        }
        final_started = time.perf_counter()
        save_conversation(args, tenant_id=tenant_id, conversation_id=conversation_id, messages=[user_message, final_message], doc_ids=doc_ids)
        final_save_ms = elapsed_ms(final_started)
        answer_chars = len(str(response.get("answer") or ""))
        return WorkflowSample(
            index=index,
            tenant_id=tenant_id,
            ok=True,
            total_ms=elapsed_ms(started),
            upload_ms=upload_ms,
            source_poll_ms=source_poll_ms,
            pending_save_ms=pending_save_ms,
            query_ms=query_ms,
            final_save_ms=final_save_ms,
            first_event_ms=first_event_ms,
            answer_chars=answer_chars,
            stage_events=stage_events,
            stage_latency_ms=stage_latency_ms,
        )
    except HTTPError as exc:
        return failure_sample(index, tenant_id, started, exc.code, safe_error_body(exc), upload_ms, source_poll_ms, pending_save_ms, query_ms, final_save_ms)
    except (OSError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
        return failure_sample(index, tenant_id, started, 0, str(exc) or exc.__class__.__name__, upload_ms, source_poll_ms, pending_save_ms, query_ms, final_save_ms)


def upload_source(args: argparse.Namespace, *, tenant_id: str, index: int) -> tuple[list[dict[str, Any]], str]:
    boundary = f"----production-rag-workflow-{uuid.uuid4().hex}"
    filename = f"workflow-{index:04d}-{uuid.uuid4().hex[:8]}.txt"
    body = multipart_body(
        boundary=boundary,
        fields={"tenant_id": tenant_id, "acl_groups": args.acl_groups, "language": "zh"},
        files={
            "file": (
                filename,
                "text/plain",
                synthetic_document(index=index, size_bytes=max(128, args.file_size_bytes)),
            )
        },
    )
    payload = request_json(
        args,
        "/sources/upload",
        tenant_id=tenant_id,
        method="POST",
        body=body,
        extra_headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    return list(payload.get("sources") or []), filename


def wait_for_sources(
    args: argparse.Namespace,
    *,
    tenant_id: str,
    uploaded_doc_ids: list[str],
    uploaded_title: str,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + max(1.0, args.source_poll_timeout)
    last_sources: list[dict[str, Any]] = []
    expected_ids = set(uploaded_doc_ids)
    while time.monotonic() < deadline:
        payload = request_json(args, f"/sources?{urlencode({'tenant_id': tenant_id})}", tenant_id=tenant_id)
        all_sources = list(payload.get("sources") or [])
        sources = [source for source in all_sources if source_matches_upload(source, expected_ids, uploaded_title)]
        last_sources = sources
        failed = [source for source in sources if source.get("status") == "failed"]
        if failed:
            raise RuntimeError(str(failed[0].get("error") or "source ingestion failed"))
        ready = [source for source in sources if source.get("status") == "ready"]
        if ready:
            return ready
        time.sleep(max(0.1, args.source_poll_interval))
    raise TimeoutError(
        f"sources did not become ready before timeout; expected_ids={sorted(expected_ids)} "
        f"title={uploaded_title!r} last={last_sources[:3]}"
    )


def source_matches_upload(source: dict[str, Any], expected_ids: set[str], uploaded_title: str) -> bool:
    doc_id = str(source.get("doc_id") or "")
    title = str(source.get("title") or "")
    return doc_id in expected_ids or title == uploaded_title


def save_conversation(
    args: argparse.Namespace,
    *,
    tenant_id: str,
    conversation_id: str,
    messages: list[dict[str, Any]],
    doc_ids: list[str],
) -> dict[str, Any]:
    return request_json(
        args,
        "/conversations",
        tenant_id=tenant_id,
        method="POST",
        json_payload={
            "id": conversation_id,
            "tenant_id": tenant_id,
            "title": "workflow load conversation",
            "messages": messages,
            "source_doc_ids": doc_ids,
        },
    )


def query_stream(
    args: argparse.Namespace,
    *,
    tenant_id: str,
    doc_ids: list[str],
) -> tuple[dict[str, Any], float | None, int, dict[str, float]]:
    body = json.dumps(
        {
            "query": args.question,
            "query_mode": "text",
            "history": [],
            "tenant_id": tenant_id,
            "acl_groups": acl_groups(args),
            "doc_ids": doc_ids,
            "candidate_limit": 20,
            "context_limit": 5,
            "request_id": f"workflow-{uuid.uuid4().hex}",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        urljoin(args.base_url.rstrip("/") + "/", "query/stream"),
        data=body,
        headers=headers(args, tenant_id=tenant_id, content_type="application/json", accept="application/x-ndjson"),
        method="POST",
    )
    started = time.perf_counter()
    first_event_ms: float | None = None
    stage_events = 0
    stage_latency_ms: dict[str, float] = {}
    result: dict[str, Any] | None = None
    with urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            if first_event_ms is None:
                first_event_ms = elapsed_ms(started)
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "stage":
                stage_events += 1
                if event.get("status") == "done" and event.get("latency_ms") is not None:
                    stage_latency_ms[str(event.get("stage"))] = float(event["latency_ms"])
            elif event.get("type") == "result":
                result = event
            elif event.get("type") == "error":
                raise RuntimeError(str(event.get("detail") or "stream error"))
    if result is None:
        raise RuntimeError("stream ended without result")
    return result, first_event_ms, stage_events, stage_latency_ms


def request_json(
    args: argparse.Namespace,
    path: str,
    *,
    tenant_id: str,
    method: str = "GET",
    json_payload: dict[str, Any] | None = None,
    body: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    if json_payload is not None:
        body = json.dumps(json_payload, ensure_ascii=False).encode("utf-8")
    request_headers = headers(
        args,
        tenant_id=tenant_id,
        content_type="application/json" if json_payload is not None else None,
    )
    for key, value in (extra_headers or {}).items():
        request_headers[key] = value
    request = Request(
        urljoin(args.base_url.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        headers=request_headers,
        method=method,
    )
    with urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def headers(
    args: argparse.Namespace,
    *,
    tenant_id: str,
    content_type: str | None = None,
    accept: str | None = None,
) -> dict[str, str]:
    output = {
        "X-RAG-Tenant-ID": tenant_id,
        "X-RAG-ACL-Groups": args.acl_groups,
    }
    if content_type:
        output["Content-Type"] = content_type
    if accept:
        output["Accept"] = accept
    if args.token:
        output["Authorization"] = f"Bearer {args.token}"
    return output


def build_summary(args: argparse.Namespace, samples: list[WorkflowSample], *, wall_ms: float) -> dict[str, Any]:
    ok_samples = [sample for sample in samples if sample.ok]
    failed_samples = [sample for sample in samples if not sample.ok]
    return {
        "target": args.base_url,
        "users": args.users,
        "concurrency": args.concurrency,
        "skip_upload": args.skip_upload,
        "skip_query": args.skip_query,
        "wall_ms": wall_ms,
        "success": len(ok_samples),
        "failed": len(failed_samples),
        "failure_rate": round(len(failed_samples) / max(1, len(samples)), 4),
        "latency_ms": summarize_values([sample.total_ms for sample in ok_samples]),
        "upload_ms": summarize_values([sample.upload_ms for sample in ok_samples if sample.upload_ms > 0]),
        "source_poll_ms": summarize_values([sample.source_poll_ms for sample in ok_samples if sample.source_poll_ms > 0]),
        "pending_save_ms": summarize_values([sample.pending_save_ms for sample in ok_samples]),
        "query_ms": summarize_values([sample.query_ms for sample in ok_samples if sample.query_ms > 0]),
        "final_save_ms": summarize_values([sample.final_save_ms for sample in ok_samples]),
        "first_event_ms": summarize_values([sample.first_event_ms for sample in ok_samples if sample.first_event_ms is not None]),
        "answer_chars": summarize_values([sample.answer_chars for sample in ok_samples]),
        "failed_samples": [sample.__dict__ for sample in failed_samples[:20]],
        "samples": [sample.__dict__ for sample in samples[:20]],
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


def failure_sample(
    index: int,
    tenant_id: str,
    started: float,
    status_code: int,
    error: str,
    upload_ms: float,
    source_poll_ms: float,
    pending_save_ms: float,
    query_ms: float,
    final_save_ms: float,
) -> WorkflowSample:
    return WorkflowSample(
        index=index,
        tenant_id=tenant_id,
        ok=False,
        total_ms=elapsed_ms(started),
        upload_ms=upload_ms,
        source_poll_ms=source_poll_ms,
        pending_save_ms=pending_save_ms,
        query_ms=query_ms,
        final_save_ms=final_save_ms,
        status_code=status_code,
        error=error[:1000],
    )


def acl_groups(args: argparse.Namespace) -> list[str]:
    return [item.strip() for item in args.acl_groups.split(",") if item.strip()]


def now_ms() -> int:
    return int(time.time() * 1000)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def safe_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:1000]
    except Exception:
        return str(exc)


if __name__ == "__main__":
    main()
