from __future__ import annotations

import time
import uuid
import base64
import json
import mimetypes
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import Full, Queue
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from answer import answer_query
from answer_multimodal import answer_multimodal_query
from rag_core.artifacts import (
    MindMapArtifact,
    build_llm_table,
    build_mindmap_root,
    delete_artifact,
    delete_metadata_artifact,
    fail_metadata_artifact,
    list_artifacts,
    list_metadata_artifacts,
    load_artifact,
    load_metadata_artifact,
    save_metadata_artifact,
)
from rag_core.auth import bearer_credential_id, build_auth_context, validate_bearer_token
from rag_core.app_version import app_version
from rag_core.config import load_config
from rag_core.conversations import (
    ConversationMessage,
    ConversationTenantConflictError,
    delete_conversation,
    list_conversation_items,
    load_conversation,
    save_conversation,
)
from rag_core.events import append_event, event_log_limits_snapshot, hit_event_summaries
from rag_core.ingestion_jobs import submit_upload_ingestion_job
from rag_core.jsonl_store import (
    parse_s3_uri,
    read_object_bytes_by_uri,
    s3_bucket,
    s3_key,
    unquote_object_uri,
)
from rag_core.model_api_retry import model_api_metrics_snapshot
from rag_core.milvus_store import milvus_client_metrics_snapshot
from rag_core.pipeline import retrieve_and_rerank
from rag_core.query_admission import (
    acquire_query_admission_lease,
    QueryAdmissionLeaseGuard,
    QueryAdmissionRejected,
    query_admission_metrics_snapshot,
)
from rag_core.readiness import readiness_report
from rag_core.sources import (
    count_active_source_tasks,
    count_source_tasks_by_status,
    create_source_task,
    delete_source,
    discard_uploaded_file,
    fail_source_task,
    get_source,
    get_source_content,
    list_sources,
    rename_source,
    resolve_metadata_display_block_urls,
    save_uploaded_file,
    SourceTaskNotFoundError,
    SourceTaskNotRetryableError,
    source_task_lease_metrics_snapshot,
    source_task_recovery_metrics_snapshot,
    retry_failed_source_task,
    UploadReservationLostError,
    UploadTooLargeError,
)
from rag_core.upload_admission import (
    acquire_upload_admission_reservation,
    release_upload_admission_reservation,
    UploadAdmissionRejected,
    upload_admission_metrics_snapshot,
)
from rag_core.user_auth import (
    authenticate_token,
    auth_token_cache_metrics_snapshot,
    bearer_token,
    bulk_update_users,
    change_user_password,
    count_public_users,
    create_announcement,
    delete_announcement,
    ensure_default_test_account,
    is_registration_enabled,
    list_announcements,
    list_public_users,
    login_user,
    logout_user,
    register_user,
    refresh_session_token,
    set_registration_enabled,
    set_user_status,
    update_user_profile,
)
from rag_core.database import metadata_pool_metrics_snapshot
from search_multimodal import retrieve_multimodal


_QUERY_STREAM_EXECUTOR_LOCK = threading.Lock()
_QUERY_STREAM_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_QUERY_STREAM_SEMAPHORE_LOCK = threading.Lock()
_QUERY_STREAM_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_QUERY_STREAM_TENANT_SEMAPHORE_LOCK = threading.Lock()
_QUERY_STREAM_TENANT_SEMAPHORES: dict[tuple[int, str], threading.BoundedSemaphore] = {}
_QUERY_STREAM_USER_SEMAPHORE_LOCK = threading.Lock()
_QUERY_STREAM_USER_SEMAPHORES: dict[tuple[int, str], threading.BoundedSemaphore] = {}
_QUERY_STREAM_METRICS_LOCK = threading.Lock()
_QUERY_STREAM_METRICS = {
    "active": 0,
    "accepted_total": 0,
    "completed_total": 0,
    "errored_total": 0,
    "rejected_global_total": 0,
    "rejected_tenant_total": 0,
    "rejected_user_total": 0,
    "event_queue_backpressure_total": 0,
}
_QUERY_STREAM_ACTIVE_BY_TENANT: dict[str, int] = {}
_QUERY_STREAM_ACTIVE_BY_USER: dict[str, int] = {}
_HTTP_METRICS_LOCK = threading.Lock()
_HTTP_METRICS: dict[str, dict[str, Any]] = {}
_HTTP_ACTIVE_TOTAL = 0
_HTTP_LATENCY_BUCKETS_MS = (
    10.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    30_000.0,
    60_000.0,
    120_000.0,
)
_QUERY_IMAGE_METRICS_LOCK = threading.Lock()
_QUERY_IMAGE_BUCKET_LIMITS = (
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
)
_QUERY_IMAGE_METRICS: dict[str, Any] = {
    "accepted_total": 0,
    "accepted_estimated_bytes_total": 0,
    "accepted_estimated_bytes_max": 0,
    "accepted_size_buckets": {},
    "rejected_oversized_total": 0,
    "rejected_estimated_bytes_total": 0,
    "rejected_estimated_bytes_max": 0,
    "rejected_size_buckets": {},
    "invalid_total": 0,
}


def query_stream_executor(max_workers: int) -> ThreadPoolExecutor:
    with _QUERY_STREAM_EXECUTOR_LOCK:
        executor = _QUERY_STREAM_EXECUTORS.get(max_workers)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-query-stream")
            _QUERY_STREAM_EXECUTORS[max_workers] = executor
        return executor


def query_stream_semaphore(limit: int) -> threading.BoundedSemaphore:
    with _QUERY_STREAM_SEMAPHORE_LOCK:
        semaphore = _QUERY_STREAM_SEMAPHORES.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _QUERY_STREAM_SEMAPHORES[limit] = semaphore
        return semaphore


def query_stream_tenant_semaphore(limit: int, tenant_id: str) -> threading.BoundedSemaphore:
    key = (limit, tenant_id)
    with _QUERY_STREAM_TENANT_SEMAPHORE_LOCK:
        semaphore = _QUERY_STREAM_TENANT_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _QUERY_STREAM_TENANT_SEMAPHORES[key] = semaphore
        return semaphore


def query_stream_user_semaphore(limit: int, user_key: str) -> threading.BoundedSemaphore:
    key = (limit, user_key)
    with _QUERY_STREAM_USER_SEMAPHORE_LOCK:
        semaphore = _QUERY_STREAM_USER_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _QUERY_STREAM_USER_SEMAPHORES[key] = semaphore
        return semaphore


def query_stream_max_workers() -> int:
    return env_int("RAG_QUERY_STREAM_WORKERS", 64)


def query_stream_queue_limit() -> int:
    return env_int("RAG_QUERY_STREAM_QUEUE_LIMIT", 256)


def query_stream_tenant_queue_limit() -> int:
    return env_int("RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT", 64)


def query_stream_user_queue_limit() -> int:
    return env_int("RAG_QUERY_STREAM_USER_QUEUE_LIMIT", 8)


def query_stream_event_queue_limit() -> int:
    return env_int("RAG_QUERY_STREAM_EVENT_QUEUE_LIMIT", 128)


def query_shared_admission_enabled() -> bool:
    return env_bool("RAG_QUERY_SHARED_ADMISSION", True)


def query_shared_admission_lease_ms() -> int:
    return env_int("RAG_QUERY_SHARED_ADMISSION_LEASE_MS", 15 * 60 * 1000)


def ingest_backlog_limit() -> int:
    return env_int("RAG_INGEST_BACKLOG_LIMIT", 100_000)


def ingest_tenant_backlog_limit() -> int:
    return env_int("RAG_INGEST_TENANT_BACKLOG_LIMIT", 1_000)


def ingest_upload_reservation_ms() -> int:
    return env_int("RAG_INGEST_UPLOAD_RESERVATION_MS", 15 * 60 * 1000)


def max_upload_request_bytes(config) -> int:
    # Multipart adds a small amount of framing around the file bytes.
    return int(config.max_upload_bytes) + 1024 * 1024


def max_query_request_bytes(config) -> int:
    # JSON/base64 framing can be larger than decoded image bytes.
    default_limit = int(config.max_query_image_bytes) * 2 + 256 * 1024
    return env_int("RAG_MAX_QUERY_REQUEST_BYTES", default_limit)


def max_conversation_request_bytes(config) -> int:
    # Conversation saves include message history, citations, RAG progress, and optionally one compressed user image.
    return env_int("RAG_MAX_CONVERSATION_REQUEST_BYTES", max_query_request_bytes(config) * 2)


def max_conversation_images() -> int:
    return env_int("RAG_MAX_CONVERSATION_IMAGES", 4)


def max_conversation_image_bytes(config) -> int:
    return env_int("RAG_MAX_CONVERSATION_IMAGE_BYTES", int(config.max_query_image_bytes) * 2)


def estimate_base64_decoded_bytes(encoded: str) -> int:
    stripped = encoded.strip()
    if not stripped:
        return 0
    padding = len(stripped) - len(stripped.rstrip("="))
    return max(0, (len(stripped) * 3) // 4 - padding)


def record_query_image_size(estimated_bytes: int, *, accepted: bool) -> None:
    prefix = "accepted" if accepted else "rejected"
    count_key = "accepted_total" if accepted else "rejected_oversized_total"
    bucket = query_image_size_bucket(estimated_bytes)
    with _QUERY_IMAGE_METRICS_LOCK:
        _QUERY_IMAGE_METRICS[count_key] += 1
        _QUERY_IMAGE_METRICS[f"{prefix}_estimated_bytes_total"] += estimated_bytes
        _QUERY_IMAGE_METRICS[f"{prefix}_estimated_bytes_max"] = max(
            int(_QUERY_IMAGE_METRICS[f"{prefix}_estimated_bytes_max"]),
            estimated_bytes,
        )
        buckets = _QUERY_IMAGE_METRICS[f"{prefix}_size_buckets"]
        buckets[bucket] = int(buckets.get(bucket, 0)) + 1


def record_invalid_query_image() -> None:
    with _QUERY_IMAGE_METRICS_LOCK:
        _QUERY_IMAGE_METRICS["invalid_total"] += 1


def query_image_size_bucket(estimated_bytes: int) -> str:
    for limit in _QUERY_IMAGE_BUCKET_LIMITS:
        if estimated_bytes <= limit:
            return f"le_{limit}"
    return f"gt_{_QUERY_IMAGE_BUCKET_LIMITS[-1]}"


def query_image_metrics_snapshot() -> dict[str, object]:
    with _QUERY_IMAGE_METRICS_LOCK:
        snapshot = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in _QUERY_IMAGE_METRICS.items()
        }
    accepted = int(snapshot["accepted_total"])
    rejected = int(snapshot["rejected_oversized_total"])
    snapshot["accepted_estimated_bytes_avg"] = round(
        int(snapshot["accepted_estimated_bytes_total"]) / accepted,
        2,
    ) if accepted else 0.0
    snapshot["rejected_estimated_bytes_avg"] = round(
        int(snapshot["rejected_estimated_bytes_total"]) / rejected,
        2,
    ) if rejected else 0.0
    snapshot["bucket_limits_bytes"] = list(_QUERY_IMAGE_BUCKET_LIMITS)
    return snapshot


def query_stream_user_key(auth_context) -> str:
    user_id = str(getattr(auth_context, "user_id", "") or "").strip()
    if user_id:
        return f"{auth_context.tenant_id}:user:{user_id}"
    credential_id = str(getattr(auth_context, "credential_id", "") or "").strip()
    if credential_id:
        return f"{auth_context.tenant_id}:api_token:{credential_id}"
    return ""


def http_route_key(method: str, path: str) -> str:
    normalized = re.sub(r"/[^/?]+@[A-Za-z0-9_.~:%+-]+", "/{doc_id}", path)
    normalized = re.sub(r"/(conv|conversation|artifact|mindmap|table)-[A-Za-z0-9_.~:%+-]+", r"/\1-{id}", normalized)
    normalized = re.sub(r"/user-[A-Za-z0-9_.~:%+-]+", "/user-{id}", normalized)
    normalized = re.sub(r"/[0-9a-f]{8,}(?=/|$)", "/{id}", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"/[^/]*sha256-[A-Za-z0-9_.~:%+-]+", "/{doc_id}", normalized)
    return f"{method.upper()} {normalized}"


def record_http_started(route_key: str) -> None:
    global _HTTP_ACTIVE_TOTAL
    with _HTTP_METRICS_LOCK:
        metrics = _HTTP_METRICS.setdefault(route_key, new_http_route_metrics())
        metrics["active"] = int(metrics["active"]) + 1
        _HTTP_ACTIVE_TOTAL += 1


def record_http_finished(route_key: str, *, status_code: int, latency_ms: float) -> None:
    global _HTTP_ACTIVE_TOTAL
    status_family = f"{max(1, min(5, status_code // 100))}xx"
    with _HTTP_METRICS_LOCK:
        metrics = _HTTP_METRICS.setdefault(route_key, new_http_route_metrics())
        metrics["active"] = max(0, int(metrics["active"]) - 1)
        metrics["requests_total"] = int(metrics["requests_total"]) + 1
        metrics[f"{status_family}_total"] = int(metrics.get(f"{status_family}_total", 0)) + 1
        metrics["latency_total_ms"] = float(metrics["latency_total_ms"]) + latency_ms
        metrics["latency_max_ms"] = max(float(metrics["latency_max_ms"]), latency_ms)
        for upper_bound_ms in _HTTP_LATENCY_BUCKETS_MS:
            if latency_ms <= upper_bound_ms:
                buckets = metrics["latency_buckets_ms"]
                buckets[upper_bound_ms] = int(buckets[upper_bound_ms]) + 1
        _HTTP_ACTIVE_TOTAL = max(0, _HTTP_ACTIVE_TOTAL - 1)


def new_http_route_metrics() -> dict[str, Any]:
    return {
        "active": 0,
        "requests_total": 0,
        "2xx_total": 0,
        "3xx_total": 0,
        "4xx_total": 0,
        "5xx_total": 0,
        "latency_total_ms": 0.0,
        "latency_max_ms": 0.0,
        "latency_buckets_ms": {
            upper_bound_ms: 0
            for upper_bound_ms in _HTTP_LATENCY_BUCKETS_MS
        },
    }


def http_metrics_snapshot() -> dict[str, object]:
    with _HTTP_METRICS_LOCK:
        routes = {
            route: {
                **metrics,
                "latency_avg_ms": (
                    round(float(metrics["latency_total_ms"]) / int(metrics["requests_total"]), 2)
                    if int(metrics["requests_total"])
                    else 0.0
                ),
                "latency_max_ms": round(float(metrics["latency_max_ms"]), 2),
                "latency_total_ms": round(float(metrics["latency_total_ms"]), 2),
            }
            for route, metrics in sorted(_HTTP_METRICS.items())
        }
        return {
            "active_total": _HTTP_ACTIVE_TOTAL,
            "routes": routes,
        }


def record_query_stream_accepted(tenant_id: str) -> None:
    with _QUERY_STREAM_METRICS_LOCK:
        _QUERY_STREAM_METRICS["active"] += 1
        _QUERY_STREAM_METRICS["accepted_total"] += 1
        _QUERY_STREAM_ACTIVE_BY_TENANT[tenant_id] = _QUERY_STREAM_ACTIVE_BY_TENANT.get(tenant_id, 0) + 1


def record_query_stream_user_accepted(user_key: str) -> None:
    if not user_key:
        return
    with _QUERY_STREAM_METRICS_LOCK:
        _QUERY_STREAM_ACTIVE_BY_USER[user_key] = _QUERY_STREAM_ACTIVE_BY_USER.get(user_key, 0) + 1


def record_query_stream_finished(tenant_id: str, user_key: str = "", *, errored: bool) -> None:
    with _QUERY_STREAM_METRICS_LOCK:
        _QUERY_STREAM_METRICS["active"] = max(0, _QUERY_STREAM_METRICS["active"] - 1)
        _QUERY_STREAM_METRICS["completed_total"] += 1
        if errored:
            _QUERY_STREAM_METRICS["errored_total"] += 1
        current = max(0, _QUERY_STREAM_ACTIVE_BY_TENANT.get(tenant_id, 0) - 1)
        if current:
            _QUERY_STREAM_ACTIVE_BY_TENANT[tenant_id] = current
        else:
            _QUERY_STREAM_ACTIVE_BY_TENANT.pop(tenant_id, None)
        if user_key:
            current_user = max(0, _QUERY_STREAM_ACTIVE_BY_USER.get(user_key, 0) - 1)
            if current_user:
                _QUERY_STREAM_ACTIVE_BY_USER[user_key] = current_user
            else:
                _QUERY_STREAM_ACTIVE_BY_USER.pop(user_key, None)


def record_query_stream_rejected(kind: str) -> None:
    if kind == "user":
        key = "rejected_user_total"
    elif kind == "tenant":
        key = "rejected_tenant_total"
    else:
        key = "rejected_global_total"
    with _QUERY_STREAM_METRICS_LOCK:
        _QUERY_STREAM_METRICS[key] += 1


def record_query_stream_event_queue_backpressure() -> None:
    with _QUERY_STREAM_METRICS_LOCK:
        _QUERY_STREAM_METRICS["event_queue_backpressure_total"] += 1


def query_stream_metrics_snapshot() -> dict[str, object]:
    with _QUERY_STREAM_METRICS_LOCK:
        return {
            **_QUERY_STREAM_METRICS,
            "active_by_tenant": dict(sorted(_QUERY_STREAM_ACTIVE_BY_TENANT.items())),
            "active_by_user": dict(sorted(_QUERY_STREAM_ACTIVE_BY_USER.items())),
            "queue_limit": query_stream_queue_limit(),
            "tenant_queue_limit": query_stream_tenant_queue_limit(),
            "user_queue_limit": query_stream_user_queue_limit(),
            "event_queue_limit": query_stream_event_queue_limit(),
            "workers": query_stream_max_workers(),
        }


def prometheus_metrics_text(config) -> str:
    http = http_metrics_snapshot()
    query_stream = query_stream_metrics_snapshot()
    query_images = query_image_metrics_snapshot()
    model_api = model_api_metrics_snapshot()
    metadata = metadata_pool_metrics_snapshot()
    ingestion = count_source_tasks_by_status(config=config, tenant_id=None)
    ingestion_leases = source_task_lease_metrics_snapshot(config=config, tenant_id=None)
    ingestion_recovery = source_task_recovery_metrics_snapshot(config=config, tenant_id=None)
    upload_admission = upload_admission_metrics_snapshot(config=config)
    shared_query_admission = query_admission_metrics_snapshot(config=config)
    lines = [
        "# HELP rag_http_active_requests Current in-flight HTTP requests.",
        "# TYPE rag_http_active_requests gauge",
        f"rag_http_active_requests {int(http['active_total'])}",
        "# HELP rag_http_requests_total Completed HTTP requests by normalized route.",
        "# TYPE rag_http_requests_total counter",
        "# HELP rag_http_responses_total Completed HTTP responses by status family.",
        "# TYPE rag_http_responses_total counter",
        "# HELP rag_http_request_latency_seconds_total Cumulative HTTP request latency.",
        "# TYPE rag_http_request_latency_seconds_total counter",
        "# HELP rag_http_request_latency_seconds_max Maximum observed HTTP request latency.",
        "# TYPE rag_http_request_latency_seconds_max gauge",
        "# HELP rag_http_request_latency_seconds HTTP request latency histogram.",
        "# TYPE rag_http_request_latency_seconds histogram",
    ]
    for route, metrics in http["routes"].items():
        route_label = prometheus_label(str(route))
        lines.append(f'rag_http_requests_total{{route="{route_label}"}} {int(metrics["requests_total"])}')
        for status_family in ("2xx", "3xx", "4xx", "5xx"):
            lines.append(
                "rag_http_responses_total"
                f'{{route="{route_label}",status_family="{status_family}"}} '
                f'{int(metrics[f"{status_family}_total"])}'
            )
        lines.append(
            f'rag_http_request_latency_seconds_total{{route="{route_label}"}} '
            f'{float(metrics["latency_total_ms"]) / 1000.0:.6f}'
        )
        lines.append(
            f'rag_http_request_latency_seconds_max{{route="{route_label}"}} '
            f'{float(metrics["latency_max_ms"]) / 1000.0:.6f}'
        )
        for upper_bound_ms in _HTTP_LATENCY_BUCKETS_MS:
            upper_bound_seconds = prometheus_number(upper_bound_ms / 1000.0)
            count = int(metrics["latency_buckets_ms"][upper_bound_ms])
            lines.append(
                f'rag_http_request_latency_seconds_bucket{{route="{route_label}",le="{upper_bound_seconds}"}} '
                f"{count}"
            )
        lines.append(
            f'rag_http_request_latency_seconds_bucket{{route="{route_label}",le="+Inf"}} '
            f'{int(metrics["requests_total"])}'
        )
        lines.append(
            f'rag_http_request_latency_seconds_sum{{route="{route_label}"}} '
            f'{float(metrics["latency_total_ms"]) / 1000.0:.6f}'
        )
        lines.append(
            f'rag_http_request_latency_seconds_count{{route="{route_label}"}} '
            f'{int(metrics["requests_total"])}'
        )

    lines.extend(
        [
            "# HELP rag_query_stream_active Current accepted query streams.",
            "# TYPE rag_query_stream_active gauge",
            f"rag_query_stream_active {int(query_stream['active'])}",
            "# HELP rag_query_stream_events_total Query-stream lifecycle and rejection events.",
            "# TYPE rag_query_stream_events_total counter",
        ]
    )
    for event, key in (
        ("accepted", "accepted_total"),
        ("completed", "completed_total"),
        ("errored", "errored_total"),
        ("rejected_global", "rejected_global_total"),
        ("rejected_tenant", "rejected_tenant_total"),
        ("rejected_user", "rejected_user_total"),
        ("event_queue_backpressure", "event_queue_backpressure_total"),
    ):
        lines.append(f'rag_query_stream_events_total{{event="{event}"}} {int(query_stream[key])}')

    lines.extend(
        [
            "# HELP rag_query_shared_admission_slots Current database-backed query admission slots.",
            "# TYPE rag_query_shared_admission_slots gauge",
            (
                'rag_query_shared_admission_slots{scope="global"} '
                f'{shared_query_admission["global_slots"]}'
            ),
            (
                'rag_query_shared_admission_slots{scope="tenant"} '
                f'{shared_query_admission["tenant_slots"]}'
            ),
            (
                'rag_query_shared_admission_slots{scope="user"} '
                f'{shared_query_admission["user_slots"]}'
            ),
            (
                'rag_query_shared_admission_slots{scope="expired"} '
                f'{shared_query_admission["expired_slots"]}'
            ),
        ]
    )

    lines.extend(
        [
            "# HELP rag_query_image_payloads_total Query image validation outcomes.",
            "# TYPE rag_query_image_payloads_total counter",
            f'rag_query_image_payloads_total{{outcome="accepted"}} {int(query_images["accepted_total"])}',
            (
                'rag_query_image_payloads_total{outcome="rejected_oversized"} '
                f'{int(query_images["rejected_oversized_total"])}'
            ),
            f'rag_query_image_payloads_total{{outcome="invalid"}} {int(query_images["invalid_total"])}',
            "# HELP rag_query_image_payload_bytes Estimated decoded query image payload size.",
            "# TYPE rag_query_image_payload_bytes histogram",
        ]
    )
    for outcome, prefix, count_key in (
        ("accepted", "accepted", "accepted_total"),
        ("rejected_oversized", "rejected", "rejected_oversized_total"),
    ):
        cumulative = 0
        size_buckets = query_images[f"{prefix}_size_buckets"]
        for upper_bound_bytes in _QUERY_IMAGE_BUCKET_LIMITS:
            cumulative += int(size_buckets.get(f"le_{upper_bound_bytes}", 0))
            lines.append(
                "rag_query_image_payload_bytes_bucket"
                f'{{outcome="{outcome}",le="{upper_bound_bytes}"}} {cumulative}'
            )
        lines.append(
            "rag_query_image_payload_bytes_bucket"
            f'{{outcome="{outcome}",le="+Inf"}} {int(query_images[count_key])}'
        )
        lines.append(
            f'rag_query_image_payload_bytes_sum{{outcome="{outcome}"}} '
            f'{int(query_images[f"{prefix}_estimated_bytes_total"])}'
        )
        lines.append(
            f'rag_query_image_payload_bytes_count{{outcome="{outcome}"}} '
            f'{int(query_images[count_key])}'
        )

    lines.extend(
        [
            "# HELP rag_model_api_active Current external model API calls.",
            "# TYPE rag_model_api_active gauge",
            f"rag_model_api_active {int(model_api['active'])}",
            "# HELP rag_model_api_events_total External model API admission events.",
            "# TYPE rag_model_api_events_total counter",
            f'rag_model_api_events_total{{event="acquired"}} {int(model_api["acquired_total"])}',
            f'rag_model_api_events_total{{event="rejected"}} {int(model_api["rejected_total"])}',
            "# HELP rag_model_api_operation_calls_total Logical external model calls by operation and outcome.",
            "# TYPE rag_model_api_operation_calls_total counter",
            "# HELP rag_model_api_operation_attempts_total External model attempts by operation.",
            "# TYPE rag_model_api_operation_attempts_total counter",
            "# HELP rag_model_api_operation_retries_total External model retries by operation.",
            "# TYPE rag_model_api_operation_retries_total counter",
            "# HELP rag_model_api_operation_latency_seconds_total Cumulative logical-call latency by operation.",
            "# TYPE rag_model_api_operation_latency_seconds_total counter",
            "# HELP rag_model_api_operation_latency_seconds_max Maximum logical-call latency by operation.",
            "# TYPE rag_model_api_operation_latency_seconds_max gauge",
        ]
    )
    for operation, metrics in model_api["operations"].items():
        operation_label = prometheus_label(operation)
        lines.append(
            f'rag_model_api_operation_calls_total{{operation="{operation_label}",outcome="success"}} '
            f'{int(metrics["successes_total"])}'
        )
        lines.append(
            f'rag_model_api_operation_calls_total{{operation="{operation_label}",outcome="failure"}} '
            f'{int(metrics["failures_total"])}'
        )
        lines.append(
            f'rag_model_api_operation_attempts_total{{operation="{operation_label}"}} '
            f'{int(metrics["attempts_total"])}'
        )
        lines.append(
            f'rag_model_api_operation_retries_total{{operation="{operation_label}"}} '
            f'{int(metrics["retries_total"])}'
        )
        lines.append(
            f'rag_model_api_operation_latency_seconds_total{{operation="{operation_label}"}} '
            f'{float(metrics["latency_total_ms"]) / 1000.0:.6f}'
        )
        lines.append(
            f'rag_model_api_operation_latency_seconds_max{{operation="{operation_label}"}} '
            f'{float(metrics["latency_max_ms"]) / 1000.0:.6f}'
        )

    pool_fields = {
        "total": "rag_metadata_pool_connections",
        "idle": "rag_metadata_pool_idle_connections",
        "borrowed": "rag_metadata_pool_borrowed_connections",
        "waits_total": "rag_metadata_pool_waits_total",
        "timeouts_total": "rag_metadata_pool_timeouts_total",
    }
    for field, metric_name in pool_fields.items():
        value = sum(int(pool[field]) for pool in metadata["pools"])
        metric_type = "counter" if field.endswith("_total") else "gauge"
        lines.extend([f"# TYPE {metric_name} {metric_type}", f"{metric_name} {value}"])

    lines.extend(
        [
            "# HELP rag_ingestion_tasks Current source-task rows by status.",
            "# TYPE rag_ingestion_tasks gauge",
        ]
    )
    for status in sorted({"queued", "processing", "ready", "failed", *ingestion.keys()}):
        lines.append(
            f'rag_ingestion_tasks{{status="{prometheus_label(status)}"}} '
            f"{int(ingestion.get(status, 0))}"
        )
    lines.extend(
        [
            "# HELP rag_ingestion_task_leases Current processing-task leases by state.",
            "# TYPE rag_ingestion_task_leases gauge",
            f'rag_ingestion_task_leases{{state="active"}} {ingestion_leases["active_leases"]}',
            f'rag_ingestion_task_leases{{state="expired"}} {ingestion_leases["expired_leases"]}',
            "# HELP rag_ingestion_task_attempts Attempt counts retained on current source tasks.",
            "# TYPE rag_ingestion_task_attempts gauge",
            f'rag_ingestion_task_attempts{{stat="sum"}} {ingestion_leases["attempts_recorded"]}',
            f'rag_ingestion_task_attempts{{stat="max"}} {ingestion_leases["max_attempt_count"]}',
            "# HELP rag_ingestion_task_recovery Current retry and dead-letter task state.",
            "# TYPE rag_ingestion_task_recovery gauge",
            f'rag_ingestion_task_recovery{{state="retry_waiting"}} {ingestion_recovery["retry_waiting"]}',
            f'rag_ingestion_task_recovery{{state="dead_lettered"}} {ingestion_recovery["dead_lettered"]}',
            f'rag_ingestion_task_recovery{{state="retries_recorded"}} {ingestion_recovery["retries_recorded"]}',
            "# HELP rag_ingestion_upload_reservations Current shared upload admission reservations.",
            "# TYPE rag_ingestion_upload_reservations gauge",
            (
                'rag_ingestion_upload_reservations{scope="global"} '
                f'{upload_admission["global_reservations"]}'
            ),
            (
                'rag_ingestion_upload_reservations{scope="tenant"} '
                f'{upload_admission["tenant_reservations"]}'
            ),
            (
                'rag_ingestion_upload_reservations{scope="expired"} '
                f'{upload_admission["expired_reservations"]}'
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def prometheus_number(value: float) -> str:
    return f"{value:g}"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    query_mode: str = Field(default="text", pattern="^(text|multimodal)$")
    image_data_url: str | None = None
    history: list[str] = Field(default_factory=list)
    tenant_id: str = "team_a"
    acl_groups: list[str] = Field(default_factory=list)
    doc_version: int | None = None
    doc_ids: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    include_all_sources: bool = False
    candidate_limit: int = Field(default=20, ge=1, le=100)
    context_limit: int = Field(default=5, ge=1, le=20)
    request_id: str | None = None


class SearchRequest(QueryRequest):
    pass


class HitResponse(BaseModel):
    doc_id: str
    title: str
    source_uri: str
    source_type: str
    chunk_index: int
    score: float
    rerank_score: float | None = None
    acl_groups: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    text_preview: str = ""


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[HitResponse]
    trace: dict[str, object] | None = None


class SearchResponse(BaseModel):
    request_id: str
    hits: list[HitResponse]
    trace: dict[str, object]


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str = ""
    selected_doc_ids: list[str] = Field(default_factory=list)
    tenant_id: str = "team_a"
    acl_groups: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


class SourceResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    source_uri: str
    doc_version: int
    chunk_count: int
    acl_groups: list[str]
    status: str
    current: bool
    created_at: int | None = None
    updated_at: int | None = None
    child_doc_ids: list[str] = Field(default_factory=list)
    error: str = ""
    retryable: bool = False
    attempt_count: int = 0
    next_attempt_at: int = 0
    dead_lettered: bool = False


class SourceListResponse(BaseModel):
    sources: list[SourceResponse]


class SourceUploadResponse(BaseModel):
    status: str
    sources: list[SourceResponse]
    document_count: int
    chunk_count: int


class RetrySourceResponse(BaseModel):
    status: str
    source: SourceResponse


class SourceContentResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    source_uri: str
    doc_version: int
    child_doc_ids: list[str]
    guide: str
    tags: list[str]
    text: str
    blocks: list[dict[str, str]] = Field(default_factory=list)
    suggested_title: str = ""


class DeleteSourceResponse(BaseModel):
    status: str
    doc_id: str
    detail: dict[str, object]


class RenameSourceRequest(BaseModel):
    title: str


class RenameSourceResponse(BaseModel):
    status: str
    doc_id: str
    title: str


class MindMapRequest(BaseModel):
    title: str = "思维导图"
    tenant_id: str = "team_a"
    workspace_id: str = ""
    acl_groups: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    doc_version: int | None = None
    context_limit: int = Field(default=5, ge=1, le=20)


class MindMapArtifactResponse(BaseModel):
    id: str
    title: str
    status: str
    tenant_id: str
    workspace_id: str = ""
    source_doc_ids: list[str]
    created_at: int
    updated_at: int
    artifact_type: str = "mindmap"
    root: dict[str, Any] | None = None
    table: dict[str, Any] | None = None
    error: str = ""


class ArtifactListResponse(BaseModel):
    artifacts: list[MindMapArtifactResponse]


class DeleteArtifactResponse(BaseModel):
    status: str
    artifact_id: str


class RenameArtifactRequest(BaseModel):
    title: str


class RenameArtifactResponse(BaseModel):
    status: str
    artifact_id: str
    title: str


class ConversationMessageRequest(BaseModel):
    id: str
    role: str = Field(pattern="^(user|assistant)$")
    content: str
    status: str = Field(default="done", pattern="^(sending|done|failed)$")
    request_id: str | None = None
    citations: list[HitResponse] = Field(default_factory=list)
    image_data_url: str | None = None
    created_at: int | None = None
    feedback_rating: int | None = Field(default=None, ge=-1, le=1)
    rag_progress: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("rag_progress", mode="before")
    @classmethod
    def normalize_rag_progress(cls, value):
        return [] if value is None else value


class ConversationUpsertRequest(BaseModel):
    id: str | None = None
    tenant_id: str = "team_a"
    title: str = ""
    messages: list[ConversationMessageRequest]
    source_doc_ids: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    id: str
    tenant_id: str
    title: str
    messages: list[ConversationMessageRequest]
    source_doc_ids: list[str]
    created_at: int
    updated_at: int


class ConversationListItemResponse(BaseModel):
    id: str
    tenant_id: str
    title: str
    message_count: int
    source_doc_ids: list[str]
    created_at: int
    updated_at: int


class ConversationListResponse(BaseModel):
    conversations: list[ConversationListItemResponse]


class DeleteConversationResponse(BaseModel):
    status: str
    conversation_id: str


class UserResponse(BaseModel):
    id: str
    username: str
    display_name: str
    role: str
    tenant_id: str
    created_at: int
    avatar_url: str = ""
    status: str = "active"
    profile_name_edit_allowed: bool = True
    avatar_edit_allowed: bool = True
    last_login_at: int | None = None


class AuthRequest(BaseModel):
    username: str
    password: str
    display_name: str | None = None


class AuthResponse(BaseModel):
    user: UserResponse
    token: str
    expires_at: int


class ProfileUpdateRequest(BaseModel):
    username: str
    display_name: str
    avatar_url: str = ""


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class UserStatusRequest(BaseModel):
    status: str = Field(pattern="^(active|banned)$")


class AdminUserUpdateItem(BaseModel):
    user_id: str
    status: str | None = Field(default=None, pattern="^(active|banned)$")
    profile_name_edit_allowed: bool | None = None
    avatar_edit_allowed: bool | None = None


class AdminUserBulkUpdateRequest(BaseModel):
    users: list[AdminUserUpdateItem] = Field(default_factory=list, min_length=1, max_length=50)


class AnnouncementRequest(BaseModel):
    title: str
    content: str
    link_url: str = ""
    link_label: str = ""


class AnnouncementResponse(BaseModel):
    id: str
    title: str
    content: str
    link_url: str = ""
    link_label: str = ""
    author_id: str
    author_name: str | None = None
    created_at: int


class AnnouncementListResponse(BaseModel):
    announcements: list[AnnouncementResponse]


class AdminSettingsResponse(BaseModel):
    registration_enabled: bool
    latest_announcement: AnnouncementResponse | None = None


class RegistrationSettingsRequest(BaseModel):
    registration_enabled: bool


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
    limit: int
    offset: int
    query: str = ""


def create_app():
    app = FastAPI(title="Production RAG", version=app_version())
    ensure_default_test_account(load_config())

    @app.middleware("http")
    async def record_http_metrics(request, call_next):
        route_key = http_route_key(request.method, request.url.path)
        start = time.perf_counter()
        record_http_started(route_key)
        status_code = 500
        try:
            if request.method.upper() == "POST" and request.url.path == "/sources/upload":
                config = load_config()
                content_length = request.headers.get("content-length")
                try:
                    request_size = int(content_length) if content_length else 0
                except ValueError:
                    request_size = 0
                if request_size > max_upload_request_bytes(config):
                    status_code = 413
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                "Uploaded file is too large. "
                                f"RAG_MAX_UPLOAD_BYTES={config.max_upload_bytes}"
                            )
                        },
                    )
            if request.method.upper() == "POST" and request.url.path in {"/query", "/query/stream", "/search"}:
                config = load_config()
                content_length = request.headers.get("content-length")
                try:
                    request_size = int(content_length) if content_length else 0
                except ValueError:
                    request_size = 0
                query_request_limit = max_query_request_bytes(config)
                if request_size > query_request_limit:
                    status_code = 413
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                "Query request body is too large. "
                                f"RAG_MAX_QUERY_REQUEST_BYTES={query_request_limit}"
                            )
                        },
                    )
            if request.method.upper() == "POST" and request.url.path == "/conversations":
                config = load_config()
                content_length = request.headers.get("content-length")
                try:
                    request_size = int(content_length) if content_length else 0
                except ValueError:
                    request_size = 0
                conversation_request_limit = max_conversation_request_bytes(config)
                if request_size > conversation_request_limit:
                    status_code = 413
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                "Conversation request body is too large. "
                                f"RAG_MAX_CONVERSATION_REQUEST_BYTES={conversation_request_limit}"
                            )
                        },
                    )
            response = await call_next(request)
            status_code = int(getattr(response, "status_code", 500))
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            record_http_finished(route_key, status_code=status_code, latency_ms=latency_ms)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        config = load_config()
        report = readiness_report(config)
        if report["status"] != "ok":
            raise HTTPException(status_code=503, detail=report)
        return report

    @app.get("/runtime-metrics")
    def runtime_metrics(tenant_id: str | None = None) -> dict[str, object]:
        config = load_config()
        return {
            "http": http_metrics_snapshot(),
            "query_stream": query_stream_metrics_snapshot(),
            "query_shared_admission": {
                "enabled": query_shared_admission_enabled(),
                "lease_ms": query_shared_admission_lease_ms(),
                **query_admission_metrics_snapshot(config=config),
            },
            "query": {
                "max_query_image_bytes": config.max_query_image_bytes,
                "max_query_request_bytes": max_query_request_bytes(config),
                "image_payloads": query_image_metrics_snapshot(),
            },
            "conversation": {
                "max_conversation_request_bytes": max_conversation_request_bytes(config),
                "max_conversation_images": max_conversation_images(),
                "max_conversation_image_bytes": max_conversation_image_bytes(config),
            },
            "model_api": model_api_metrics_snapshot(),
            "milvus_client": milvus_client_metrics_snapshot(),
            "metadata_db": metadata_pool_metrics_snapshot(),
            "auth_token_cache": auth_token_cache_metrics_snapshot(),
            "event_log": event_log_limits_snapshot(),
            "ingestion": {
                "source_tasks_by_status": count_source_tasks_by_status(config=config, tenant_id=tenant_id),
                "task_leases": source_task_lease_metrics_snapshot(config=config, tenant_id=tenant_id),
                "task_recovery": source_task_recovery_metrics_snapshot(config=config, tenant_id=tenant_id),
                "upload_admission": {
                    "reservation_ms": ingest_upload_reservation_ms(),
                    **upload_admission_metrics_snapshot(config=config),
                },
                "active_source_tasks": count_active_source_tasks(config=config, tenant_id=None),
                "tenant_active_source_tasks": count_active_source_tasks(config=config, tenant_id=tenant_id),
                "backlog_limit": ingest_backlog_limit(),
                "tenant_backlog_limit": ingest_tenant_backlog_limit(),
                "max_upload_bytes": config.max_upload_bytes,
                "tenant_id": tenant_id or "",
            },
        }

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(
            content=prometheus_metrics_text(load_config()),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/auth/register", response_model=AuthResponse)
    def register(request: AuthRequest) -> AuthResponse:
        config = load_config()
        try:
            user = register_user(
                config,
                username=request.username,
                password=request.password,
                display_name=request.display_name,
            )
            user, token, expires_at = login_user(
                config,
                username=request.username,
                password=request.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AuthResponse(user=user_to_response(user), token=token, expires_at=expires_at)

    @app.post("/auth/login", response_model=AuthResponse)
    def login(request: AuthRequest) -> AuthResponse:
        config = load_config()
        try:
            user, token, expires_at = login_user(
                config,
                username=request.username,
                password=request.password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return AuthResponse(user=user_to_response(user), token=token, expires_at=expires_at)

    @app.post("/auth/logout")
    def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
        config = load_config()
        token = bearer_token(authorization)
        if token:
            logout_user(config, token=token)
        return {"status": "ok"}

    @app.post("/auth/token/refresh", response_model=AuthResponse)
    def refresh_token(authorization: str | None = Header(default=None)) -> AuthResponse:
        config = load_config()
        token = bearer_token(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="请先登录")
        try:
            user, next_token, expires_at = refresh_session_token(config, current_token=token)
        except ValueError as exc:
            status_code = 401 if str(exc) == "请先登录" else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return AuthResponse(user=user_to_response(user), token=next_token, expires_at=expires_at)

    @app.get("/auth/me", response_model=UserResponse)
    def me(authorization: str | None = Header(default=None)) -> UserResponse:
        user = require_current_user(authorization=authorization)
        return user_to_response(user)

    @app.patch("/auth/me", response_model=UserResponse)
    def update_me(
        request: ProfileUpdateRequest,
        authorization: str | None = Header(default=None),
    ) -> UserResponse:
        config = load_config()
        current = require_current_user(authorization=authorization)
        try:
            user = update_user_profile(
                config,
                user_id=current.id,
                username=request.username,
                display_name=request.display_name,
                avatar_url=request.avatar_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return user_to_response(user)

    @app.patch("/auth/password")
    def change_password(
        request: PasswordChangeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        config = load_config()
        current = require_current_user(authorization=authorization)
        try:
            change_user_password(
                config,
                user_id=current.id,
                current_password=request.current_password,
                new_password=request.new_password,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get("/admin/users", response_model=UserListResponse)
    def admin_users(
        authorization: str | None = Header(default=None),
        q: str = Query(default="", max_length=80),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> UserListResponse:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        users = list_public_users(config, query=q, limit=limit, offset=offset)
        total = count_public_users(config, query=q)
        return UserListResponse(
            users=[user_to_response(user) for user in users],
            total=total,
            limit=limit,
            offset=offset,
            query=q,
        )

    @app.patch("/admin/users/{user_id}/status", response_model=UserResponse)
    def admin_update_user_status(
        user_id: str,
        request: UserStatusRequest,
        authorization: str | None = Header(default=None),
    ) -> UserResponse:
        config = load_config()
        actor = require_admin(config=config, authorization=authorization)
        try:
            user = set_user_status(config, actor_id=actor.id, user_id=user_id, status=request.status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return user_to_response(user)

    @app.patch("/admin/users/bulk", response_model=UserListResponse)
    def admin_bulk_update_users(
        request: AdminUserBulkUpdateRequest,
        authorization: str | None = Header(default=None),
    ) -> UserListResponse:
        config = load_config()
        actor = require_admin(config=config, authorization=authorization)
        try:
            users = bulk_update_users(
                config,
                actor_id=actor.id,
                updates=[item.model_dump(exclude_unset=True) for item in request.users],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UserListResponse(
            users=[user_to_response(user) for user in users],
            total=len(users),
            limit=len(users),
            offset=0,
            query="",
        )

    @app.get("/admin/settings", response_model=AdminSettingsResponse)
    def admin_settings(authorization: str | None = Header(default=None)) -> AdminSettingsResponse:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        return admin_settings_response(config)

    @app.patch("/admin/settings/registration", response_model=AdminSettingsResponse)
    def admin_update_registration_settings(
        request: RegistrationSettingsRequest,
        authorization: str | None = Header(default=None),
    ) -> AdminSettingsResponse:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        set_registration_enabled(config, enabled=request.registration_enabled)
        return admin_settings_response(config)

    @app.post("/admin/announcements", response_model=AnnouncementResponse)
    def admin_create_announcement(
        request: AnnouncementRequest,
        authorization: str | None = Header(default=None),
    ) -> AnnouncementResponse:
        config = load_config()
        user = require_admin(config=config, authorization=authorization)
        try:
            row = create_announcement(
                config,
                title=request.title,
                content=request.content,
                author_id=user.id,
                link_url=request.link_url,
                link_label=request.link_label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AnnouncementResponse(**row, author_name=user.display_name)

    @app.delete("/admin/announcements/{announcement_id}")
    def admin_delete_announcement(
        announcement_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        config = load_config()
        require_admin(config=config, authorization=authorization)
        removed = delete_announcement(config, announcement_id=announcement_id)
        return {"status": "deleted" if removed else "not_found", "announcement_id": announcement_id}

    @app.get("/announcements", response_model=AnnouncementListResponse)
    def public_announcements(limit: int = 5) -> AnnouncementListResponse:
        config = load_config()
        return AnnouncementListResponse(
            announcements=[AnnouncementResponse(**row) for row in list_announcements(config, limit=limit)]
        )

    @app.get("/sources", response_model=SourceListResponse)
    def sources(
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceListResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        return SourceListResponse(
            sources=[
                source_to_response(source)
                for source in list_sources(config=config, tenant_id=auth_context.tenant_id)
            ]
        )

    @app.post("/sources/upload", response_model=SourceUploadResponse)
    def upload_source(
        file: UploadFile = File(...),
        tenant_id: str = Form("team_a"),
        acl_groups: str = Form("engineering"),
        doc_version: int | None = Form(default=None),
        language: str = Form("zh"),
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceUploadResponse:
        config = load_config()
        body_acl_groups = [item.strip() for item in acl_groups.split(",") if item.strip()]
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=body_acl_groups,
        )
        backlog_limit = ingest_backlog_limit()
        tenant_backlog_limit = ingest_tenant_backlog_limit()
        try:
            reservation = acquire_upload_admission_reservation(
                config=config,
                tenant_id=auth_context.tenant_id,
                global_limit=backlog_limit,
                tenant_limit=tenant_backlog_limit,
                lease_ms=ingest_upload_reservation_ms(),
            )
        except UploadAdmissionRejected as exc:
            limit_name = (
                "RAG_INGEST_TENANT_BACKLOG_LIMIT"
                if exc.kind == "tenant"
                else "RAG_INGEST_BACKLOG_LIMIT"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Ingestion {exc.kind} backlog is full. Please retry later. "
                    f"active_source_tasks={exc.active_tasks} "
                    f"{limit_name}={exc.limit}"
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Ingestion admission is unavailable. Please retry later.",
            ) from exc
        saved_path = None
        try:
            saved_path = save_uploaded_file(
                config=config,
                tenant_id=auth_context.tenant_id,
                filename=file.filename or "upload.txt",
                content=file.file,
            )
            pending_source = create_source_task(
                config=config,
                tenant_id=auth_context.tenant_id,
                path=saved_path,
                acl_groups=auth_context.acl_groups or body_acl_groups or ["engineering"],
                doc_version=doc_version,
                upload_reservation_owner=reservation.owner,
            )
            reservation = None
        except Exception as exc:
            if saved_path is not None:
                discard_uploaded_file(config=config, path=saved_path)
            if reservation is not None:
                release_upload_admission_reservation(config=config, reservation=reservation)
            if isinstance(exc, UploadTooLargeError):
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            if isinstance(exc, UploadReservationLostError):
                raise HTTPException(
                    status_code=503,
                    detail="Upload reservation expired. Please retry.",
                ) from exc
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise
        try:
            submit_upload_ingestion_job(
                pending_source=pending_source,
                saved_path=saved_path,
                tenant_id=auth_context.tenant_id,
                acl_groups=auth_context.acl_groups or body_acl_groups or ["engineering"],
                doc_version=doc_version,
                language=language,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SourceUploadResponse(
            status="queued",
            sources=[source_to_response(pending_source)],
            document_count=0,
            chunk_count=0,
        )

    @app.get("/sources/content/{doc_id:path}", response_model=SourceContentResponse)
    def source_content(
        doc_id: str,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceContentResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        content = get_source_content(
            config=config,
            tenant_id=auth_context.tenant_id,
            doc_id=doc_id,
            doc_version=doc_version,
        )
        if content is None:
            raise HTTPException(status_code=404, detail="Source not found")
        return SourceContentResponse(**content.__dict__)

    @app.get("/source-assets/{asset_path:path}")
    def source_asset(
        asset_path: str,
        tenant_id: str = "team_a",
        token: str | None = None,
        authorization: str | None = Header(default=None),
    ) -> Response:
        config = load_config()
        auth_context = resolve_asset_auth_context(
            config=config,
            authorization=authorization,
            token=token,
            tenant_id=tenant_id,
        )
        if asset_path.startswith("__s3__/"):
            object_uri = unquote_object_uri(asset_path[len("__s3__/") :])
            if not s3_asset_belongs_to_tenant(object_uri, auth_context.tenant_id):
                raise HTTPException(status_code=404, detail="Asset not found")
            try:
                body = read_object_bytes_by_uri(object_uri)
            except Exception:
                raise HTTPException(status_code=404, detail="Asset not found") from None
            media_type = mimetypes.guess_type(object_uri)[0] or "application/octet-stream"
            if not media_type.startswith("image/"):
                raise HTTPException(status_code=404, detail="Asset not found")
            return Response(content=body, media_type=media_type)
        asset_parts = asset_path.split("/")
        if len(asset_parts) < 3 or asset_parts[0] != "uploads":
            raise HTTPException(status_code=404, detail="Asset not found")
        requested_tenant = asset_parts[1]
        if requested_tenant != auth_context.tenant_id:
            raise HTTPException(status_code=404, detail="Asset not found")
        try:
            object_store_dir = config.object_store_dir.expanduser().resolve()
            path = (object_store_dir / asset_path).expanduser().resolve()
            path.relative_to(object_store_dir)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="Asset not found") from None
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not media_type.startswith("image/"):
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(path, media_type=media_type)

    @app.get("/sources/{doc_id:path}", response_model=SourceResponse)
    def source_detail(
        doc_id: str,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SourceResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        source = get_source(
            config=config,
            tenant_id=auth_context.tenant_id,
            doc_id=doc_id,
            doc_version=doc_version,
        )
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        return source_to_response(source)

    @app.post("/sources/{doc_id:path}/retry", response_model=RetrySourceResponse)
    def retry_source_ingestion(
        doc_id: str,
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> RetrySourceResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        try:
            reservation = acquire_upload_admission_reservation(
                config=config,
                tenant_id=auth_context.tenant_id,
                global_limit=ingest_backlog_limit(),
                tenant_limit=ingest_tenant_backlog_limit(),
                lease_ms=ingest_upload_reservation_ms(),
            )
        except UploadAdmissionRejected as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Ingestion {exc.kind} backlog is full. Please retry later.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Ingestion admission is unavailable. Please retry later.",
            ) from exc
        try:
            queued = retry_failed_source_task(
                config=config,
                tenant_id=auth_context.tenant_id,
                task_id=doc_id,
                upload_reservation_owner=reservation.owner,
            )
        except SourceTaskNotFoundError as exc:
            release_upload_admission_reservation(config=config, reservation=reservation)
            raise HTTPException(status_code=404, detail="Source task not found") from exc
        except SourceTaskNotRetryableError as exc:
            release_upload_admission_reservation(config=config, reservation=reservation)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UploadReservationLostError as exc:
            release_upload_admission_reservation(config=config, reservation=reservation)
            raise HTTPException(
                status_code=503,
                detail="Retry reservation expired. Please retry.",
            ) from exc
        source = queued.source
        submit_upload_ingestion_job(
            pending_source=source,
            saved_path=Path(source.source_uri),
            tenant_id=auth_context.tenant_id,
            acl_groups=source.acl_groups,
            doc_version=queued.requested_doc_version,
            language="zh",
        )
        return RetrySourceResponse(status="queued", source=source_to_response(source))

    @app.delete("/sources/{doc_id:path}", response_model=DeleteSourceResponse)
    def remove_source(
        doc_id: str,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> DeleteSourceResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        detail = delete_source(
            config=config,
            tenant_id=auth_context.tenant_id,
            doc_id=doc_id,
            doc_version=doc_version,
        )
        return DeleteSourceResponse(status="deleted", doc_id=doc_id, detail=detail)

    @app.patch("/sources/{doc_id:path}", response_model=RenameSourceResponse)
    def rename_source_endpoint(
        doc_id: str,
        request: RenameSourceRequest,
        tenant_id: str = "team_a",
        doc_version: int | None = None,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> RenameSourceResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        try:
            source = rename_source(
                config=config,
                tenant_id=auth_context.tenant_id,
                doc_id=doc_id,
                doc_version=doc_version,
                title=request.title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RenameSourceResponse(status="renamed", doc_id=source.doc_id, title=source.title)

    @app.post("/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> SearchResponse:
        config = load_config()
        validate_query_image_data_url(request, config)
        auth_context = resolve_auth_context(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            request=request,
        )
        result = resolve_search_result(request, auth_context)
        response = SearchResponse(
            request_id=result.request_id,
            hits=[hit_to_response(hit, config=config, tenant_id=auth_context.tenant_id) for hit in result.hits],
            trace=result.trace.__dict__,
        )
        append_event(
            config.runtime_dir,
            "retrieval_events",
            {
                "request_id": result.request_id,
                "query": request.query,
                "query_mode": request.query_mode,
                "history_len": len(request.history),
                "doc_version": request.doc_version,
                "doc_ids": request.doc_ids,
                "source_types": request.source_types,
                "auth_context": auth_context.summary(),
                "trace": result.trace,
                "raw_hits": hit_event_summaries(result.candidates),
                "rerank_hits": hit_event_summaries(result.reranked),
                "final_context": hit_event_summaries(result.hits),
            },
        )
        return response

    @app.post("/query", response_model=QueryResponse)
    def query(
        request: QueryRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> QueryResponse:
        config = load_config()
        validate_query_image_data_url(request, config)
        auth_context = resolve_auth_context(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            request=request,
        )
        result = resolve_answer_result(request, auth_context)
        response = QueryResponse(
            request_id=result.request_id,
            answer=result.answer,
            citations=[hit_to_response(hit, config=config, tenant_id=auth_context.tenant_id) for hit in result.hits],
            trace=result.trace.__dict__,
        )
        append_event(
            config.runtime_dir,
            "answer_events",
            {
                "request_id": result.request_id,
                "query": request.query,
                "query_mode": request.query_mode,
                "history_len": len(request.history),
                "auth_context": auth_context.summary(),
                "doc_version": request.doc_version,
                "doc_ids": request.doc_ids,
                "source_types": request.source_types,
                "trace": result.trace,
                "raw_hits": hit_event_summaries(result.candidates),
                "rerank_hits": hit_event_summaries(result.reranked),
                "final_context": hit_event_summaries(result.hits),
                "llm": result.generation,
            },
        )
        return response

    @app.post("/query/stream")
    def query_stream(
        request: QueryRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> StreamingResponse:
        config = load_config()
        validate_query_image_data_url(request, config)
        auth_context = resolve_auth_context(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            request=request,
        )
        stream_slot = query_stream_semaphore(query_stream_queue_limit())
        if not stream_slot.acquire(blocking=False):
            record_query_stream_rejected("global")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Query service is busy. Please retry later. "
                    f"RAG_QUERY_STREAM_QUEUE_LIMIT={query_stream_queue_limit()}"
                ),
            )
        tenant_slot = query_stream_tenant_semaphore(
            query_stream_tenant_queue_limit(),
            auth_context.tenant_id,
        )
        if not tenant_slot.acquire(blocking=False):
            stream_slot.release()
            record_query_stream_rejected("tenant")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Query service is busy for this tenant. Please retry later. "
                    f"tenant_id={auth_context.tenant_id} "
                    f"RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT={query_stream_tenant_queue_limit()}"
                ),
            )
        user_key = query_stream_user_key(auth_context)
        user_slot = None
        if user_key:
            user_slot = query_stream_user_semaphore(query_stream_user_queue_limit(), user_key)
            if not user_slot.acquire(blocking=False):
                tenant_slot.release()
                stream_slot.release()
                record_query_stream_rejected("user")
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Query service is busy for this user. Please retry later. "
                        f"principal={'user' if auth_context.user_id else 'api_token'} "
                        f"RAG_QUERY_STREAM_USER_QUEUE_LIMIT={query_stream_user_queue_limit()}"
                    ),
                )
        shared_guard = None
        if query_shared_admission_enabled():
            try:
                shared_lease = acquire_query_admission_lease(
                    config=config,
                    tenant_id=auth_context.tenant_id,
                    user_key=user_key,
                    global_limit=query_stream_queue_limit(),
                    tenant_limit=query_stream_tenant_queue_limit(),
                    user_limit=query_stream_user_queue_limit(),
                    lease_ms=query_shared_admission_lease_ms(),
                )
            except QueryAdmissionRejected as exc:
                if user_slot is not None:
                    user_slot.release()
                tenant_slot.release()
                stream_slot.release()
                record_query_stream_rejected(exc.kind)
                raise HTTPException(
                    status_code=503,
                    detail=f"Query service shared {exc.kind} capacity is busy. Please retry later.",
                ) from exc
            except Exception as exc:
                if user_slot is not None:
                    user_slot.release()
                tenant_slot.release()
                stream_slot.release()
                record_query_stream_rejected("global")
                raise HTTPException(
                    status_code=503,
                    detail="Query service shared admission is unavailable. Please retry later.",
                ) from exc
            shared_guard = QueryAdmissionLeaseGuard(
                config=config,
                lease=shared_lease,
                lease_ms=query_shared_admission_lease_ms(),
            )
            try:
                shared_guard.start()
            except Exception as exc:
                shared_guard.close()
                if user_slot is not None:
                    user_slot.release()
                tenant_slot.release()
                stream_slot.release()
                record_query_stream_rejected("global")
                raise HTTPException(
                    status_code=503,
                    detail="Query service shared admission is unavailable. Please retry later.",
                ) from exc
        record_query_stream_accepted(auth_context.tenant_id)
        record_query_stream_user_accepted(user_key)

        def stream_events():
            client_disconnected = threading.Event()
            event_queue: Queue[dict[str, object] | None] = Queue(maxsize=query_stream_event_queue_limit())

            def enqueue_event(event: dict[str, object] | None) -> bool:
                while not client_disconnected.is_set():
                    try:
                        event_queue.put(event, timeout=0.5)
                        return True
                    except Full:
                        record_query_stream_event_queue_backpressure()
                return False

            def emit(event_type: str, payload: dict[str, object]) -> bool:
                return enqueue_event({"type": event_type, **payload})

            def emit_stage_event(payload: dict[str, object]) -> None:
                emit("stage", payload)

            def run_query() -> None:
                errored = False
                try:
                    emit(
                        "stage",
                        {
                            "stage": "start",
                            "status": "done",
                            "label": "接收问题",
                            "detail": "已收到问题，正在准备 RAG 调用链。",
                        },
                    )
                    result = resolve_answer_result(request, auth_context, stage_callback=emit_stage_event)
                    if shared_guard is not None and not shared_guard.valid.is_set():
                        raise RuntimeError("Query admission lease was lost before completion")
                    response = QueryResponse(
                        request_id=result.request_id,
                        answer=result.answer,
                        citations=[
                            hit_to_response(hit, config=config, tenant_id=auth_context.tenant_id)
                            for hit in result.hits
                        ],
                        trace=result.trace.__dict__,
                    )
                    append_event(
                        config.runtime_dir,
                        "answer_events",
                        {
                            "request_id": result.request_id,
                            "query": request.query,
                            "query_mode": request.query_mode,
                            "history_len": len(request.history),
                            "auth_context": auth_context.summary(),
                            "doc_version": request.doc_version,
                            "doc_ids": request.doc_ids,
                            "source_types": request.source_types,
                            "trace": result.trace,
                            "raw_hits": hit_event_summaries(result.candidates),
                            "rerank_hits": hit_event_summaries(result.reranked),
                            "final_context": hit_event_summaries(result.hits),
                            "llm": result.generation,
                        },
                    )
                    emit("result", response.model_dump())
                except Exception as exc:  # noqa: BLE001 - streamed API must serialize failures.
                    errored = True
                    emit("error", {"detail": str(exc) or exc.__class__.__name__})
                finally:
                    record_query_stream_finished(auth_context.tenant_id, user_key, errored=errored)
                    if shared_guard is not None:
                        shared_guard.close()
                    if user_slot is not None:
                        user_slot.release()
                    tenant_slot.release()
                    stream_slot.release()
                    enqueue_event(None)

            try:
                query_stream_executor(query_stream_max_workers()).submit(run_query)
            except Exception as exc:  # noqa: BLE001 - streamed API must serialize failures.
                record_query_stream_finished(auth_context.tenant_id, user_key, errored=True)
                if shared_guard is not None:
                    shared_guard.close()
                if user_slot is not None:
                    user_slot.release()
                tenant_slot.release()
                stream_slot.release()
                emit("error", {"detail": str(exc) or exc.__class__.__name__})
                enqueue_event(None)
            try:
                while True:
                    event = event_queue.get()
                    if event is None:
                        break
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            finally:
                client_disconnected.set()

        return StreamingResponse(stream_events(), media_type="application/x-ndjson")

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(
        request: FeedbackRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> FeedbackResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=request.tenant_id,
            acl_groups=request.acl_groups,
        )
        append_event(
            config.runtime_dir,
            "feedback_events",
            {
                **request.model_dump(),
                "tenant_id": auth_context.tenant_id,
                "acl_groups": auth_context.acl_groups,
                "auth_context": auth_context.summary(),
            },
        )
        return FeedbackResponse(
            status="accepted",
            request_id=request.request_id,
        )

    @app.get("/conversations", response_model=ConversationListResponse)
    def conversations(
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ConversationListResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        return ConversationListResponse(
            conversations=[
                conversation_item_to_response(item)
                for item in list_conversation_items(config, tenant_id=auth_context.tenant_id)
            ]
        )

    @app.post("/conversations", response_model=ConversationResponse)
    def upsert_conversation(
        request: ConversationUpsertRequest,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ConversationResponse:
        config = load_config()
        validate_conversation_image_data_urls(request, config)
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=request.tenant_id,
            acl_groups=[],
        )
        try:
            conversation = save_conversation(
                config,
                tenant_id=auth_context.tenant_id,
                conversation_id=request.id,
                title=request.title,
                messages=[message_request_to_domain(message) for message in request.messages],
                source_doc_ids=request.source_doc_ids,
            )
        except ConversationTenantConflictError as exc:
            raise HTTPException(status_code=409, detail="Conversation ID is unavailable") from exc
        return conversation_to_response(conversation)

    @app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(
        conversation_id: str,
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ConversationResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        conversation = load_conversation(
            config,
            tenant_id=auth_context.tenant_id,
            conversation_id=conversation_id,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation_to_response(conversation)

    @app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse)
    def remove_conversation(
        conversation_id: str,
        tenant_id: str = "team_a",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> DeleteConversationResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        removed = delete_conversation(
            config,
            tenant_id=auth_context.tenant_id,
            conversation_id=conversation_id,
        )
        return DeleteConversationResponse(
            status="deleted" if removed else "not_found",
            conversation_id=conversation_id,
        )

    @app.get("/artifacts", response_model=ArtifactListResponse)
    def artifacts(
        tenant_id: str = "team_a",
        workspace_id: str = "",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> ArtifactListResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        migrate_legacy_artifacts(config, tenant_id=auth_context.tenant_id)
        return ArtifactListResponse(
            artifacts=[
                artifact_to_response(artifact)
                for artifact in list_metadata_artifacts(
                    config,
                    tenant_id=auth_context.tenant_id,
                    workspace_id=workspace_id,
                )
            ]
        )

    @app.post("/artifacts/mindmap", response_model=MindMapArtifactResponse)
    def create_mindmap(
        request: MindMapRequest,
        background_tasks: BackgroundTasks,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> MindMapArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=request.tenant_id,
            acl_groups=request.acl_groups,
        )
        artifact = pending_artifact(
            title=request.title,
            tenant_id=auth_context.tenant_id,
            workspace_id=request.workspace_id,
            source_doc_ids=request.source_doc_ids,
            artifact_type="mindmap",
        )
        save_metadata_artifact(config, artifact)
        background_tasks.add_task(
            build_mindmap_background,
            artifact,
            request.context_limit,
        )
        return artifact_to_response(artifact)

    @app.post("/artifacts/table", response_model=MindMapArtifactResponse)
    def create_table(
        request: MindMapRequest,
        background_tasks: BackgroundTasks,
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> MindMapArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=request.tenant_id,
            acl_groups=request.acl_groups,
        )
        artifact = pending_artifact(
            title=request.title,
            tenant_id=auth_context.tenant_id,
            workspace_id=request.workspace_id,
            source_doc_ids=request.source_doc_ids,
            artifact_type="table",
        )
        save_metadata_artifact(config, artifact)
        background_tasks.add_task(build_table_background, artifact)
        return artifact_to_response(artifact)

    @app.get("/artifacts/{artifact_id}", response_model=MindMapArtifactResponse)
    def get_artifact(
        artifact_id: str,
        tenant_id: str = "team_a",
        workspace_id: str = "",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> MindMapArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        artifact = load_metadata_artifact(
            config,
            tenant_id=auth_context.tenant_id,
            artifact_id=artifact_id,
            workspace_id=workspace_id,
        )
        if artifact is None:
            artifact = None if workspace_id else load_artifact(config, tenant_id=auth_context.tenant_id, artifact_id=artifact_id)
            if artifact is not None:
                save_metadata_artifact(config, artifact)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return artifact_to_response(artifact)

    @app.delete("/artifacts/{artifact_id}", response_model=DeleteArtifactResponse)
    def remove_artifact(
        artifact_id: str,
        tenant_id: str = "team_a",
        workspace_id: str = "",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> DeleteArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        removed = delete_metadata_artifact(
            config,
            tenant_id=auth_context.tenant_id,
            artifact_id=artifact_id,
            workspace_id=workspace_id,
        )
        legacy_removed = False if workspace_id else delete_artifact(config, tenant_id=auth_context.tenant_id, artifact_id=artifact_id)
        return DeleteArtifactResponse(
            status="deleted" if removed or legacy_removed else "not_found",
            artifact_id=artifact_id,
        )

    @app.patch("/artifacts/{artifact_id}", response_model=RenameArtifactResponse)
    def rename_artifact(
        artifact_id: str,
        request: RenameArtifactRequest,
        tenant_id: str = "team_a",
        workspace_id: str = "",
        authorization: str | None = Header(default=None),
        x_rag_tenant_id: str | None = Header(default=None),
        x_rag_acl_groups: str | None = Header(default=None),
    ) -> RenameArtifactResponse:
        config = load_config()
        auth_context = resolve_auth_context_from_values(
            config=config,
            authorization=authorization,
            x_rag_tenant_id=x_rag_tenant_id,
            x_rag_acl_groups=x_rag_acl_groups,
            tenant_id=tenant_id,
            acl_groups=[],
        )
        artifact = load_metadata_artifact(
            config,
            tenant_id=auth_context.tenant_id,
            artifact_id=artifact_id,
            workspace_id=workspace_id,
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        updated_artifact = replace(
            artifact,
            title=request.title,
            updated_at=int(time.time() * 1000)
        )
        save_metadata_artifact(config, updated_artifact)

        return RenameArtifactResponse(
            status="renamed",
            artifact_id=artifact_id,
            title=updated_artifact.title,
        )

    return app


def resolve_auth_context(
    *,
    config,
    authorization: str | None,
    x_rag_tenant_id: str | None,
    x_rag_acl_groups: str | None,
    request: QueryRequest,
):
    from fastapi import HTTPException

    try:
        user = authenticate_token(config, token=bearer_token(authorization))
        if user is not None:
            return build_auth_context(
                config=config,
                header_tenant_id=user.tenant_id,
                header_acl_groups="engineering",
                body_tenant_id=request.tenant_id,
                body_acl_groups=request.acl_groups,
                user_id=user.id,
                username=user.username,
            )
        if not config.api_token:
            raise ValueError("请先登录")
        validate_bearer_token(config=config, authorization=authorization)
        return build_auth_context(
            config=config,
            header_tenant_id=x_rag_tenant_id,
            header_acl_groups=x_rag_acl_groups,
            body_tenant_id=request.tenant_id,
            body_acl_groups=request.acl_groups,
            credential_id=bearer_credential_id(authorization),
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def pending_artifact(
    *,
    title: str,
    tenant_id: str,
    workspace_id: str,
    source_doc_ids: list[str],
    artifact_type: str,
) -> MindMapArtifact:
    timestamp = int(time.time() * 1000)
    prefix = "table" if artifact_type == "table" else "mindmap"
    return MindMapArtifact(
        id=f"{prefix}-{uuid.uuid4().hex[:12]}",
        title=title,
        status="generating",
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_doc_ids=source_doc_ids,
        created_at=timestamp,
        updated_at=timestamp,
        artifact_type=artifact_type,
        root=None,
        table=None,
    )


def build_mindmap_background(artifact: MindMapArtifact, context_limit: int) -> None:
    config = load_config()
    try:
        root = build_mindmap_root(
            title=artifact.title,
            config=config,
            tenant_id=artifact.tenant_id,
            source_doc_ids=artifact.source_doc_ids,
            batch_chunk_count=context_limit,
        )
        save_metadata_artifact(
            config,
            replace(artifact, status="ready", root=root, updated_at=int(time.time() * 1000)),
        )
    except Exception as exc:
        fail_metadata_artifact(config, artifact, str(exc))


def build_table_background(artifact: MindMapArtifact) -> None:
    config = load_config()
    try:
        table = build_llm_table(
            title=artifact.title,
            config=config,
            tenant_id=artifact.tenant_id,
            source_doc_ids=artifact.source_doc_ids,
        )
        save_metadata_artifact(
            config,
            replace(artifact, status="ready", table=table, updated_at=int(time.time() * 1000)),
        )
    except Exception as exc:
        fail_metadata_artifact(config, artifact, str(exc))


def migrate_legacy_artifacts(config, *, tenant_id: str) -> None:
    for artifact in list_artifacts(config, tenant_id=tenant_id):
        if load_metadata_artifact(config, tenant_id=tenant_id, artifact_id=artifact.id) is None:
            save_metadata_artifact(config, artifact)


def resolve_auth_context_from_values(
    *,
    config,
    authorization: str | None,
    x_rag_tenant_id: str | None,
    x_rag_acl_groups: str | None,
    tenant_id: str,
    acl_groups: list[str],
):
    from fastapi import HTTPException

    try:
        user = authenticate_token(config, token=bearer_token(authorization))
        if user is not None:
            return build_auth_context(
                config=config,
                header_tenant_id=user.tenant_id,
                header_acl_groups="engineering",
                body_tenant_id=tenant_id,
                body_acl_groups=acl_groups,
                user_id=user.id,
                username=user.username,
            )
        if not config.api_token:
            raise ValueError("请先登录")
        validate_bearer_token(config=config, authorization=authorization)
        return build_auth_context(
            config=config,
            header_tenant_id=x_rag_tenant_id,
            header_acl_groups=x_rag_acl_groups,
            body_tenant_id=tenant_id,
            body_acl_groups=acl_groups,
            credential_id=bearer_credential_id(authorization),
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def resolve_asset_auth_context(
    *,
    config,
    authorization: str | None,
    token: str | None,
    tenant_id: str,
):
    from fastapi import HTTPException

    query_authorization = f"Bearer {token}" if token else None
    resolved_authorization = authorization or query_authorization
    try:
        user = authenticate_token(config, token=bearer_token(resolved_authorization))
        if user is not None:
            return build_auth_context(
                config=config,
                header_tenant_id=user.tenant_id,
                header_acl_groups="engineering",
                body_tenant_id=tenant_id,
                body_acl_groups=[],
                user_id=user.id,
                username=user.username,
            )
        if not config.api_token:
            raise ValueError("请先登录")
        validate_bearer_token(config=config, authorization=resolved_authorization)
        return build_auth_context(
            config=config,
            header_tenant_id=tenant_id,
            header_acl_groups="engineering",
            body_tenant_id=tenant_id,
            body_acl_groups=[],
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def s3_asset_belongs_to_tenant(object_uri: str, tenant_id: str) -> bool:
    try:
        bucket, key = parse_s3_uri(object_uri)
    except ValueError:
        return False
    if bucket != s3_bucket():
        return False
    expected_prefix = s3_key(Path("uploads") / tenant_id).rstrip("/") + "/"
    return key.startswith(expected_prefix) and len(key) > len(expected_prefix)


def validate_query_image_data_url(request: QueryRequest, config) -> None:
    if not request.image_data_url:
        return
    prefix, separator, encoded = request.image_data_url.partition(",")
    if separator != "," or not prefix.startswith("data:image/"):
        record_invalid_query_image()
        raise HTTPException(status_code=400, detail="image_data_url must be a data:image URL")
    estimated_bytes = estimate_base64_decoded_bytes(encoded)
    if estimated_bytes > config.max_query_image_bytes:
        record_query_image_size(estimated_bytes, accepted=False)
        raise HTTPException(
            status_code=413,
            detail=f"Query image is too large. RAG_MAX_QUERY_IMAGE_BYTES={config.max_query_image_bytes}",
        )
    record_query_image_size(estimated_bytes, accepted=True)


def validate_conversation_image_data_urls(request: ConversationUpsertRequest, config) -> None:
    image_count = 0
    total_image_bytes = 0
    image_count_limit = max_conversation_images()
    image_bytes_limit = max_conversation_image_bytes(config)
    for message in request.messages:
        if not message.image_data_url:
            continue
        image_count += 1
        if image_count > image_count_limit:
            raise HTTPException(
                status_code=413,
                detail=f"Too many conversation images. RAG_MAX_CONVERSATION_IMAGES={image_count_limit}",
            )
        prefix, separator, encoded = message.image_data_url.partition(",")
        if separator != "," or not prefix.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="message.image_data_url must be a data:image URL")
        decoded_bytes = estimate_base64_decoded_bytes(encoded)
        if decoded_bytes > config.max_query_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Conversation image is too large. RAG_MAX_QUERY_IMAGE_BYTES={config.max_query_image_bytes}",
            )
        total_image_bytes += decoded_bytes
        if total_image_bytes > image_bytes_limit:
            raise HTTPException(
                status_code=413,
                detail=f"Conversation images are too large. RAG_MAX_CONVERSATION_IMAGE_BYTES={image_bytes_limit}",
            )


def materialize_query_image(request: QueryRequest, config=None) -> str | None:
    if not request.image_data_url:
        return None
    config = config or load_config()
    prefix, separator, encoded = request.image_data_url.partition(",")
    if separator != "," or not prefix.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image_data_url must be a data:image URL")
    media_type = prefix.removeprefix("data:").split(";", 1)[0]
    extension = media_type.split("/", 1)[1].lower()
    if extension == "jpeg":
        extension = "jpg"
    if extension not in {"png", "jpg", "webp", "gif"}:
        raise HTTPException(status_code=400, detail="Unsupported query image type")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid query image data") from exc
    if len(image_bytes) > config.max_query_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Query image is too large. RAG_MAX_QUERY_IMAGE_BYTES={config.max_query_image_bytes}",
        )
    query_dir = config.runtime_dir / "query_images"
    query_dir.mkdir(parents=True, exist_ok=True)
    image_path = query_dir / f"{uuid.uuid4().hex}.{extension}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def resolve_search_result(request: SearchRequest, auth_context):
    if request.query_mode == "multimodal":
        image_query_path = materialize_query_image(request)
        return retrieve_multimodal(
            request.query,
            text_query=request.query,
            image_query_path=image_query_path,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            doc_ids=request.doc_ids or None,
            source_types=request.source_types or None,
            include_all_sources=request.include_all_sources,
            history=request.history,
            request_id=request.request_id,
        )
    return retrieve_and_rerank(
        request.query,
        tenant_id=auth_context.tenant_id,
        candidate_limit=request.candidate_limit,
        context_limit=request.context_limit,
        acl_groups=auth_context.acl_groups or None,
        doc_version=request.doc_version,
        doc_ids=request.doc_ids or None,
        source_types=request.source_types or None,
        include_all_sources=request.include_all_sources,
        history=request.history,
        request_id=request.request_id,
    )


def resolve_answer_result(request: QueryRequest, auth_context, stage_callback=None):
    if request.query_mode == "multimodal":
        image_query_path = materialize_query_image(request)
        return answer_multimodal_query(
            request.query,
            text_query=request.query,
            image_query_path=image_query_path,
            tenant_id=auth_context.tenant_id,
            candidate_limit=request.candidate_limit,
            context_limit=request.context_limit,
            acl_groups=auth_context.acl_groups or None,
            doc_version=request.doc_version,
            doc_ids=request.doc_ids or None,
            source_types=request.source_types or None,
            include_all_sources=request.include_all_sources,
            history=request.history,
            request_id=request.request_id,
            answer_query=request.query,
            stage_callback=stage_callback,
        )
    return answer_query(
        request.query,
        tenant_id=auth_context.tenant_id,
        candidate_limit=request.candidate_limit,
        context_limit=request.context_limit,
        acl_groups=auth_context.acl_groups or None,
        doc_version=request.doc_version,
        doc_ids=request.doc_ids or None,
        source_types=request.source_types or None,
        include_all_sources=request.include_all_sources,
        history=request.history,
        request_id=request.request_id,
        stage_callback=stage_callback,
    )


def hit_to_response(hit, *, config=None, tenant_id: str = "") -> HitResponse:
    metadata = hit.metadata
    if config is not None and tenant_id:
        metadata = resolve_metadata_display_block_urls(config=config, tenant_id=tenant_id, metadata=metadata)
    return HitResponse(
        doc_id=hit.doc_id,
        title=hit.title,
        source_uri=hit.source_uri,
        source_type=hit.source_type,
        chunk_index=hit.chunk_index,
        score=hit.score,
        rerank_score=hit.rerank_score,
        acl_groups=hit.acl_groups,
        metadata=metadata,
        text=hit.text,
        text_preview=hit.text[:360],
    )


def source_to_response(source) -> SourceResponse:
    return SourceResponse(
        doc_id=source.doc_id,
        title=source.title,
        source_type=source.source_type,
        source_uri=source.source_uri,
        doc_version=source.doc_version,
        chunk_count=source.chunk_count,
        acl_groups=source.acl_groups,
        status=source.status,
        current=source.current,
        created_at=source.created_at,
        updated_at=source.updated_at,
        child_doc_ids=source.child_doc_ids,
        error=getattr(source, "error", ""),
        retryable=source.status == "failed",
        attempt_count=getattr(source, "attempt_count", 0),
        next_attempt_at=getattr(source, "next_attempt_at", 0),
        dead_lettered=getattr(source, "dead_lettered", False),
    )


def artifact_to_response(artifact) -> MindMapArtifactResponse:
    return MindMapArtifactResponse(
        id=artifact.id,
        title=artifact.title,
        status=artifact.status,
        tenant_id=artifact.tenant_id,
        workspace_id=artifact.workspace_id,
        source_doc_ids=artifact.source_doc_ids,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
        artifact_type=artifact.artifact_type,
        root=artifact.root,
        table=artifact.table,
        error=artifact.error,
    )


def message_request_to_domain(message: ConversationMessageRequest) -> ConversationMessage:
    return ConversationMessage(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        content=message.content,
        status=message.status,  # type: ignore[arg-type]
        request_id=message.request_id,
        citations=[citation.model_dump() for citation in message.citations],
        image_data_url=message.image_data_url,
        created_at=message.created_at,
        feedback_rating=message.feedback_rating,
        rag_progress=message.rag_progress,
    )


def conversation_to_response(conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        tenant_id=conversation.tenant_id,
        title=conversation.title,
        messages=[
            ConversationMessageRequest(
                id=message.id,
                role=message.role,
                content=message.content,
                status=message.status,
                request_id=message.request_id,
                citations=[HitResponse(**citation) for citation in message.citations],
                image_data_url=message.image_data_url,
                created_at=message.created_at,
                feedback_rating=message.feedback_rating,
                rag_progress=message.rag_progress,
            )
            for message in conversation.messages
        ],
        source_doc_ids=conversation.source_doc_ids,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def conversation_to_list_item(conversation) -> ConversationListItemResponse:
    return ConversationListItemResponse(
        id=conversation.id,
        tenant_id=conversation.tenant_id,
        title=conversation.title,
        message_count=len(conversation.messages),
        source_doc_ids=conversation.source_doc_ids,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def conversation_item_to_response(item) -> ConversationListItemResponse:
    return ConversationListItemResponse(
        id=item.id,
        tenant_id=item.tenant_id,
        title=item.title,
        message_count=item.message_count,
        source_doc_ids=item.source_doc_ids,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def user_to_response(user) -> UserResponse:
    return UserResponse(**user.public_dict())


def admin_settings_response(config) -> AdminSettingsResponse:
    latest = list_announcements(config, limit=1)
    return AdminSettingsResponse(
        registration_enabled=is_registration_enabled(config),
        latest_announcement=AnnouncementResponse(**latest[0]) if latest else None,
    )


def require_current_user(*, authorization: str | None):
    from fastapi import HTTPException

    config = load_config()
    user = authenticate_token(config, token=bearer_token(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def require_admin(*, config, authorization: str | None):
    from fastapi import HTTPException

    user = authenticate_token(config, token=bearer_token(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=8008, reload=False)
