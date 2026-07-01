from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.ingestion_operations import append_ingestion_operation_audit  # noqa: E402
from rag_core.model_api_retry import call_model_api_with_retries  # noqa: E402
from rag_core.model_usage import record_model_usage  # noqa: E402
from rag_core.query_rate_limits import acquire_query_rate_limit  # noqa: E402


def main() -> None:
    with isolated_runtime():
        seed_ingestion_stage_stats()
        seed_query_result_cache()
        seed_query_rate_limit()
        seed_ingestion_operator_audit()
        seed_model_usage()
        api = TestClient(serve.create_app())
        assert call_model_api_with_retries("metrics_smoke", lambda: "ok") == "ok"
        serve.record_query_image_size(64 * 1024, accepted=True)
        serve.record_query_image_size(3 * 1024 * 1024, accepted=False)
        assert api.get("/health").status_code == 200
        assert api.get("/sources/private-doc@sha256-secret?tenant_id=metrics-smoke").status_code >= 400
        response = api.get("/metrics")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "# TYPE rag_http_requests_total counter" in text
    assert "# TYPE rag_http_request_latency_seconds histogram" in text
    assert 'rag_http_requests_total{route="GET /health"}' in text
    assert 'rag_http_request_latency_seconds_bucket{route="GET /health",le="0.01"}' in text
    assert 'rag_http_request_latency_seconds_bucket{route="GET /health",le="+Inf"} 1' in text
    assert 'rag_http_request_latency_seconds_count{route="GET /health"} 1' in text
    assert "private-doc" not in text
    assert "sha256-secret" not in text
    assert 'rag_query_stream_events_total{event="rejected_user"}' in text
    assert 'rag_query_rate_limit_config{scope="global"} 7' in text
    assert 'rag_query_rate_limit_requests{scope="global"} 1' in text
    assert 'rag_query_rate_limit_active_keys{scope="tenant"} 1' in text
    assert 'rag_query_rate_limit_events_total{event="accepted"}' in text
    assert "metrics-private-rate-tenant" not in text
    assert 'rag_query_result_cache_entries{status="processing"} 1' in text
    assert 'rag_query_result_cache_entries{status="completed"} 1' in text
    assert 'rag_query_result_cache_entries{status="failed"} 0' in text
    assert "rag_query_result_cache_expired_entries 0" in text
    assert "rag_query_result_stale_processing_entries 1" in text
    assert "rag_query_result_events 1" in text
    assert "metrics-private-tenant" not in text
    assert "metrics-private-request" not in text
    assert 'rag_query_shared_admission_slots{scope="global"}' in text
    assert 'rag_query_shared_admission_slots{scope="tenant"}' in text
    assert 'rag_query_shared_admission_slots{scope="user"}' in text
    assert 'rag_query_shared_admission_slots{scope="expired"}' in text
    assert 'rag_query_image_payloads_total{outcome="accepted"}' in text
    assert "# TYPE rag_query_image_payload_bytes histogram" in text
    assert 'rag_query_image_payload_bytes_bucket{outcome="accepted",le="65536"}' in text
    assert 'rag_query_image_payload_bytes_bucket{outcome="rejected_oversized",le="+Inf"}' in text
    assert 'rag_query_image_payload_bytes_sum{outcome="accepted"}' in text
    assert 'rag_query_image_payload_bytes_count{outcome="rejected_oversized"}' in text
    assert (
        'rag_model_api_operation_calls_total{operation="metrics_smoke",outcome="success"} 1'
        in text
    )
    assert 'rag_model_api_operation_retries_total{operation="metrics_smoke"} 0' in text
    assert 'rag_model_usage_daily{workload="query",kind="total_tokens"} 17' in text
    assert 'rag_model_usage_recording_events_total{outcome="write_failure"}' in text
    assert "metrics-private-model-tenant" not in text
    assert "rag_metadata_pool_timeouts_total" in text
    assert 'rag_ingestion_tasks{status="queued"}' in text
    assert 'rag_ingestion_task_leases{state="active"}' in text
    assert 'rag_ingestion_task_leases{state="expired"}' in text
    assert 'rag_ingestion_task_attempts{stat="sum"}' in text
    assert 'rag_ingestion_task_attempts{stat="max"}' in text
    assert 'rag_ingestion_task_recovery{state="retry_waiting"}' in text
    assert 'rag_ingestion_task_recovery{state="dead_lettered"}' in text
    assert (
        'rag_ingestion_operator_audit_events{operation="bulk_redrive",outcome="queued"} 1'
        in text
    )
    assert "metrics-private-operator-tenant" not in text
    assert "metrics-private-operator-task" not in text
    assert 'rag_ingestion_task_recovery{state="retries_recorded"}' in text
    assert 'rag_ingestion_stage_samples{source_type="txt",stage="text_embedding"} 2' in text
    assert (
        'rag_ingestion_stage_duration_seconds_average{source_type="txt",stage="text_embedding"} 4.000'
        in text
    )
    assert 'rag_ingestion_upload_reservations{scope="global"}' in text
    assert 'rag_ingestion_upload_reservations{scope="tenant"}' in text
    assert 'rag_ingestion_upload_reservations{scope="expired"}' in text
    validate_metric_lines(text)
    validate_health_histogram(text)
    validate_query_image_histogram(text)
    print("smoke_prometheus_metrics=ok")


def seed_ingestion_stage_stats() -> None:
    config = load_config()
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO ingestion_stage_stats(
                source_type, stage, sample_count, total_duration_ms, updated_at
            )
            VALUES ('txt', 'text_embedding', 2, 8000, 1)
            """
        )


def seed_query_result_cache() -> None:
    config = load_config()
    timestamp = int(time.time() * 1000)
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            INSERT INTO query_result_cache(
                tenant_id, request_id, request_fingerprint, status, lease_owner,
                lease_expires_at, response_json, error, created_at, updated_at, expires_at
            )
            VALUES (?, ?, ?, 'completed', '', 0, '{}', '', ?, ?, ?)
            """,
            (
                "metrics-private-tenant",
                "metrics-private-request",
                "metrics-private-fingerprint",
                timestamp,
                timestamp,
                timestamp + 60_000,
            ),
        )
        conn.execute(
            """
            INSERT INTO query_result_events(
                tenant_id, request_id, sequence, event_json, created_at
            )
            VALUES (?, ?, 1, '{}', ?)
            """,
            ("metrics-private-tenant", "metrics-private-request", timestamp),
        )
        conn.execute(
            """
            INSERT INTO query_result_cache(
                tenant_id, request_id, request_fingerprint, status, lease_owner,
                lease_expires_at, response_json, error, created_at, updated_at, expires_at
            )
            VALUES (?, ?, ?, 'processing', 'expired-owner', ?, '', '', ?, ?, ?)
            """,
            (
                "metrics-private-tenant",
                "metrics-private-stale-request",
                "metrics-private-stale-fingerprint",
                timestamp - 1,
                timestamp - 60_000,
                timestamp - 60_000,
                timestamp + 60_000,
            ),
        )


def seed_query_rate_limit() -> None:
    os.environ["RAG_QUERY_RATE_LIMIT_GLOBAL"] = "7"
    os.environ["RAG_QUERY_RATE_LIMIT_TENANT"] = "5"
    os.environ["RAG_QUERY_RATE_LIMIT_USER"] = "3"
    acquire_query_rate_limit(
        config=load_config(),
        tenant_id="metrics-private-rate-tenant",
        user_key="metrics-private-rate-user",
        global_limit=7,
        tenant_limit=5,
        user_limit=3,
    )
    serve.record_query_rate_limit_event("accepted")


def seed_ingestion_operator_audit() -> None:
    append_ingestion_operation_audit(
        config=load_config(),
        actor_user_id="metrics-private-operator-user",
        tenant_id="metrics-private-operator-tenant",
        task_id="metrics-private-operator-task",
        operation="bulk_redrive",
        outcome="queued",
    )


def seed_model_usage() -> None:
    record_model_usage(
        config=load_config(),
        tenant_id="metrics-private-model-tenant",
        principal_key="user:metrics-private-model-user",
        workload="query",
        provider="synthetic-provider",
        model="synthetic-model",
        operation="answer_generation",
        usage={
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
        },
    )


def validate_metric_lines(text: str) -> None:
    sample_pattern = re.compile(
        r'^[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[a-zA-Z_][a-zA-Z0-9_]*="(?:\\.|[^"\\])*"'
        r'(?:,[a-zA-Z_][a-zA-Z0-9_]*="(?:\\.|[^"\\])*")*\})?\s+'
        r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]Inf)$"
    )
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        assert sample_pattern.fullmatch(line), line


def validate_health_histogram(text: str) -> None:
    counts = []
    for line in text.splitlines():
        if not line.startswith('rag_http_request_latency_seconds_bucket{route="GET /health"'):
            continue
        counts.append(int(line.rsplit(" ", 1)[1]))
    assert counts
    assert counts == sorted(counts)
    assert counts[-1] == 1


def validate_query_image_histogram(text: str) -> None:
    for outcome in ("accepted", "rejected_oversized"):
        bucket_pattern = re.compile(
            rf'^rag_query_image_payload_bytes_bucket\{{outcome="{outcome}",le="([^"]+)"\}}\s+(\d+)$',
        )
        buckets = [
            (match.group(1), int(match.group(2)))
            for line in text.splitlines()
            if (match := bucket_pattern.match(line))
        ]
        assert buckets[-1][0] == "+Inf"
        counts = [count for _, count in buckets]
        assert counts == sorted(counts)
        count_pattern = re.compile(
            rf'^rag_query_image_payload_bytes_count\{{outcome="{outcome}"\}}\s+(\d+)$',
        )
        count = next(
            int(match.group(1))
            for line in text.splitlines()
            if (match := count_pattern.match(line))
        )
        assert counts[-1] == count


@contextmanager
def isolated_runtime():
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_metadata_url = os.environ.get("RAG_METADATA_DATABASE_URL")
    old_rate_limits = {
        name: os.environ.get(name)
        for name in (
            "RAG_QUERY_RATE_LIMIT_GLOBAL",
            "RAG_QUERY_RATE_LIMIT_TENANT",
            "RAG_QUERY_RATE_LIMIT_USER",
        )
    }
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        try:
            yield
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_METADATA_DATABASE_URL", old_metadata_url)
            for name, value in old_rate_limits.items():
                restore_env(name, value)


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
