from __future__ import annotations

import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.model_api_retry import call_model_api_with_retries  # noqa: E402


def main() -> None:
    with isolated_runtime():
        seed_ingestion_stage_stats()
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
    assert "rag_metadata_pool_timeouts_total" in text
    assert 'rag_ingestion_tasks{status="queued"}' in text
    assert 'rag_ingestion_task_leases{state="active"}' in text
    assert 'rag_ingestion_task_leases{state="expired"}' in text
    assert 'rag_ingestion_task_attempts{stat="sum"}' in text
    assert 'rag_ingestion_task_attempts{stat="max"}' in text
    assert 'rag_ingestion_task_recovery{state="retry_waiting"}' in text
    assert 'rag_ingestion_task_recovery{state="dead_lettered"}' in text
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
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        try:
            yield
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_METADATA_DATABASE_URL", old_metadata_url)


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
