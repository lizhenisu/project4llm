from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve
from answer import AnswerResult
from rag_core.auth import AuthContext
from rag_core.answering import AnswerGeneration
from rag_core.config import load_config
from rag_core.model_api_retry import model_api_slot
from rag_core.sources import SourceSummary, save_source_task_for_tenant
from rag_core.text_utils import now_ms
from rag_core.types import TraceInfo


def main() -> None:
    with isolated_runtime():
        test_runtime_metrics_exposes_runtime_counters_and_ingestion_counts()
        test_runtime_metrics_records_http_route_counts()
        test_runtime_metrics_records_query_stream_acceptance_and_completion()
        test_runtime_metrics_records_query_stream_rejections()
    print("smoke_runtime_metrics=ok")


def test_runtime_metrics_exposes_runtime_counters_and_ingestion_counts() -> None:
    config = load_config()
    tenant_id = f"tenant-runtime-metrics-{now_ms()}"
    save_source_task_for_tenant(
        config=config,
        tenant_id=tenant_id,
        source=SourceSummary(
            doc_id="runtime-metrics-doc",
            title="runtime metrics doc.txt",
            source_type="txt",
            source_uri="memory://runtime-metrics-doc",
            doc_version=1,
            chunk_count=0,
            acl_groups=["engineering"],
            status="queued",
            current=False,
            created_at=now_ms(),
            updated_at=now_ms(),
            child_doc_ids=[],
        ),
    )

    api = TestClient(serve.create_app())
    with model_api_slot("runtime-metrics-smoke"):
        response = api.get(f"/runtime-metrics?tenant_id={tenant_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query_stream"]["queue_limit"] >= 1
    assert body["query_stream"]["tenant_queue_limit"] >= 1
    assert body["query_stream"]["user_queue_limit"] >= 1
    assert body["query_stream"]["event_queue_limit"] >= 1
    assert body["query_stream"]["workers"] >= 1
    assert body["query"]["max_query_image_bytes"] >= 1
    assert body["query"]["max_query_request_bytes"] >= body["query"]["max_query_image_bytes"]
    assert body["conversation"]["max_conversation_request_bytes"] >= body["query"]["max_query_request_bytes"]
    assert body["conversation"]["max_conversation_images"] >= 1
    assert body["conversation"]["max_conversation_image_bytes"] >= body["query"]["max_query_image_bytes"]
    assert body["model_api"]["active"] >= 1
    assert body["model_api"]["acquired_total"] >= 1
    assert body["milvus_client"]["created_total"] >= 0
    assert body["milvus_client"]["reused_total"] >= 0
    assert body["milvus_client"]["thread_cached_clients"] >= 0
    assert isinstance(body["milvus_client"]["cache_enabled"], bool)
    assert body["metadata_db"]["pool_count"] >= 0
    assert body["metadata_db"]["default_pool_size"] >= 1
    assert body["metadata_db"]["acquire_timeout_seconds"] >= 0.0
    assert body["auth_token_cache"]["hits_total"] >= 0
    assert body["auth_token_cache"]["misses_total"] >= 0
    assert body["auth_token_cache"]["entries"] >= 0
    assert body["auth_token_cache"]["ttl_seconds"] >= 0.0
    assert body["event_log"]["max_json_bytes"] >= 1
    assert body["event_log"]["max_string_chars"] >= 1
    assert body["event_log"]["max_list_items"] >= 1
    assert body["event_log"]["max_dict_items"] >= 1
    assert body["ingestion"]["tenant_id"] == tenant_id
    assert body["ingestion"]["source_tasks_by_status"]["queued"] == 1
    assert body["ingestion"]["active_source_tasks"] >= 1
    assert body["ingestion"]["tenant_active_source_tasks"] == 1
    assert body["ingestion"]["backlog_limit"] >= 1
    assert body["ingestion"]["tenant_backlog_limit"] >= 1
    assert body["ingestion"]["max_upload_bytes"] >= 1


def test_runtime_metrics_records_http_route_counts() -> None:
    api = TestClient(serve.create_app())
    assert api.get("/health").status_code == 200
    doc_response = api.get("/sources/not-a-real-doc@sha256-deadbeef?tenant_id=tenant-runtime-metrics")
    assert 400 <= doc_response.status_code < 500
    response = api.get("/runtime-metrics")
    assert response.status_code == 200, response.text
    http = response.json()["http"]
    assert http["active_total"] >= 0
    assert http["routes"]["GET /health"]["requests_total"] >= 1
    assert http["routes"]["GET /health"]["2xx_total"] >= 1
    assert http["routes"]["GET /health"]["latency_avg_ms"] >= 0.0
    assert http["routes"]["GET /sources/{doc_id}"]["4xx_total"] >= 1
    assert not any("sha256-deadbeef" in route for route in http["routes"])


def test_runtime_metrics_records_query_stream_acceptance_and_completion() -> None:
    old_event_queue_limit = os.environ.get("RAG_QUERY_STREAM_EVENT_QUEUE_LIMIT")
    os.environ["RAG_QUERY_STREAM_EVENT_QUEUE_LIMIT"] = "1"
    api = TestClient(serve.create_app())
    tenant_id = f"tenant-runtime-query-{now_ms()}"
    before = api.get("/runtime-metrics").json()["query_stream"]
    try:
        with (
            patch(
                "serve.resolve_auth_context",
                return_value=AuthContext(tenant_id, ["engineering"], "smoke", user_id="user-runtime-query"),
            ),
            patch("serve.resolve_answer_result", return_value=fake_answer_result()),
        ):
            response = api.post(
                "/query/stream",
                json={
                    "query": "smoke accepted query stream",
                    "tenant_id": tenant_id,
                    "acl_groups": ["engineering"],
                    "request_id": "smoke-runtime-metrics-accepted",
                },
            )
        assert response.status_code == 200, response.text
        assert '"type": "result"' in response.text
        after = api.get("/runtime-metrics").json()["query_stream"]
        assert after["accepted_total"] == before["accepted_total"] + 1
        assert after["completed_total"] == before["completed_total"] + 1
        assert after["active"] == before["active"]
        assert after["event_queue_limit"] == 1
        assert tenant_id not in after["active_by_tenant"]
        assert f"{tenant_id}:user-runtime-query" not in after["active_by_user"]
    finally:
        restore_env("RAG_QUERY_STREAM_EVENT_QUEUE_LIMIT", old_event_queue_limit)


def test_runtime_metrics_records_query_stream_rejections() -> None:
    old_limit = os.environ.get("RAG_QUERY_STREAM_QUEUE_LIMIT")
    os.environ["RAG_QUERY_STREAM_QUEUE_LIMIT"] = "1"
    semaphore = serve.query_stream_semaphore(1)
    acquired = semaphore.acquire(blocking=False)
    assert acquired
    try:
        api = TestClient(serve.create_app())
        before = api.get("/runtime-metrics").json()["query_stream"]["rejected_global_total"]
        with patch("serve.resolve_auth_context", return_value=AuthContext("tenant-runtime-metrics", ["engineering"], "smoke")):
            response = api.post(
                "/query/stream",
                json={
                    "query": "smoke runtime metrics backpressure",
                    "tenant_id": "tenant-runtime-metrics",
                    "acl_groups": ["engineering"],
                    "request_id": "smoke-runtime-metrics-backpressure",
                },
            )
        after = api.get("/runtime-metrics").json()["query_stream"]["rejected_global_total"]
        assert response.status_code == 503, response.text
        assert after == before + 1
    finally:
        semaphore.release()
        restore_env("RAG_QUERY_STREAM_QUEUE_LIMIT", old_limit)


def fake_answer_result() -> AnswerResult:
    trace = TraceInfo(
        request_id="smoke-runtime-metrics-accepted",
        original_query="smoke accepted query stream",
        rewritten_query="smoke accepted query stream",
        rewrite_backend="smoke",
        tenant_id="tenant-runtime-query",
        acl_groups=["engineering"],
        doc_version=None,
        current_versions={},
        embedding_model="smoke",
        source_types=[],
        doc_ids=[],
        filter_expr="",
        retrieval_mode="smoke",
        candidate_count=0,
        reranked_count=0,
        context_count=0,
        dropped_by_score=0,
        dropped_by_doc_limit=0,
        dropped_by_budget=0,
        stage_latency_ms={},
    )
    generation = AnswerGeneration(
        answer="smoke answer",
        llm_model="smoke",
        llm_backend="smoke",
        latency_ms=1.0,
        token_usage={},
    )
    return AnswerResult(
        request_id="smoke-runtime-metrics-accepted",
        answer="smoke answer",
        hits=[],
        candidates=[],
        reranked=[],
        trace=trace,
        generation=generation,
    )


class isolated_runtime:
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_env = {
            "RAG_RUNTIME_DIR": os.environ.get("RAG_RUNTIME_DIR"),
            "RAG_OBJECT_STORE_DIR": os.environ.get("RAG_OBJECT_STORE_DIR"),
            "RAG_METADATA_DATABASE_URL": os.environ.get("RAG_METADATA_DATABASE_URL"),
        }
        root = Path(self.tmp.name)
        os.environ["RAG_RUNTIME_DIR"] = str(root / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(root / "object_store")
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for name, value in self.old_env.items():
            restore_env(name, value)
        self.tmp.cleanup()


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
