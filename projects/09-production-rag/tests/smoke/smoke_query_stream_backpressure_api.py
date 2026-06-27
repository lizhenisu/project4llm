from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve
from rag_core.auth import AuthContext


def main() -> None:
    test_query_stream_route_returns_503_when_queue_slot_is_unavailable()
    test_query_stream_route_returns_503_when_tenant_slot_is_unavailable()
    test_query_stream_route_returns_503_when_user_slot_is_unavailable()
    print("smoke_query_stream_backpressure_api=ok")


def test_query_stream_route_returns_503_when_queue_slot_is_unavailable() -> None:
    old_limit = os.environ.get("RAG_QUERY_STREAM_QUEUE_LIMIT")
    os.environ["RAG_QUERY_STREAM_QUEUE_LIMIT"] = "1"
    semaphore = serve.query_stream_semaphore(1)
    acquired = semaphore.acquire(blocking=False)
    assert acquired
    try:
        with patch("serve.resolve_auth_context", return_value=AuthContext("team_a", ["engineering"], "smoke")):
            api = TestClient(serve.create_app())
            response = api.post(
                "/query/stream",
                json={
                    "query": "smoke backpressure",
                    "tenant_id": "team_a",
                    "acl_groups": ["engineering"],
                    "request_id": "smoke-query-stream-backpressure",
                },
            )
        assert response.status_code == 503, response.text
        assert "Query service is busy" in response.text
    finally:
        semaphore.release()
        restore_env("RAG_QUERY_STREAM_QUEUE_LIMIT", old_limit)


def test_query_stream_route_returns_503_when_tenant_slot_is_unavailable() -> None:
    old_global_limit = os.environ.get("RAG_QUERY_STREAM_QUEUE_LIMIT")
    old_tenant_limit = os.environ.get("RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT")
    os.environ["RAG_QUERY_STREAM_QUEUE_LIMIT"] = "8"
    os.environ["RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT"] = "1"
    semaphore = serve.query_stream_tenant_semaphore(1, "tenant-smoke")
    acquired = semaphore.acquire(blocking=False)
    assert acquired
    try:
        with patch("serve.resolve_auth_context", return_value=AuthContext("tenant-smoke", ["engineering"], "smoke")):
            api = TestClient(serve.create_app())
            response = api.post(
                "/query/stream",
                json={
                    "query": "smoke tenant backpressure",
                    "tenant_id": "tenant-smoke",
                    "acl_groups": ["engineering"],
                    "request_id": "smoke-query-stream-tenant-backpressure",
                },
            )
        assert response.status_code == 503, response.text
        assert "Query service is busy for this tenant" in response.text
        assert "RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT=1" in response.text
    finally:
        semaphore.release()
        restore_env("RAG_QUERY_STREAM_QUEUE_LIMIT", old_global_limit)
        restore_env("RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT", old_tenant_limit)


def test_query_stream_route_returns_503_when_user_slot_is_unavailable() -> None:
    old_global_limit = os.environ.get("RAG_QUERY_STREAM_QUEUE_LIMIT")
    old_tenant_limit = os.environ.get("RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT")
    old_user_limit = os.environ.get("RAG_QUERY_STREAM_USER_QUEUE_LIMIT")
    os.environ["RAG_QUERY_STREAM_QUEUE_LIMIT"] = "8"
    os.environ["RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT"] = "8"
    os.environ["RAG_QUERY_STREAM_USER_QUEUE_LIMIT"] = "1"
    auth_context = AuthContext(
        "tenant-user-smoke",
        ["engineering"],
        "smoke",
        user_id="user-smoke",
        username="user_smoke",
    )
    user_key = serve.query_stream_user_key(auth_context)
    semaphore = serve.query_stream_user_semaphore(1, user_key)
    acquired = semaphore.acquire(blocking=False)
    assert acquired
    try:
        with patch("serve.resolve_auth_context", return_value=auth_context):
            api = TestClient(serve.create_app())
            before = api.get("/runtime-metrics").json()["query_stream"]["rejected_user_total"]
            response = api.post(
                "/query/stream",
                json={
                    "query": "smoke user backpressure",
                    "tenant_id": "tenant-user-smoke",
                    "acl_groups": ["engineering"],
                    "request_id": "smoke-query-stream-user-backpressure",
                },
            )
            after = api.get("/runtime-metrics").json()["query_stream"]["rejected_user_total"]
        assert response.status_code == 503, response.text
        assert "Query service is busy for this user" in response.text
        assert "RAG_QUERY_STREAM_USER_QUEUE_LIMIT=1" in response.text
        assert after == before + 1
    finally:
        semaphore.release()
        restore_env("RAG_QUERY_STREAM_QUEUE_LIMIT", old_global_limit)
        restore_env("RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT", old_tenant_limit)
        restore_env("RAG_QUERY_STREAM_USER_QUEUE_LIMIT", old_user_limit)


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
