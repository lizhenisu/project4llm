from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve
from rag_core.auth import AuthContext
from rag_core.config import load_config
from rag_core.query_admission import acquire_query_admission_lease, release_query_admission_lease


def main() -> None:
    test_query_stream_route_returns_503_when_queue_slot_is_unavailable()
    test_query_stream_route_returns_503_when_tenant_slot_is_unavailable()
    test_query_stream_route_returns_503_when_user_slot_is_unavailable()
    test_api_token_auth_context_gets_redacted_credential_principal()
    test_query_stream_route_applies_user_limit_to_api_token_principal()
    test_query_stream_route_applies_shared_user_limit()
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


def test_query_stream_route_applies_user_limit_to_api_token_principal() -> None:
    old_global_limit = os.environ.get("RAG_QUERY_STREAM_QUEUE_LIMIT")
    old_tenant_limit = os.environ.get("RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT")
    old_user_limit = os.environ.get("RAG_QUERY_STREAM_USER_QUEUE_LIMIT")
    os.environ["RAG_QUERY_STREAM_QUEUE_LIMIT"] = "8"
    os.environ["RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT"] = "8"
    os.environ["RAG_QUERY_STREAM_USER_QUEUE_LIMIT"] = "1"
    auth_context = AuthContext(
        "tenant-api-token-smoke",
        ["engineering"],
        "headers",
        credential_id="0123456789abcdef",
    )
    user_key = serve.query_stream_user_key(auth_context)
    assert user_key == "tenant-api-token-smoke:api_token:0123456789abcdef"
    semaphore = serve.query_stream_user_semaphore(1, user_key)
    acquired = semaphore.acquire(blocking=False)
    assert acquired
    try:
        with patch("serve.resolve_auth_context", return_value=auth_context):
            api = TestClient(serve.create_app())
            response = api.post(
                "/query/stream",
                json={
                    "query": "smoke api token fairness",
                    "tenant_id": "tenant-api-token-smoke",
                    "acl_groups": ["engineering"],
                    "request_id": "smoke-query-stream-api-token-backpressure",
                },
            )
        assert response.status_code == 503, response.text
        assert "principal=api_token" in response.text
        assert "0123456789abcdef" not in response.text
    finally:
        semaphore.release()
        restore_env("RAG_QUERY_STREAM_QUEUE_LIMIT", old_global_limit)
        restore_env("RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT", old_tenant_limit)
        restore_env("RAG_QUERY_STREAM_USER_QUEUE_LIMIT", old_user_limit)


def test_api_token_auth_context_gets_redacted_credential_principal() -> None:
    config = SimpleNamespace(api_token="smoke-secret-token", require_auth_context=True)
    request = serve.QueryRequest(query="smoke", tenant_id="ignored")
    with patch("serve.authenticate_token", return_value=None):
        auth_context = serve.resolve_auth_context(
            config=config,
            authorization="Bearer smoke-secret-token",
            x_rag_tenant_id="tenant-api-token-smoke",
            x_rag_acl_groups="engineering",
            request=request,
        )
    assert auth_context.credential_id
    assert auth_context.credential_id != "smoke-secret-token"
    assert "smoke-secret-token" not in str(auth_context.summary())
    assert "credential_id" not in auth_context.summary()
    assert serve.query_stream_user_key(auth_context).startswith(
        "tenant-api-token-smoke:api_token:"
    )


def test_query_stream_route_applies_shared_user_limit() -> None:
    old_values = {
        name: os.environ.get(name)
        for name in (
            "RAG_QUERY_STREAM_QUEUE_LIMIT",
            "RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT",
            "RAG_QUERY_STREAM_USER_QUEUE_LIMIT",
            "RAG_QUERY_SHARED_ADMISSION",
        )
    }
    os.environ["RAG_QUERY_STREAM_QUEUE_LIMIT"] = "8"
    os.environ["RAG_QUERY_STREAM_TENANT_QUEUE_LIMIT"] = "8"
    os.environ["RAG_QUERY_STREAM_USER_QUEUE_LIMIT"] = "1"
    os.environ["RAG_QUERY_SHARED_ADMISSION"] = "1"
    auth_context = AuthContext(
        "tenant-shared-user-smoke",
        ["engineering"],
        "smoke",
        user_id="shared-user-smoke",
        username="shared_user_smoke",
    )
    user_key = serve.query_stream_user_key(auth_context)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                load_config(),
                metadata_database_url=None,
                runtime_dir=Path(tmp) / "runtime",
                object_store_dir=Path(tmp) / "object_store",
            )
            occupied = acquire_query_admission_lease(
                config=config,
                tenant_id=auth_context.tenant_id,
                user_key=user_key,
                global_limit=8,
                tenant_limit=8,
                user_limit=1,
                lease_ms=60_000,
            )
            try:
                with (
                    patch("serve.resolve_auth_context", return_value=auth_context),
                    patch("serve.load_config", return_value=config),
                ):
                    api = TestClient(serve.create_app())
                    response = api.post(
                        "/query/stream",
                        json={
                            "query": "smoke shared user admission",
                            "tenant_id": auth_context.tenant_id,
                            "acl_groups": ["engineering"],
                            "request_id": "smoke-shared-user-admission",
                        },
                    )
                assert response.status_code == 503, response.text
                assert "shared user capacity is busy" in response.text
                assert user_key not in response.text
            finally:
                assert release_query_admission_lease(config=config, lease=occupied)
    finally:
        for name, value in old_values.items():
            restore_env(name, value)


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
