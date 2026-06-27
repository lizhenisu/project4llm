from __future__ import annotations

import base64
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve
from rag_core.auth import AuthContext


@contextmanager
def patched_env(**values: str):
    old_values = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def main() -> None:
    test_query_routes_reject_oversized_request_bodies_before_parsing_work()
    test_query_routes_reject_oversized_images_before_rag_work()
    test_query_routes_reject_invalid_image_data_url()
    test_query_image_metrics_record_accepted_rejected_and_invalid_payloads()
    print("smoke_query_image_limits=ok")


def test_query_routes_reject_oversized_request_bodies_before_parsing_work() -> None:
    payload = {
        "query": "describe this image",
        "query_mode": "multimodal",
        "tenant_id": "tenant-query-image-limit",
        "image_data_url": "data:image/png;base64," + ("a" * 512),
    }
    body = json.dumps(payload).encode("utf-8")
    with patched_env(RAG_MAX_QUERY_REQUEST_BYTES="128"):
        api = TestClient(serve.create_app())
        with (
            patch("serve.resolve_auth_context") as resolve_auth,
            patch("serve.resolve_answer_result") as resolve_answer,
            patch("serve.resolve_search_result") as resolve_search,
        ):
            for route in ("/query", "/query/stream", "/search"):
                response = api.post(route, content=body, headers={"content-type": "application/json"})
                assert response.status_code == 413, response.text
                assert "RAG_MAX_QUERY_REQUEST_BYTES=128" in response.text

            resolve_auth.assert_not_called()
            resolve_answer.assert_not_called()
            resolve_search.assert_not_called()


def test_query_routes_reject_oversized_images_before_rag_work() -> None:
    payload = {
        "query": "describe this image",
        "query_mode": "multimodal",
        "tenant_id": "tenant-query-image-limit",
        "image_data_url": "data:image/png;base64," + base64.b64encode(b"12345").decode("ascii"),
    }
    with patched_env(RAG_MAX_QUERY_IMAGE_BYTES="4"):
        api = TestClient(serve.create_app())
        with (
            patch("serve.resolve_auth_context") as resolve_auth,
            patch("serve.resolve_answer_result") as resolve_answer,
            patch("serve.resolve_search_result") as resolve_search,
        ):
            response = api.post("/query", json=payload)
            assert response.status_code == 413, response.text
            assert "RAG_MAX_QUERY_IMAGE_BYTES=4" in response.text

            stream_response = api.post("/query/stream", json=payload)
            assert stream_response.status_code == 413, stream_response.text
            assert "RAG_MAX_QUERY_IMAGE_BYTES=4" in stream_response.text

            search_response = api.post("/search", json=payload)
            assert search_response.status_code == 413, search_response.text
            assert "RAG_MAX_QUERY_IMAGE_BYTES=4" in search_response.text

            resolve_auth.assert_not_called()
            resolve_answer.assert_not_called()
            resolve_search.assert_not_called()


def test_query_routes_reject_invalid_image_data_url() -> None:
    payload = {
        "query": "describe this image",
        "query_mode": "multimodal",
        "tenant_id": "tenant-query-image-limit",
        "image_data_url": "not-a-data-url",
    }
    api = TestClient(serve.create_app())
    with patch("serve.resolve_auth_context", return_value=AuthContext("tenant-query-image-limit", ["engineering"], "smoke")):
        response = api.post("/query/stream", json=payload)
    assert response.status_code == 400, response.text
    assert "image_data_url must be a data:image URL" in response.text


def test_query_image_metrics_record_accepted_rejected_and_invalid_payloads() -> None:
    before = serve.query_image_metrics_snapshot()
    config = type("Config", (), {"max_query_image_bytes": 4})()

    accepted = serve.QueryRequest(
        query="accepted",
        image_data_url="data:image/png;base64," + base64.b64encode(b"1234").decode("ascii"),
    )
    serve.validate_query_image_data_url(accepted, config)

    rejected = serve.QueryRequest(
        query="rejected",
        image_data_url="data:image/png;base64," + base64.b64encode(b"12345").decode("ascii"),
    )
    try:
        serve.validate_query_image_data_url(rejected, config)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 413
    else:
        raise AssertionError("oversized query image was not rejected")

    invalid = serve.QueryRequest(query="invalid", image_data_url="not-a-data-url")
    try:
        serve.validate_query_image_data_url(invalid, config)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
    else:
        raise AssertionError("invalid query image was not rejected")

    after = serve.query_image_metrics_snapshot()
    assert after["accepted_total"] == before["accepted_total"] + 1
    assert after["rejected_oversized_total"] == before["rejected_oversized_total"] + 1
    assert after["invalid_total"] == before["invalid_total"] + 1
    assert after["accepted_estimated_bytes_max"] >= 4
    assert after["rejected_estimated_bytes_max"] >= 5
    assert after["accepted_size_buckets"]["le_65536"] >= 1
    assert after["rejected_size_buckets"]["le_65536"] >= 1


if __name__ == "__main__":
    main()
