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
    test_conversation_save_rejects_oversized_request_before_auth_or_write()
    test_conversation_save_rejects_oversized_message_image_before_auth_or_write()
    test_conversation_save_rejects_too_many_images_before_auth_or_write()
    test_conversation_save_rejects_excess_total_image_bytes_before_auth_or_write()
    test_conversation_save_rejects_invalid_message_image_url()
    print("smoke_conversation_request_limits=ok")


def conversation_payload(*, image_data_url: str | None = None, image_data_urls: list[str] | None = None) -> dict[str, object]:
    message: dict[str, object] = {
        "id": "msg-1",
        "role": "user",
        "content": "hello",
        "created_at": 1,
    }
    if image_data_urls is not None:
        messages = [
            {
                "id": f"msg-{index + 1}",
                "role": "user",
                "content": f"hello {index + 1}",
                "created_at": index + 1,
                "image_data_url": item,
            }
            for index, item in enumerate(image_data_urls)
        ]
    else:
        message["image_data_url"] = image_data_url
        messages = [message]
    return {
        "id": "conv-limit-smoke",
        "tenant_id": "tenant-conversation-limit",
        "title": "limit smoke",
        "messages": messages,
        "source_doc_ids": [],
    }


def test_conversation_save_rejects_oversized_request_before_auth_or_write() -> None:
    body = json.dumps(conversation_payload(image_data_url="data:image/png;base64," + ("a" * 512))).encode("utf-8")
    with patched_env(RAG_MAX_CONVERSATION_REQUEST_BYTES="128"):
        api = TestClient(serve.create_app())
        with (
            patch("serve.resolve_auth_context_from_values") as resolve_auth,
            patch("serve.save_conversation") as save_conversation,
        ):
            response = api.post("/conversations", content=body, headers={"content-type": "application/json"})
            assert response.status_code == 413, response.text
            assert "RAG_MAX_CONVERSATION_REQUEST_BYTES=128" in response.text
            resolve_auth.assert_not_called()
            save_conversation.assert_not_called()


def test_conversation_save_rejects_oversized_message_image_before_auth_or_write() -> None:
    payload = conversation_payload(
        image_data_url="data:image/png;base64," + base64.b64encode(b"12345").decode("ascii")
    )
    with patched_env(RAG_MAX_QUERY_IMAGE_BYTES="4"):
        api = TestClient(serve.create_app())
        with (
            patch("serve.resolve_auth_context_from_values") as resolve_auth,
            patch("serve.save_conversation") as save_conversation,
        ):
            response = api.post("/conversations", json=payload)
            assert response.status_code == 413, response.text
            assert "RAG_MAX_QUERY_IMAGE_BYTES=4" in response.text
            resolve_auth.assert_not_called()
            save_conversation.assert_not_called()


def test_conversation_save_rejects_too_many_images_before_auth_or_write() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"1").decode("ascii")
    payload = conversation_payload(image_data_urls=[image, image])
    with patched_env(RAG_MAX_CONVERSATION_IMAGES="1"):
        api = TestClient(serve.create_app())
        with (
            patch("serve.resolve_auth_context_from_values") as resolve_auth,
            patch("serve.save_conversation") as save_conversation,
        ):
            response = api.post("/conversations", json=payload)
            assert response.status_code == 413, response.text
            assert "RAG_MAX_CONVERSATION_IMAGES=1" in response.text
            resolve_auth.assert_not_called()
            save_conversation.assert_not_called()


def test_conversation_save_rejects_excess_total_image_bytes_before_auth_or_write() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"1234").decode("ascii")
    payload = conversation_payload(image_data_urls=[image, image])
    with patched_env(RAG_MAX_CONVERSATION_IMAGE_BYTES="7"):
        api = TestClient(serve.create_app())
        with (
            patch("serve.resolve_auth_context_from_values") as resolve_auth,
            patch("serve.save_conversation") as save_conversation,
        ):
            response = api.post("/conversations", json=payload)
            assert response.status_code == 413, response.text
            assert "RAG_MAX_CONVERSATION_IMAGE_BYTES=7" in response.text
            resolve_auth.assert_not_called()
            save_conversation.assert_not_called()


def test_conversation_save_rejects_invalid_message_image_url() -> None:
    payload = conversation_payload(image_data_url="not-a-data-url")
    api = TestClient(serve.create_app())
    with (
        patch("serve.resolve_auth_context_from_values") as resolve_auth,
        patch("serve.save_conversation") as save_conversation,
    ):
        response = api.post("/conversations", json=payload)
        assert response.status_code == 400, response.text
        assert "message.image_data_url must be a data:image URL" in response.text
        resolve_auth.assert_not_called()
        save_conversation.assert_not_called()


if __name__ == "__main__":
    main()
