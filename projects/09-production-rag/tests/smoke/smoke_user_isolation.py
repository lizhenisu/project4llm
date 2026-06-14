from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from serve import create_app


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as temp_dir:
        runtime_dir = Path(temp_dir) / "runtime"
        os.environ["RAG_RUNTIME_DIR"] = str(runtime_dir)
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(temp_dir) / "object_store")
        try:
            run_smoke(runtime_dir)
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)


def run_smoke(runtime_dir: Path) -> None:
    api = TestClient(create_app())
    alice = register(api, "alice")
    bob = register(api, "bob")
    alice_headers = {"Authorization": f"Bearer {alice['token']}"}
    bob_headers = {"Authorization": f"Bearer {bob['token']}"}

    created = api.post(
        "/conversations",
        headers=alice_headers,
        json={
            "tenant_id": bob["user"]["tenant_id"],
            "title": "Alice 私有会话",
            "messages": [
                {"id": "m1", "role": "user", "content": "只属于 Alice", "status": "done"},
            ],
            "source_doc_ids": [],
        },
    )
    assert created.status_code == 200, created.text
    assert created.json()["tenant_id"] == alice["user"]["tenant_id"]

    alice_rows = api.get(
        f"/conversations?tenant_id={bob['user']['tenant_id']}",
        headers=alice_headers,
    )
    assert alice_rows.status_code == 200, alice_rows.text
    assert len(alice_rows.json()["conversations"]) == 1

    bob_rows = api.get(
        f"/conversations?tenant_id={alice['user']['tenant_id']}",
        headers=bob_headers,
    )
    assert bob_rows.status_code == 200, bob_rows.text
    assert bob_rows.json()["conversations"] == []

    feedback = api.post(
        "/feedback",
        headers=alice_headers,
        json={
            "request_id": "alice-feedback",
            "rating": 1,
            "tenant_id": bob["user"]["tenant_id"],
            "acl_groups": ["spoofed"],
            "selected_doc_ids": ["private-doc"],
        },
    )
    assert feedback.status_code == 200, feedback.text
    event = json.loads((runtime_dir / "feedback_events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert event["tenant_id"] == alice["user"]["tenant_id"]
    assert event["auth_context"]["tenant_id"] == alice["user"]["tenant_id"]
    assert event["auth_context"]["source"] == "headers"
    print("user isolation smoke passed")


def register(api: TestClient, username: str) -> dict:
    response = api.post(
        "/auth/register",
        json={"username": username, "password": "strong-password", "display_name": username.title()},
    )
    assert response.status_code == 200, response.text
    return response.json()


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
