from __future__ import annotations

import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve  # noqa: E402


API_TOKEN = "synthetic-conversation-persistence-token"
TENANT_COUNT = 12
RUN_ID = uuid.uuid4().hex[:12]


def main() -> None:
    with isolated_runtime():
        api = TestClient(serve.create_app())
        with ThreadPoolExecutor(max_workers=TENANT_COUNT) as executor:
            saved = list(executor.map(lambda index: save_tenant_conversation(api, index), range(TENANT_COUNT)))
        assert all(response.status_code == 200 for response in saved), [response.text for response in saved]

        with ThreadPoolExecutor(max_workers=TENANT_COUNT) as executor:
            loaded = list(executor.map(lambda index: load_tenant_conversation(api, index), range(TENANT_COUNT)))
        for index, response in enumerate(loaded):
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["tenant_id"] == tenant_id(index)
            assert body["messages"][-1]["content"] == f"persisted answer {index}"

        verify_cross_tenant_id_collision_is_rejected(api)
        verify_cross_workspace_isolation(api)
        cleanup_conversations(api)
    print("smoke_conversation_multi_tenant_persistence=ok")


def save_tenant_conversation(api: TestClient, index: int):
    return api.post(
        "/conversations",
        headers=tenant_headers(index),
        json=conversation_payload(index),
    )


def load_tenant_conversation(api: TestClient, index: int):
    return api.get(
        f"/conversations/{conversation_id(index)}?tenant_id={tenant_id(index)}"
        f"&workspace_id={workspace_id(index)}",
        headers=tenant_headers(index),
    )


def verify_cross_tenant_id_collision_is_rejected(api: TestClient) -> None:
    shared_id = shared_conversation_id()
    owner = api.post(
        "/conversations",
        headers=tenant_headers(100),
        json=conversation_payload(100, override_id=shared_id),
    )
    assert owner.status_code == 200, owner.text

    conflict = api.post(
        "/conversations",
        headers=tenant_headers(101),
        json=conversation_payload(101, override_id=shared_id),
    )
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"] == "Conversation ID is unavailable"
    assert tenant_id(100) not in conflict.text

    owner_reload = api.get(
        f"/conversations/{shared_id}?tenant_id={tenant_id(100)}&workspace_id={workspace_id(100)}",
        headers=tenant_headers(100),
    )
    assert owner_reload.status_code == 200, owner_reload.text
    assert owner_reload.json()["messages"][-1]["content"] == "persisted answer 100"

    attacker_reload = api.get(
        f"/conversations/{shared_id}?tenant_id={tenant_id(101)}&workspace_id={workspace_id(101)}",
        headers=tenant_headers(101),
    )
    assert attacker_reload.status_code == 404, attacker_reload.text


def cleanup_conversations(api: TestClient) -> None:
    for index in range(TENANT_COUNT):
        response = api.delete(
            f"/conversations/{conversation_id(index)}?tenant_id={tenant_id(index)}"
            f"&workspace_id={workspace_id(index)}",
            headers=tenant_headers(index),
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "deleted"
    owner_cleanup = api.delete(
        f"/conversations/{shared_conversation_id()}?tenant_id={tenant_id(100)}"
        f"&workspace_id={workspace_id(100)}",
        headers=tenant_headers(100),
    )
    assert owner_cleanup.status_code == 200, owner_cleanup.text
    assert owner_cleanup.json()["status"] == "deleted"


def verify_cross_workspace_isolation(api: TestClient) -> None:
    index = 200
    headers = tenant_headers(index)
    first = api.post(
        "/conversations",
        headers=headers,
        json=conversation_payload(
            index,
            override_id=workspace_conversation_id("a"),
            override_workspace="workspace-a",
        ),
    )
    second = api.post(
        "/conversations",
        headers=headers,
        json=conversation_payload(
            index,
            override_id=workspace_conversation_id("b"),
            override_workspace="workspace-b",
        ),
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    workspace_a = api.get(
        f"/conversations?tenant_id={tenant_id(index)}&workspace_id=workspace-a",
        headers=headers,
    )
    assert workspace_a.status_code == 200, workspace_a.text
    assert [row["id"] for row in workspace_a.json()["conversations"]] == [workspace_conversation_id("a")]

    wrong_workspace = (
        f"/conversations/{workspace_conversation_id('a')}?tenant_id={tenant_id(index)}"
        "&workspace_id=workspace-b"
    )
    assert api.get(wrong_workspace, headers=headers).status_code == 404
    assert api.patch(wrong_workspace, headers=headers, json={"title": "forbidden"}).status_code == 404
    assert api.delete(wrong_workspace, headers=headers).json()["status"] == "not_found"

    collision = api.post(
        "/conversations",
        headers=headers,
        json=conversation_payload(
            index,
            override_id=workspace_conversation_id("a"),
            override_workspace="workspace-b",
        ),
    )
    assert collision.status_code == 409, collision.text

    for suffix, workspace in (("a", "workspace-a"), ("b", "workspace-b")):
        cleanup = api.delete(
            f"/conversations/{workspace_conversation_id(suffix)}?tenant_id={tenant_id(index)}"
            f"&workspace_id={workspace}",
            headers=headers,
        )
        assert cleanup.json()["status"] == "deleted"


def conversation_payload(
    index: int,
    *,
    override_id: str | None = None,
    override_workspace: str | None = None,
) -> dict[str, object]:
    resolved_conversation_id = override_id or conversation_id(index)
    return {
        "id": resolved_conversation_id,
        "tenant_id": tenant_id(index),
        "workspace_id": override_workspace or workspace_id(index),
        "title": f"Synthetic persistence {index}",
        "messages": [
            {
                "id": f"{resolved_conversation_id}-message-user",
                "role": "user",
                "content": f"question {index}",
                "status": "done",
            },
            {
                "id": f"{resolved_conversation_id}-message-assistant",
                "role": "assistant",
                "content": f"persisted answer {index}",
                "status": "done",
            },
        ],
        "source_doc_ids": [],
    }


def tenant_headers(index: int) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "X-RAG-Tenant-ID": tenant_id(index),
        "X-RAG-ACL-Groups": "engineering",
    }


def tenant_id(index: int) -> str:
    return f"synthetic-persistence-{RUN_ID}-tenant-{index}"


def conversation_id(index: int) -> str:
    return f"synthetic-persistence-{RUN_ID}-conversation-{index}"


def workspace_id(index: int) -> str:
    return f"synthetic-persistence-{RUN_ID}-workspace-{index}"


def workspace_conversation_id(suffix: str) -> str:
    return f"synthetic-persistence-{RUN_ID}-workspace-conversation-{suffix}"


def shared_conversation_id() -> str:
    return f"synthetic-persistence-{RUN_ID}-shared-conversation"


@contextmanager
def isolated_runtime():
    names = ("RAG_RUNTIME_DIR", "RAG_METADATA_DATABASE_URL", "RAG_API_TOKEN")
    previous = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        os.environ["RAG_METADATA_DATABASE_URL"] = os.environ.get("SMOKE_METADATA_DATABASE_URL", "")
        os.environ["RAG_API_TOKEN"] = API_TOKEN
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    main()
