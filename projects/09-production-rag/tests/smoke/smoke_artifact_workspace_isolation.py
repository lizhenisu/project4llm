from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_core.artifacts import (
    list_metadata_artifacts,
    load_metadata_artifact,
    MindMapArtifact,
    save_metadata_artifact,
)
from rag_core.config import load_config
from rag_core.text_utils import now_ms
from serve import create_app


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(temp_dir) / "object_store")
        try:
            seed_legacy_artifacts_table(Path(temp_dir) / "runtime" / "db" / "metadata.db")
            run_smoke()
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)


def seed_legacy_artifacts_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE artifacts (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_type TEXT NOT NULL DEFAULT 'mindmap',
                source_doc_ids TEXT NOT NULL DEFAULT '[]',
                root TEXT,
                table_json TEXT,
                error TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX idx_artifacts_tenant_updated ON artifacts(tenant_id, updated_at DESC);
            """
        )


def run_smoke() -> None:
    api = TestClient(create_app())
    registered = api.post(
        "/auth/register",
        json={"username": "admin_user", "password": "strong-password"},
    )
    assert registered.status_code == 200, registered.text
    body = registered.json()
    headers = {"Authorization": f"Bearer {body['token']}"}
    tenant_id = body["user"]["tenant_id"]
    second_registered = api.post(
        "/auth/register",
        json={"username": "artifact_other_user", "password": "strong-password"},
    )
    assert second_registered.status_code == 200, second_registered.text
    second_body = second_registered.json()
    second_headers = {"Authorization": f"Bearer {second_body['token']}"}
    second_tenant_id = second_body["user"]["tenant_id"]

    config = load_config()
    artifact_a = make_artifact(tenant_id=tenant_id, workspace_id="workspace-a", artifact_id="table-a")
    artifact_b = make_artifact(tenant_id=tenant_id, workspace_id="workspace-b", artifact_id="table-b")
    legacy_artifact = make_artifact(tenant_id=tenant_id, workspace_id="", artifact_id="legacy-table")
    save_metadata_artifact(config, artifact_a)
    save_metadata_artifact(config, artifact_b)
    save_metadata_artifact(config, legacy_artifact)

    conflicting_artifact = make_artifact(
        tenant_id=second_tenant_id,
        workspace_id="workspace-a",
        artifact_id="table-a",
    )
    with patch("serve.pending_artifact", return_value=conflicting_artifact):
        collision_response = api.post(
            "/artifacts/table",
            headers=second_headers,
            json={
                "title": "collision",
                "tenant_id": second_tenant_id,
                "workspace_id": "workspace-a",
                "source_doc_ids": [],
            },
        )
    assert collision_response.status_code == 409, collision_response.text
    assert "Artifact ID is unavailable" in collision_response.text
    assert tenant_id not in collision_response.text

    cross_tenant_get = api.get(
        f"/artifacts/table-a?tenant_id={tenant_id}&workspace_id=workspace-a",
        headers=second_headers,
    )
    assert cross_tenant_get.status_code == 404, cross_tenant_get.text
    cross_tenant_rename = api.patch(
        f"/artifacts/table-a?tenant_id={tenant_id}&workspace_id=workspace-a",
        headers=second_headers,
        json={"title": "cross tenant overwrite"},
    )
    assert cross_tenant_rename.status_code == 404, cross_tenant_rename.text
    cross_tenant_delete = api.delete(
        f"/artifacts/table-a?tenant_id={tenant_id}&workspace_id=workspace-a",
        headers=second_headers,
    )
    assert cross_tenant_delete.status_code == 200, cross_tenant_delete.text
    assert cross_tenant_delete.json()["status"] == "not_found"
    assert load_metadata_artifact(
        config,
        tenant_id=tenant_id,
        workspace_id="workspace-a",
        artifact_id="table-a",
    ) is not None
    assert list_metadata_artifacts(config, tenant_id=second_tenant_id) == []

    listed_a = api.get(f"/artifacts?tenant_id={tenant_id}&workspace_id=workspace-a", headers=headers)
    assert listed_a.status_code == 200, listed_a.text
    listed_a_ids = [artifact["id"] for artifact in listed_a.json()["artifacts"]]
    assert listed_a_ids == ["table-a"]

    listed_b = api.get(f"/artifacts?tenant_id={tenant_id}&workspace_id=workspace-b", headers=headers)
    assert listed_b.status_code == 200, listed_b.text
    assert [artifact["id"] for artifact in listed_b.json()["artifacts"]] == ["table-b"]

    hidden_get = api.get(f"/artifacts/table-b?tenant_id={tenant_id}&workspace_id=workspace-a", headers=headers)
    assert hidden_get.status_code == 404, hidden_get.text
    get_b = api.get(f"/artifacts/table-b?tenant_id={tenant_id}&workspace_id=workspace-b", headers=headers)
    assert get_b.status_code == 200, get_b.text
    assert get_b.json()["workspace_id"] == "workspace-b"

    wrong_rename = api.patch(
        f"/artifacts/table-b?tenant_id={tenant_id}&workspace_id=workspace-a",
        headers=headers,
        json={"title": "Wrong scope rename"},
    )
    assert wrong_rename.status_code == 404, wrong_rename.text

    wrong_delete = api.delete(f"/artifacts/table-b?tenant_id={tenant_id}&workspace_id=workspace-a", headers=headers)
    assert wrong_delete.status_code == 200, wrong_delete.text
    assert wrong_delete.json()["status"] == "not_found"
    assert load_metadata_artifact(
        config,
        tenant_id=tenant_id,
        workspace_id="workspace-b",
        artifact_id="table-b",
    ) is not None

    delete_b = api.delete(f"/artifacts/table-b?tenant_id={tenant_id}&workspace_id=workspace-b", headers=headers)
    assert delete_b.status_code == 200, delete_b.text
    assert delete_b.json()["status"] == "deleted"
    assert load_metadata_artifact(
        config,
        tenant_id=tenant_id,
        workspace_id="workspace-b",
        artifact_id="table-b",
    ) is None

    print("artifact workspace isolation smoke passed")


def make_artifact(*, tenant_id: str, workspace_id: str, artifact_id: str) -> MindMapArtifact:
    timestamp = now_ms()
    return MindMapArtifact(
        id=artifact_id,
        title=artifact_id,
        status="ready",
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_doc_ids=["same-source-doc"],
        created_at=timestamp,
        updated_at=timestamp,
        artifact_type="table",
        table={"title": artifact_id, "columns": ["主题"], "rows": [[artifact_id]], "summary": ""},
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
