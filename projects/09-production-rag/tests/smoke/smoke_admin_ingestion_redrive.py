from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import serve  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.sources import SourceSummary, save_source_task_for_tenant  # noqa: E402
from rag_core.text_utils import now_ms  # noqa: E402
from rag_core.upload_admission import upload_admission_metrics_snapshot  # noqa: E402


def main() -> None:
    old_env = {
        name: os.environ.get(name)
        for name in (
            "RAG_RUNTIME_DIR",
            "RAG_OBJECT_STORE_DIR",
            "RAG_METADATA_DATABASE_URL",
            "RAG_INGEST_BACKLOG_LIMIT",
            "RAG_INGEST_TENANT_BACKLOG_LIMIT",
            "RAG_INGEST_EXECUTION_MODE",
        )
    }
    with tempfile.TemporaryDirectory(prefix="rag-admin-redrive-") as tmp:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_METADATA_DATABASE_URL"] = (
            os.environ.get("SMOKE_METADATA_DATABASE_URL") or ""
        )
        os.environ["RAG_INGEST_BACKLOG_LIMIT"] = "100"
        os.environ["RAG_INGEST_TENANT_BACKLOG_LIMIT"] = "100"
        os.environ["RAG_INGEST_EXECUTION_MODE"] = "external"
        try:
            if os.environ["RAG_METADATA_DATABASE_URL"]:
                with patch(
                    "serve.require_admin",
                    return_value=SimpleNamespace(id="postgres-redrive-admin"),
                ):
                    test_admin_lists_and_redrives_dead_letters(mocked_admin=True)
            else:
                test_admin_lists_and_redrives_dead_letters(mocked_admin=False)
        finally:
            for name, value in old_env.items():
                restore_env(name, value)
    print("smoke_admin_ingestion_redrive=ok")


def test_admin_lists_and_redrives_dead_letters(*, mocked_admin: bool) -> None:
    api = TestClient(serve.create_app())
    if mocked_admin:
        cleanup(load_config(), delete_users=False)
        admin_id = "postgres-redrive-admin"
        admin_headers = {"Authorization": "Bearer synthetic-postgres-admin"}
    else:
        admin = api.post(
            "/auth/register",
            json={
                "username": "redrive_admin",
                "password": "strong-password",
                "display_name": "Redrive Admin",
            },
        )
        assert admin.status_code == 200, admin.text
        assert admin.json()["user"]["role"] == "admin"
        admin_id = admin.json()["user"]["id"]
        admin_headers = {"Authorization": f"Bearer {admin.json()['token']}"}

        user = api.post(
            "/auth/register",
            json={
                "username": "redrive_user",
                "password": "strong-password",
                "display_name": "Redrive User",
            },
        )
        assert user.status_code == 200, user.text
        user_headers = {"Authorization": f"Bearer {user.json()['token']}"}

    config = load_config()
    first = save_failed_task(config, "redrive-tenant-a", "dead-a", dead_lettered=True)
    second = save_failed_task(config, "redrive-tenant-b", "dead-b", dead_lettered=True)
    ordinary = save_failed_task(
        config,
        "redrive-tenant-a",
        "ordinary-failed",
        dead_lettered=False,
    )

    if not mocked_admin:
        assert api.get("/admin/ingestion/dead-letters").status_code == 401
        assert (
            api.get("/admin/ingestion/dead-letters", headers=user_headers).status_code
            == 403
        )
    listed = api.get(
        "/admin/ingestion/dead-letters?limit=1&offset=0",
        headers=admin_headers,
    )
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 2
    assert len(listed.json()["tasks"]) == 1
    assert listed.json()["tasks"][0]["dead_lettered_at"] > 0

    payload = {
        "tasks": [
            {"tenant_id": "redrive-tenant-a", "task_id": first.doc_id},
            {"tenant_id": "redrive-tenant-b", "task_id": second.doc_id},
            {"tenant_id": "redrive-tenant-a", "task_id": ordinary.doc_id},
            {"tenant_id": "redrive-tenant-a", "task_id": "missing-task"},
        ]
    }
    with patch("serve.submit_upload_ingestion_job", return_value=True) as submit_job:
        response = api.post(
            "/admin/ingestion/dead-letters/redrive",
            headers=admin_headers,
            json=payload,
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["queued"] == 2
    assert body["rejected"] == 2
    assert [result["outcome"] for result in body["results"]] == [
        "queued",
        "queued",
        "not_retryable",
        "not_found",
    ]
    assert submit_job.call_count == 2
    assert task_status(config, "redrive-tenant-a", first.doc_id) == "queued"
    assert task_status(config, "redrive-tenant-b", second.doc_id) == "queued"
    assert task_status(config, "redrive-tenant-a", ordinary.doc_id) == "failed"

    audit = api.get("/admin/ingestion/audit?limit=10", headers=admin_headers)
    assert audit.status_code == 200, audit.text
    events = audit.json()["events"]
    assert audit.json()["total"] == 4
    assert {event["outcome"] for event in events} == {
        "queued",
        "not_retryable",
        "not_found",
    }
    assert all(event["operation"] == "bulk_redrive" for event in events)
    assert all(event["actor_user_id"] == admin_id for event in events)
    runtime_metrics = api.get("/runtime-metrics")
    assert runtime_metrics.status_code == 200, runtime_metrics.text
    operator_metrics = runtime_metrics.json()["ingestion"]["operator_operations"]
    assert operator_metrics["audit_events"] == 4
    assert operator_metrics["bulk_redrive_outcomes"]["queued"] == 2
    assert operator_metrics["bulk_redrive_outcomes"]["not_retryable"] == 1
    assert operator_metrics["bulk_redrive_outcomes"]["not_found"] == 1
    assert "redrive-tenant-a" not in runtime_metrics.text
    assert first.doc_id not in runtime_metrics.text
    prometheus = api.get("/metrics")
    assert prometheus.status_code == 200, prometheus.text
    assert (
        'rag_ingestion_operator_audit_events{operation="bulk_redrive",outcome="queued"} 2'
        in prometheus.text
    )
    assert "redrive-tenant-a" not in prometheus.text
    assert first.doc_id not in prometheus.text
    assert upload_admission_metrics_snapshot(config=config) == {
        "global_reservations": 0,
        "tenant_reservations": 0,
        "expired_reservations": 0,
    }

    duplicate = api.post(
        "/admin/ingestion/dead-letters/redrive",
        headers=admin_headers,
        json={"tasks": [payload["tasks"][0], payload["tasks"][0]]},
    )
    assert duplicate.status_code == 422
    cleanup(config, delete_users=not mocked_admin)


def save_failed_task(
    config,
    tenant_id: str,
    label: str,
    *,
    dead_lettered: bool,
) -> SourceSummary:
    timestamp = now_ms()
    source = SourceSummary(
        doc_id=f"{label}-{timestamp}",
        title=f"{label}.txt",
        source_type="txt",
        source_uri=str(config.object_store_dir / f"{label}.txt"),
        doc_version=1,
        chunk_count=0,
        acl_groups=["engineering"],
        status="failed",
        current=False,
        created_at=timestamp,
        updated_at=timestamp,
        child_doc_ids=[],
        error="synthetic terminal failure",
    )
    save_source_task_for_tenant(
        config=config,
        tenant_id=tenant_id,
        source=source,
        error=source.error,
        requested_doc_version=None,
    )
    with connect_metadata_db(config) as conn:
        conn.execute(
            """
            UPDATE source_tasks
            SET attempt_count = 3, dead_lettered_at = ?
            WHERE tenant_id = ? AND id = ?
            """,
            (timestamp if dead_lettered else 0, tenant_id, source.doc_id),
        )
    return source


def task_status(config, tenant_id: str, task_id: str) -> str:
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            "SELECT status FROM source_tasks WHERE tenant_id = ? AND id = ?",
            (tenant_id, task_id),
        ).fetchone()
    assert row is not None
    return str(row["status"])


def cleanup(config, *, delete_users: bool) -> None:
    with connect_metadata_db(config) as conn:
        conn.execute(
            "DELETE FROM source_tasks WHERE tenant_id LIKE ?",
            ("redrive-tenant-%",),
        )
        conn.execute(
            "DELETE FROM ingestion_operation_audit WHERE tenant_id LIKE ?",
            ("redrive-tenant-%",),
        )
        if delete_users:
            conn.execute(
                "DELETE FROM sessions WHERE user_id IN "
                "(SELECT id FROM users WHERE username IN (?, ?))",
                ("redrive_admin", "redrive_user"),
            )
            conn.execute(
                "DELETE FROM users WHERE username IN (?, ?)",
                ("redrive_admin", "redrive_user"),
            )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
