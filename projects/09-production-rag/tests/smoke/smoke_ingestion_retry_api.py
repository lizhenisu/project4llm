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

import serve  # noqa: E402
from rag_core.config import load_config  # noqa: E402
from rag_core.database import connect_metadata_db  # noqa: E402
from rag_core.sources import SourceSummary, save_source_task_for_tenant  # noqa: E402
from rag_core.text_utils import now_ms  # noqa: E402
from rag_core.upload_admission import upload_admission_metrics_snapshot  # noqa: E402


def main() -> None:
    old_env = {name: os.environ.get(name) for name in (
        "RAG_RUNTIME_DIR",
        "RAG_OBJECT_STORE_DIR",
        "RAG_METADATA_DATABASE_URL",
        "RAG_API_TOKEN",
    )}
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        os.environ["RAG_API_TOKEN"] = "synthetic-retry-token"
        try:
            test_retry_failed_task_is_tenant_scoped_and_atomic()
        finally:
            for name, value in old_env.items():
                restore_env(name, value)
    print("smoke_ingestion_retry_api=ok")


def test_retry_failed_task_is_tenant_scoped_and_atomic() -> None:
    config = load_config()
    tenant_a = "synthetic-retry-a"
    tenant_b = "synthetic-retry-b"
    failed = save_failed_task(config, tenant_a, "failed-a")
    foreign = save_failed_task(config, tenant_b, "failed-b")
    headers = {
        "Authorization": "Bearer synthetic-retry-token",
        "X-RAG-Tenant-ID": tenant_a,
        "X-RAG-ACL-Groups": "engineering",
    }
    api = TestClient(serve.create_app())
    listed = api.get(f"/sources?tenant_id={tenant_a}", headers=headers)
    assert listed.status_code == 200, listed.text
    listed_failed = next(
        source
        for source in listed.json()["sources"]
        if source["doc_id"] == failed.doc_id
    )
    assert listed_failed["attempt_count"] == 3
    assert listed_failed["next_attempt_at"] == 0
    assert listed_failed["dead_lettered"] is True
    assert listed_failed["retryable"] is True
    assert listed_failed["ingestion_stage"] == "failed"
    assert listed_failed["progress_percent"] == 0
    assert listed_failed["progress_detail"] == ""

    with patch("serve.submit_upload_ingestion_job", return_value=True) as submit_job:
        response = api.post(
            f"/sources/{failed.doc_id}/retry?tenant_id={tenant_a}",
            headers=headers,
        )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    assert response.json()["source"]["status"] == "queued"
    assert response.json()["source"]["retryable"] is False
    assert response.json()["source"]["ingestion_stage"] == "queued"
    assert response.json()["source"]["progress_percent"] == 0
    assert response.json()["source"]["progress_detail"] == ""
    submit_job.assert_called_once()
    assert submit_job.call_args.kwargs["tenant_id"] == tenant_a
    assert submit_job.call_args.kwargs["doc_version"] is None

    row = task_row(config, tenant_a, failed.doc_id)
    assert row["status"] == "queued"
    assert row["error"] == ""
    assert int(row["attempt_count"]) == 0
    assert int(row["next_attempt_at"]) == 0
    assert int(row["dead_lettered_at"]) == 0

    repeated = api.post(
        f"/sources/{failed.doc_id}/retry?tenant_id={tenant_a}",
        headers=headers,
    )
    assert repeated.status_code == 409
    cross_tenant = api.post(
        f"/sources/{foreign.doc_id}/retry?tenant_id={tenant_a}",
        headers=headers,
    )
    assert cross_tenant.status_code == 404
    assert foreign.doc_id not in cross_tenant.text
    assert upload_admission_metrics_snapshot(config=config) == {
        "global_reservations": 0,
        "tenant_reservations": 0,
        "expired_reservations": 0,
    }


def save_failed_task(config, tenant_id: str, label: str) -> SourceSummary:
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
            (timestamp, tenant_id, source.doc_id),
        )
    return source


def task_row(config, tenant_id: str, task_id: str):
    with connect_metadata_db(config) as conn:
        row = conn.execute(
            """
            SELECT status, error, attempt_count, next_attempt_at, dead_lettered_at,
                   ingestion_stage, progress_percent
            FROM source_tasks
            WHERE tenant_id = ? AND id = ?
            """,
            (tenant_id, task_id),
        ).fetchone()
    assert row is not None
    return row


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
