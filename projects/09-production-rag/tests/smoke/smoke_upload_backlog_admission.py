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
from rag_core.auth import AuthContext
from rag_core.config import load_config
from rag_core.sources import delete_source_task, discard_uploaded_file, list_source_tasks
from rag_core.upload_admission import upload_admission_metrics_snapshot
from rag_core.upload_admission import UploadAdmissionRejected, UploadAdmissionReservation


def main() -> None:
    test_upload_returns_503_before_saving_when_global_backlog_is_full()
    test_upload_returns_503_before_saving_when_tenant_backlog_is_full()
    test_upload_returns_413_before_auth_when_request_body_is_too_large()
    test_upload_returns_413_and_cleans_partial_file_when_file_is_too_large()
    test_task_creation_failure_releases_reservation_and_discards_file()
    test_successful_upload_atomically_converts_reservation_to_task()
    print("smoke_upload_backlog_admission=ok")


def test_upload_returns_503_before_saving_when_global_backlog_is_full() -> None:
    with patched_env(
        RAG_INGEST_BACKLOG_LIMIT="1",
        RAG_INGEST_TENANT_BACKLOG_LIMIT="10",
    ):
        with (
            patch(
                "serve.resolve_auth_context_from_values",
                return_value=AuthContext("tenant-upload-smoke", ["engineering"], "smoke"),
            ),
            patch(
                "serve.acquire_upload_admission_reservation",
                side_effect=UploadAdmissionRejected("global", active_tasks=1, limit=1),
            ) as acquire_reservation,
            patch("serve.save_uploaded_file") as save_uploaded,
            patch("serve.create_source_task") as create_task,
            patch("serve.submit_upload_ingestion_job") as submit_job,
        ):
            response = upload_smoke_file()
    assert response.status_code == 503, response.text
    assert "Ingestion global backlog is full" in response.text
    assert "RAG_INGEST_BACKLOG_LIMIT=1" in response.text
    acquire_reservation.assert_called_once()
    save_uploaded.assert_not_called()
    create_task.assert_not_called()
    submit_job.assert_not_called()


def test_upload_returns_503_before_saving_when_tenant_backlog_is_full() -> None:
    with patched_env(
        RAG_INGEST_BACKLOG_LIMIT="10",
        RAG_INGEST_TENANT_BACKLOG_LIMIT="1",
    ):
        with (
            patch(
                "serve.resolve_auth_context_from_values",
                return_value=AuthContext("tenant-upload-smoke", ["engineering"], "smoke"),
            ),
            patch(
                "serve.acquire_upload_admission_reservation",
                side_effect=UploadAdmissionRejected("tenant", active_tasks=1, limit=1),
            ) as acquire_reservation,
            patch("serve.save_uploaded_file") as save_uploaded,
            patch("serve.create_source_task") as create_task,
            patch("serve.submit_upload_ingestion_job") as submit_job,
        ):
            response = upload_smoke_file()
    assert response.status_code == 503, response.text
    assert "Ingestion tenant backlog is full" in response.text
    assert "RAG_INGEST_TENANT_BACKLOG_LIMIT=1" in response.text
    acquire_reservation.assert_called_once()
    save_uploaded.assert_not_called()
    create_task.assert_not_called()
    submit_job.assert_not_called()


def test_upload_returns_413_before_auth_when_request_body_is_too_large() -> None:
    with patched_env(RAG_MAX_UPLOAD_BYTES="1"):
        with patch("serve.resolve_auth_context_from_values") as resolve_auth:
            response = upload_smoke_file(content=b"x" * (2 * 1024 * 1024))
    assert response.status_code == 413, response.text
    assert "RAG_MAX_UPLOAD_BYTES=1" in response.text
    resolve_auth.assert_not_called()


def test_upload_returns_413_and_cleans_partial_file_when_file_is_too_large() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        object_store_dir = Path(tmp) / "object_store"
        with patched_env(
            RAG_MAX_UPLOAD_BYTES="8",
            RAG_OBJECT_STORE_DIR=str(object_store_dir),
            RAG_RUNTIME_DIR=str(Path(tmp) / "runtime"),
            RAG_METADATA_DATABASE_URL="",
        ):
            with (
                patch(
                    "serve.resolve_auth_context_from_values",
                    return_value=AuthContext("tenant-upload-smoke", ["engineering"], "smoke"),
                ),
                patch("serve.create_source_task") as create_task,
                patch("serve.submit_upload_ingestion_job") as submit_job,
            ):
                response = upload_smoke_file(content=b"this file is too large")
        assert response.status_code == 413, response.text
        assert "RAG_MAX_UPLOAD_BYTES=8" in response.text
        create_task.assert_not_called()
        submit_job.assert_not_called()
        leftover_files = [path for path in object_store_dir.rglob("*") if path.is_file()]
        assert leftover_files == []


def test_task_creation_failure_releases_reservation_and_discards_file() -> None:
    reservation = UploadAdmissionReservation(
        owner="failed-task-reservation",
        tenant_id="tenant-upload-smoke",
        expires_at=9999999999999,
    )
    saved_path = Path("/tmp/synthetic-failed-upload.txt")
    with (
        patch(
            "serve.resolve_auth_context_from_values",
            return_value=AuthContext("tenant-upload-smoke", ["engineering"], "smoke"),
        ),
        patch("serve.acquire_upload_admission_reservation", return_value=reservation),
        patch("serve.save_uploaded_file", return_value=saved_path),
        patch("serve.create_source_task", side_effect=ValueError("synthetic task failure")),
        patch("serve.discard_uploaded_file") as discard_uploaded,
        patch("serve.release_upload_admission_reservation", return_value=True) as release_reservation,
        patch("serve.submit_upload_ingestion_job") as submit_job,
    ):
        response = upload_smoke_file()
    assert response.status_code == 400, response.text
    discard_uploaded.assert_called_once()
    assert discard_uploaded.call_args.kwargs["path"] == saved_path
    release_reservation.assert_called_once()
    assert release_reservation.call_args.kwargs["reservation"] == reservation
    submit_job.assert_not_called()


def test_successful_upload_atomically_converts_reservation_to_task() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with patched_env(
            RAG_OBJECT_STORE_BACKEND="local",
            RAG_OBJECT_STORE_DIR=str(Path(tmp) / "object_store"),
            RAG_RUNTIME_DIR=str(Path(tmp) / "runtime"),
            RAG_METADATA_DATABASE_URL="",
        ):
            with (
                patch(
                    "serve.resolve_auth_context_from_values",
                    return_value=AuthContext("tenant-upload-success", ["engineering"], "smoke"),
                ),
                patch("serve.submit_upload_ingestion_job", return_value=True),
            ):
                api = TestClient(serve.create_app())
                response = api.post(
                    "/sources/upload",
                    data={"tenant_id": "tenant-upload-success", "acl_groups": "engineering"},
                    files={"file": ("synthetic-success.txt", b"synthetic success", "text/plain")},
                )
            assert response.status_code == 200, response.text
            task_id = response.json()["sources"][0]["doc_id"]
            config = load_config()
            task = next(
                item
                for item in list_source_tasks(config=config, tenant_id="tenant-upload-success")
                if item.doc_id == task_id
            )
            assert task.status == "queued"
            assert upload_admission_metrics_snapshot(config=config) == {
                "global_reservations": 0,
                "tenant_reservations": 0,
                "expired_reservations": 0,
            }
            discard_uploaded_file(config=config, path=Path(task.source_uri))
            assert delete_source_task(
                config=config,
                tenant_id="tenant-upload-success",
                task_id=task_id,
            )


def upload_smoke_file(content: bytes = b"smoke upload backlog admission"):
    api = TestClient(serve.create_app())
    return api.post(
        "/sources/upload",
        data={
            "tenant_id": "tenant-upload-smoke",
            "acl_groups": "engineering",
            "language": "zh",
        },
        files={"file": ("smoke.txt", content, "text/plain")},
    )


class patched_env:
    def __init__(self, **values: str) -> None:
        self.values = values
        self.old_values: dict[str, str | None] = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.old_values[name] = os.environ.get(name)
            os.environ[name] = value
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for name, value in self.old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
