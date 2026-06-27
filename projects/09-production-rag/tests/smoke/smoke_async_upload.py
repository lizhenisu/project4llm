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
from rag_core.sources import SourceSummary


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_token = os.environ.get("RAG_API_TOKEN")

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(temp_dir) / "object_store")
        os.environ["RAG_API_TOKEN"] = "smoke-token"
        pending_source = SourceSummary(
            doc_id="async-upload",
            title="async.md",
            source_type="md",
            source_uri=str(Path(temp_dir) / "object_store" / "async.md"),
            doc_version=1,
            chunk_count=0,
            acl_groups=["engineering"],
            status="queued",
            current=False,
            created_at=1,
            updated_at=1,
        )
        try:
            api = TestClient(serve.create_app())
            with (
                patch("serve.count_active_source_tasks", return_value=0),
                patch("serve.save_uploaded_file", return_value=Path(pending_source.source_uri)),
                patch("serve.create_source_task", return_value=pending_source),
                patch("serve.submit_upload_ingestion_job") as submit_job,
            ):
                response = api.post(
                    "/sources/upload",
                    headers={
                        "Authorization": "Bearer smoke-token",
                        "X-RAG-Tenant-ID": "team_a",
                        "X-RAG-ACL-Groups": "engineering",
                    },
                    data={"tenant_id": "team_a", "acl_groups": "engineering"},
                    files={"file": ("async.md", b"# Async upload\n\ncontent", "text/markdown")},
                )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "queued"
            assert body["sources"][0]["status"] == "queued"
            submit_job.assert_called_once()
            assert submit_job.call_args.kwargs["tenant_id"] == "team_a"
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("RAG_API_TOKEN", old_token)

    print("smoke_async_upload=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
