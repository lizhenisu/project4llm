from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import serve


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_ingest = serve.ingest_uploaded_path
    calls: list[dict] = []

    def fake_ingest_uploaded_path(**kwargs):
        calls.append(kwargs)

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(temp_dir) / "object_store")
        serve.ingest_uploaded_path = fake_ingest_uploaded_path
        try:
            api = TestClient(serve.create_app())
            response = api.post(
                "/sources/upload",
                data={"tenant_id": "team_a", "acl_groups": "engineering"},
                files={"file": ("async.md", b"# Async upload\n\ncontent", "text/markdown")},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "processing"
            assert body["sources"][0]["status"] == "processing"
            assert calls and calls[0]["tenant_id"] == "team_a"
        finally:
            serve.ingest_uploaded_path = old_ingest
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)

    print("async upload smoke passed")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
