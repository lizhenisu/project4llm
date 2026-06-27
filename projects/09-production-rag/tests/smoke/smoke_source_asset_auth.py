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


def main() -> None:
    old_env = {
        "RAG_RUNTIME_DIR": os.environ.get("RAG_RUNTIME_DIR"),
        "RAG_OBJECT_STORE_DIR": os.environ.get("RAG_OBJECT_STORE_DIR"),
        "RAG_METADATA_DATABASE_URL": os.environ.get("RAG_METADATA_DATABASE_URL"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        object_store = root / "object_store"
        image_path = object_store / "uploads" / "tenant-asset-smoke" / "upload-1" / "image.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(tiny_png())
        os.environ["RAG_RUNTIME_DIR"] = str(root / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(object_store)
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        try:
            user = SimpleNamespace(
                id="user-asset-smoke",
                username="asset_smoke",
                tenant_id="tenant-asset-smoke",
            )
            def fake_authenticate_token(config, *, token):
                return user if token == "session-token" else None

            with patch("serve.authenticate_token", side_effect=fake_authenticate_token):
                api = TestClient(serve.create_app())
                missing_token = api.get("/source-assets/uploads/tenant-asset-smoke/upload-1/image.png")
                assert missing_token.status_code == 401, missing_token.text

                wrong_tenant = api.get(
                    "/source-assets/uploads/other-tenant/upload-1/image.png"
                    "?tenant_id=other-tenant&token=session-token"
                )
                assert wrong_tenant.status_code == 404, wrong_tenant.text

                ok = api.get(
                    "/source-assets/uploads/tenant-asset-smoke/upload-1/image.png"
                    "?tenant_id=tenant-asset-smoke&token=session-token"
                )
                assert ok.status_code == 200, ok.text
                assert ok.headers["content-type"].startswith("image/png")
        finally:
            for name, value in old_env.items():
                restore_env(name, value)
    print("smoke_source_asset_auth=ok")


def tiny_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
