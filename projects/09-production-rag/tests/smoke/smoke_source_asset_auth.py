from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
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
        "RAG_S3_BUCKET": os.environ.get("RAG_S3_BUCKET"),
        "RAG_S3_PREFIX": os.environ.get("RAG_S3_PREFIX"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        object_store = root / "object_store"
        image_path = object_store / "uploads" / "tenant-asset-smoke" / "upload-1" / "image.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(tiny_png())
        other_image = object_store / "uploads" / "other-tenant" / "upload-2" / "private.png"
        other_image.parent.mkdir(parents=True)
        other_image.write_bytes(tiny_png())
        symlink_path = image_path.parent / "other-tenant-link.png"
        symlink_path.symlink_to(other_image)
        os.environ["RAG_RUNTIME_DIR"] = str(root / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(object_store)
        os.environ["RAG_METADATA_DATABASE_URL"] = ""
        os.environ["RAG_S3_BUCKET"] = "asset-smoke-bucket"
        os.environ["RAG_S3_PREFIX"] = "app"
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
                auth_headers = {"Authorization": "Bearer session-token"}
                missing_token = api.get("/source-assets/uploads/tenant-asset-smoke/upload-1/image.png")
                assert missing_token.status_code == 401, missing_token.text
                query_token = api.get(
                    "/source-assets/uploads/tenant-asset-smoke/upload-1/image.png"
                    "?tenant_id=tenant-asset-smoke&token=session-token"
                )
                assert query_token.status_code == 401, query_token.text

                wrong_tenant = api.get(
                    "/source-assets/uploads/other-tenant/upload-1/image.png"
                    "?tenant_id=other-tenant",
                    headers=auth_headers,
                )
                assert wrong_tenant.status_code == 404, wrong_tenant.text

                ok = api.get(
                    "/source-assets/uploads/tenant-asset-smoke/upload-1/image.png"
                    "?tenant_id=tenant-asset-smoke",
                    headers=auth_headers,
                )
                assert ok.status_code == 200, ok.text
                assert ok.headers["content-type"].startswith("image/png")

                legacy_path = api.get(
                    "/source-assets/legacy/tenant-asset-smoke/image.png"
                    "?tenant_id=tenant-asset-smoke",
                    headers=auth_headers,
                )
                assert legacy_path.status_code == 404, legacy_path.text
                traversal_path = serve.resolve_local_source_asset(
                    object_store_dir=object_store,
                    asset_path=(
                        "uploads/tenant-asset-smoke/../other-tenant/upload-2/private.png"
                    ),
                    tenant_id="tenant-asset-smoke",
                )
                assert traversal_path is None
                symlink_escape = api.get(
                    "/source-assets/uploads/tenant-asset-smoke/upload-1/"
                    "other-tenant-link.png?tenant_id=tenant-asset-smoke",
                    headers=auth_headers,
                )
                assert symlink_escape.status_code == 404, symlink_escape.text

                valid_s3_uri = "s3://asset-smoke-bucket/app/uploads/tenant-asset-smoke/upload-1/image.png"
                invalid_s3_uris = [
                    "s3://other-bucket/app/uploads/tenant-asset-smoke/upload-1/image.png",
                    "s3://asset-smoke-bucket/legacy/uploads/tenant-asset-smoke/upload-1/image.png",
                    "s3://asset-smoke-bucket/app/archive/uploads/tenant-asset-smoke/image.png",
                    "s3://asset-smoke-bucket/app/uploads/other-tenant/upload-1/image.png",
                ]
                with patch("serve.read_object_bytes_by_uri", return_value=tiny_png()) as read_s3:
                    valid_s3 = api.get(s3_asset_url(valid_s3_uri), headers=auth_headers)
                    assert valid_s3.status_code == 200, valid_s3.text
                    assert valid_s3.headers["content-type"].startswith("image/png")
                    for invalid_uri in invalid_s3_uris:
                        invalid_s3 = api.get(s3_asset_url(invalid_uri), headers=auth_headers)
                        assert invalid_s3.status_code == 404, (invalid_uri, invalid_s3.text)
                    read_s3.assert_called_once_with(valid_s3_uri)
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


def s3_asset_url(uri: str) -> str:
    return (
        f"/source-assets/__s3__/{quote(uri, safe='')}"
        "?tenant_id=tenant-asset-smoke"
    )


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
