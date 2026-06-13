from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from serve import create_app


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(temp_dir) / "object_store")
        try:
            run_smoke()
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)


def run_smoke() -> None:
    api = TestClient(create_app())

    first = api.post(
        "/auth/register",
        json={"username": "admin_user", "password": "strong-password", "display_name": "管理员"},
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["user"]["role"] == "admin"
    assert first_body["token"]

    second = api.post(
        "/auth/register",
        json={"username": "normal_user", "password": "strong-password", "display_name": "普通用户"},
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["user"]["role"] == "user"
    assert second_body["user"]["tenant_id"] != first_body["user"]["tenant_id"]

    user_headers = {"Authorization": f"Bearer {second_body['token']}"}
    admin_headers = {"Authorization": f"Bearer {first_body['token']}"}

    me = api.get("/auth/me", headers=user_headers)
    assert me.status_code == 200, me.text
    assert me.json()["username"] == "normal_user"

    forbidden = api.get("/admin/users", headers=user_headers)
    assert forbidden.status_code == 403, forbidden.text

    users = api.get("/admin/users", headers=admin_headers)
    assert users.status_code == 200, users.text
    assert [row["username"] for row in users.json()["users"]] == ["normal_user", "admin_user"]

    announcement = api.post(
        "/admin/announcements",
        headers=admin_headers,
        json={"title": "系统维护", "content": "今晚 23:00 进行例行维护。"},
    )
    assert announcement.status_code == 200, announcement.text
    assert announcement.json()["author_name"] == "管理员"

    announcements = api.get("/announcements")
    assert announcements.status_code == 200, announcements.text
    assert announcements.json()["announcements"][0]["title"] == "系统维护"

    logout = api.post("/auth/logout", headers=user_headers)
    assert logout.status_code == 200, logout.text
    expired = api.get("/auth/me", headers=user_headers)
    assert expired.status_code == 401, expired.text

    print("user auth smoke passed")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
