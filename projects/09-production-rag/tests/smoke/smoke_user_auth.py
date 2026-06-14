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
    admin_headers = {"Authorization": f"Bearer {first_body['token']}"}

    anonymous_sources = api.get("/sources?tenant_id=team_a")
    assert anonymous_sources.status_code == 401, anonymous_sources.text
    anonymous_upload = api.post(
        "/sources/upload",
        data={"tenant_id": "team_a", "acl_groups": "engineering"},
        files={"file": ("anonymous.md", b"# Anonymous\n\nblocked", "text/markdown")},
    )
    assert anonymous_upload.status_code == 401, anonymous_upload.text
    anonymous_query = api.post(
        "/query",
        json={"query": "匿名请求", "tenant_id": "team_a", "acl_groups": ["engineering"]},
    )
    assert anonymous_query.status_code == 401, anonymous_query.text

    settings = api.get("/admin/settings", headers=admin_headers)
    assert settings.status_code == 200, settings.text
    assert settings.json()["registration_enabled"] is True
    assert settings.json()["latest_announcement"] is None

    disabled_settings = api.patch(
        "/admin/settings/registration",
        headers=admin_headers,
        json={"registration_enabled": False},
    )
    assert disabled_settings.status_code == 200, disabled_settings.text
    assert disabled_settings.json()["registration_enabled"] is False

    blocked = api.post(
        "/auth/register",
        json={"username": "blocked_user", "password": "strong-password", "display_name": "禁用注册用户"},
    )
    assert blocked.status_code == 400, blocked.text
    assert "注册" in blocked.json()["detail"]

    enabled_settings = api.patch(
        "/admin/settings/registration",
        headers=admin_headers,
        json={"registration_enabled": True},
    )
    assert enabled_settings.status_code == 200, enabled_settings.text
    assert enabled_settings.json()["registration_enabled"] is True

    second = api.post(
        "/auth/register",
        json={"username": "normal_user", "password": "strong-password", "display_name": "普通用户"},
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["user"]["role"] == "user"
    assert second_body["user"]["tenant_id"] != first_body["user"]["tenant_id"]

    user_headers = {"Authorization": f"Bearer {second_body['token']}"}

    me = api.get("/auth/me", headers=user_headers)
    assert me.status_code == 200, me.text
    assert me.json()["username"] == "normal_user"
    assert me.json()["status"] == "active"
    assert me.json()["created_at"]

    profile = api.patch(
        "/auth/me",
        headers=user_headers,
        json={
            "username": "renamed_user",
            "display_name": "改名用户",
            "avatar_url": "https://example.com/avatar.png",
        },
    )
    assert profile.status_code == 200, profile.text
    assert profile.json()["username"] == "renamed_user"
    assert profile.json()["display_name"] == "改名用户"
    assert profile.json()["avatar_url"] == "https://example.com/avatar.png"

    password = api.patch(
        "/auth/password",
        headers=user_headers,
        json={"current_password": "strong-password", "new_password": "stronger-password"},
    )
    assert password.status_code == 200, password.text
    relogin = api.post(
        "/auth/login",
        json={"username": "renamed_user", "password": "stronger-password"},
    )
    assert relogin.status_code == 200, relogin.text
    user_headers = {"Authorization": f"Bearer {relogin.json()['token']}"}

    forbidden = api.get("/admin/users", headers=user_headers)
    assert forbidden.status_code == 403, forbidden.text

    users = api.get("/admin/users", headers=admin_headers)
    assert users.status_code == 200, users.text
    assert [row["username"] for row in users.json()["users"]] == ["renamed_user", "admin_user"]

    banned = api.patch(
        f"/admin/users/{second_body['user']['id']}/status",
        headers=admin_headers,
        json={"status": "banned"},
    )
    assert banned.status_code == 200, banned.text
    assert banned.json()["status"] == "banned"
    assert banned.json()["role"] == "user"

    banned_me = api.get("/auth/me", headers=user_headers)
    assert banned_me.status_code == 401, banned_me.text
    banned_login = api.post(
        "/auth/login",
        json={"username": "renamed_user", "password": "stronger-password"},
    )
    assert banned_login.status_code == 401, banned_login.text

    active = api.patch(
        f"/admin/users/{second_body['user']['id']}/status",
        headers=admin_headers,
        json={"status": "active"},
    )
    assert active.status_code == 200, active.text
    assert active.json()["status"] == "active"

    relogin = api.post(
        "/auth/login",
        json={"username": "renamed_user", "password": "stronger-password"},
    )
    assert relogin.status_code == 200, relogin.text
    user_headers = {"Authorization": f"Bearer {relogin.json()['token']}"}

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

    settings = api.get("/admin/settings", headers=admin_headers)
    assert settings.status_code == 200, settings.text
    assert settings.json()["latest_announcement"]["title"] == "系统维护"
    assert settings.json()["latest_announcement"]["content"] == "今晚 23:00 进行例行维护。"

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
