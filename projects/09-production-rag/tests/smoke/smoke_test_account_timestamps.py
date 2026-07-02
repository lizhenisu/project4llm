from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.database import connect_metadata_db
from rag_core.user_auth import (
    DEFAULT_ADMIN_INITIALIZED_KEY,
    DEFAULT_ADMIN_USERNAME,
    LEGACY_TEST_ACCOUNT_CREATED_AT,
    TEST_ACCOUNT_USERNAME,
    authenticate_token,
    change_user_password,
    ensure_default_admin_account,
    ensure_default_test_account,
    login_user,
    refresh_session_token,
    list_public_users,
)


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_token = os.environ.get("RAG_FIXED_TEST_LOGIN_TOKEN")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        os.environ["RAG_FIXED_TEST_LOGIN_TOKEN"] = "production-rag-fixed-test-login-token"
        try:
            config = load_config()
            created_user = ensure_default_test_account(config)
            created_admin = ensure_default_admin_account(config)
            assert created_user.created_at != LEGACY_TEST_ACCOUNT_CREATED_AT
            assert created_user.last_login_at is None
            assert created_admin.username == DEFAULT_ADMIN_USERNAME
            assert created_admin.role == "admin"
            assert ensure_default_admin_account(config).id == created_admin.id

            logged_in_admin, admin_token, _ = login_user(
                config,
                username="admin",
                password="admin",
            )
            assert logged_in_admin.id == created_admin.id
            refreshed_admin, refreshed_token, _ = refresh_session_token(
                config,
                current_token=admin_token,
            )
            assert refreshed_admin.id == created_admin.id
            assert refreshed_token != admin_token
            assert authenticate_token(config, token=admin_token) is None
            assert authenticate_token(config, token=refreshed_token).id == created_admin.id

            change_user_password(
                config,
                user_id=created_admin.id,
                current_password="admin",
                new_password="stronger-password",
            )
            assert ensure_default_admin_account(config).id == created_admin.id
            assert login_user(
                config,
                username="admin",
                password="stronger-password",
            )[0].id == created_admin.id
            try:
                login_user(config, username="admin", password="admin")
            except ValueError:
                pass
            else:
                raise AssertionError("repeated initialization must not reset a changed admin password")

            with connect_metadata_db(config) as conn:
                conn.execute(
                    "DELETE FROM schema_meta WHERE key = ?",
                    (DEFAULT_ADMIN_INITIALIZED_KEY,),
                )
            assert ensure_default_admin_account(config).id == created_admin.id
            assert login_user(config, username="admin", password="admin")[0].id == created_admin.id
            try:
                login_user(config, username="admin", password="stronger-password")
            except ValueError:
                pass
            else:
                raise AssertionError("legacy admin migration must apply the requested initial password once")

            authenticated_user = authenticate_token(
                config,
                token=config.fixed_test_login_token,
            )
            assert authenticated_user is not None
            assert authenticated_user.username == TEST_ACCOUNT_USERNAME
            assert authenticated_user.last_login_at is not None
            assert authenticated_user.last_login_at >= created_user.created_at

            listed_user = next(
                user for user in list_public_users(config) if user.username == TEST_ACCOUNT_USERNAME
            )
            assert listed_user.created_at == created_user.created_at
            assert listed_user.last_login_at == authenticated_user.last_login_at

            with connect_metadata_db(config) as conn:
                conn.execute(
                    "UPDATE users SET created_at = ? WHERE username = ?",
                    (LEGACY_TEST_ACCOUNT_CREATED_AT, TEST_ACCOUNT_USERNAME),
                )
            migrated_user = ensure_default_test_account(config)
            assert migrated_user.created_at != LEGACY_TEST_ACCOUNT_CREATED_AT
            assert migrated_user.created_at >= created_user.created_at
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_FIXED_TEST_LOGIN_TOKEN", old_token)
    print("smoke_test_account_timestamps=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
