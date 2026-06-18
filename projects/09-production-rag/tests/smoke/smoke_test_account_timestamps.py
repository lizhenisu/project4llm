from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.database import connect_metadata_db
from rag_core.user_auth import (
    LEGACY_TEST_ACCOUNT_CREATED_AT,
    TEST_ACCOUNT_USERNAME,
    authenticate_token,
    ensure_default_test_account,
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
            assert created_user.created_at != LEGACY_TEST_ACCOUNT_CREATED_AT
            assert created_user.last_login_at is None

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
