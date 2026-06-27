from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.user_auth import (  # noqa: E402
    authenticate_token,
    auth_token_cache_metrics_snapshot,
    invalidate_auth_token_cache,
    login_user,
    refresh_session_token,
    register_user,
    set_user_status,
)


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_ttl = os.environ.get("RAG_AUTH_TOKEN_CACHE_TTL_SECONDS")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        os.environ["RAG_AUTH_TOKEN_CACHE_TTL_SECONDS"] = "60"
        try:
            config = load_config()
            invalidate_auth_token_cache(config)
            admin = register_user(
                config,
                username="cache-admin",
                password="strong-password",
                display_name="Cache Admin",
            )
            user = register_user(
                config,
                username="cache-user",
                password="strong-password",
                display_name="Cache User",
            )

            _, token, _ = login_user(config, username=user.username, password="strong-password")
            invalidate_auth_token_cache(config)

            first = authenticate_token(config, token=token)
            second = authenticate_token(config, token=token)
            assert first is not None
            assert second is not None
            assert first.id == user.id
            assert second.id == user.id
            metrics = auth_token_cache_metrics_snapshot()
            assert metrics["misses_total"] >= 1
            assert metrics["hits_total"] >= 1
            assert metrics["entries"] >= 1

            _, next_token, _ = refresh_session_token(config, current_token=token)
            assert authenticate_token(config, token=token) is None
            assert authenticate_token(config, token=next_token) is not None

            set_user_status(config, actor_id=admin.id, user_id=user.id, status="banned")
            assert authenticate_token(config, token=next_token) is None
        finally:
            invalidate_auth_token_cache()
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_AUTH_TOKEN_CACHE_TTL_SECONDS", old_ttl)
    print("smoke_auth_token_cache=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
