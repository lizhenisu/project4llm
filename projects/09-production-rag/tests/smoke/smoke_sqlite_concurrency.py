from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_core.config import load_config
from rag_core.user_auth import list_public_users, register_user


def main() -> None:
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        try:
            config = load_config()
            with ThreadPoolExecutor(max_workers=12) as executor:
                futures = [
                    executor.submit(
                        register_user,
                        config,
                        username=f"user_{index:02d}",
                        password="strong-password",
                        display_name=f"用户 {index:02d}",
                    )
                    for index in range(30)
                ]
                users = [future.result() for future in futures]
            rows = list_public_users(config)
            assert len(rows) == 30
            assert sum(1 for user in users if user.role == "admin") == 1
            assert sum(1 for user in users if user.role == "user") == 29
        finally:
            restore_env("RAG_RUNTIME_DIR", old_runtime)
    print("sqlite concurrency smoke passed")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
