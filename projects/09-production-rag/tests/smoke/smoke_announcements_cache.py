from __future__ import annotations

import os
import sys
from pathlib import Path
import tempfile


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.user_auth import (  # noqa: E402
    create_announcement,
    invalidate_announcement_cache,
    list_announcements,
    register_user,
)


def main() -> None:
    old_ttl = os.environ.get("RAG_ANNOUNCEMENT_CACHE_TTL_SECONDS")
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["RAG_ANNOUNCEMENT_CACHE_TTL_SECONDS"] = "60"
        os.environ["RAG_RUNTIME_DIR"] = str(Path(temp_dir) / "runtime")
        config = load_config()
        try:
            invalidate_announcement_cache(config)
            admin = register_user(config, username="cache-admin", password="12345678", display_name="Cache Admin")
            first = create_announcement(
                config,
                title="First",
                content="First announcement",
                author_id=admin.id,
            )
            assert [row["id"] for row in list_announcements(config, limit=5)] == [first["id"]]

            second = create_announcement(
                config,
                title="Second",
                content="Second announcement",
                author_id=admin.id,
            )
            rows = list_announcements(config, limit=5)
            assert [row["id"] for row in rows] == [second["id"], first["id"]]

            # Mutating the returned rows must not mutate cached rows.
            rows[0]["title"] = "mutated"
            cached_rows = list_announcements(config, limit=5)
            assert cached_rows[0]["title"] == "Second"
        finally:
            invalidate_announcement_cache(config)
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_ANNOUNCEMENT_CACHE_TTL_SECONDS", old_ttl)
    print("smoke_announcements_cache=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
