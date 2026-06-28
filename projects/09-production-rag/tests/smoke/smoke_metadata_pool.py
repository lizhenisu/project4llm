from __future__ import annotations

import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core import database


def main() -> None:
    test_postgres_pool_reuses_connections_and_records_waits()
    test_postgres_pool_times_out_when_exhausted()
    print("smoke_metadata_pool=ok")


def test_postgres_pool_reuses_connections_and_records_waits() -> None:
    url = fake_database_url("reuse")
    with patched_pool_env(pool_size="2", timeout_seconds="2"):
        with patch("rag_core.database.open_postgres_raw_connection", side_effect=lambda _: FakeRawConnection()):
            with patch("rag_core.database.ensure_postgres_schema_initialized"):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(lambda _: hold_connection(url), range(4)))
                snapshot = database.postgres_connection_pool(url).snapshot()
        assert results == [True, True, True, True]
        assert snapshot["max_size"] == 2
        assert snapshot["created_total"] == 2
        assert snapshot["reused_total"] >= 2
        assert snapshot["waits_total"] >= 1
        assert snapshot["timeouts_total"] == 0
        assert snapshot["borrowed"] == 0
        assert snapshot["idle"] == 2


def test_postgres_pool_times_out_when_exhausted() -> None:
    url = fake_database_url("timeout")
    with patched_pool_env(pool_size="1", timeout_seconds="0.01"):
        with patch("rag_core.database.open_postgres_raw_connection", side_effect=lambda _: FakeRawConnection()):
            with patch("rag_core.database.ensure_postgres_schema_initialized"):
                with database.connect_postgres_metadata_db(url):
                    try:
                        with database.connect_postgres_metadata_db(url):
                            raise AssertionError("second checkout should not succeed")
                    except TimeoutError as exc:
                        assert "RAG_METADATA_POOL_SIZE=1" in str(exc)
                snapshot = database.postgres_connection_pool(url).snapshot()
        assert snapshot["timeouts_total"] == 1
        assert snapshot["borrowed"] == 0
        assert snapshot["idle"] == 1


def hold_connection(url: str) -> bool:
    with database.connect_postgres_metadata_db(url):
        time.sleep(0.02)
    return True


def fake_database_url(label: str) -> str:
    path = Path(tempfile.gettempdir()) / f"production-rag-metadata-pool-{label}-{time.time_ns()}"
    return f"postgresql://rag:secret@127.0.0.1/{path.name}"


class FakeRawConnection:
    closed = False

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class patched_pool_env:
    def __init__(self, *, pool_size: str, timeout_seconds: str) -> None:
        self.values = {
            "RAG_METADATA_POOL_SIZE": pool_size,
            "RAG_METADATA_POOL_TIMEOUT_SECONDS": timeout_seconds,
        }
        self.old_values: dict[str, str | None] = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.old_values[name] = os.environ.get(name)
            os.environ[name] = value
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for name, value in self.old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
