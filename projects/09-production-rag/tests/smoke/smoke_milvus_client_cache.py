from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.milvus_store import (  # noqa: E402
    close_thread_milvus_clients,
    connect,
    milvus_client_metrics_snapshot,
)


class FakeMilvusClient:
    created: list[tuple[str, str | None, dict[str, int]]] = []
    closed = 0

    def __init__(self, *, uri: str, token: str | None = None, grpc_options: dict[str, int] | None = None) -> None:
        self.uri = uri
        self.token = token
        FakeMilvusClient.created.append((uri, token, dict(grpc_options or {})))

    def close(self) -> None:
        FakeMilvusClient.closed += 1


def main() -> None:
    old_cache = os.environ.get("RAG_MILVUS_CLIENT_CACHE")
    old_keepalive_time = os.environ.get("RAG_MILVUS_GRPC_KEEPALIVE_TIME_MS")
    old_keepalive_timeout = os.environ.get("RAG_MILVUS_GRPC_KEEPALIVE_TIMEOUT_MS")
    old_keepalive_without_calls = os.environ.get("RAG_MILVUS_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS")
    config = load_config()
    close_thread_milvus_clients()
    try:
        with patch("rag_core.milvus_store.MilvusClient", FakeMilvusClient):
            cached_config = replace(config, milvus_uri="http://milvus-cache-smoke:19530", milvus_token=None)
            first = connect(cached_config)
            second = connect(cached_config)
            assert first is second
            assert FakeMilvusClient.created == [
                (
                    "http://milvus-cache-smoke:19530",
                    None,
                    {
                        "grpc.keepalive_time_ms": 60_000,
                        "grpc.keepalive_timeout_ms": 20_000,
                        "grpc.keepalive_permit_without_calls": 0,
                    },
                )
            ]
            metrics = milvus_client_metrics_snapshot()
            assert metrics["created_total"] >= 1
            assert metrics["reused_total"] >= 1
            assert metrics["thread_cached_clients"] == 1

            other = connect(replace(cached_config, milvus_uri="http://milvus-cache-smoke-2:19530"))
            assert other is not first
            assert len(FakeMilvusClient.created) == 2
            assert milvus_client_metrics_snapshot()["thread_cached_clients"] == 2

            os.environ["RAG_MILVUS_CLIENT_CACHE"] = "0"
            uncached_a = connect(cached_config)
            uncached_b = connect(cached_config)
            assert uncached_a is not uncached_b
            assert len(FakeMilvusClient.created) == 4
            assert milvus_client_metrics_snapshot()["cache_enabled"] is False
            assert milvus_client_metrics_snapshot()["cache_disabled_total"] >= 2

            os.environ["RAG_MILVUS_GRPC_KEEPALIVE_TIME_MS"] = "120000"
            os.environ["RAG_MILVUS_GRPC_KEEPALIVE_TIMEOUT_MS"] = "30000"
            os.environ["RAG_MILVUS_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "1"
            custom = connect(replace(cached_config, milvus_uri="http://milvus-cache-smoke-3:19530"))
            assert custom is not None
            assert FakeMilvusClient.created[-1][2] == {
                "grpc.keepalive_time_ms": 120_000,
                "grpc.keepalive_timeout_ms": 30_000,
                "grpc.keepalive_permit_without_calls": 1,
            }
    finally:
        close_thread_milvus_clients()
        restore_env("RAG_MILVUS_CLIENT_CACHE", old_cache)
        restore_env("RAG_MILVUS_GRPC_KEEPALIVE_TIME_MS", old_keepalive_time)
        restore_env("RAG_MILVUS_GRPC_KEEPALIVE_TIMEOUT_MS", old_keepalive_timeout)
        restore_env("RAG_MILVUS_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", old_keepalive_without_calls)
    print("smoke_milvus_client_cache=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
