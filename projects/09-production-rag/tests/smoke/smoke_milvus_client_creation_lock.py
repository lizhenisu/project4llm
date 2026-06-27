from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_core.config import load_config  # noqa: E402
from rag_core.milvus_store import close_thread_milvus_clients, connect  # noqa: E402


class ContendedMilvusClient:
    active_creations = 0
    max_active_creations = 0
    created = 0
    lock = threading.Lock()

    def __init__(
        self,
        *,
        uri: str,
        token: str | None = None,
        grpc_options: dict[str, int] | None = None,
    ) -> None:
        self.uri = uri
        self.token = token
        self.grpc_options = dict(grpc_options or {})
        with ContendedMilvusClient.lock:
            ContendedMilvusClient.active_creations += 1
            ContendedMilvusClient.max_active_creations = max(
                ContendedMilvusClient.max_active_creations,
                ContendedMilvusClient.active_creations,
            )
        try:
            time.sleep(0.03)
            ContendedMilvusClient.created += 1
        finally:
            with ContendedMilvusClient.lock:
                ContendedMilvusClient.active_creations -= 1


def main() -> None:
    config = replace(load_config(), milvus_uri="http://milvus-client-lock-smoke:19530", milvus_token=None)
    close_thread_milvus_clients()
    try:
        with patch("rag_core.milvus_store.MilvusClient", ContendedMilvusClient):
            with ThreadPoolExecutor(max_workers=8) as executor:
                clients = list(executor.map(lambda _: connect(config), range(8)))
        assert len(clients) == 8
        assert ContendedMilvusClient.created == 8
        assert ContendedMilvusClient.max_active_creations == 1, ContendedMilvusClient.max_active_creations
    finally:
        close_thread_milvus_clients()
    print("smoke_milvus_client_creation_lock=ok")


if __name__ == "__main__":
    main()
