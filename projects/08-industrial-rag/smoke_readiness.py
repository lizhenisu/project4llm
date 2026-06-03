from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.milvus_store import connect, ensure_collection
from serve import create_app


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "readiness.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_readiness"
        try:
            config = load_config()
            client = connect(config)
            ensure_collection(client, config, reset=True)

            api = TestClient(create_app())
            health = api.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            ready = api.get("/ready")
            assert ready.status_code == 200, ready.text
            body = ready.json()
            assert body["status"] == "ok"
            assert body["checks"]["milvus_connect"]["ok"] is True
            assert body["checks"]["collection_exists"]["ok"] is True
            assert body["checks"]["schema"]["ok"] is True
            assert body["checks"]["schema"]["text_dense_vector_dim"] == config.embedding_dim
            assert body["checks"]["schema"]["image_dense_vector_dim"] == config.image_embedding_dim
            assert body["checks"]["schema"]["text_analyzer_enabled"] is True

            os.environ["RAG_COLLECTION"] = "rag_smoke_readiness_missing"
            missing = TestClient(create_app()).get("/ready")
            assert missing.status_code == 503
            assert missing.json()["detail"]["checks"]["collection_exists"]["ok"] is False
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)

    print("smoke_readiness=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
