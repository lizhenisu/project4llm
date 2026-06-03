from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
EXPECTED_OBJECT_STORE_PATH = "/app/projects/08-industrial-rag/object_store"
EXPECTED_RUNTIME_PATH = "/app/projects/08-industrial-rag/runtime"


def main() -> None:
    env_example = parse_env_example(PROJECT_DIR / ".env.example")
    required_env_keys = {
        "RAG_MILVUS_URI",
        "MILVUS_URI",
        "MILVUS_TOKEN",
        "RAG_COLLECTION",
        "RAG_OBJECT_STORE_DIR",
        "RAG_RUNTIME_DIR",
        "RAG_REQUIRE_AUTH_CONTEXT",
        "RAG_API_TOKEN",
        "RAG_EMBEDDING_BACKEND",
        "RAG_RERANK_BACKEND",
        "RAG_IMAGE_EMBEDDING_BACKEND",
        "EMBEDDING_MODEL",
        "RERANK_MODEL",
        "IMAGE_EMBEDDING_MODEL",
        "EMBEDDING_DIM",
        "IMAGE_EMBEDDING_DIM",
        "RAG_MODEL_DEVICE",
        "RAG_MODEL_DTYPE",
        "RAG_EMBED_BATCH_SIZE",
        "RAG_EMBED_MAX_LENGTH",
        "RAG_RERANK_BATCH_SIZE",
        "RAG_RERANK_MAX_LENGTH",
        "RAG_IMAGE_EMBED_BATCH_SIZE",
        "RAG_QUERY_REWRITE_BACKEND",
        "RAG_MAX_CONTEXT_CHARS",
        "RAG_MAX_CHUNKS_PER_DOC",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_MODEL",
        "HF_ENDPOINT",
        "HF_HUB_DISABLE_XET",
        "HF_ENABLE_PARALLEL_LOADING",
        "HF_PARALLEL_LOADING_WORKERS",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
    }
    assert required_env_keys.issubset(env_example.keys())

    compose = yaml.safe_load((PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"milvus", "rag-api", "rag-ingest", "minio"}.issubset(services.keys())

    minio_env = services["minio"]["environment"]
    assert "MINIO_ROOT_USER" in minio_env
    assert "MINIO_ROOT_PASSWORD" in minio_env
    assert "MINIO_ACCESS_KEY" not in minio_env
    assert "MINIO_SECRET_KEY" not in minio_env

    rag_api = services["rag-api"]
    rag_api_env = rag_api["environment"]
    assert rag_api_env["MILVUS_URI"] == "http://milvus:19530"
    assert rag_api_env["RAG_OBJECT_STORE_DIR"] == EXPECTED_OBJECT_STORE_PATH
    assert rag_api_env["RAG_RUNTIME_DIR"] == EXPECTED_RUNTIME_PATH
    assert has_volume(rag_api["volumes"], EXPECTED_OBJECT_STORE_PATH)
    assert has_volume(rag_api["volumes"], EXPECTED_RUNTIME_PATH)

    rag_ingest = services["rag-ingest"]
    rag_ingest_env = rag_ingest["environment"]
    assert rag_ingest_env["MILVUS_URI"] == "http://milvus:19530"
    assert rag_ingest_env["RAG_OBJECT_STORE_DIR"] == EXPECTED_OBJECT_STORE_PATH
    assert has_volume(rag_ingest["volumes"], EXPECTED_OBJECT_STORE_PATH)

    dockerfile = (PROJECT_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["./scripts/start_api.sh"]' in dockerfile
    assert "COPY projects/08-industrial-rag /app/projects/08-industrial-rag" in dockerfile

    assert (PROJECT_DIR / "scripts" / "start_api.sh").exists()
    assert (PROJECT_DIR / "scripts" / "start_ingest.sh").exists()

    print("smoke_container_config=ok")


def parse_env_example(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def has_volume(volumes: list[str], container_path: str) -> bool:
    return any(volume.split(":", 1)[1] == container_path for volume in volumes)


if __name__ == "__main__":
    main()
