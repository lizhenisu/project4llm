from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPECTED_OBJECT_STORE_PATH = "/app/projects/09-production-rag/object_store"
EXPECTED_RUNTIME_PATH = "/app/projects/09-production-rag/runtime"


def main() -> None:
    env_example = parse_env_example(PROJECT_DIR / ".env.example")
    required_env_keys = {
        "RAG_MILVUS_URI",
        "MILVUS_URI",
        "MILVUS_TOKEN",
        "RAG_COLLECTION",
        "RAG_OBJECT_STORE_DIR",
        "RAG_OBJECT_STORE_BACKEND",
        "RAG_S3_ENDPOINT_URL",
        "RAG_S3_ACCESS_KEY_ID",
        "RAG_S3_SECRET_ACCESS_KEY",
        "RAG_S3_BUCKET",
        "RAG_S3_PREFIX",
        "RAG_RUNTIME_DIR",
        "RAG_TEXT_INPUT",
        "RAG_IMAGE_INPUT",
        "RAG_REQUIRE_AUTH_CONTEXT",
        "RAG_QUERY_RATE_LIMIT_WINDOW_SECONDS",
        "RAG_QUERY_RATE_LIMIT_GLOBAL",
        "RAG_QUERY_RATE_LIMIT_TENANT",
        "RAG_QUERY_RATE_LIMIT_USER",
        "RAG_API_TOKEN",
        "RAG_EMBEDDING_BACKEND",
        "RAG_RERANK_BACKEND",
        "RAG_IMAGE_EMBEDDING_BACKEND",
        "EMBEDDING_MODEL",
        "RERANK_MODEL",
        "SILICONFLOW_URL",
        "SILICONFLOW_API_KEY",
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
        "RAG_QUERY_REWRITE_HISTORY_TURNS",
        "RAG_QUERY_REWRITE_MAX_TOKENS",
        "RAG_ANSWER_BACKEND",
        "RAG_MAX_CONTEXT_CHARS",
        "RAG_MAX_CHUNKS_PER_DOC",
        "RAG_LLM_BASE_URL",
        "RAG_LLM_API_KEY",
        "NEW_API_URL",
        "NEW_API_KEY",
        "LLM_MODEL",
        "HF_ENDPOINT",
        "HF_HUB_DISABLE_XET",
        "HF_ENABLE_PARALLEL_LOADING",
        "HF_PARALLEL_LOADING_WORKERS",
        "MINIO_ROOT_USER",
        "MINIO_ROOT_PASSWORD",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "RAG_METADATA_DATABASE_URL",
        "PGBOUNCER_MAX_CLIENT_CONN",
        "PGBOUNCER_DEFAULT_POOL_SIZE",
        "PGBOUNCER_RESERVE_POOL_SIZE",
        "PGBOUNCER_MAX_DB_CONNECTIONS",
    }
    assert required_env_keys.issubset(env_example.keys())

    compose = yaml.safe_load((PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"milvus", "rag-api", "rag-worker", "rag-ingest", "minio", "postgres", "pgbouncer"}.issubset(
        services.keys()
    )

    minio_env = services["minio"]["environment"]
    assert "MINIO_ROOT_USER" in minio_env
    assert "MINIO_ROOT_PASSWORD" in minio_env
    assert "MINIO_ACCESS_KEY" not in minio_env
    assert "MINIO_SECRET_KEY" not in minio_env

    postgres_env = services["postgres"]["environment"]
    assert postgres_env["POSTGRES_DB"] == "${POSTGRES_DB:-production_rag}"
    assert postgres_env["POSTGRES_USER"] == "${POSTGRES_USER:-rag}"
    assert postgres_env["POSTGRES_PASSWORD"] == "${POSTGRES_PASSWORD:-rag_password}"

    pgbouncer = services["pgbouncer"]
    pgbouncer_env = pgbouncer["environment"]
    assert "image" not in pgbouncer
    assert pgbouncer["build"]["context"] == "./ops/pgbouncer"
    assert pgbouncer_env["POSTGRES_HOST"] == "postgres"
    assert pgbouncer_env["PGBOUNCER_MAX_CLIENT_CONN"] == "${PGBOUNCER_MAX_CLIENT_CONN:-200}"
    assert pgbouncer_env["PGBOUNCER_DEFAULT_POOL_SIZE"] == "${PGBOUNCER_DEFAULT_POOL_SIZE:-20}"
    assert pgbouncer_env["PGBOUNCER_MAX_DB_CONNECTIONS"] == "${PGBOUNCER_MAX_DB_CONNECTIONS:-60}"
    assert pgbouncer["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "pg_isready" in pgbouncer["healthcheck"]["test"][1]

    rag_api = services["rag-api"]
    rag_api_env = rag_api["environment"]
    assert "container_name" not in rag_api
    assert "ports" not in rag_api
    assert rag_api["expose"] == ["8008"]
    assert rag_api_env["MILVUS_URI"] == "http://milvus:19530"
    assert rag_api_env["RAG_OBJECT_STORE_DIR"] == EXPECTED_OBJECT_STORE_PATH
    assert rag_api_env["RAG_OBJECT_STORE_BACKEND"] == "${RAG_OBJECT_STORE_BACKEND:-s3}"
    assert rag_api_env["RAG_S3_ENDPOINT_URL"] == "${RAG_S3_ENDPOINT_URL:-http://minio:9000}"
    assert rag_api_env["RAG_S3_BUCKET"] == "${RAG_S3_BUCKET:-production-rag}"
    assert rag_api_env["RAG_RUNTIME_DIR"] == EXPECTED_RUNTIME_PATH
    assert "./.env" in rag_api.get("env_file", [])
    assert rag_api_env["RAG_QUERY_REWRITE_BACKEND"] == "${RAG_QUERY_REWRITE_BACKEND:-llm}"
    assert rag_api_env["RAG_QUERY_REWRITE_HISTORY_TURNS"] == "6"
    assert rag_api_env["RAG_QUERY_REWRITE_MAX_TOKENS"] == "256"
    assert rag_api_env["RAG_ANSWER_BACKEND"] == "${RAG_ANSWER_BACKEND:-llm}"
    assert rag_api_env["RAG_IMAGE_EMBEDDING_BACKEND"] == "${RAG_IMAGE_EMBEDDING_BACKEND:-siliconflow}"
    assert "@pgbouncer:6432/" in rag_api_env["RAG_METADATA_DATABASE_URL"]
    assert rag_api_env["RAG_INGEST_EXECUTION_MODE"] == "${RAG_INGEST_EXECUTION_MODE:-external}"
    assert rag_api_env["RAG_QUERY_RATE_LIMIT_WINDOW_SECONDS"] == "${RAG_QUERY_RATE_LIMIT_WINDOW_SECONDS:-60}"
    assert rag_api_env["RAG_QUERY_RATE_LIMIT_GLOBAL"] == "${RAG_QUERY_RATE_LIMIT_GLOBAL:-600}"
    assert rag_api_env["RAG_QUERY_RATE_LIMIT_TENANT"] == "${RAG_QUERY_RATE_LIMIT_TENANT:-120}"
    assert rag_api_env["RAG_QUERY_RATE_LIMIT_USER"] == "${RAG_QUERY_RATE_LIMIT_USER:-30}"
    assert rag_api_env["NEW_API_URL"] == "${NEW_API_URL:-}"
    assert rag_api_env["NEW_API_KEY"] == "${NEW_API_KEY:-}"
    assert rag_api_env["LLM_MODEL"] == "${LLM_MODEL:-gemini-3-flash-preview}"
    assert has_volume(rag_api["volumes"], EXPECTED_OBJECT_STORE_PATH)
    assert has_volume(rag_api["volumes"], EXPECTED_RUNTIME_PATH)
    assert rag_api["depends_on"]["rag-worker"]["condition"] == "service_started"
    assert rag_api["depends_on"]["pgbouncer"]["condition"] == "service_healthy"

    rag_worker = services["rag-worker"]
    rag_worker_env = rag_worker["environment"]
    assert rag_worker["command"] == ["./scripts/start_worker.sh"]
    assert rag_worker["restart"] == "unless-stopped"
    assert "container_name" not in rag_worker
    assert rag_worker_env["MILVUS_URI"] == "http://milvus:19530"
    assert rag_worker_env["RAG_OBJECT_STORE_BACKEND"] == "${RAG_OBJECT_STORE_BACKEND:-s3}"
    assert "@pgbouncer:6432/" in rag_worker_env["RAG_METADATA_DATABASE_URL"]
    assert rag_worker_env["RAG_EMBEDDING_BACKEND"] == "${RAG_EMBEDDING_BACKEND:-siliconflow}"
    assert rag_worker_env["RAG_RERANK_BACKEND"] == "${RAG_RERANK_BACKEND:-siliconflow}"
    assert has_volume(rag_worker["volumes"], EXPECTED_OBJECT_STORE_PATH)
    assert has_volume(rag_worker["volumes"], EXPECTED_RUNTIME_PATH)
    assert rag_worker["depends_on"]["pgbouncer"]["condition"] == "service_healthy"

    rag_ingest = services["rag-ingest"]
    rag_ingest_env = rag_ingest["environment"]
    assert rag_ingest_env["MILVUS_URI"] == "http://milvus:19530"
    assert rag_ingest_env["RAG_OBJECT_STORE_DIR"] == EXPECTED_OBJECT_STORE_PATH
    assert rag_ingest_env["RAG_OBJECT_STORE_BACKEND"] == "${RAG_OBJECT_STORE_BACKEND:-s3}"
    assert rag_ingest_env["RAG_S3_ENDPOINT_URL"] == "${RAG_S3_ENDPOINT_URL:-http://minio:9000}"
    assert rag_ingest_env["RAG_S3_BUCKET"] == "${RAG_S3_BUCKET:-production-rag}"
    assert "./.env" in rag_ingest.get("env_file", [])
    assert "RAG_TEXT_INPUT" in rag_ingest_env
    assert "RAG_IMAGE_INPUT" in rag_ingest_env
    assert rag_ingest_env["RAG_IMAGE_EMBEDDING_BACKEND"] == "${RAG_IMAGE_EMBEDDING_BACKEND:-siliconflow}"
    assert "@pgbouncer:6432/" in rag_ingest_env["RAG_METADATA_DATABASE_URL"]
    assert has_volume(rag_ingest["volumes"], EXPECTED_OBJECT_STORE_PATH)
    assert has_volume(rag_ingest["volumes"], "/data")
    assert rag_ingest["depends_on"]["pgbouncer"]["condition"] == "service_healthy"

    pgbouncer_dockerfile = (PROJECT_DIR / "ops" / "pgbouncer" / "Dockerfile").read_text(encoding="utf-8")
    assert "apt-get" in pgbouncer_dockerfile
    assert "install -y --no-install-recommends" in pgbouncer_dockerfile
    assert "pgbouncer" in pgbouncer_dockerfile
    assert "USER pgbouncer" in pgbouncer_dockerfile

    dockerfile = (PROJECT_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["./scripts/start_api.sh"]' in dockerfile
    assert "COPY projects/09-production-rag /app/projects/09-production-rag" in dockerfile
    assert "requirements-api.txt" in dockerfile
    assert "ARG PIP_INDEX_URL=" in dockerfile
    assert '--index-url "${PIP_INDEX_URL}"' in dockerfile
    assert "uv sync" not in dockerfile

    start_api = (PROJECT_DIR / "scripts" / "start_api.sh").read_text(encoding="utf-8")
    assert "python check_config.py" in start_api
    assert (PROJECT_DIR / "scripts" / "start_ingest.sh").exists()
    start_worker = (PROJECT_DIR / "scripts" / "start_worker.sh").read_text(encoding="utf-8")
    assert "python ingestion_worker.py" in start_worker

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
    return any(volume.split(":")[1] == container_path for volume in volumes)


if __name__ == "__main__":
    main()
