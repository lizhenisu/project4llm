from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RUNTIME_DIR = PROJECT_DIR / "runtime"
OBJECT_STORE_DIR = PROJECT_DIR / "object_store"
DEFAULT_MILVUS_DB = PROJECT_DIR / "industrial_rag_demo.db"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float_or_none(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


@dataclass(frozen=True)
class RagConfig:
    milvus_uri: str
    milvus_token: str | None
    collection_name: str
    embedding_model: str
    embedding_backend: str
    embedding_dim: int
    rerank_model: str
    rerank_backend: str
    image_embedding_backend: str
    image_embedding_model: str
    image_embedding_dim: int
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    reset_collection: bool
    runtime_dir: Path
    object_store_dir: Path
    pii_policy: str
    max_context_chars: int
    max_chunks_per_doc: int
    min_rerank_score: float | None
    query_rewrite_backend: str
    require_auth_context: bool
    api_token: str | None


def load_config() -> RagConfig:
    embedding_backend = os.environ.get("RAG_EMBEDDING_BACKEND", "hash").lower()
    image_backend = os.environ.get("RAG_IMAGE_EMBEDDING_BACKEND", "hash").lower()

    return RagConfig(
        milvus_uri=os.environ.get("MILVUS_URI", str(DEFAULT_MILVUS_DB)),
        milvus_token=os.environ.get("MILVUS_TOKEN") or None,
        collection_name=os.environ.get("RAG_COLLECTION", "rag_chunks_v1"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_backend=embedding_backend,
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_backend=os.environ.get("RAG_RERANK_BACKEND", "lexical").lower(),
        image_embedding_backend=image_backend,
        image_embedding_model=os.environ.get(
            "IMAGE_EMBEDDING_MODEL",
            "openai/clip-vit-base-patch32",
        ),
        image_embedding_dim=_env_int("IMAGE_EMBEDDING_DIM", _env_int("EMBEDDING_DIM", 1024)),
        llm_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        llm_api_key=os.environ.get("OPENAI_API_KEY") or None,
        llm_model=os.environ.get("LLM_MODEL", "gemini-3-flash-preview"),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 700),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 100),
        reset_collection=_env_bool("RAG_RESET_COLLECTION", False),
        runtime_dir=Path(os.environ.get("RAG_RUNTIME_DIR", str(RUNTIME_DIR))),
        object_store_dir=Path(
            os.environ.get("RAG_OBJECT_STORE_DIR", str(OBJECT_STORE_DIR))
        ),
        pii_policy=os.environ.get("RAG_PII_POLICY", "warn").lower(),
        max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 6000),
        max_chunks_per_doc=_env_int("RAG_MAX_CHUNKS_PER_DOC", 2),
        min_rerank_score=_env_float_or_none("RAG_MIN_RERANK_SCORE"),
        query_rewrite_backend=os.environ.get("RAG_QUERY_REWRITE_BACKEND", "none").lower(),
        require_auth_context=_env_bool("RAG_REQUIRE_AUTH_CONTEXT", False),
        api_token=os.environ.get("RAG_API_TOKEN") or None,
    )
