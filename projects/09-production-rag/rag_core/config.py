from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIXTURE_DATA_DIR = PROJECT_DIR / "tests" / "fixtures" / "data"
RUNTIME_DIR = PROJECT_DIR / "runtime"
OBJECT_STORE_DIR = PROJECT_DIR / "object_store"
DEFAULT_MILVUS_DB = PROJECT_DIR / "production_rag.db"
MODELSCOPE_CACHE = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "BAAI"
_ENV_LOADED = False


def load_dotenv_if_present() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    for path in (PROJECT_DIR.parents[1] / ".env", PROJECT_DIR / ".env"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    _ENV_LOADED = True


def _newapi_base_url() -> str | None:
    value = os.environ.get("NEW_API_URL")
    if not value:
        return None
    base_url = value.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _resolve_model_path(model_id: str, *, ms_subdir: str) -> str:
    """Return local path if the model exists in the ModelScope cache."""
    candidate = MODELSCOPE_CACHE / ms_subdir
    if candidate.is_dir() and (
        (candidate / "pytorch_model.bin").exists()
        or (candidate / "model.safetensors").exists()
    ):
        return str(candidate)
    return model_id


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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _validate_backend(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        allowed_text = "/".join(sorted(allowed))
        raise ValueError(f"Unsupported {name}={value!r}; use {allowed_text}")


@dataclass(frozen=True)
class RagConfig:
    milvus_uri: str
    milvus_token: str | None
    collection_name: str
    embedding_model: str
    embedding_backend: str
    embedding_dim: int
    embedding_batch_size: int
    embedding_max_length: int
    rerank_model: str
    rerank_backend: str
    rerank_batch_size: int
    rerank_max_length: int
    image_embedding_backend: str
    image_embedding_model: str
    image_embedding_dim: int
    image_embedding_batch_size: int
    model_device: str
    model_dtype: str
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str
    siliconflow_base_url: str
    siliconflow_api_key: str | None
    answer_backend: str
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
    query_rewrite_history_turns: int
    query_rewrite_max_tokens: int
    require_auth_context: bool
    api_token: str | None
    dense_hnsw_m: int
    dense_hnsw_ef_construction: int
    dense_search_ef: int
    image_hnsw_m: int
    image_hnsw_ef_construction: int
    image_search_ef: int
    sparse_drop_ratio_build: float
    sparse_drop_ratio_search: float


def load_config() -> RagConfig:
    load_dotenv_if_present()
    embedding_backend = os.environ.get("RAG_EMBEDDING_BACKEND", "siliconflow").lower()
    image_backend = os.environ.get("RAG_IMAGE_EMBEDDING_BACKEND", "none").lower()
    rerank_backend = os.environ.get("RAG_RERANK_BACKEND", "siliconflow").lower()
    answer_backend = os.environ.get("RAG_ANSWER_BACKEND", "llm").lower()
    query_rewrite_backend = os.environ.get("RAG_QUERY_REWRITE_BACKEND", "llm").lower()
    _validate_backend("RAG_EMBEDDING_BACKEND", embedding_backend, {"siliconflow", "bge"})
    _validate_backend("RAG_IMAGE_EMBEDDING_BACKEND", image_backend, {"clip", "none", "siliconflow"})
    _validate_backend("RAG_RERANK_BACKEND", rerank_backend, {"siliconflow", "bge"})
    _validate_backend("RAG_ANSWER_BACKEND", answer_backend, {"llm"})
    _validate_backend("RAG_QUERY_REWRITE_BACKEND", query_rewrite_backend, {"llm"})
    embedding_dim = _env_int("EMBEDDING_DIM", 1024)
    milvus_uri = (
        os.environ.get("RAG_MILVUS_URI")
        or os.environ.get("MILVUS_URI")
        or str(DEFAULT_MILVUS_DB)
    )

    return RagConfig(
        milvus_uri=milvus_uri,
        milvus_token=os.environ.get("MILVUS_TOKEN") or None,
        collection_name=os.environ.get("RAG_COLLECTION", "rag_chunks_v1"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_backend=embedding_backend,
        embedding_dim=embedding_dim,
        embedding_batch_size=_env_int("RAG_EMBED_BATCH_SIZE", 8),
        embedding_max_length=_env_int("RAG_EMBED_MAX_LENGTH", 8192),
        rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_backend=rerank_backend,
        rerank_batch_size=_env_int("RAG_RERANK_BATCH_SIZE", 8),
        rerank_max_length=_env_int("RAG_RERANK_MAX_LENGTH", 1024),
        image_embedding_backend=image_backend,
        image_embedding_model=os.environ.get(
            "IMAGE_EMBEDDING_MODEL",
            "Qwen/Qwen3-VL-Embedding-8B" if image_backend == "siliconflow" else "disabled-image-embedding",
        ),
        image_embedding_dim=_env_int("IMAGE_EMBEDDING_DIM", embedding_dim),
        image_embedding_batch_size=_env_int("RAG_IMAGE_EMBED_BATCH_SIZE", 8),
        model_device=os.environ.get("RAG_MODEL_DEVICE", "auto").lower(),
        model_dtype=os.environ.get("RAG_MODEL_DTYPE", "auto").lower(),
        llm_base_url=_newapi_base_url(),
        llm_api_key=os.environ.get("NEW_API_KEY") or None,
        llm_model=os.environ.get("LLM_MODEL", "gemini-3-flash-preview"),
        siliconflow_base_url=os.environ.get(
            "SILICONFLOW_URL",
            "https://api.siliconflow.cn",
        ).rstrip("/"),
        siliconflow_api_key=os.environ.get("SILICONFLOW_API_KEY") or None,
        answer_backend=answer_backend,
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
        query_rewrite_backend=query_rewrite_backend,
        query_rewrite_history_turns=_env_int("RAG_QUERY_REWRITE_HISTORY_TURNS", 6),
        query_rewrite_max_tokens=_env_int("RAG_QUERY_REWRITE_MAX_TOKENS", 256),
        require_auth_context=_env_bool("RAG_REQUIRE_AUTH_CONTEXT", False),
        api_token=os.environ.get("RAG_API_TOKEN") or None,
        dense_hnsw_m=_env_int("RAG_DENSE_HNSW_M", 16),
        dense_hnsw_ef_construction=_env_int("RAG_DENSE_HNSW_EF_CONSTRUCTION", 100),
        dense_search_ef=_env_int("RAG_DENSE_SEARCH_EF", 128),
        image_hnsw_m=_env_int("RAG_IMAGE_HNSW_M", 16),
        image_hnsw_ef_construction=_env_int("RAG_IMAGE_HNSW_EF_CONSTRUCTION", 100),
        image_search_ef=_env_int("RAG_IMAGE_SEARCH_EF", 128),
        sparse_drop_ratio_build=_env_float("RAG_SPARSE_DROP_RATIO_BUILD", 0.2),
        sparse_drop_ratio_search=_env_float("RAG_SPARSE_DROP_RATIO_SEARCH", 0.0),
    )
