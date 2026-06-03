from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rag_core.config import RagConfig
from rag_core.milvus_store import connect


REQUIRED_FIELDS = {
    "id",
    "tenant_id",
    "doc_id",
    "doc_version",
    "chunk_index",
    "source_type",
    "source_uri",
    "title",
    "text",
    "language",
    "acl_groups",
    "created_at",
    "updated_at",
    "is_active",
    "embedding_model",
    "embedding_dim",
    "content_hash",
    "text_dense_vector",
    "bm25_sparse_vector",
    "image_dense_vector",
    "metadata",
}

SECRET_KEYS = {"milvus_token", "llm_api_key", "api_token"}


def readiness_report(config: RagConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "ok",
        "collection": config.collection_name,
        "milvus_uri": config.milvus_uri,
        "embedding_backend": config.embedding_backend,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "image_embedding_backend": config.image_embedding_backend,
        "image_embedding_dim": config.image_embedding_dim,
        "rerank_backend": config.rerank_backend,
        "require_auth_context": config.require_auth_context,
        "checks": {},
    }
    checks = report["checks"]

    try:
        client = connect(config)
        checks["milvus_connect"] = {"ok": True}
    except Exception as exc:  # pragma: no cover - exercised in real deployment smoke
        checks["milvus_connect"] = {"ok": False, "error": str(exc)}
        report["status"] = "error"
        return report

    try:
        collection_exists = client.has_collection(config.collection_name)
        checks["collection_exists"] = {"ok": collection_exists}
        if not collection_exists:
            report["status"] = "error"
            return report

        description = client.describe_collection(config.collection_name)
        field_map = extract_field_map(description)
        missing_fields = sorted(REQUIRED_FIELDS - set(field_map))
        text_dim = extract_dim(field_map.get("text_dense_vector", {}))
        image_dim = extract_dim(field_map.get("image_dense_vector", {}))
        text_analyzer_enabled = extract_bool_param(
            field_map.get("text", {}),
            "enable_analyzer",
        )
        schema_ok = (
            not missing_fields
            and text_dim == config.embedding_dim
            and image_dim == config.image_embedding_dim
            and text_analyzer_enabled is True
        )
        checks["schema"] = {
            "ok": schema_ok,
            "missing_fields": missing_fields,
            "text_dense_vector_dim": text_dim,
            "image_dense_vector_dim": image_dim,
            "text_analyzer_enabled": text_analyzer_enabled,
        }
        if not schema_ok:
            report["status"] = "error"
    except Exception as exc:  # pragma: no cover - exercised in real deployment smoke
        checks["schema"] = {"ok": False, "error": str(exc)}
        report["status"] = "error"

    return report


def extract_field_map(description: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields = description.get("fields") or description.get("schema", {}).get("fields") or []
    return {field["name"]: field for field in fields if "name" in field}


def extract_dim(field: dict[str, Any]) -> int | None:
    params = field.get("params") or field.get("type_params") or {}
    dim = params.get("dim")
    if dim is None:
        return None
    return int(dim)


def extract_bool_param(field: dict[str, Any], name: str) -> bool | None:
    params = field.get("params") or field.get("type_params") or {}
    value = params.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    if value is None:
        return None
    return bool(value)


def redacted_config(config: RagConfig) -> dict[str, Any]:
    values = asdict(config)
    return {
        key: ("***" if key in SECRET_KEYS and value else value)
        for key, value in values.items()
    }
