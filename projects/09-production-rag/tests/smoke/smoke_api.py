from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions
from serve import create_app


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    old_token = os.environ.get("RAG_API_TOKEN")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "api.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_api"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        os.environ["RAG_API_TOKEN"] = "smoke-token"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("RAG_RUNTIME_DIR", old_runtime)
            restore_env("RAG_API_TOKEN", old_token)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    doc = SourceDocument(
        tenant_id="team_a",
        doc_id="api-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://api-runbook",
        title="API Runbook",
        text="RAG 检索变慢时应该先检查 rewrite、Milvus search 和 rerank。",
        acl_groups=["ops"],
    )
    chunks = chunk_document(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    text_model = build_embedding_model(config)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = zero_image_vector(config)
    upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=[
            chunk_to_entity(
                chunk,
                dense_vector=dense_vector,
                image_vector=zero_image,
                embedding_model=text_model.model_name,
                embedding_dim=text_model.dim,
            )
            for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
        ],
    )
    publish_current_versions(config.object_store_dir, [doc])

    api = TestClient(create_app())
    headers = {
        "Authorization": "Bearer smoke-token",
        "X-RAG-Tenant-ID": "team_a",
        "X-RAG-ACL-Groups": "ops",
    }

    health = api.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    sources = api.get("/sources?tenant_id=team_a", headers=headers)
    assert sources.status_code == 200, sources.text
    assert sources.json()["sources"][0]["title"] == "api-runbook"

    renamed = api.patch(
        "/sources/api-runbook?tenant_id=team_a",
        headers=headers,
        json={"title": "API Runbook Renamed"},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["title"] == "API Runbook Renamed"

    sources = api.get("/sources?tenant_id=team_a", headers=headers)
    assert sources.status_code == 200, sources.text
    assert sources.json()["sources"][0]["title"] == "API Runbook Renamed"

    search = api.post(
        "/search",
        headers=headers,
        json={
            "query": "RAG 检索变慢时应该排查什么",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-api-search",
        },
    )
    assert search.status_code == 200, search.text
    search_body = search.json()
    assert search_body["request_id"] == "smoke-api-search"
    assert search_body["hits"]
    assert search_body["hits"][0]["doc_id"] == "api-runbook"
    assert search_body["trace"]["filter_expr"]

    query = api.post(
        "/query",
        headers=headers,
        json={
            "query": "RAG 检索变慢时应该排查什么",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
        },
    )
    assert query.status_code == 200, query.text
    query_body = query.json()
    assert query_body["request_id"]
    assert query_body["answer"]
    assert query_body["citations"]
    assert query_body["citations"][0]["doc_id"] == "api-runbook"

    asset_path = config.object_store_dir / "uploads" / "team_a" / "asset-smoke" / "figure.png"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x04\x00\x00\x00\xb5\x1c\x0c\x02\x00\x00\x00\x0bIDATx\xdac\xfc\xff\x1f\x00"
        b"\x03\x03\x02\x00\xef\xbf\xa7\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    asset = api.get("/source-assets/uploads/team_a/asset-smoke/figure.png?tenant_id=team_a")
    assert asset.status_code == 200, asset.text
    assert asset.headers["content-type"].startswith("image/png")
    cross_tenant_asset = api.get("/source-assets/uploads/team_a/asset-smoke/figure.png?tenant_id=other")
    assert cross_tenant_asset.status_code == 404

    feedback = api.post(
        "/feedback",
        headers=headers,
        json={
            "request_id": query_body["request_id"],
            "rating": 1,
            "comment": "smoke ok",
            "selected_doc_ids": [query_body["citations"][0]["doc_id"]],
        },
    )
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["status"] == "accepted"

    print("smoke_api=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
