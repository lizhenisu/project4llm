from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions
from serve import create_app


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "auth_context.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_auth_context"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="team-a-runbook",
            doc_version=1,
            source_type="json",
            source_uri="memory://team-a-runbook",
            title="Team A Runbook",
            text="team_a 的 RAG runbook 要检查 embedding、Milvus 和 rerank。",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_b",
            doc_id="team-b-secret",
            doc_version=1,
            source_type="json",
            source_uri="memory://team-b-secret",
            title="Team B Secret",
            text="team_b 的私有报销规则不可被 team_a 查询。",
            acl_groups=["finance"],
        ),
    ]
    chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(
            doc,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
    ]
    text_model = build_embedding_model(config)
    image_model = build_image_embedding_model(config)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = image_model.encode(["no image"])[0]
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
    publish_current_versions(config.object_store_dir, docs)

    old_require = os.environ.get("RAG_REQUIRE_AUTH_CONTEXT")
    old_token = os.environ.get("RAG_API_TOKEN")
    os.environ["RAG_REQUIRE_AUTH_CONTEXT"] = "1"
    os.environ["RAG_API_TOKEN"] = "dev-secret"
    try:
        api = TestClient(create_app())
        unauthorized = api.post(
            "/search",
            json={"query": "RAG runbook", "tenant_id": "team_a"},
        )
        assert unauthorized.status_code == 401

        missing_headers = api.post(
            "/search",
            headers={"Authorization": "Bearer dev-secret"},
            json={"query": "RAG runbook", "tenant_id": "team_a"},
        )
        assert missing_headers.status_code == 401

        response = api.post(
            "/search",
            headers={
                "Authorization": "Bearer dev-secret",
                "X-RAG-Tenant-ID": "team_a",
                "X-RAG-ACL-Groups": "ops",
            },
            json={
                "query": "RAG runbook",
                "tenant_id": "team_b",
                "acl_groups": ["finance"],
                "candidate_limit": 5,
                "context_limit": 3,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["trace"]["tenant_id"] == "team_a"
        assert all(hit["doc_id"] != "team-b-secret" for hit in body["hits"])
        assert body["hits"] and body["hits"][0]["doc_id"] == "team-a-runbook"
    finally:
        restore_env("RAG_REQUIRE_AUTH_CONTEXT", old_require)
        restore_env("RAG_API_TOKEN", old_token)

    print("smoke_auth_context=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
