from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions
from serve import create_app


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "source_filter.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_source_filter"
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
            doc_id="source-md",
            doc_version=1,
            source_type="md",
            source_uri="memory://source-md",
            title="Markdown Source",
            text="source filter exact phrase shared evidence markdown only",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="source-image",
            doc_version=1,
            source_type="image",
            source_uri="memory://source-image",
            title="Image Source",
            text="source filter exact phrase shared evidence image only",
            acl_groups=["ops"],
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
    publish_current_versions(config.object_store_dir, docs)

    md_result = retrieve_and_rerank(
        "source filter exact phrase shared evidence",
        tenant_id="team_a",
        acl_groups=["ops"],
        source_types=["md"],
        candidate_limit=10,
        context_limit=5,
    )
    assert md_result.hits
    assert {hit.source_type for hit in md_result.hits} == {"md"}
    assert md_result.trace.source_types == ["md"]
    assert 'source_type in ["md"]' in md_result.trace.filter_expr

    api = TestClient(create_app())
    image_response = api.post(
        "/search",
        json={
            "query": "source filter exact phrase shared evidence",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "source_types": ["image"],
            "candidate_limit": 10,
            "context_limit": 5,
        },
    )
    assert image_response.status_code == 200, image_response.text
    body = image_response.json()
    assert body["hits"]
    assert {hit["source_type"] for hit in body["hits"]} == {"image"}
    assert body["trace"]["source_types"] == ["image"]
    print("smoke_source_filter=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
