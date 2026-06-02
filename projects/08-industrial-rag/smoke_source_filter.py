from __future__ import annotations

import os

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from serve import create_app


def main() -> None:
    old_collection = os.environ.get("RAG_COLLECTION")
    os.environ["RAG_COLLECTION"] = "rag_smoke_source_filter"
    try:
        run_smoke()
    finally:
        restore_env("RAG_COLLECTION", old_collection)


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
