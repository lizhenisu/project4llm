from __future__ import annotations

import os

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument


def main() -> None:
    old_collection = os.environ.get("RAG_COLLECTION")
    os.environ["RAG_COLLECTION"] = "rag_smoke_embedding_model_filter"
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
            doc_id="old-vector-space",
            doc_version=1,
            source_type="md",
            source_uri="memory://old-vector-space",
            title="Old Vector Space",
            text="legacy model contains exact query term stale embedding model upgrade",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="current-vector-space",
            doc_version=1,
            source_type="md",
            source_uri="memory://current-vector-space",
            title="Current Vector Space",
            text="current model contains exact query term stale embedding model upgrade",
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
    entities = []
    for chunk, dense_vector in zip(chunks, dense_vectors, strict=True):
        model_name = (
            "hash:legacy-embedding-model"
            if chunk.doc_id == "old-vector-space"
            else text_model.model_name
        )
        entities.append(
            chunk_to_entity(
                chunk,
                dense_vector=dense_vector,
                image_vector=zero_image,
                embedding_model=model_name,
                embedding_dim=text_model.dim,
            )
        )
    upsert_entities(client, collection_name=config.collection_name, entities=entities)

    result = retrieve_and_rerank(
        "stale embedding model upgrade",
        tenant_id="team_a",
        acl_groups=["ops"],
        candidate_limit=10,
        context_limit=5,
        request_id="smoke-embedding-model-filter",
    )
    assert result.hits
    assert {hit.doc_id for hit in result.candidates} == {"current-vector-space"}
    assert {hit.doc_id for hit in result.hits} == {"current-vector-space"}
    assert f'embedding_model == "{text_model.model_name}"' in result.trace.filter_expr
    assert result.trace.embedding_model == text_model.model_name
    print("smoke_embedding_model_filter=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
