from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import (
    build_filter_expr,
    chunk_to_entity,
    connect,
    ensure_collection,
    hybrid_search,
    upsert_entities,
)
from rag_core.object_store import archive_source_documents, load_archived_source_documents
from rag_core.text_utils import chunk_document, sparse_embedding
from rag_core.types import SourceDocument


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "object_store_rebuild.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_object_rebuild"
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
            doc_id="archive-runbook",
            doc_version=1,
            source_type="md",
            source_uri="object://raw/archive-runbook.md",
            title="Archive Runbook",
            text="对象存储保存 canonical text，Milvus 只是索引，可以从归档重建。",
            acl_groups=["ops"],
            metadata={"object_key": "raw/archive-runbook.md"},
        )
    ]
    archived = archive_source_documents(config.object_store_dir, docs, replace=True)
    assert archived == 1
    assert load_archived_source_documents(config.object_store_dir)[0].doc_id == "archive-runbook"

    ensure_collection(client, config, reset=True)
    archived_docs = load_archived_source_documents(config.object_store_dir)
    chunks = [
        chunk
        for doc in archived_docs
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

    query = "Milvus 从 canonical text 归档重建"
    hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=text_model.encode([query])[0],
        query_sparse=sparse_embedding(query),
        filter_expr=build_filter_expr(
            tenant_id="team_a",
            allowed_acl_groups=["ops"],
        ),
        limit=3,
    )
    assert hits and hits[0].doc_id == "archive-runbook"
    print("smoke_object_store_rebuild=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
