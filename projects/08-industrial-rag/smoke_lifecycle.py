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
from rag_core.text_utils import chunk_document, sparse_embedding
from rag_core.types import SourceDocument


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "lifecycle.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_lifecycle"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="release-note",
            doc_version=1,
            source_type="json",
            source_uri="memory://release-note-v1",
            title="Release Note v1",
            text="v1 文档说明 hybrid search 只启用 dense recall。",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="release-note",
            doc_version=2,
            source_type="json",
            source_uri="memory://release-note-v2",
            title="Release Note v2",
            text="v2 文档说明 hybrid search 启用 dense sparse rerank 全链路。",
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

    query = "dense sparse rerank 全链路"
    version_2_hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=text_model.encode([query])[0],
        query_sparse=sparse_embedding(query),
        filter_expr=build_filter_expr(
            tenant_id="team_a",
            allowed_acl_groups=["ops"],
            doc_version=2,
        ),
        limit=3,
    )
    assert version_2_hits
    assert all(hit.doc_id == "release-note" for hit in version_2_hits)

    missing_version_hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=text_model.encode([query])[0],
        query_sparse=sparse_embedding(query),
        filter_expr=build_filter_expr(
            tenant_id="team_a",
            allowed_acl_groups=["ops"],
            doc_version=999,
        ),
        limit=3,
    )
    assert missing_version_hits == []
    print("smoke_lifecycle=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
