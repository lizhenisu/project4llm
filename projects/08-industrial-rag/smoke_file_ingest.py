from __future__ import annotations

import tempfile
import os
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_file_documents
from rag_core.milvus_store import (
    build_filter_expr,
    chunk_to_entity,
    connect,
    ensure_collection,
    hybrid_search,
    upsert_entities,
)
from rag_core.text_utils import chunk_document, sparse_embedding


def main() -> None:
    old_collection = os.environ.get("RAG_COLLECTION")
    os.environ["RAG_COLLECTION"] = "rag_smoke_file_ingest"
    try:
        run_smoke()
    finally:
        restore_env("RAG_COLLECTION", old_collection)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "runbook.html").write_text(
            """
            <html>
              <head><title>Latency Runbook</title></head>
              <body>
                <h1>RAG 延迟排障</h1>
                <p>先检查 embedding batch，再检查 Milvus hybrid search 和 rerank。</p>
              </body>
            </html>
            """,
            encoding="utf-8",
        )
        (root / "policy.txt").write_text(
            "退款材料需要订单号、付款凭证和问题截图。",
            encoding="utf-8",
        )

        docs = load_file_documents(
            root,
            tenant_id="team_a",
            doc_version=7,
            acl_groups=["ops"],
        )

    assert {doc.source_type for doc in docs} == {"html", "txt"}
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
    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=zero_image,
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
    ]
    upsert_entities(client, collection_name=config.collection_name, entities=entities)

    query = "Milvus hybrid search rerank 延迟怎么排查"
    hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=text_model.encode([query])[0],
        query_sparse=sparse_embedding(query),
        filter_expr=build_filter_expr(
            tenant_id="team_a",
            allowed_acl_groups=["ops"],
            doc_version=7,
        ),
        limit=3,
    )
    assert hits and hits[0].doc_id == "runbook"
    print("smoke_file_ingest=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
