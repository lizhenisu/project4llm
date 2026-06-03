from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_image_documents, load_source_documents, read_jsonl
from rag_core.milvus_store import (
    build_filter_expr,
    chunk_to_entity,
    connect,
    ensure_collection,
    hybrid_search,
    image_search,
    upsert_entities,
)
from rag_core.rerankers import build_reranker
from rag_core.text_utils import chunk_document, sparse_embedding
from rag_core.types import Chunk


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "e2e.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_e2e"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
 

def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    text_model = build_embedding_model(config)
    image_model = build_image_embedding_model(config)

    docs = load_source_documents(config_path("sample_docs.jsonl"))
    text_chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(
            doc,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
    ]

    image_docs = load_image_documents(config_path("sample_images.jsonl"))
    image_chunks = [
        Chunk(
            tenant_id=doc.tenant_id,
            doc_id=doc.doc_id,
            doc_version=doc.doc_version,
            chunk_index=0,
            source_type="image",
            source_uri=doc.source_uri,
            title=doc.title,
            text=(
                f"标题路径: {doc.title}\n来源: image\nOCR:\n{doc.ocr_text}\n"
                f"图片描述:\n{doc.caption}"
            ),
            language=doc.language,
            acl_groups=doc.acl_groups,
            metadata=doc.metadata | {"caption": doc.caption, "ocr_text": doc.ocr_text},
        )
        for doc in image_docs
    ]

    all_chunks = [*text_chunks, *image_chunks]
    dense_vectors = text_model.encode([chunk.text for chunk in all_chunks])
    image_vectors = image_model.encode(
        [
            f"{chunk.source_uri}\n{chunk.title}\n{chunk.text}"
            if chunk.source_type == "image"
            else "no image"
            for chunk in all_chunks
        ]
    )
    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=image_vector,
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk, dense_vector, image_vector in zip(
            all_chunks, dense_vectors, image_vectors, strict=True
        )
    ]
    upserted = upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=entities,
    )
    print(f"upserted={upserted}")

    query = "RAG 检索变慢时应该排查哪些环节？"
    query_vector = text_model.encode([query])[0]
    filter_expr = build_filter_expr(
        tenant_id="team_a",
        allowed_acl_groups=["support"],
    )
    hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=query_vector,
        query_sparse=sparse_embedding(query),
        filter_expr=filter_expr,
        limit=5,
    )
    reranked = build_reranker(config).rerank(query, hits, limit=3)
    print("hybrid_rerank=", [hit.doc_id for hit in reranked])
    assert reranked and reranked[0].doc_id == "rag-runbook"

    image_query_vector = image_model.encode(["RAG Dashboard latency recall"])[0]
    image_hits = image_search(
        client,
        collection_name=config.collection_name,
        image_query_vector=image_query_vector,
        filter_expr=build_filter_expr(
            tenant_id="team_a",
            allowed_acl_groups=["ops"],
            source_types=["image"],
        ),
        limit=3,
    )
    print("image_hits=", [hit.doc_id for hit in image_hits])
    assert image_hits and image_hits[0].doc_id == "dashboard-screenshot"

    eval_rows = read_jsonl(config_path("eval_queries.jsonl"))
    assert eval_rows
    print("smoke_e2e=ok")


def config_path(filename: str):
    from rag_core.config import DATA_DIR

    return DATA_DIR / filename


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
