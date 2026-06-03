from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    previous = os.environ.get("RAG_QUERY_REWRITE_BACKEND")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "rewrite.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_rewrite"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_QUERY_REWRITE_BACKEND"] = "heuristic"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("RAG_QUERY_REWRITE_BACKEND", previous)

    print("smoke_rewrite=ok")


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    doc = SourceDocument(
        tenant_id="team_a",
        doc_id="rewrite-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://rewrite-runbook",
        title="Rewrite Runbook",
        text="用户在问 RAG 检索延迟升高的问题，排查时先看 Milvus search 和 rerank。",
        acl_groups=["ops"],
    )
    chunks = chunk_document(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
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
    publish_current_versions(config.object_store_dir, [doc])

    result = retrieve_and_rerank(
        "怎么排查？",
        history=["用户在问 RAG 检索延迟升高的问题"],
        tenant_id="team_a",
        acl_groups=["ops"],
        candidate_limit=5,
        context_limit=3,
        request_id="smoke-rewrite",
    )
    assert result.trace.original_query == "怎么排查？"
    assert "RAG" in result.trace.rewritten_query
    assert result.trace.rewrite_backend == "heuristic"
    assert result.hits


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
