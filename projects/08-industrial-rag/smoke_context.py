from __future__ import annotations

import os
import tempfile
from pathlib import Path

from answer import answer_query
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
    previous = os.environ.get("RAG_MIN_RERANK_SCORE")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "context.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_context"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_MIN_RERANK_SCORE"] = "999"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("RAG_MIN_RERANK_SCORE", previous)

    print("smoke_context=ok")


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    doc = SourceDocument(
        tenant_id="team_a",
        doc_id="context-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://context-runbook",
        title="Context Runbook",
        text="RAG 检索变慢时应该检查 rewrite、embedding、Milvus search 和 rerank。",
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

    retrieval = retrieve_and_rerank(
        "RAG 检索变慢时应该排查什么",
        tenant_id="team_a",
        acl_groups=["ops"],
        candidate_limit=5,
        context_limit=3,
        request_id="smoke-context",
    )
    assert retrieval.trace.dropped_by_score > 0
    assert retrieval.trace.context_count == 0

    result = answer_query(
        "RAG 检索变慢时应该排查什么",
        tenant_id="team_a",
        acl_groups=["ops"],
        candidate_limit=5,
        context_limit=3,
    )
    assert result.answer == "当前知识库没有足够证据。"


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
