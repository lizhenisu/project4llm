from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model, zero_image_vector
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    if os.environ.get("RAG_RUN_MODEL_SMOKE") != "1":
        print("smoke_models=skipped; set RAG_RUN_MODEL_SMOKE=1 to load model backends")
        return

    with tempfile.TemporaryDirectory() as tmp:
        env_overrides = {
            "RAG_MILVUS_URI": str(Path(tmp) / "model_smoke.db"),
            "RAG_OBJECT_STORE_DIR": str(Path(tmp) / "object_store"),
            "RAG_RUNTIME_DIR": str(Path(tmp) / "runtime"),
            "RAG_COLLECTION": "rag_smoke_models",
            "RAG_QUERY_REWRITE_BACKEND": "none",
        }
        with temporary_env(env_overrides):
            run_model_smoke()


def run_model_smoke() -> None:
    config = load_config()

    embedding_model = build_embedding_model(config)
    vectors = embedding_model.encode(["Milvus hybrid search smoke"])
    assert vectors and len(vectors[0]) == embedding_model.dim
    print(
        "embedding=ok "
        f"backend={config.embedding_backend} dim={embedding_model.dim} "
        f"device={config.model_device} dtype={config.model_dtype} "
        f"batch={config.embedding_batch_size}"
    )

    image_model = build_image_embedding_model(config)
    image_vectors = image_model.encode(["RAG dashboard screenshot"])
    assert image_vectors and len(image_vectors[0]) == image_model.dim
    print(
        f"image_embedding=ok backend={config.image_embedding_backend} "
        f"dim={image_model.dim} batch={config.image_embedding_batch_size}"
    )

    client = connect(config)
    try:
        ensure_collection(client, config, reset=True)
        docs = [
            SourceDocument(
                tenant_id="team_a",
                doc_id="rag-runbook",
                doc_version=1,
                source_type="md",
                source_uri="memory://rag-runbook",
                title="RAG Runbook",
                text="RAG 检索延迟升高时检查 embedding、Milvus search 和 rerank。",
                acl_groups=["ops"],
            ),
            SourceDocument(
                tenant_id="team_a",
                doc_id="refund-policy",
                doc_version=1,
                source_type="md",
                source_uri="memory://refund-policy",
                title="Refund Policy",
                text="退款申请需要提交工单，并附上合同编号和管理员邮箱。",
                acl_groups=["support"],
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
        dense_vectors = embedding_model.encode([chunk.text for chunk in chunks])
        zero_image = zero_image_vector(config)
        upsert_entities(
            client,
            collection_name=config.collection_name,
            entities=[
                chunk_to_entity(
                    chunk,
                    dense_vector=dense_vector,
                    image_vector=zero_image,
                    embedding_model=embedding_model.model_name,
                    embedding_dim=embedding_model.dim,
                )
                for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
            ],
        )
        publish_current_versions(config.object_store_dir, docs)
    finally:
        client.close()

    result = retrieve_and_rerank(
        "RAG 检索变慢排查什么",
        tenant_id="team_a",
        candidate_limit=5,
        context_limit=3,
        acl_groups=["ops"],
    )
    assert result.reranked
    assert result.hits
    assert result.reranked[0].doc_id == "rag-runbook"
    print(
        f"reranker=ok backend={config.rerank_backend} "
        f"batch={config.rerank_batch_size} "
        f"top_doc={result.reranked[0].doc_id}"
    )
    print(
        "pipeline=ok "
        f"retrieval_mode={result.trace.retrieval_mode} "
        f"stage_latency_ms={result.trace.stage_latency_ms}"
    )


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
