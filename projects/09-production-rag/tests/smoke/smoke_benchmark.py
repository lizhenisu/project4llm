from __future__ import annotations

import os
import tempfile
from pathlib import Path

from benchmark_latency import benchmark_query
from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import publish_current_versions


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_rewrite_backend = os.environ.get("RAG_QUERY_REWRITE_BACKEND")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "benchmark.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_benchmark"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_QUERY_REWRITE_BACKEND"] = "llm"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("RAG_QUERY_REWRITE_BACKEND", old_rewrite_backend)
    print("smoke_benchmark=ok")


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)
    text_model = build_embedding_model(config)

    text_doc = SourceDocument(
        tenant_id="team_a",
        doc_id="latency-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://latency-runbook",
        title="延迟排障",
        text="RAG 检索变慢时，需要检查 query rewrite、embedding、Milvus search、rerank 和 context packing。",
        acl_groups=["ops"],
    )
    text_chunks = chunk_document(
        text_doc,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    zero_image = zero_image_vector(config)
    image_chunk = Chunk(
        tenant_id="team_a",
        doc_id="dashboard-screenshot",
        doc_version=1,
        chunk_index=0,
        source_type="image",
        source_uri="memory://rag-dashboard.png",
        title="RAG 监控面板截图",
        text=(
            "标题路径: RAG 监控面板截图\n"
            "来源: image\n"
            "OCR:\n"
            "RAG Dashboard p95 latency recall@50 rerank latency error rate\n"
            "图片描述:\n"
            "一张 RAG 线上监控面板截图，展示检索延迟、召回率、rerank 耗时和错误率。"
        ),
        language="zh",
        acl_groups=["ops"],
        metadata={
            "image_uri": "memory://rag-dashboard.png",
            "caption": "一张 RAG 线上监控面板截图，展示检索延迟、召回率、rerank 耗时和错误率。",
            "ocr_text": "RAG Dashboard p95 latency recall@50 rerank latency error rate",
        },
    )
    upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=[
            *[
                chunk_to_entity(
                    chunk,
                    dense_vector=text_model.encode([chunk.text])[0],
                    image_vector=zero_image,
                    embedding_model=text_model.model_name,
                    embedding_dim=text_model.dim,
                )
                for chunk in text_chunks
            ],
            chunk_to_entity(
                image_chunk,
                dense_vector=text_model.encode([image_chunk.text])[0],
                image_vector=image_model.encode(
                    [f"{image_chunk.source_uri}\n{image_chunk.title}\n{image_chunk.text}"]
                )[0],
                embedding_model=text_model.model_name,
                embedding_dim=text_model.dim,
            ),
        ],
    )
    publish_current_versions(
        config.object_store_dir,
        [
            text_doc,
            SourceDocument(
                tenant_id=image_chunk.tenant_id,
                doc_id=image_chunk.doc_id,
                doc_version=image_chunk.doc_version,
                source_type=image_chunk.source_type,
                source_uri=image_chunk.source_uri,
                title=image_chunk.title,
                text=image_chunk.text,
                language=image_chunk.language,
                acl_groups=image_chunk.acl_groups,
                metadata=image_chunk.metadata,
            ),
        ],
    )

    text_metrics = benchmark_query(
        query="它呢",
        query_mode="text",
        tenant_id="team_a",
        acl_groups=["ops"],
        source_types=["md"],
        history=["RAG 检索变慢时应该排查什么"],
        candidate_limit=5,
        context_limit=3,
        runs=2,
    )
    assert text_metrics["query_mode"] == "text"
    text_stages = text_metrics["stage_latency_ms"]
    assert "rewrite" in text_stages
    assert "embedding" in text_stages
    assert "milvus_search" in text_stages
    assert "rerank" in text_stages
    assert "context_pack" in text_stages
    assert "answer" in text_stages
    assert "total" in text_stages

    multimodal_metrics = benchmark_query(
        query="它呢",
        query_mode="multimodal",
        tenant_id="team_a",
        acl_groups=["ops"],
        source_types=["image"],
        history=["RAG Dashboard latency recall"],
        candidate_limit=5,
        context_limit=3,
        runs=2,
    )
    assert multimodal_metrics["query_mode"] == "multimodal"
    multimodal_stages = multimodal_metrics["stage_latency_ms"]
    assert "rewrite" in multimodal_stages
    assert "text_search" in multimodal_stages
    assert "image_search" in multimodal_stages
    assert "fusion" in multimodal_stages
    assert "context_pack" in multimodal_stages
    assert "answer" in multimodal_stages
    assert "total" in multimodal_stages


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
