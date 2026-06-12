from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import publish_current_versions
from search_multimodal import run_multimodal_search


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "multimodal_search.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_multimodal_search"
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
    text_model = build_embedding_model(config)
    image_model = build_image_embedding_model(config)

    chunk = Chunk(
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
            "caption": "一张 RAG 线上监控面板截图，展示检索延迟、召回率、rerank 耗时和错误率。",
            "ocr_text": "RAG Dashboard p95 latency recall@50 rerank latency error rate",
            "image_uri": "memory://rag-dashboard.png",
            "bbox": [],
            "linked_doc_id": "rag-runbook",
        },
    )
    upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=[
            chunk_to_entity(
                chunk,
                dense_vector=text_model.encode([chunk.text])[0],
                image_vector=image_model.encode(
                    [f"{chunk.source_uri}\n{chunk.title}\n{chunk.text}"]
                )[0],
                embedding_model=text_model.model_name,
                embedding_dim=text_model.dim,
            )
        ],
    )
    publish_current_versions(
        config.object_store_dir,
        [
            SourceDocument(
                tenant_id=chunk.tenant_id,
                doc_id=chunk.doc_id,
                doc_version=chunk.doc_version,
                source_type=chunk.source_type,
                source_uri=chunk.source_uri,
                title=chunk.title,
                text=chunk.text,
                language=chunk.language,
                acl_groups=chunk.acl_groups,
                metadata=chunk.metadata,
            )
        ],
    )

    hits = run_multimodal_search(
        "RAG Dashboard latency recall",
        tenant_id="team_a",
        acl_groups=["ops"],
        limit=3,
    )
    assert hits
    assert hits[0].doc_id == "dashboard-screenshot"
    fusion = hits[0].metadata["fusion"]
    assert "text_hybrid" in fusion["channels"]
    assert "image_vector" in fusion["channels"]
    assert hits[0].metadata["linked_doc_id"] == "rag-runbook"
    print("smoke_multimodal_search=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
