from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.types import Chunk, SourceDocument
from rag_core.versioning import publish_current_versions
from serve import create_app


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_rewrite_backend = os.environ.get("RAG_QUERY_REWRITE_BACKEND")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "api_multimodal.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_api_multimodal"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_QUERY_REWRITE_BACKEND"] = "llm"
        try:
            run_smoke()
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("RAG_QUERY_REWRITE_BACKEND", old_rewrite_backend)


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
            "image_uri": "memory://rag-dashboard.png",
            "caption": "一张 RAG 线上监控面板截图，展示检索延迟、召回率、rerank 耗时和错误率。",
            "ocr_text": "RAG Dashboard p95 latency recall@50 rerank latency error rate",
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

    api = TestClient(create_app())
    search = api.post(
        "/search",
        json={
            "query": "它呢",
            "query_mode": "multimodal",
            "history": ["RAG Dashboard latency recall"],
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "source_types": ["image"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-api-multimodal-search",
        },
    )
    assert search.status_code == 200, search.text
    search_body = search.json()
    assert search_body["request_id"] == "smoke-api-multimodal-search"
    assert search_body["hits"]
    assert search_body["trace"]["retrieval_mode"] == "multimodal_text_image_fusion"
    assert search_body["trace"]["original_query"] == "它呢"
    assert "RAG Dashboard latency recall" in search_body["trace"]["rewritten_query"]
    assert search_body["hits"][0]["doc_id"] == "dashboard-screenshot"
    assert search_body["hits"][0]["metadata"]["image_uri"] == "memory://rag-dashboard.png"
    fusion = search_body["hits"][0]["metadata"]["fusion"]
    assert "text_hybrid" in fusion["channels"]
    assert "image_vector" in fusion["channels"]

    query = api.post(
        "/query",
        json={
            "query": "它呢",
            "query_mode": "multimodal",
            "history": ["RAG Dashboard latency recall"],
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "source_types": ["image"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-api-multimodal-query",
        },
    )
    assert query.status_code == 200, query.text
    query_body = query.json()
    assert query_body["request_id"] == "smoke-api-multimodal-query"
    assert query_body["answer"]
    assert query_body["citations"]
    assert query_body["trace"]["retrieval_mode"] == "multimodal_text_image_fusion"
    assert query_body["trace"]["original_query"] == "它呢"
    assert "RAG Dashboard latency recall" in query_body["trace"]["rewritten_query"]
    assert query_body["citations"][0]["source_type"] == "image"
    assert query_body["citations"][0]["metadata"]["image_uri"] == "memory://rag-dashboard.png"
    query_fusion = query_body["citations"][0]["metadata"]["fusion"]
    assert "text_hybrid" in query_fusion["channels"]
    assert "image_vector" in query_fusion["channels"]

    print("smoke_api_multimodal=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
