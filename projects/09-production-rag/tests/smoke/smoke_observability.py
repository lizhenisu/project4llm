from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.io import read_jsonl
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions
from serve import create_app


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "observability.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_observability"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        os.environ["RAG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        try:
            run_smoke(Path(os.environ["RAG_RUNTIME_DIR"]))
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)
            restore_env("RAG_RUNTIME_DIR", old_runtime)


def run_smoke(runtime_dir: Path) -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    doc = SourceDocument(
        tenant_id="team_a",
        doc_id="observability-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://observability-runbook",
        title="Observability Runbook",
        text="RAG 排障需要记录 raw hits、rerank hits、final context 和 LLM latency。",
        acl_groups=["ops"],
    )
    chunks = chunk_document(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    text_model = build_embedding_model(config)
    dense_vectors = text_model.encode([chunk.text for chunk in chunks])
    zero_image = zero_image_vector(config)
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

    api = TestClient(create_app())
    search_response = api.post(
        "/search",
        json={
            "query": "RAG 排障要记录什么",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-observability-search",
        },
    )
    assert search_response.status_code == 200, search_response.text
    assert search_response.json()["trace"]["stage_latency_ms"]["milvus_search"] >= 0

    query_response = api.post(
        "/query",
        json={
            "query": "RAG 排障要记录什么",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-observability-query",
        },
    )
    assert query_response.status_code == 200, query_response.text
    assert query_response.json()["trace"]["context_count"] > 0

    retrieval_events = read_jsonl(runtime_dir / "retrieval_events.jsonl")
    answer_events = read_jsonl(runtime_dir / "answer_events.jsonl")
    retrieval_event = retrieval_events[-1]
    answer_event = answer_events[-1]
    assert retrieval_event["raw_hits"]
    assert retrieval_event["rerank_hits"]
    assert retrieval_event["final_context"]
    assert "text_preview" in retrieval_event["final_context"][0]
    assert answer_event["llm"]["llm_model"]
    assert answer_event["llm"]["latency_ms"] >= 0
    assert answer_event["trace"]["stage_latency_ms"]["rerank"] >= 0
    print("smoke_observability=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
