from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import read_jsonl
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rag_core.versioning import publish_current_versions
from serve import create_app


RAW_EMAIL = "leak@example.com"
RAW_PHONE = "13800138000"
RAW_KEY = "ak-testSECRETSECRET1234"


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    old_runtime = os.environ.get("RAG_RUNTIME_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "event_redaction.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_event_redaction"
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
        doc_id="redaction-runbook",
        doc_version=1,
        source_type="md",
        source_uri="memory://redaction-runbook",
        title="Redaction Runbook",
        text=f"联系 {RAW_EMAIL} 或 {RAW_PHONE} 排查日志脱敏。",
        acl_groups=["ops"],
        metadata={"owner_email": RAW_EMAIL, "debug_key": RAW_KEY},
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

    api = TestClient(create_app())
    search = api.post(
        "/search",
        json={
            "query": f"请联系 {RAW_EMAIL} {RAW_PHONE} {RAW_KEY}",
            "tenant_id": "team_a",
            "acl_groups": ["ops"],
            "candidate_limit": 5,
            "context_limit": 3,
            "request_id": "smoke-event-redaction-search",
        },
    )
    assert search.status_code == 200, search.text

    feedback = api.post(
        "/feedback",
        json={
            "request_id": "smoke-event-redaction-search",
            "rating": 1,
            "comment": f"user comment {RAW_EMAIL} {RAW_PHONE} {RAW_KEY}",
            "selected_doc_ids": ["redaction-runbook"],
        },
    )
    assert feedback.status_code == 200, feedback.text

    event_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            runtime_dir / "retrieval_events.jsonl",
            runtime_dir / "feedback_events.jsonl",
        ]
    )
    assert RAW_EMAIL not in event_text
    assert RAW_PHONE not in event_text
    assert RAW_KEY not in event_text
    assert "[REDACTED_EMAIL]" in event_text
    assert "[REDACTED_PHONE_CN]" in event_text
    assert "[REDACTED_API_KEY]" in event_text

    retrieval_event = read_jsonl(runtime_dir / "retrieval_events.jsonl")[-1]
    assert retrieval_event["final_context"]
    assert RAW_EMAIL not in retrieval_event["final_context"][0]["text_preview"]
    assert RAW_EMAIL not in str(retrieval_event["final_context"][0]["metadata"])
    print("smoke_event_redaction=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
