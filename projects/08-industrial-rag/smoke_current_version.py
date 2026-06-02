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
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RAG_COLLECTION"] = "rag_smoke_current_version"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        try:
            run_smoke()
        finally:
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="versioned-runbook",
            doc_version=1,
            source_type="md",
            source_uri="memory://versioned-runbook-v1",
            title="Versioned Runbook v1",
            text="旧版本说明只检查 legacy dense recall。",
            acl_groups=["ops"],
        ),
        SourceDocument(
            tenant_id="team_a",
            doc_id="versioned-runbook",
            doc_version=2,
            source_type="md",
            source_uri="memory://versioned-runbook-v2",
            title="Versioned Runbook v2",
            text="当前版本说明检查 current sparse rerank release gate。",
            acl_groups=["ops"],
        ),
    ]
    publish_current_versions(config.object_store_dir, docs)

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

    default_result = retrieve_and_rerank(
        "legacy dense current sparse rerank release gate",
        tenant_id="team_a",
        acl_groups=["ops"],
        candidate_limit=10,
        context_limit=5,
        request_id="smoke-current-version-default",
    )
    assert default_result.hits
    assert {hit.source_uri for hit in default_result.hits} == {
        "memory://versioned-runbook-v2"
    }
    assert default_result.trace.current_versions == {"versioned-runbook": 2}

    historical_result = retrieve_and_rerank(
        "legacy dense recall",
        tenant_id="team_a",
        acl_groups=["ops"],
        doc_version=1,
        candidate_limit=10,
        context_limit=5,
        request_id="smoke-current-version-history",
    )
    assert historical_result.hits
    assert {hit.source_uri for hit in historical_result.hits} == {
        "memory://versioned-runbook-v1"
    }
    assert historical_result.trace.current_versions == {}
    print("smoke_current_version=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
