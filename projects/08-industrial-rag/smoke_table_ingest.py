from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.io import load_table_documents
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import archive_source_documents
from rag_core.pipeline import retrieve_and_rerank
from rag_core.text_utils import chunk_document
from rag_core.versioning import publish_current_versions


def main() -> None:
    old_env = {
        "MILVUS_URI": os.environ.get("MILVUS_URI"),
        "RAG_COLLECTION": os.environ.get("RAG_COLLECTION"),
        "RAG_OBJECT_STORE_DIR": os.environ.get("RAG_OBJECT_STORE_DIR"),
    }
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as object_store:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "table_ingest.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_table_ingest"
        os.environ["RAG_OBJECT_STORE_DIR"] = object_store
        try:
            run_smoke()
        finally:
            for name, value in old_env.items():
                restore_env(name, value)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "ticket_sla.csv").write_text(
            "\n".join(
                [
                    "sku,priority,sla_minutes,owner",
                    "sku_neon_42,gold,15,ops_table_team",
                    "sku_basic_09,standard,120,support_team",
                    "sku_urgent_77,platinum,5,incident_team",
                ]
            ),
            encoding="utf-8",
        )

        docs = load_table_documents(
            root,
            tenant_id="team_a",
            doc_version=3,
            acl_groups=["ops"],
            rows_per_document=2,
        )

    assert len(docs) == 2
    assert docs[0].doc_id == "ticket_sla/part-000"
    assert docs[0].source_type == "csv"
    assert "| sku | priority | sla_minutes | owner |" in docs[0].text
    assert "sku_neon_42" in docs[0].text
    assert docs[0].metadata["columns"] == ["sku", "priority", "sla_minutes", "owner"]
    assert docs[0].metadata["row_count"] == 2
    assert docs[1].metadata["row_start"] == 3

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
    archived = archive_source_documents(config.object_store_dir, docs)
    current = publish_current_versions(config.object_store_dir, docs)
    assert archived == 2
    assert current["team_a"]["ticket_sla/part-000"] == 3

    result = retrieve_and_rerank(
        "sku_neon_42 gold sla_minutes ops_table_team",
        tenant_id="team_a",
        acl_groups=["ops"],
        doc_version=3,
        source_types=["csv"],
        candidate_limit=5,
        context_limit=3,
    )
    assert result.hits
    assert result.hits[0].doc_id == "ticket_sla/part-000"
    assert {hit.source_type for hit in result.hits} == {"csv"}
    assert result.trace.source_types == ["csv"]
    print("smoke_table_ingest=ok")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
