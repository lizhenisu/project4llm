from __future__ import annotations

import os
import tempfile
from pathlib import Path

import rebuild_from_object_store
from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, zero_image_vector
from rag_core.milvus_store import (
    build_filter_expr,
    chunk_to_entity,
    connect,
    ensure_collection,
    hybrid_search,
    upsert_entities,
)
from rag_core.object_store import archive_source_documents, load_archived_source_documents
from rag_core.text_utils import chunk_document
from rag_core.types import SourceDocument
from rebuild_from_object_store import chunks_for_rebuild, image_vectors_for_rebuild


def main() -> None:
    old_milvus_uri = os.environ.get("MILVUS_URI")
    old_collection = os.environ.get("RAG_COLLECTION")
    old_object_store = os.environ.get("RAG_OBJECT_STORE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MILVUS_URI"] = str(Path(tmp) / "object_store_rebuild.db")
        os.environ["RAG_COLLECTION"] = "rag_smoke_object_rebuild"
        os.environ["RAG_OBJECT_STORE_DIR"] = str(Path(tmp) / "object_store")
        try:
            run_smoke()
            test_rebuild_encodes_archived_image_vectors(Path(tmp))
        finally:
            restore_env("MILVUS_URI", old_milvus_uri)
            restore_env("RAG_COLLECTION", old_collection)
            restore_env("RAG_OBJECT_STORE_DIR", old_object_store)


def run_smoke() -> None:
    config = load_config()
    client = connect(config)
    ensure_collection(client, config, reset=True)

    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="archive-runbook",
            doc_version=1,
            source_type="md",
            source_uri="object://raw/archive-runbook.md",
            title="Archive Runbook",
            text="对象存储保存 canonical text，Milvus 只是索引，可以从归档重建。",
            acl_groups=["ops"],
            metadata={"object_key": "raw/archive-runbook.md"},
        )
    ]
    archived = archive_source_documents(config.object_store_dir, docs, replace=True)
    assert archived == 1
    assert load_archived_source_documents(config.object_store_dir)[0].doc_id == "archive-runbook"

    ensure_collection(client, config, reset=True)
    archived_docs = load_archived_source_documents(config.object_store_dir)
    chunks = [
        chunk
        for doc in archived_docs
        for chunk in chunk_document(
            doc,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
    ]
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

    query = "Milvus 从 canonical text 归档重建"
    hits = hybrid_search(
        client,
        collection_name=config.collection_name,
        query_vector=text_model.encode([query])[0],
        query_text=query,
        filter_expr=build_filter_expr(
            tenant_id="team_a",
            allowed_acl_groups=["ops"],
        ),
        limit=3,
    )
    assert hits and hits[0].doc_id == "archive-runbook"
    print("smoke_object_store_rebuild=ok")


def test_rebuild_encodes_archived_image_vectors(tmp_path: Path) -> None:
    config = load_config()
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"fake-png")
    docs = [
        SourceDocument(
            tenant_id="team_a",
            doc_id="paper/page-1",
            doc_version=1,
            source_type="pdf",
            source_uri="object://raw/paper.pdf",
            title="Paper p1",
            text="页面正文",
            acl_groups=["ops"],
            metadata={
                "page_no": 1,
                "display_blocks": [
                    {
                        "type": "image",
                        "title": "Figure 1",
                        "path": str(image_path),
                        "media_type": "image/png",
                    }
                ],
            },
        )
    ]
    text_model = build_embedding_model(config)
    chunks = chunks_for_rebuild(config=config, docs=docs, text_model=text_model)
    assert any(chunk.source_type == "image" for chunk in chunks)

    old_builder = rebuild_from_object_store.build_image_embedding_model
    try:
        rebuild_from_object_store.build_image_embedding_model = lambda _config: FakeImageEmbeddingModel(config.image_embedding_dim)
        vectors = image_vectors_for_rebuild(config=config, chunks=chunks)
    finally:
        rebuild_from_object_store.build_image_embedding_model = old_builder
    image_vector = vectors[[chunk.source_type for chunk in chunks].index("image")]
    assert image_vector == [0.5] * config.image_embedding_dim


class FakeImageEmbeddingModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [[0.5] * self.dim for _ in image_paths]


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    main()
