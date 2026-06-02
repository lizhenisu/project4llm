from __future__ import annotations

import argparse

from rag_core.config import load_config
from rag_core.embeddings import build_embedding_model, build_image_embedding_model
from rag_core.milvus_store import chunk_to_entity, connect, ensure_collection, upsert_entities
from rag_core.object_store import load_archived_source_documents
from rag_core.text_utils import chunk_document
from rag_core.versioning import publish_current_versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Milvus index from archived canonical SourceDocument rows."
    )
    parser.add_argument("--reset", action="store_true", help="Drop collection before rebuild.")
    args = parser.parse_args()

    config = load_config()
    docs = load_archived_source_documents(config.object_store_dir)
    if not docs:
        raise SystemExit(f"No archived documents under {config.object_store_dir}")

    client = connect(config)
    ensure_collection(client, config, reset=args.reset)
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
    entities = [
        chunk_to_entity(
            chunk,
            dense_vector=dense_vector,
            image_vector=zero_image,
            embedding_model=text_model.model_name,
            embedding_dim=text_model.dim,
        )
        for chunk, dense_vector in zip(chunks, dense_vectors, strict=True)
    ]
    upserted = upsert_entities(
        client,
        collection_name=config.collection_name,
        entities=entities,
    )
    current_versions = publish_current_versions(config.object_store_dir, docs)
    print(f"Archived documents: {len(docs)}")
    print(f"Rebuilt chunks: {upserted}")
    print(f"Published current versions: {current_versions}")
    print(f"Collection: {config.collection_name}")


if __name__ == "__main__":
    main()
